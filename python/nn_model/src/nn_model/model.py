"""Audio classifier"""

import math

import librosa
import torch
import torch.nn.functional as tf
from torch import nn

torch.set_float32_matmul_precision("high")


class AudioClassifier(nn.Module):
    """Classifies an audio fragment and returns the probabilities of belonging to a particular class (scores)"""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        n_history: int,
        hop_length: int = 256,
        n_convs: int = 6,
        n_channels: int = 256,
        n_attns: int = 2,
        d_model: int = 128,
        n_heads: int = 48,
        n_classes: int = 4,
        n_hidden: int = 256,
        log_eps: float = 1e-6,
    ):
        """Create AudioClassifier

        Args:
            sample_rate (int): Sample rate
            n_fft (int): Size of fft window
            n_mels (int): Number of mels
            n_history (int): Max number of history frames
            hop_length (int): Hop length
            n_convs (int): Number of convolution blocks
            n_channels (int): Number of intermediate convolution channels
            n_attns (int): Number of attention blocks
            d_model (int): The size of the attention block embedding
            n_heads (int): Number of attention block heads
            n_classes (int): Number of output scores
            n_hidden (int): Classifier's hidden layer size
            log_eps (float, optional): The epsilon used for calucations

        """
        super().__init__()
        self.log_eps = log_eps

        self.mel_spec = MelExtractor(
            sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length
        )

        self.compressor = MelCompressor(
            threshold=-54,
            ratio=8,
            knee=40,
            attack_frames=(1 / 1000) * (sample_rate / hop_length),  # 1ms attack
            release_frames=(40 / 1000) * (sample_rate / hop_length),  # 40ms release
            makeup=30,
            eps=log_eps,
        )

        self.encoder = AudioEncoder(
            n_mels,
            conv_layers=n_convs,
            conv_channels=n_channels,
            trans_dim=d_model,
            trans_heads=n_heads,
            trans_layers=n_attns,
            n_history=n_history,
        )
        self.classifier = ClassifierHead(
            d_model, n_history, n_classes, n_inner=n_hidden
        )

    def _prepare_mels(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mel_spec(x)  # (B,mels,frames)

        x = x.transpose(2, 1)  # (B,frames,mels)
        x = self.compressor(x)  # (B,mels,frames)
        x = x.transpose(2, 1)  # (B,mels,frames)
        return torch.log(x + self.log_eps)  # log-mel

    def forward_train(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Train the model

        Args:
            x (torch.Tensor): mono audio (B, n_samples); (float32; range [-1.0;1.0])

        Returns:
            tuple[torch.Tensor, torch.Tensor]: scores,logits (B, n_classes); (float32; range [0.0;1.0])

        """
        x = self._prepare_mels(x)  # (B,mels,frames)
        x = self.encoder(x)  # (B,d_model,frames)
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input

        Args:
            x (torch.Tensor): mono audio segment (n_samples,); (float32; range [-1.0;1.0])

        Returns:
            torch.Tensor: scores (n_classes,)

        """
        x = x.unsqueeze(0)  # (1,n_samples)
        x = self._prepare_mels(x)  # (1,mels,frames)
        x = self.encoder(x)  # (1,d_model,frames)
        x, _ = self.classifier(x)  # (1,n_classes)
        return x.squeeze(0)  # (n_classes,)


class MelExtractor(nn.Module):
    """Extracts a sequence of mels out of the audio segment"""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        hop_length: int,
    ):
        """Create MelExtractor

        Args:
            sample_rate (int): Sample rate
            n_fft (int): Size of fft window
            n_mels (int): Number of mels
            hop_length (int): Hop length

        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq = n_fft // 2 + 1

        mel_fb = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
        )
        self.register_buffer("mel_fb_t", torch.tensor(mel_fb).transpose(0, 1))

        hann_window = torch.hann_window(n_fft, periodic=True)
        self.register_buffer("window", hann_window)

        n = torch.arange(n_fft)
        k = torch.arange(self.n_freq).unsqueeze(1)
        a = -2.0 * math.pi * k * n / float(n_fft)  # (n_freq, n_fft)
        real_basis = torch.cos(a) * hann_window.unsqueeze(0)
        imag_basis = torch.sin(a) * hann_window.unsqueeze(0)

        forward_basis = torch.cat([real_basis, imag_basis], dim=0).unsqueeze(1)
        stft_conv = torch.nn.Conv1d(
            1, forward_basis.shape[0], kernel_size=n_fft, stride=hop_length, bias=False
        )
        stft_conv.weight.data.copy_(forward_basis)
        stft_conv.weight.requires_grad_(mode=False)
        self.stft_conv = stft_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input

        Args:
            x (torch.Tensor): mono audio segment (B, n_samples)

        Returns:
            torch.Tensor: mel frames (B, n_mels, n_frames)

        """
        x = x.unsqueeze(1)  # (B, 1, n_samples)
        x = self.stft_conv(x)  # (B, 2*n_freq, n_frames)
        real, imag = x.split(self.n_freq, dim=1)
        x = real.pow(2) + imag.pow(2)

        x = x.transpose(2, 1)
        x = torch.matmul(x, self.mel_fb_t)  # ty:ignore[invalid-argument-type]
        return x.transpose(2, 1)


class MelCompressor(nn.Module):
    """Compressor operating on mel frames"""

    def __init__(
        self,
        threshold: float = -30.0,
        ratio: float = 4.0,
        knee: float = 6.0,
        attack_frames: float = 1.0,
        release_frames: float = 10.0,
        makeup: float = 0.0,
        eps: float = 1e-6,
    ):
        """Create MelCompressor

        Args:
            threshold (float, optional): Input level threshold in db. Defaults to -30.0.
            ratio (float, optional): Compression ratio. Defaults to 4.0.
            knee (float, optional): Soft knee width in db. Defaults to 6.0.
            attack_frames (float, optional): Attack duration (number of frames). Defaults to 1.0.
            release_frames (float, optional): Release duration (number of frames). Defaults to 10.0.
            makeup (float, optional): Additional gain. Defaults to 0.0.
            eps (float, optional): Epsilon used for linear/db conversion. Defaults to 1e-6.

        """
        super().__init__()
        self.threshold = threshold
        self.ratio = ratio
        self.half_knee = knee / 2.0
        self.eps = eps

        def _calc_coef(tau) -> float:
            return math.exp(-1.0 / max(1e-6, tau))

        self.register_buffer(
            "_attack_coef",
            torch.tensor(_calc_coef(attack_frames)),
            persistent=False,
        )
        self.register_buffer(
            "_release_coef",
            torch.tensor(_calc_coef(release_frames)),
            persistent=False,
        )
        self.register_buffer(
            "_makeup",
            self._db_to_linear(torch.tensor(makeup)),
            persistent=False,
        )

    def _linear_to_db(self, x: torch.Tensor) -> torch.Tensor:
        return 10.0 * torch.log10(x.clamp(min=self.eps))

    def _db_to_linear(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(10.0, x / 10.0)

    def _target_level(self, level: torch.Tensor) -> torch.Tensor:
        level = 1.0 + (level - 1.0).sum(
            dim=-1
        )  # (n_frames,n_mels) -> (n_frames,); energies sum
        level = self._linear_to_db(level)
        over_thr = level - self.threshold

        coef = 1.0 / self.ratio - 1.0

        out = torch.zeros_like(over_thr)
        out = torch.where(over_thr >= self.half_knee, over_thr * coef, out)

        o_mid = over_thr + self.half_knee
        out = torch.where(
            ~((over_thr <= -self.half_knee) | (over_thr >= self.half_knee)),
            coef * (o_mid * o_mid) / (4.0 * self.half_knee),
            out,
        )
        return self._db_to_linear(out)

    def _apply_level(self, inp: torch.Tensor, lvl: torch.Tensor) -> torch.Tensor:
        return inp * lvl.unsqueeze(-1) * self._makeup  # ty:ignore[unsupported-operator]

    def forward(self, x) -> torch.Tensor:
        n_batches, n_frames, _ = x.shape
        tgt_lvl = self._target_level(x)  # (B,n_frames)

        out_lvl = torch.empty_like(tgt_lvl)
        lvl = torch.ones((n_batches,), device=tgt_lvl.device, dtype=tgt_lvl.dtype)
        for frame in range(n_frames):
            tgt = tgt_lvl[:, frame]  # (B,)

            coef: torch.Tensor = torch.where(
                tgt < lvl, self._attack_coef, self._release_coef
            )  # ty:ignore[no-matching-overload]

            lvl = torch.lerp(tgt, lvl, coef)
            out_lvl[:, frame] = lvl

        return self._apply_level(x, out_lvl)


class ConvBlock(nn.Module):
    """Casual convolution block"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        """Create ConvBlock

        Args:
            in_ch (int): Number of input channels
            out_ch (int): Number of output channels
            kernel (int, optional): Kernel size
            dilation (int, optional): Dilation

        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel, padding=0, dilation=dilation
        )
        self.norm = nn.LayerNorm(out_ch)
        self.pad = (kernel - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = tf.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = tf.silu(x)
        return x.transpose(1, 2)


class AttnBlock(nn.Module):
    """Casual attention block"""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        """Create AttnBlock

        Args:
            d_model (int): Total dimension of the model.
            n_heads (int): Number of parallel attention heads.
            dropout (float): dropout: Dropout probability on MultiheadAttention output.

        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _causal_mask(self, seq_len: int, device: torch.device):
        return torch.triu(
            torch.full([seq_len, seq_len], -torch.inf, device=device),
            diagonal=1,
        )

    def _attn(
        self, q: torch.Tensor, kv: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.attn(
            q,
            kv,
            kv,
            attn_mask=attn_mask,
            need_weights=False,
            is_causal=attn_mask is not None,
        )[0]

    def forward(self, x: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        x = x + pos_enc
        attn_mask = self._causal_mask(x.shape[1], x.device)
        x = self.norm1(x + self._attn(x, x, attn_mask))
        return self.norm2(x + self.feedforward(x))


class AudioEncoder(nn.Module):
    """Audio encoder module"""

    def __init__(
        self,
        n_mels: int,
        n_history: int,
        conv_channels: int = 64,
        conv_layers: int = 2,
        trans_dim: int = 128,
        trans_heads: int = 4,
        trans_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Create AudioEncoder

        Args:
            n_mels (int): Number of mels
            n_history (int): Max number of history frames
            conv_channels (int, optional): Number of intermediate convolution channels
            conv_layers (int, optional): Number of convolution blocks
            trans_dim (int, optional): The size of the attention block embedding
            trans_heads (int, optional): Number of attention block heads
            trans_layers (int, optional): Number of attention blocks
            dropout (float, optional): The dropout value

        """
        self.conv_proj = nn.Conv1d(n_mels, conv_channels, 1)
        self.convs = nn.ModuleList(
            [
                ConvBlock(conv_channels, conv_channels, kernel=3, dilation=2**i)
                for i in range(conv_layers)
            ]
        )
        if conv_channels != trans_dim:
            self.trans_proj = nn.Conv1d(conv_channels, trans_dim, 1)
        self.attns = nn.ModuleList(
            [
                AttnBlock(
                    d_model=trans_dim,
                    n_heads=trans_heads,
                    dropout=dropout,
                )
                for _ in range(trans_layers)
            ]
        )

        dim = torch.arange(0, trans_dim, 2)
        inv_freq = 1.0 / (10000 ** (dim / trans_dim))
        pos = torch.arange(0, n_history).unsqueeze(1)
        table = torch.zeros((n_history, trans_dim))
        table[:, 0::2] = torch.sin(pos * inv_freq)
        table[:, 1::2] = torch.cos(pos * inv_freq)
        self.register_buffer("_pos_table", table.unsqueeze(0).clone(), persistent=False)

    def causal_mask(self, seq_len: int, device: torch.device):
        return torch.triu(
            torch.full(
                [seq_len, seq_len], -torch.inf, dtype=torch.float32, device=device
            ),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input

        Args:
            x (torch.Tensor): mel frames (B, n_mels, n_frames)

        Returns:
            torch.Tensor: transformer encoder output (B, trans_dim, n_frames)

        """
        x = self.conv_proj(x)
        for c in self.convs:
            x = c(x) + x

        if hasattr(self, "trans_proj"):
            x = self.trans_proj(x)

        x = x.transpose(1, 2)
        for a in self.attns:
            x = a(x, self._pos_table) + x
        return x.transpose(1, 2)


class ClassifierHead(nn.Module):
    """Classifier head"""

    def __init__(self, n_channels: int, n_history: int, n_classes: int, n_inner: int):
        """Create ClassifierHead

        Args:
            n_channels (int): Number of input channels
            n_history (int): The size of the history
            n_classes (int): Number of output classes
            n_inner (int): The size of the hidden layer

        """
        super().__init__()
        self.conv = nn.Conv1d(
            n_channels, n_inner, kernel_size=n_history, padding=0, bias=True
        )
        self.norm = nn.LayerNorm(n_inner)
        self.head = nn.Linear(n_inner, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)  # (B, n_inner, 1)
        x = x.squeeze(-1)  # (B, n_inner)
        x = self.norm(x)
        x = tf.silu(x)
        logits = self.head(x)  # (B, n_classes)
        scores = torch.sigmoid(logits)
        return scores, logits
