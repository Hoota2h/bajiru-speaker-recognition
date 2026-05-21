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
        conv_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_classes: int = 4,
        dropout: float = 0.1,
    ):
        """Create AudioClassifier

        Args:
            sample_rate (int): Sample rate
            n_fft (int): Size of fft window
            n_mels (int): Number of mels
            n_history (int): Max number of history frames
            hop_length (int): Hop length
            conv_channels (int): Number of intermediate conv channels
            d_model (int): The size of the embedding
            n_heads (int): Number of encoder heads
            n_layers (int): Number of encoder layers
            n_classes (int): Number of output scores
            dropout (float, optional): The dropout value

        """
        super().__init__()

        self.ext = MelExtractor(sample_rate, n_fft, n_mels, hop_length)
        self.prep = AudioPreprocessor(sample_rate, n_mels, hop_length)

        self.conv1 = Conv1dBlock(n_mels, conv_channels, kernel=3, dilation=1)
        self.conv2 = Conv1dBlock(conv_channels, d_model, kernel=3, dilation=2)

        self.encoder = AudioEncoder(
            n_mels, n_history, conv_channels, d_model, n_heads, n_layers, dropout
        )

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input

        Args:
            x (torch.Tensor): mono audio segment (B, n_samples); (float32; range [-1.0;1.0])

        Returns:
            torch.Tensor: scores (B, n_classes)

        """
        mel_power = self.ext(x)  # (B, n_mels, n_frames)

        mel_power = self.prep(mel_power)

        log_mel = torch.log1p(mel_power)

        x = self.encoder(log_mel)  # (B, n_frames, d_model)

        x = x[:, -1, :]  # (B, d_model)
        x = self.norm(x)
        return self.head(x)  # (B, n_classes)


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
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.register_buffer("hann_window", torch.hann_window(n_fft), persistent=False)
        mel_fb = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
        )
        self.register_buffer("mel_fb", torch.tensor(mel_fb, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input

        Args:
            x (torch.Tensor): mono audio segment (B, n_samples)

        Returns:
            torch.Tensor: mel frames (B, n_mels, n_frames)

        """
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.hann_window,  # ty:ignore[invalid-argument-type]
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        x = torch.abs(x) ** 2

        return torch.matmul(self.mel_fb, x)  # ty:ignore[invalid-argument-type]


class Conv1dBlock(nn.Module):
    """Casual convolution block"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        """Create Conv1dBlock

        Args:
            in_ch (int): Number of input channels
            out_ch (int): Number of output channels
            kernel (int, optional): Kernel size
            dilation (int, optional): Dilation

        """
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel, padding=0, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = tf.pad(x, (self.pad, 0))
        x = self.conv(x)
        return self.act(self.bn(x))


class AudioEncoder(nn.Module):
    """Audio encoder module"""

    def __init__(
        self,
        n_mels: int,
        n_history: int,
        conv_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout=0.1,
    ):
        """Create AudioEncoder

        Args:
            n_mels (int): Number of mels
            n_history (int): Max number of history frames
            conv_channels (int, optional): Number of intermediate conv channels
            d_model (int, optional): The size of the embedding
            n_heads (int, optional): Number of encoder heads
            n_layers (int, optional): Number of encoder layers
            dropout (float, optional): The dropout value

        """
        super().__init__()

        self.conv1 = Conv1dBlock(n_mels, conv_channels, kernel=3, dilation=1)
        self.conv2 = Conv1dBlock(conv_channels, d_model, kernel=3, dilation=2)

        self.pos_emb = nn.Embedding(n_history, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation=tf.gelu,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

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
            torch.Tensor: transformer encoder output (B, n_frames, d_model)

        """
        x = self.conv1(x)  # (B, conv_channels, n_frames)
        x = self.conv2(x)  # (B, d_model, n_frames)
        x = x.permute(0, 2, 1)  # (B, n_frames, d_model)

        pos = self.pos_emb(
            torch.arange(x.shape[1], device=x.device)
            .unsqueeze(0)
            .expand(x.shape[0], -1)
        )
        x = x + pos

        src_mask = self.causal_mask(x.shape[1], x.device)
        return self.transformer(x, mask=src_mask, is_causal=True)


class AudioPreprocessor(nn.Module):
    """Audio preprocessing module"""

    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        hop_length: int,
    ):
        """Create AudioPreprocessor

        Args:
            sample_rate (int): Sample rate
            n_mels (int): Number of mels
            hop_length (int): Hop length

        """
        super().__init__()

        self.log_mel_gain = nn.Parameter(torch.zeros(n_mels))

        self.compressor = TimeConstantCompressor(n_mels, sample_rate, hop_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input

        Args:
            x (torch.Tensor): mel frames (B, n_mels, n_frames)

        Returns:
            torch.Tensor: mel frames (B, n_mels, n_frames)

        """
        mel_gain = torch.exp(self.log_mel_gain).view(1, -1, 1)  # (1, n_mels, 1)
        x = x * mel_gain

        return self.compressor(x)


# AI-generated
class TimeConstantCompressor(nn.Module):
    """Audio compressor"""

    def __init__(
        self,
        n_mels: int,
        sample_rate: int,
        hop_length: int,
        tau=0.05,  # seconds
        threshold_db=-30.0,  # dB
        ratio=4.0,
        truncate_eps=1e-4,
    ):
        """Create TimeConstantCompressor

        Args:
            n_mels (int): Number of mels
            sample_rate (int): Sample rate
            hop_length (int): Hop length
            tau (float, optional): Tau
            threshold_db (float, optional): Threshold
            ratio (float, optional): Ratio
            truncate_eps (float, optional): Truncate epsilon

        """
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.truncate_eps = truncate_eps
        self.hop_time = float(hop_length) / sample_rate

        # store log(tau) to keep positivity via exp
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau), dtype=torch.float32))

        self.threshold_db = nn.Parameter(
            torch.tensor(threshold_db, dtype=torch.float32)
        )

        self.log_ratio = nn.Parameter(
            torch.tensor(math.log(ratio - 1.0), dtype=torch.float32)
        )

    def forward(self, mel_power: torch.Tensor) -> torch.Tensor:
        # mel_power: (B, M, T) power (|X|^2)
        batch_size, n_mels, n_frames = mel_power.shape
        device = mel_power.device
        dtype = mel_power.dtype

        # derive actual params (positive constraints)
        tau = torch.exp(self.log_tau)  # seconds
        alpha = torch.exp(-self.hop_time / tau).clamp(min=1e-8, max=0.999999)

        # compute truncation length L where alpha^L < truncate_eps
        # use scalar alpha (take item for math.log), but keep tensor-safe branch if alpha is tensor
        a_val = alpha.detach().cpu().item()
        if a_val <= 0.0:
            kernel_size = 1
        else:
            kernel_size = max(
                1, int(math.ceil(math.log(self.truncate_eps) / math.log(a_val)) * -1)
            )
            kernel_size = min(kernel_size, n_frames)

        # kernel h[n] = (1-alpha) * alpha^n for n=0..L-1
        n = torch.arange(0, kernel_size, device=device, dtype=dtype)
        h = (1.0 - alpha) * (alpha**n)  # (L,)

        # input for grouped conv: shape (N=1, C_in=B*M, T)
        x = mel_power.reshape(
            1, batch_size * n_mels, n_frames
        )  # channels = one channel per band across batch

        # expand kernel to (B*M, 1, L) so conv is grouped and parallel
        kernel = (
            h.view(1, 1, kernel_size)
            .expand(batch_size * n_mels, 1, kernel_size)
            .contiguous()
        )

        # pad left for causal conv
        pad = (kernel_size - 1, 0)
        smoothed = tf.conv1d(
            tf.pad(x, pad), kernel, groups=batch_size * n_mels
        )  # (B*M, 1, T)
        smoothed = smoothed.view(batch_size, n_mels, n_frames)

        # dB level from power
        eps = 1e-12
        level_db = 10.0 * torch.log10(smoothed + eps)

        # compute gain in dB (only negative reductions)
        ratio = 1.0 + torch.exp(self.log_ratio)  # >1
        target_db = self.threshold_db + (level_db - self.threshold_db) / ratio
        gain_db = torch.minimum(torch.zeros_like(level_db), target_db - level_db)

        # convert to linear amplitude gain, then to power gain (since mel_power is power)
        gain_lin = 10.0 ** (gain_db / 20.0)

        return (gain_lin**2) * mel_power
