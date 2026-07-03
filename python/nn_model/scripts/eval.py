import argparse
import time
from functools import wraps

import model_preset
import numpy as np
import scipy.io.wavfile as wav
import torch
import torch.nn.functional as tf
from torch.utils.data import DataLoader, Dataset

torch.backends.nnpack.set_flags(False)


class AudioDataset(Dataset):
    """Dataset for evaluation"""

    def __init__(
        self,
        audio: np.ndarray,
        segment_samples: int,
        hop_length: int,
        win_length: int,
        device: str | torch.device,
        audio_mult: float = 1.0,
    ):
        """Create AudioDataset

        Args:
            audio (np.ndarray): Audio data
            segment_samples (int): The size of the output audio segment
            hop_length (int): Audio data step interval
            win_length (int): The size of fft window
            device (str | torch.device): Device to store the data
            audio_mult (float, optional): Multiplier to adjust the audio to a range of -1.0:1.0

        """
        self.segment_samples = segment_samples
        self.hop = hop_length
        self.win = win_length
        self.mult = audio_mult

        total = audio.shape[0]
        pad_left = (total - self.win) % self.hop
        padded = np.pad(audio, (pad_left, 0), mode="reflect")
        self.audio = torch.as_tensor(padded, device=device)
        total += pad_left

        self.sample_count = ((total - self.win) // self.hop) + 1

    def __len__(self) -> int:
        """Return the number of items

        Returns:
            int: Number of items in the dataset

        """
        return self.sample_count

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the audio slice by index

        Args:
            index (int): Index of the slize

        Returns:
            torch.Tensor: Audio slice

        """
        index = index * self.hop
        pad = self.segment_samples - self.win
        if pad > index:
            seg = tf.pad(
                self.audio[: index + self.win],
                (pad - index, 0),
                mode="constant",
            )
        else:
            seg = self.audio[index - pad : index + self.win]
        return seg.float() * self.mult


def rate_logger(interval=1.0):
    def dec(func):
        count = 0
        t0 = time.perf_counter()

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal count, t0
            count += 1
            now = time.perf_counter()
            if now - t0 >= interval:
                print(f"{func.__name__}: {count / (now - t0):.2f} calls/sec")
                count = 0
                t0 = now
            return func(*args, **kwargs)

        return wrapper

    return dec


@rate_logger()
def run_model(
    model,
    device,
    scores_list,
    seg_batch,
):
    seg_batch = seg_batch.squeeze(0)
    seg_batch = seg_batch.to(device)
    scores = model(seg_batch)
    scores = scores.detach()
    scores = scores.unsqueeze(0)

    scores = scores.cpu().numpy()
    scores_list.append(scores)


def process_data(
    audio: np.ndarray,
    model: torch.nn.Module,
    device="cpu",
    segment_samples: int = 16000,
    hop_length: int = 256,
    win_length: int = 512,
    audio_mult: float = 1.0,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    ds = AudioDataset(
        audio, segment_samples, hop_length, win_length, device, audio_mult
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    scores_list = []
    with torch.no_grad():
        for seg_batch in loader:
            run_model(
                model,
                device,
                scores_list,
                seg_batch,
            )

    return np.concatenate(scores_list, axis=0)


device = "cuda" if torch.cuda.is_available() else "cpu"

segment_samples = model_preset.segment_samples
win_length = model_preset.win_length
hop_length = model_preset.hop_length
model = model_preset.model
model = model.to(device)
model.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("outputs", nargs=model_preset.n_classes)
    args = parser.parse_args()

    model.load_state_dict(torch.load(args.checkpoint)["model"])

    sample_rate, audio_data = wav.read(args.input)
    if sample_rate != model_preset.sample_rate:
        msg = f"Expected sample rate of {model_preset.sample_rate}"
        raise ValueError(msg)

    match audio_data.dtype.name:
        case "int16":
            audio_mult = 1.0 / 0x7FFF
        case "int32":
            audio_mult = 1.0 / 0x7FFFFFFF
        case "float32":
            audio_mult = 1.0
        case _:
            msg = f"Unexpected audio data type: {audio_data.dtype}"
            raise TypeError(msg)
    scores = process_data(
        audio_data,
        model,
        device=device,
        segment_samples=segment_samples,
        hop_length=hop_length,
        win_length=win_length,
        audio_mult=audio_mult,
    )

    out_sample_rate = sample_rate // hop_length
    for i in range(model_preset.n_classes):
        wav.write(args.outputs[i], out_sample_rate, scores[:, i])
