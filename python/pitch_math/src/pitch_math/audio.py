"""Audio I/O: microphone recording and file loading."""

import logging

import librosa
import numpy as np
import sounddevice as sd

from pitch_math.config import SAMPLE_RATE

logger = logging.getLogger(__name__)


def record(
    duration_s: float,
    *,
    sample_rate: int = SAMPLE_RATE,
    device: int | str | None = None,
) -> np.ndarray:
    """Record audio from a microphone input device.

    Blocks until the full recording duration has elapsed.

    Args:
        duration_s: Recording length in seconds.
        sample_rate: Desired sample rate in Hz.
        device: PortAudio device index or name. ``None`` selects the system
            default input device.

    Returns:
        Float32 mono audio array of length ``duration_s * sample_rate``.

    """
    frames = int(sample_rate * duration_s)
    audio = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    logger.debug(
        "Recorded %.1fs at %d Hz (device=%s)",
        duration_s,
        sample_rate,
        device,
    )
    return audio.flatten()


def load_file(path: str, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and resample it to *sample_rate*.

    Args:
        path: Path to any audio format supported by librosa (MP3, WAV, FLAC…).
        sample_rate: Target sample rate in Hz.

    Returns:
        Float32 mono audio array resampled to *sample_rate*.

    """
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    logger.debug("Loaded %s — %d samples at %d Hz", path, len(audio), sample_rate)
    return audio
