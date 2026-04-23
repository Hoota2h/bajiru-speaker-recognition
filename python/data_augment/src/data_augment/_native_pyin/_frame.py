"""Framing and FFT autocorrelation helpers for native pyin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.fft

if TYPE_CHECKING:
    from numpy.typing import NDArray


def frame(y: NDArray[np.float64], frame_length: int, hop_length: int) -> NDArray[np.float64]:
    """Strided framing along the last axis → shape (frame_length, n_frames)."""
    n_frames = 1 + (y.shape[-1] - frame_length) // hop_length
    stride = y.strides[-1]
    framed: NDArray[np.float64] = np.lib.stride_tricks.as_strided(
        y,
        shape=(frame_length, n_frames),
        strides=(stride, hop_length * stride),
        writeable=False,
    )
    return framed


def autocorrelate(y_frames: NDArray[np.float64], max_size: int) -> NDArray[np.float64]:
    """FFT-based autocorrelation along axis -2, truncated to max_size."""
    n = y_frames.shape[-2]
    max_size = min(max_size, n)
    n_pad = scipy.fft.next_fast_len(2 * n - 1, real=True)
    powspec = np.abs(scipy.fft.rfft(y_frames, n=n_pad, axis=-2)) ** 2
    autocorr = scipy.fft.irfft(powspec, n=n_pad, axis=-2)
    result: NDArray[np.float64] = autocorr[..., :max_size, :]
    return result
