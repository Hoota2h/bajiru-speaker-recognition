"""Sinc resampling with Kaiser-windowed kernel.

This helps with reconstruction quality (over speed, I wouldn't do this if realtime).
Default kernel is 4097 taps (odd, perfectly symmetric) which gives approx. 200 dB stopband
attenuation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _kaiser_sinc_kernel(num_taps: int, cutoff: float, beta: float = 14.0) -> NDArray[np.float64]:
    """Symmetric Kaiser-windowed sinc lowpass kernel.

    Args:
        num_taps: Filter length. Should be odd so the kernel has a well-defined
            center tap; an even value is accepted but gives a fractional-sample
            delay.
        cutoff: Cutoff normalized to Nyquist (0..1). Passband is 0 to cutoff/2
            cycles/sample.
        beta: Kaiser window shape parameter. 14 → ~130 dB sidelobe suppression.

    """
    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.sin(np.pi * cutoff * n) / (np.pi * n)
    # Fix the center tap (limit as n→0 of sinc is cutoff). Only exact for odd num_taps.
    if num_taps % 2 == 1:
        kernel[num_taps // 2] = cutoff
    kernel *= np.kaiser(num_taps, beta)
    kernel /= np.sum(kernel)
    return kernel


def sinc_upsample(signal: NDArray[np.float64], factor: int, num_taps: int = 4097) -> NDArray[np.float64]:
    """Upsample by integer factor with sinc interpolation.

    Zero-stuffs then applies a Kaiser-windowed sinc lowpass to reconstruct the
    inter-sample values. Edges are reflection-padded so the filter sees a
    continuation of the signal (avoids the zero→signal step that would otherwise
    appear as broadband transients at boundaries).

    Args:
        signal: 1-D audio signal.
        factor: Upsample ratio (e.g., 4 for 16kHz → 64kHz).
        num_taps: Sinc kernel length (should be odd).

    """
    if factor == 1:
        return signal.copy()

    upsampled = np.zeros(len(signal) * factor, dtype=np.float64)
    upsampled[::factor] = signal

    # Lowpass at original Nyquist, normalized to the oversampled rate's Nyquist.
    kernel = _kaiser_sinc_kernel(num_taps, cutoff=1.0 / factor)

    pad = num_taps // 2
    padded = np.pad(upsampled, pad, mode="reflect")
    filtered = np.convolve(padded, kernel, mode="same")

    filtered = filtered[pad : pad + len(upsampled)]
    return filtered * factor


def sinc_downsample(signal: NDArray[np.float64], factor: int, num_taps: int = 4097) -> NDArray[np.float64]:
    """Downsample by integer factor with sinc anti-alias filtering.

    Applies a Kaiser-windowed sinc lowpass at the target Nyquist, then decimates.

    Args:
        signal: 1-D audio signal (at ``factor * target_rate``).
        factor: Downsample ratio.
        num_taps: Sinc kernel length (should be odd).

    """
    if factor == 1:
        return signal.copy()

    kernel = _kaiser_sinc_kernel(num_taps, cutoff=1.0 / factor)

    pad = num_taps // 2
    padded = np.pad(signal, pad, mode="reflect")
    filtered = np.convolve(padded, kernel, mode="same")
    filtered = filtered[pad : pad + len(signal)]

    return filtered[::factor]
