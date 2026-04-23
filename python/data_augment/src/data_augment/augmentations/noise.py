"""Additive colored-noise augmentation at a given SNR (single-mode per call)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from data_augment.augmentations._common import AugmentContext

NoiseKind = Literal["white", "pink", "brown"]
_EPS = 1e-20

_SHAPERS: dict[NoiseKind, Callable[[np.ndarray], np.ndarray]] = {
    "pink": lambda f: 1.0 / np.sqrt(f),
    "brown": lambda f: 1.0 / f,
}


def _generate_colored_noise(n: int, kind: NoiseKind, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate `n` samples of colored noise via FFT spectral shaping."""
    white: NDArray[np.float64] = rng.standard_normal(n).astype(np.float64)
    if kind == "white":
        return white

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = _EPS
    shape = _SHAPERS[kind](freqs)
    shape[0] = 0.0
    shaped: NDArray[np.float64] = np.fft.irfft(spectrum * shape, n=n)
    return shaped


def apply_noise(
    signal: NDArray[np.float64],
    config: dict[str, Any],
    ctx: AugmentContext,
) -> NDArray[np.float64]:
    """Add colored noise at ``snr_db``. Silent inputs are returned unchanged."""
    snr_db = float(config.get("snr_db", 20.0))
    kind: NoiseKind = config.get("kind", "white")

    signal_rms = float(np.sqrt(np.mean(signal**2)))
    if signal_rms < _EPS:
        return signal.copy()

    noise = _generate_colored_noise(len(signal), kind, ctx.rng)
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    if noise_rms < _EPS:
        return signal.copy()

    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    noise *= target_noise_rms / noise_rms
    return np.asarray(signal + noise, dtype=np.float64)
