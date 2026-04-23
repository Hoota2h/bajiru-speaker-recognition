"""PSOLA pitch shift — preserves length."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from data_augment.pitch_detect import detect_pitch_periods
from data_augment.psola import psola_shift
from data_augment.resample import sinc_downsample, sinc_upsample

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from data_augment.augmentations._common import AugmentContext

_SEMITONE_BASE = 2.0 ** (1.0 / 12.0)  # ratio = _SEMITONE_BASE ** semitones


def _semitones_to_ratio(semitones: float) -> float:
    return float(_SEMITONE_BASE**semitones)


def apply_pitch(
    signal: NDArray[np.float64],
    config: dict[str, Any],
    ctx: AugmentContext,
) -> NDArray[np.float64]:
    """PSOLA pitch shift: sinc upsample → detect pitch periods → TD-PSOLA → sinc downsample."""
    semitones = float(config.get("semitones", 0.0))
    if semitones == 0.0:
        return signal.copy()

    ratio = _semitones_to_ratio(semitones)
    upsampled = sinc_upsample(signal, ctx.oversample_factor, ctx.sinc_kernel_size)
    oversampled_rate = ctx.sample_rate * ctx.oversample_factor
    marks = detect_pitch_periods(upsampled, oversampled_rate)
    shifted = psola_shift(upsampled, marks, ratio)
    downsampled: NDArray[np.float64] = sinc_downsample(shifted, ctx.oversample_factor, ctx.sinc_kernel_size)
    return downsampled
