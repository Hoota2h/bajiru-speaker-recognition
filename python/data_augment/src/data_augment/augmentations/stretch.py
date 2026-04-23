"""PSOLA time-scale — changes duration, preserves pitch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from data_augment.pitch_detect import detect_pitch_periods
from data_augment.psola import psola_stretch
from data_augment.resample import sinc_downsample, sinc_upsample

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from data_augment.augmentations._common import AugmentContext


def apply_stretch(
    signal: NDArray[np.float64],
    config: dict[str, Any],
    ctx: AugmentContext,
) -> NDArray[np.float64]:
    """Time-stretch by ``factor`` (deterministic), preserving pitch."""
    factor = float(config.get("factor", 1.0))

    if factor <= 0:
        msg = f"stretch factor must be positive, got {factor}"
        raise ValueError(msg)

    if factor == 1.0:
        return signal.copy()

    upsampled = sinc_upsample(signal, ctx.oversample_factor, ctx.sinc_kernel_size)
    oversampled_rate = ctx.sample_rate * ctx.oversample_factor
    marks = detect_pitch_periods(upsampled, oversampled_rate)
    stretched = psola_stretch(upsampled, marks, factor)
    downsampled: NDArray[np.float64] = sinc_downsample(stretched, ctx.oversample_factor, ctx.sinc_kernel_size)
    return downsampled
