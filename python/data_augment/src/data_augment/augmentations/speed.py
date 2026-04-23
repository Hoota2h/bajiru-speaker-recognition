"""Resample the signal — changes duration AND pitch together."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.signal

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from data_augment.augmentations._common import AugmentContext

_MAX_RATIONAL_DENOM = 1000


def apply_speed(
    signal: NDArray[np.float64],
    config: dict[str, Any],
    ctx: AugmentContext,  # noqa: ARG001 — see gain.py for why ctx stays in the signature
) -> NDArray[np.float64]:
    """Resample by ``factor``. >1 → shorter+higher-pitch; <1 → longer+lower-pitch."""
    factor = float(config.get("factor", 1.0))

    if factor <= 0:
        msg = f"speed factor must be positive, got {factor}"
        raise ValueError(msg)

    if factor == 1.0:
        return signal.copy()

    ratio = Fraction(1.0 / factor).limit_denominator(_MAX_RATIONAL_DENOM)
    resampled: NDArray[np.float64] = scipy.signal.resample_poly(signal, ratio.numerator, ratio.denominator)
    return resampled.astype(np.float64, copy=False)
