"""Sign inversion, deterministic per mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from data_augment.augmentations._common import AugmentContext


def apply_phase_flip(
    signal: NDArray[np.float64],
    config: dict[str, Any],
    ctx: AugmentContext,  # noqa: ARG001 — see gain.py for why ctx stays in the signature
) -> NDArray[np.float64]:
    """Flip the signal's sign iff ``flip`` is True (a boolean-valued dimension)."""
    flip = bool(config.get("flip", False))
    if flip:
        return np.asarray(-signal, dtype=np.float64)
    return signal.copy()
