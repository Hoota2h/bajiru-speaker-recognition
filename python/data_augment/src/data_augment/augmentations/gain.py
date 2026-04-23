"""Linear dB gain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from data_augment.augmentations._common import AugmentContext


def apply_gain(
    signal: NDArray[np.float64],
    config: dict[str, Any],
    ctx: AugmentContext,  # noqa: ARG001 — see comment below on the uniform handler signature
) -> NDArray[np.float64]:
    """Scale by a dB amount (positive = louder).

    Keeping the uniform signature means the dispatcher doesn't special-case anything,
    and adding a new augmentation that *does* need ``ctx`` later is a zero-API-change edit.
    The ``noqa`` silences ruff's unused-argument lint specifically for this file's
    parameter list; don't remove ``ctx`` itself.

    TLDR: I didn't want to refactor the dispatcher logic, "lost the battle win the war" type shih.
    """
    db = float(config.get("db", 0.0))
    factor = 10 ** (db / 20.0)
    return np.asarray(signal * factor, dtype=np.float64)
