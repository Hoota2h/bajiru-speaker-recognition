"""Shared context + combinatorial-mode expansion used by every augmentation handler."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class AugmentContext:
    """Augmenter-level settings passed through to each handler."""

    rng: np.random.Generator
    sample_rate: int
    oversample_factor: int
    sinc_kernel_size: int


def expand_modes[T](value: T | list[T]) -> list[T]:
    """Normalize a config value into a list of discrete modes.

    A scalar becomes a one-element list; a list passes through unchanged.
    """
    if isinstance(value, list):
        return list(value)
    return [value]


def expand_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian-product expansion of a single augmentation's config dict.

    Each entry whose value is a list becomes a dimension with that many modes.
    Returns one scalar-valued dict per combination, in deterministic (insertion,
    then positional) order.
    """
    if not config:
        return [{}]
    keys = list(config.keys())
    dimensions = [expand_modes(config[k]) for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*dimensions)]
