"""All augmentation handlers. Each is ``apply_X(signal, config, ctx) -> signal``.

Config values are scalars: the pipeline expands ``list`` values into discrete
modes and feeds each concrete config to the corresponding handler one at a time
(see :func:`data_augment.augmentations._common.expand_config`).
"""

from __future__ import annotations

from data_augment.augmentations._common import AugmentContext, expand_config, expand_modes
from data_augment.augmentations.apf import apply_apf
from data_augment.augmentations.gain import apply_gain
from data_augment.augmentations.noise import apply_noise
from data_augment.augmentations.phase_flip import apply_phase_flip
from data_augment.augmentations.pitch import apply_pitch
from data_augment.augmentations.speed import apply_speed
from data_augment.augmentations.stretch import apply_stretch

__all__ = [
    "AugmentContext",
    "apply_apf",
    "apply_gain",
    "apply_noise",
    "apply_phase_flip",
    "apply_pitch",
    "apply_speed",
    "apply_stretch",
    "expand_config",
    "expand_modes",
]
