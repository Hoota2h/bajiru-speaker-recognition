"""Default augmentation chain — single source of truth.

Edit this file to tune the out-of-the-box behavior of
``Augmenter.augment(data)`` when no ``augmentations`` argument is passed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_augment.core import AugmentationSpec

DEFAULT_AUGMENTATIONS: list[AugmentationSpec] = [
    ("pitch", {"semitones": [-2.0, -1.0, 0.0, 1.0, 2.0]}),
    ("noise", {"snr_db": [20.0, 30.0], "kind": ["white", "pink"]}),
    ("phase_flip", {"flip": [False, True]}),
]

# The default chain gives 5 x 2 x 2 = 20 possible augmentations, including the identity (no change).
