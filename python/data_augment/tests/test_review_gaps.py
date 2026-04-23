"""Regression tests for issues surfaced by the harsh code review.

Covers:
- noise reproducibility across repeated augment() calls with a fixed seed
- short-signal graceful handling (no crash on clips shorter than one pyin frame)
- chained length-changing augmentations (pitch → speed, pitch → stretch)
- APF strict config validation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from data_augment import Augmenter

if TYPE_CHECKING:
    from numpy.typing import NDArray

SR = 16000


def _voice_like(f0: float, duration_s: float = 1.0, n_harmonics: int = 10) -> NDArray[np.float64]:
    t = np.arange(int(duration_s * SR)) / SR
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, n_harmonics + 1))
    return np.asarray(0.3 * x / np.max(np.abs(x)), dtype=np.float64)


# Gap #5 — noise reproducibility


def test_noise_reproducible_across_calls_with_fixed_seed() -> None:
    """Two calls on the same Augmenter with seed=X must produce byte-identical noise."""
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=42)
    spec = [("noise", {"snr_db": 20.0, "kind": "white"})]
    a = aug.augment(x, spec)[0]
    b = aug.augment(x, spec)[0]
    assert isinstance(a, np.ndarray)
    assert isinstance(b, np.ndarray)
    np.testing.assert_array_equal(a, b)


def test_noise_reproducible_across_instances_with_same_seed() -> None:
    """Two Augmenter instances with the same seed must produce the same noise output."""
    x = _voice_like(180.0)
    spec = [("noise", {"snr_db": 20.0, "kind": "pink"})]
    a = Augmenter(sample_rate=SR, seed=7).augment(x, spec)[0]
    b = Augmenter(sample_rate=SR, seed=7).augment(x, spec)[0]
    assert isinstance(a, np.ndarray)
    assert isinstance(b, np.ndarray)
    np.testing.assert_array_equal(a, b)


def test_noise_differs_across_inputs_in_same_call() -> None:
    """Within one augment() call, each input gets its own noise draw (RNG advances)."""
    x1 = _voice_like(180.0)
    x2 = _voice_like(180.0)  # same waveform
    aug = Augmenter(sample_rate=SR, seed=42)
    out = aug.augment([x1, x2], [("noise", {"snr_db": 20.0, "kind": "white"})])
    assert isinstance(out[0], np.ndarray)
    assert isinstance(out[1], np.ndarray)
    # Same source signal, same noise spec → must still differ (different noise draws).
    assert not np.array_equal(out[0], out[1])


# Gap #6 — short signals


@pytest.mark.parametrize("n", [64, 256, 512, 1023])  # all strictly below default 1024 frame_length
@pytest.mark.parametrize("op", ["pitch", "speed", "stretch", "gain", "noise", "phase_flip", "apf"])
def test_short_signal_does_not_crash(n: int, op: str) -> None:
    """Clips shorter than one pyin frame must pass through every augmentation cleanly."""
    rng = np.random.default_rng(0)
    x = 0.3 * rng.standard_normal(n).astype(np.float64)

    default_configs: dict[str, dict[str, object]] = {
        "pitch": {"semitones": 2.0},
        "speed": {"factor": 1.1},
        "stretch": {"factor": 1.1},
        "gain": {"db": 3.0},
        "noise": {"snr_db": 20.0, "kind": "white"},
        "phase_flip": {"flip": True},
        "apf": {},
    }
    aug = Augmenter(sample_rate=SR, seed=0)
    out = aug.augment(x, [(op, default_configs[op])])
    assert isinstance(out, list)
    y = out[0]
    assert isinstance(y, np.ndarray)
    # Augmenter always re-aligns to input length.
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


# Gap #7 — chained length-changing augmentations


def test_chain_pitch_then_speed_preserves_input_length() -> None:
    """Pitch (length-preserving) → speed (length-changing) → realigned back to input length."""
    x = _voice_like(180.0, duration_s=1.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    out = aug.augment(x, [("pitch", {"semitones": 2.0}), ("speed", {"factor": 1.25})])
    assert isinstance(out, list)
    y = out[0]
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape


def test_chain_pitch_then_stretch_preserves_input_length() -> None:
    x = _voice_like(180.0, duration_s=1.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    out = aug.augment(x, [("pitch", {"semitones": -2.0}), ("stretch", {"factor": 0.8})])
    assert isinstance(out, list)
    y = out[0]
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape


def test_chain_speed_then_pitch_handles_variable_intermediate_length() -> None:
    """Speed shortens the signal mid-chain; pitch runs on the shortened signal and must not crash."""
    x = _voice_like(180.0, duration_s=1.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    out = aug.augment(x, [("speed", {"factor": 1.5}), ("pitch", {"semitones": 2.0})])
    assert isinstance(out, list)
    y = out[0]
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


# Bonus — APF strict config validation (fix #3)


def test_apf_rejects_unknown_config_keys() -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    with pytest.raises(ValueError, match="Unknown apf config keys"):
        aug.augment(x, [("apf", {"sections": [{"order": 1, "freq_hz": 1000.0}], "flag": [True, False]})])
