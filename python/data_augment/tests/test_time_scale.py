"""Tests for speed and stretch augmentations (combinatorial API)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from data_augment import Augmenter, _native_pyin
from data_augment.augmentations import AugmentContext, apply_speed, apply_stretch

if TYPE_CHECKING:
    from numpy.typing import NDArray

SR = 16000


def _voice_like(f0: float, duration_s: float = 1.0, n_harmonics: int = 10) -> NDArray[np.float64]:
    t = np.arange(int(duration_s * SR)) / SR
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, n_harmonics + 1))
    return np.asarray(0.3 * x / np.max(np.abs(x)), dtype=np.float64)


def _detect_f0(signal: NDArray[np.float64]) -> float:
    f0, voiced, _ = _native_pyin.pyin(signal, fmin=50.0, fmax=800.0, sr=SR)
    voiced_f0 = f0[voiced]
    if len(voiced_f0) == 0:
        return float("nan")
    return float(np.median(voiced_f0))


def _ctx() -> AugmentContext:
    return AugmentContext(rng=np.random.default_rng(0), sample_rate=SR, oversample_factor=4, sinc_kernel_size=4097)


# Primitive-level: apply_speed / apply_stretch change length (pre-alignment).


@pytest.mark.parametrize("factor", [0.5, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0])
def test_apply_speed_changes_length(factor: float) -> None:
    x = _voice_like(180.0)
    y = apply_speed(x, {"factor": factor}, _ctx())
    assert isinstance(y, np.ndarray)
    expected_len = len(x) / factor
    assert abs(len(y) - expected_len) / expected_len < 0.01


@pytest.mark.parametrize("factor", [0.7, 0.85, 1.2, 1.5])
def test_apply_stretch_changes_length(factor: float) -> None:
    x = _voice_like(180.0)
    y = apply_stretch(x, {"factor": factor}, _ctx())
    assert isinstance(y, np.ndarray)
    expected_len = round(len(x) * factor)
    assert abs(len(y) - expected_len) <= 1


# Augmenter-level: the public API re-aligns outputs to input length.


@pytest.mark.parametrize("factor", [0.7, 0.85, 1.0, 1.2, 1.5])
def test_augment_speed_preserves_input_length(factor: float) -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("speed", {"factor": factor})])[0]
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape


@pytest.mark.parametrize("factor", [0.7, 0.85, 1.0, 1.2, 1.5])
def test_augment_stretch_preserves_input_length(factor: float) -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("stretch", {"factor": factor})])[0]
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape


# Content correctness (unaffected by length alignment since pyin skips silence).


@pytest.mark.parametrize("factor", [0.8, 0.9, 1.1, 1.2, 1.5])
def test_speed_scales_pitch_with_factor(factor: float) -> None:
    f0_in = 180.0
    x = _voice_like(f0_in, duration_s=2.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("speed", {"factor": factor})])[0]
    assert isinstance(y, np.ndarray)
    detected = _detect_f0(y)
    expected = f0_in * factor
    assert abs(detected - expected) / expected < 0.03


@pytest.mark.parametrize("factor", [0.7, 0.85, 1.2, 1.5])
def test_stretch_preserves_pitch(factor: float) -> None:
    f0_in = 180.0
    x = _voice_like(f0_in, duration_s=1.5)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("stretch", {"factor": factor})])[0]
    assert isinstance(y, np.ndarray)
    detected = _detect_f0(y)
    assert abs(detected - f0_in) / f0_in < 0.03


def test_speed_factor_1_is_copy() -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("speed", {"factor": 1.0})])[0]
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, x)


def test_stretch_factor_1_is_copy() -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("stretch", {"factor": 1.0})])[0]
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, x)


def test_speed_rejects_nonpositive_factor() -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    with pytest.raises(ValueError, match="speed factor"):
        aug.augment(x, [("speed", {"factor": 0.0})])
    with pytest.raises(ValueError, match="speed factor"):
        aug.augment(x, [("speed", {"factor": -1.0})])


def test_stretch_rejects_nonpositive_factor() -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    with pytest.raises(ValueError, match="stretch factor"):
        aug.augment(x, [("stretch", {"factor": 0.0})])
    with pytest.raises(ValueError, match="stretch factor"):
        aug.augment(x, [("stretch", {"factor": -1.0})])


def test_speed_list_produces_one_output_per_mode_all_same_length() -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    factors = [0.8, 1.0, 1.25]
    out = aug.augment(x, [("speed", {"factor": factors})])
    assert isinstance(out, list)
    assert len(out) == len(factors)
    for y in out:
        assert isinstance(y, np.ndarray)
        assert y.shape == x.shape
