"""Batch / dataset / default-chain tests for Augmenter.augment.

The combinatorial API returns ``num_inputs x num_combinations`` augmented
signals per call; these tests verify that shape across the supported input
forms (1-D signal, 2-D array, list / tuple).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from data_augment import DEFAULT_AUGMENTATIONS, Augmenter

if TYPE_CHECKING:
    from numpy.typing import NDArray

SR = 16000


def _voice_like(f0: float, duration_s: float = 1.0, n_harmonics: int = 10) -> NDArray[np.float64]:
    t = np.arange(int(duration_s * SR)) / SR
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, n_harmonics + 1))
    return np.asarray(0.3 * x / np.max(np.abs(x)), dtype=np.float64)


# Default chain


def test_default_chain_expands_to_many_variants() -> None:
    """DEFAULT_AUGMENTATIONS produces > 1 variant per input (combinatorial product)."""
    x = _voice_like(200.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    out = aug.augment(x)
    assert isinstance(out, list)
    assert len(out) > 1
    for y in out:
        assert isinstance(y, np.ndarray)
        assert np.all(np.isfinite(y))


def test_default_chain_variant_count_matches_manual_calculation() -> None:
    """Count matches the cartesian product of each augmentation's list sizes."""
    expected = 1
    for spec in DEFAULT_AUGMENTATIONS:
        config = {} if isinstance(spec, str) else spec[1]
        for v in config.values():
            expected *= len(v) if isinstance(v, list) else 1

    x = _voice_like(200.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    out = aug.augment(x)
    assert len(out) == expected


# List / tuple input


def test_augment_list_returns_flat_list() -> None:
    """K inputs x N modes → list of length K x N."""
    signals = [_voice_like(f) for f in (150.0, 200.0, 300.0)]
    aug = Augmenter(sample_rate=SR, seed=0)
    dbs = [-3.0, 3.0]
    out = aug.augment(signals, [("gain", {"db": dbs})])
    assert isinstance(out, list)
    assert len(out) == len(signals) * len(dbs)


def test_augment_tuple_input_also_returns_list() -> None:
    signals = (_voice_like(150.0), _voice_like(200.0))
    aug = Augmenter(sample_rate=SR, seed=0)
    out = aug.augment(signals, [("phase_flip", {"flip": [False, True]})])
    assert isinstance(out, list)
    assert len(out) == 2 * 2  # 2 inputs x 2 flip modes


def test_augment_list_speed_preserves_each_inputs_length() -> None:
    """Speed/stretch change length internally; augment() re-aligns each variant to its source length."""
    signals = [_voice_like(200.0, duration_s=1.0) for _ in range(3)]
    aug = Augmenter(sample_rate=SR, seed=0)
    out = aug.augment(signals, [("speed", {"factor": [0.8, 1.25]})])
    assert isinstance(out, list)
    assert len(out) == 3 * 2
    for y in out:
        assert isinstance(y, np.ndarray)
        assert y.shape == signals[0].shape


# 2-D batch input


def test_augment_2d_array_flattens_to_list() -> None:
    """2-D input (B, N) is treated as a sequence of rows → list of (B x modes) outputs."""
    batch = np.stack([_voice_like(f) for f in (150.0, 200.0, 250.0)])
    aug = Augmenter(sample_rate=SR, seed=0)
    dbs = [-6.0, 0.0, 6.0]
    out = aug.augment(batch, [("gain", {"db": dbs})])
    assert isinstance(out, list)
    assert len(out) == batch.shape[0] * len(dbs)


def test_augment_2d_per_row_deterministic() -> None:
    """Same config + same input → identical output across runs (gain is deterministic)."""
    batch = np.stack([_voice_like(200.0) for _ in range(5)])
    aug1 = Augmenter(sample_rate=SR, seed=0)
    aug2 = Augmenter(sample_rate=SR, seed=999)
    out1 = aug1.augment(batch, [("gain", {"db": 3.0})])
    out2 = aug2.augment(batch, [("gain", {"db": 3.0})])
    for a, b in zip(out1, out2, strict=True):
        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)
        np.testing.assert_array_equal(a, b)


# Chained augmentations — cartesian across the whole chain


def test_chain_cartesian_product() -> None:
    """Chained specs multiply mode counts."""
    x = _voice_like(200.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    out = aug.augment(
        x,
        [
            ("gain", {"db": [-3.0, 3.0]}),  # 2 modes
            ("phase_flip", {"flip": [False, True]}),  # 2 modes
        ],
    )
    assert isinstance(out, list)
    assert len(out) == 2 * 2


# Per-element shape validation


def test_augment_rejects_3d_array() -> None:
    bad = np.zeros((2, 2, 16000))
    aug = Augmenter(sample_rate=SR, seed=0)
    # 3-D flows as a single signal → fails the 1-D check.
    with pytest.raises(ValueError, match="1-D mono"):
        aug.augment(bad, [("gain", {"db": 0.0})])


def test_augment_list_rejects_non_1d_elements() -> None:
    bad = [_voice_like(200.0), np.zeros((2, 100))]
    aug = Augmenter(sample_rate=SR, seed=0)
    with pytest.raises(ValueError, match="1-D mono"):
        aug.augment(bad, [("gain", {"db": 0.0})])


def test_augment_rejects_transposed_2d_shape() -> None:
    """``(samples, batch)`` shape (leading dim huge, trailing dim tiny) is flagged."""
    bad = np.zeros((16000, 3), dtype=np.float64)
    aug = Augmenter(sample_rate=SR, seed=0)
    with pytest.raises(ValueError, match=r"looks transposed"):
        aug.augment(bad, [("gain", {"db": 0.0})])


def test_augment_rejects_variable_length_list() -> None:
    """List of 1-D signals with mismatched lengths must raise."""
    signals = [_voice_like(200.0, duration_s=1.0), _voice_like(200.0, duration_s=0.5)]
    aug = Augmenter(sample_rate=SR, seed=0)
    with pytest.raises(ValueError, match=r"same length"):
        aug.augment(signals, [("gain", {"db": 0.0})])


def test_augment_accepts_uniform_length_list() -> None:
    signals = [_voice_like(200.0) for _ in range(3)]
    aug = Augmenter(sample_rate=SR, seed=0)
    out = aug.augment(signals, [("gain", {"db": 3.0})])
    assert isinstance(out, list)
    assert len(out) == 3


def test_empty_chain_is_identity() -> None:
    x = _voice_like(200.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    out = aug.augment(x, [])
    assert isinstance(out, list)
    assert len(out) == 1
    y = out[0]
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, x)
