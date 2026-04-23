"""Tests for gain, phase_flip, apf augmentations (combinatorial API)."""

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


# gain


@pytest.mark.parametrize("db", [-12.0, -6.0, -3.0, 0.0, 3.0, 6.0, 12.0])
def test_gain_scales_rms_correctly(db: float) -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("gain", {"db": db})])[0]
    assert isinstance(y, np.ndarray)

    rms_in = float(np.sqrt(np.mean(x**2)))
    rms_out = float(np.sqrt(np.mean(y**2)))
    expected_ratio = 10 ** (db / 20.0)
    np.testing.assert_allclose(rms_out / rms_in, expected_ratio, rtol=1e-10)


def test_gain_zero_db_is_identity() -> None:
    x = _voice_like(200.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("gain", {"db": 0.0})])[0]
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, x)


def test_gain_list_produces_one_output_per_mode() -> None:
    """A list of dBs gives one output signal per value."""
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    dbs = [-6.0, 0.0, 6.0]
    out = aug.augment(x, [("gain", {"db": dbs})])
    assert isinstance(out, list)
    assert len(out) == len(dbs)
    rms_in = float(np.sqrt(np.mean(x**2)))
    for db, y in zip(dbs, out, strict=True):
        assert isinstance(y, np.ndarray)
        expected_ratio = 10 ** (db / 20.0)
        np.testing.assert_allclose(
            float(np.sqrt(np.mean(y**2))) / rms_in,
            expected_ratio,
            rtol=1e-10,
        )


# phase_flip


def test_phase_flip_true_inverts() -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("phase_flip", {"flip": True})])[0]
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, -x)


def test_phase_flip_false_is_identity() -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("phase_flip", {"flip": False})])[0]
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, x)


def test_phase_flip_both_modes_gives_two_variants() -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    out = aug.augment(x, [("phase_flip", {"flip": [False, True]})])
    assert isinstance(out, list)
    assert len(out) == 2
    np.testing.assert_array_equal(out[0], x)
    np.testing.assert_array_equal(out[1], -x)


# apf


@pytest.mark.parametrize(
    "sections",
    [
        [{"order": 1, "freq_hz": 1000.0}],
        [{"order": 2, "freq_hz": 1000.0, "q": 1.0}],
        [
            {"order": 1, "freq_hz": 500.0},
            {"order": 2, "freq_hz": 2000.0, "q": 1.5},
        ],
    ],
)
def test_apf_preserves_magnitude_spectrum(sections: list[dict[str, object]]) -> None:
    x = _voice_like(180.0, duration_s=2.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("apf", {"sections": sections})])[0]
    assert isinstance(y, np.ndarray)

    transient = 2000
    mag_in = np.abs(np.fft.rfft(x[transient:]))
    mag_out = np.abs(np.fft.rfft(y[transient:]))
    rel_err = np.max(np.abs(mag_out - mag_in)) / np.max(mag_in)
    assert rel_err < 0.01, f"magnitude changed by {rel_err:.1%}"


def test_apf_changes_phase() -> None:
    x = _voice_like(180.0, duration_s=2.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    sections = [{"order": 2, "freq_hz": 1500.0, "q": 1.0}]
    y = aug.augment(x, [("apf", {"sections": sections})])[0]
    assert isinstance(y, np.ndarray)
    assert not np.allclose(y, x, atol=1e-3)


def test_apf_rms_preserved() -> None:
    x = _voice_like(200.0, duration_s=2.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("apf", {})])[0]
    assert isinstance(y, np.ndarray)
    transient = 2000
    rms_in = float(np.sqrt(np.mean(x[transient:] ** 2)))
    rms_out = float(np.sqrt(np.mean(y[transient:] ** 2)))
    assert abs(rms_out - rms_in) / rms_in < 0.01


def test_apf_deterministic_across_seeds() -> None:
    """APF config is fully deterministic → output independent of Augmenter seed."""
    x = _voice_like(180.0, duration_s=1.0)
    sections = [
        {"order": 1, "freq_hz": 800.0},
        {"order": 2, "freq_hz": 2500.0, "q": 1.5},
    ]
    y1 = Augmenter(sample_rate=SR, seed=0).augment(x, [("apf", {"sections": sections})])[0]
    y2 = Augmenter(sample_rate=SR, seed=999).augment(x, [("apf", {"sections": sections})])[0]
    assert isinstance(y1, np.ndarray)
    assert isinstance(y2, np.ndarray)
    np.testing.assert_array_equal(y1, y2)


@pytest.mark.parametrize(
    "sections",
    [
        [{"order": 1, "freq_hz": SR / 2}],
        [{"order": 1, "freq_hz": 0.0}],
        [{"order": 2, "freq_hz": 1000.0, "q": 0.0}],
        [{"order": 2, "freq_hz": 1000.0, "q": -1.0}],
        [{"order": 3, "freq_hz": 1000.0}],
    ],
)
def test_apf_rejects_invalid_config(sections: list[dict[str, object]]) -> None:
    x = _voice_like(180.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    with pytest.raises(ValueError, match=r"freq_hz|q must be positive|section order"):
        aug.augment(x, [("apf", {"sections": sections})])


def test_apf_first_order_break_frequency_matches() -> None:
    target_hz = 1500.0
    n = SR * 2
    t = np.arange(n) / SR
    f_low = target_hz * 0.3
    f_high = target_hz * 3.0
    x = 0.3 * (np.sin(2 * np.pi * f_low * t) + np.sin(2 * np.pi * f_high * t))
    x = np.asarray(x, dtype=np.float64)

    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("apf", {"sections": [{"order": 1, "freq_hz": target_hz}]})])[0]
    assert isinstance(y, np.ndarray)

    transient = 4000
    spec = np.fft.rfft(y[transient:-transient])
    x_spec = np.fft.rfft(x[transient:-transient])
    freqs = np.fft.rfftfreq(len(spec) * 2 - 2, d=1 / SR)

    idx_low = int(np.argmin(np.abs(freqs - f_low)))
    idx_high = int(np.argmin(np.abs(freqs - f_high)))

    def _wrap(p: float) -> float:
        return float(((p + np.pi) % (2 * np.pi)) - np.pi)

    phase_low = _wrap(np.angle(spec[idx_low]) - np.angle(x_spec[idx_low]))
    phase_high = _wrap(np.angle(spec[idx_high]) - np.angle(x_spec[idx_high]))

    assert abs(phase_low) < 0.6, f"f_low phase shift {phase_low:.2f} rad (expect near 0)"
    assert phase_high < -1.5, f"f_high phase shift {phase_high:.2f} rad (expect near -π)"
