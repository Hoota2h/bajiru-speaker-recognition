"""Tests for the additive colored-noise augmentation (combinatorial API)."""

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


@pytest.mark.parametrize("kind", ["white", "pink", "brown"])
@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0, 30.0])
def test_noise_hits_target_snr(kind: str, snr_db: float) -> None:
    """Added noise should produce measured SNR within 0.5 dB of target."""
    x = _voice_like(180.0, duration_s=2.0)
    aug = Augmenter(sample_rate=SR, seed=42)
    y = aug.augment(x, [("noise", {"snr_db": snr_db, "kind": kind})])[0]
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape

    noise = y - x
    signal_rms = float(np.sqrt(np.mean(x**2)))
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    measured_snr = 20 * np.log10(signal_rms / noise_rms)
    assert abs(measured_snr - snr_db) < 0.5, f"kind={kind} target={snr_db} got={measured_snr:.2f}"


@pytest.mark.parametrize("kind", ["white", "pink", "brown"])
def test_noise_kind_has_expected_spectral_tilt(kind: str) -> None:
    x = np.zeros(SR * 4, dtype=np.float64)
    tiny = 1e-6 * np.sin(2 * np.pi * 100.0 * np.arange(len(x)) / SR)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(tiny, [("noise", {"snr_db": -80.0, "kind": kind})])[0]
    assert isinstance(y, np.ndarray)
    noise = y - tiny

    spectrum = np.abs(np.fft.rfft(noise))
    freqs = np.fft.rfftfreq(len(noise), d=1 / SR)

    band_low = (50.0, 500.0)
    band_high = (500.0, 5000.0)
    p_low = np.mean(spectrum[(freqs >= band_low[0]) & (freqs < band_low[1])] ** 2)
    p_high = np.mean(spectrum[(freqs >= band_high[0]) & (freqs < band_high[1])] ** 2)
    slope = np.log10(p_high / p_low)

    # Expected log10(power) slope per decade of frequency: 0 for white, -1 for pink,
    # -2 for brown.
    expected = {"white": 0.0, "pink": -1.0, "brown": -2.0}[kind]
    assert abs(slope - expected) < 0.3, f"kind={kind} expected slope ~{expected}, got {slope:.2f}"


def test_noise_silent_input_is_unchanged() -> None:
    x = np.zeros(SR, dtype=np.float64)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("noise", {"snr_db": 10.0, "kind": "white"})])[0]
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, x)


def test_noise_preserves_length() -> None:
    x = _voice_like(200.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    y = aug.augment(x, [("noise", {"snr_db": 10.0, "kind": "pink"})])[0]
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape


def test_noise_list_produces_one_output_per_mode() -> None:
    """Cartesian over snr_db x kind dimensions."""
    x = _voice_like(180.0, duration_s=1.0)
    aug = Augmenter(sample_rate=SR, seed=0)
    snrs = [15.0, 25.0]
    kinds = ["white", "pink"]
    out = aug.augment(x, [("noise", {"snr_db": snrs, "kind": kinds})])
    assert isinstance(out, list)
    assert len(out) == len(snrs) * len(kinds)  # 2 x 2 = 4
    for y in out:
        assert isinstance(y, np.ndarray)
        assert y.shape == x.shape
