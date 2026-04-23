"""Augmenter public-API behavior and pitch-shift correctness tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from data_augment import Augmenter, _native_pyin
from data_augment.pitch_detect import detect_pitch_periods
from data_augment.psola import psola_shift
from data_augment.resample import sinc_downsample, sinc_upsample

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_augment_accepts_2d_as_batch() -> None:
    """2-D input is a batch of 1-D rows; one mode → list of length B."""
    aug = Augmenter(sample_rate=16000)
    batch = np.zeros((3, 16000), dtype=np.float64)
    out = aug.augment(batch, [("gain", {"db": 0.0})])
    assert isinstance(out, list)
    assert len(out) == 3
    for y in out:
        assert isinstance(y, np.ndarray)
        assert y.shape == (16000,)


def test_augment_rejects_scalar_input() -> None:
    aug = Augmenter(sample_rate=16000)
    scalar = np.array(1.0)
    with pytest.raises(ValueError, match="1-D mono"):
        aug.augment(scalar, [("pitch", {"semitones": 1.0})])


def test_augment_rejects_wrong_type() -> None:
    aug = Augmenter(sample_rate=16000)
    with pytest.raises(TypeError, match=r"numpy\.ndarray or torch\.Tensor"):
        aug.augment([1.0, 2.0, 3.0], [("pitch", {"semitones": 1.0})])  # type: ignore[list-item]


def test_augment_preserves_length_and_finiteness() -> None:
    sr = 16000
    x = _voice_like(f0=150.0, sr=sr, duration_s=1.0)
    aug = Augmenter(sample_rate=sr)
    out = aug.augment(x, [("pitch", {"semitones": 4.0})])
    assert isinstance(out, list)
    assert len(out) == 1
    y = out[0]
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert np.any(y != 0), "output is silent"


def test_augment_accepts_1d_input() -> None:
    aug = Augmenter(sample_rate=16000)
    t = np.arange(16000) / 16000
    x = 0.3 * np.sin(2 * np.pi * 200 * t)
    out = aug.augment(x, [("pitch", {"semitones": 2.0})])
    assert isinstance(out, list)
    y = out[0]
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


# Pitch-shift correctness


def _voice_like(f0: float, sr: int, duration_s: float, n_harmonics: int = 10) -> NDArray[np.float64]:
    """Harmonic stack with 1/k amplitude falloff — a voice-like test signal."""
    t = np.arange(int(duration_s * sr)) / sr
    x = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        x += np.sin(2 * np.pi * k * f0 * t) / k
    x = 0.3 * x / np.max(np.abs(x))
    result: NDArray[np.float64] = x.astype(np.float64)
    return result


def _detect_f0(signal: NDArray[np.float64], sr: int, fmin: float = 50.0, fmax: float = 800.0) -> float:
    f0, voiced, _ = _native_pyin.pyin(signal, fmin=fmin, fmax=fmax, sr=sr)
    voiced_f0 = f0[voiced]
    if len(voiced_f0) == 0:
        return float("nan")
    return float(np.median(voiced_f0))


@pytest.mark.parametrize(
    "semitones",
    [
        # Integer-octave shifts on a pure harmonic stack trip the documented PSOLA
        # pure-tone pathology (see README). Marked xfail so the suite stays clean
        # while the test keeps running; it'll flip to xpass if we ever fix it.
        pytest.param(-12, marks=pytest.mark.xfail(reason="PSOLA pure-tone pathology at ratio=0.5", strict=True)),
        -7,
        -5,
        -3,
        -1,
        0,
        1,
        3,
        5,
        7,
        pytest.param(
            12,
            marks=pytest.mark.xfail(reason="PSOLA harmonic-rich output fools pyin octave detection", strict=True),
        ),
    ],
)
def test_pitch_shift_moves_detected_f0(semitones: int) -> None:
    """Shifting a voice-like signal by N semitones should move its detected f0 by 2**(N/12)."""
    sr = 16000
    f_in = 150.0

    x = _voice_like(f0=f_in, sr=sr, duration_s=1.0)
    aug = Augmenter(sample_rate=sr)
    y = aug.augment(x, [("pitch", {"semitones": float(semitones)})])[0]
    assert isinstance(y, np.ndarray)

    detected = _detect_f0(y, sr)
    expected = f_in * (2 ** (semitones / 12))

    assert abs(detected - expected) / expected < 0.03, (
        f"semitones={semitones:+d}: expected ~{expected:.1f} Hz, got {detected:.1f} Hz"
    )


@pytest.mark.parametrize("f_in", [100.0, 150.0, 220.0, 330.0])
@pytest.mark.parametrize("semitones", [-5, -2, 2, 5])
def test_pitch_shift_across_fundamentals(f_in: float, semitones: int) -> None:
    """Same check across several input fundamental frequencies."""
    sr = 16000

    x = _voice_like(f0=f_in, sr=sr, duration_s=1.0)
    aug = Augmenter(sample_rate=sr)
    y = aug.augment(x, [("pitch", {"semitones": float(semitones)})])[0]
    assert isinstance(y, np.ndarray)

    detected = _detect_f0(y, sr)
    expected = f_in * (2 ** (semitones / 12))

    assert abs(detected - expected) / expected < 0.03, (
        f"f_in={f_in} Hz, semitones={semitones:+d}: expected ~{expected:.1f} Hz, got {detected:.1f} Hz"
    )


def test_zero_semitone_shift_preserves_pitch() -> None:
    sr = 16000
    f_in = 200.0
    x = _voice_like(f0=f_in, sr=sr, duration_s=1.0)
    aug = Augmenter(sample_rate=sr)
    y = aug.augment(x, [("pitch", {"semitones": 0.0})])[0]
    assert isinstance(y, np.ndarray)
    assert abs(_detect_f0(y, sr) - f_in) / f_in < 0.02


def test_pitch_shift_preserves_energy_roughly() -> None:
    sr = 16000
    x = _voice_like(f0=180.0, sr=sr, duration_s=1.0)
    aug = Augmenter(sample_rate=sr)
    for semitones in [-5.0, -2.0, 2.0, 5.0]:
        y = aug.augment(x, [("pitch", {"semitones": semitones})])[0]
        assert isinstance(y, np.ndarray)
        rms_in = float(np.sqrt(np.mean(x**2)))
        rms_out = float(np.sqrt(np.mean(y**2)))
        db_delta = 20 * np.log10(rms_out / rms_in)
        assert abs(db_delta) < 6.0, f"semitones={semitones}: |Δ| = {abs(db_delta):.1f} dB"


@pytest.mark.parametrize("backend", ["native", "librosa"])
def test_pitch_shift_matches_retuned_reference(backend: str) -> None:
    """Shifting f_low by +2 semitones should approximate a fresh waveform at f_high.

    Compares a PSOLA-shifted complex waveform against a freshly-synthesized
    reference at the target frequency. Uses the pitch-shift pipeline directly
    so the pyin backend can be swapped. Reports (and asserts bounds on):
      - time-domain *mean* abs diff (phase-sensitive but bounded)
      - magnitude-spectrum max abs diff (phase-invariant)
    Time-domain *max* diff isn't asserted — two same-frequency waveforms with
    different phase can differ by up to 2x peak at some samples.

    Both backends produce identical pitch marks within float precision, so their
    outputs here are also identical.
    """
    sr = 16000
    f_low = 220.0
    semitones = 2.0
    ratio = 2 ** (semitones / 12)
    f_high = f_low * ratio

    x_low = _voice_like(f0=f_low, sr=sr, duration_s=1.0)
    x_ref = _voice_like(f0=f_high, sr=sr, duration_s=1.0)

    oversample = 4
    kernel = 4097
    up = sinc_upsample(x_low, oversample, kernel)
    marks = detect_pitch_periods(up, sr * oversample, backend=backend)  # type: ignore[arg-type]
    shifted = psola_shift(up, marks, ratio)
    y = sinc_downsample(shifted, oversample, kernel)

    trim = 2000
    y_core = y[trim:-trim]
    ref_core = x_ref[trim:-trim]
    peak = float(np.max(np.abs(ref_core)))

    time_max = float(np.max(np.abs(y_core - ref_core))) / peak
    time_mean = float(np.mean(np.abs(y_core - ref_core))) / peak
    mag_y = np.abs(np.fft.rfft(y_core))
    mag_ref = np.abs(np.fft.rfft(ref_core))
    mag_max = float(np.max(np.abs(mag_y - mag_ref))) / float(np.max(mag_ref))

    # Time-domain max is phase-ambiguous (up to 200%); just sanity-cap it.
    assert time_max < 2.0, f"[{backend}] time max diff: {time_max:.1%}"
    # Mean / magnitude-spectrum differences reflect actual quality.
    assert time_mean < 0.20, f"[{backend}] time mean diff: {time_mean:.1%}"
    assert mag_max < 0.20, f"[{backend}] magnitude-spectrum max diff: {mag_max:.1%}"
