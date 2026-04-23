"""Kaiser-sinc resampling correctness tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from data_augment.resample import sinc_downsample, sinc_upsample

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _sine(freq: float, sr: int, duration_s: float) -> NDArray[np.float64]:
    t = np.arange(int(duration_s * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float64)


def _dominant_hz(x: NDArray[np.float64], sr: int) -> float:
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    mag[0] = 0
    return float(np.argmax(mag)) * sr / len(x)


@pytest.mark.parametrize("factor", [2, 3, 4, 8])
def test_upsample_preserves_frequency(factor: int) -> None:
    """Sinc upsampling should preserve the sine's frequency at the higher rate."""
    sr = 8000
    f_in = 300.0
    x = _sine(f_in, sr, duration_s=0.5)
    y = sinc_upsample(x, factor=factor)
    assert len(y) == len(x) * factor
    f_detected = _dominant_hz(y, sr * factor)
    assert abs(f_detected - f_in) < 2.0, f"expected {f_in}, got {f_detected}"


@pytest.mark.parametrize("factor", [2, 3, 4, 8])
def test_downsample_preserves_frequency(factor: int) -> None:
    """Sinc downsampling should preserve an in-band sine's frequency."""
    sr = 32000
    f_in = 300.0  # well below any factor's Nyquist
    x = _sine(f_in, sr, duration_s=0.5)
    y = sinc_downsample(x, factor=factor)
    # Decimation `filtered[::factor]` produces ceil(N/factor) samples.
    assert len(y) == -(-len(x) // factor)
    f_detected = _dominant_hz(y, sr // factor)
    assert abs(f_detected - f_in) < 2.0, f"expected {f_in}, got {f_detected}"


def test_roundtrip_preserves_signal() -> None:
    """Upsample → downsample should reconstruct the input to high accuracy."""
    sr = 8000
    x = _sine(400.0, sr, duration_s=0.5) * 0.5
    up = sinc_upsample(x, factor=4)
    back = sinc_downsample(up, factor=4)
    # Drop filter edges where convolution incompleteness dominates
    edge = 200
    np.testing.assert_allclose(back[edge:-edge], x[edge:-edge], atol=1e-3)


def test_roundtrip_edge_artifacts_are_small() -> None:
    """Reflection padding keeps edge artifacts small even for non-zero-ending signals.

    Previously with zero-padding, the boundary step (signal → 0) leaked broadband
    transients into the first/last ~pad samples of the output.
    """
    sr = 16000
    x = _sine(300.0, sr, duration_s=1.0) * 0.3
    up = sinc_upsample(x, factor=4)
    back = sinc_downsample(up, factor=4)

    # Interior must be near-perfect
    assert np.max(np.abs((back - x)[500:-500])) < 1e-5

    # Edges: reflection keeps them within ~1% of peak (zero-pad would be ~20%+)
    peak = float(np.max(np.abs(x)))
    edge_err = max(float(np.max(np.abs((back - x)[:500]))), float(np.max(np.abs((back - x)[-500:]))))
    assert edge_err / peak < 0.02, f"edge error {edge_err:.2e} exceeds 2% of peak {peak:.2e}"


def test_downsample_rejects_aliasing() -> None:
    """A tone above the target Nyquist must be heavily attenuated (steady-state).

    Zero-padded convolution at the edges produces transients that dominate any
    short signal's RMS; evaluate the filter's real performance in the interior.
    """
    sr = 32000
    factor = 4
    target_nyquist = sr / (2 * factor)  # = 4000 Hz
    # Long enough that the kernel transient (~2048 samples at each end, ~0.064 s) is small vs total.
    duration = 2.0

    in_band = _sine(target_nyquist * 0.5, sr, duration_s=duration)
    out_band = _sine(target_nyquist * 1.5, sr, duration_s=duration)

    y_in = sinc_downsample(in_band, factor=factor)
    y_out = sinc_downsample(out_band, factor=factor)

    # Drop the first and last 0.1 s of the output (filter settling region).
    edge = int(0.1 * sr / factor)
    rms_in = float(np.sqrt(np.mean(y_in[edge:-edge] ** 2)))
    rms_out = float(np.sqrt(np.mean(y_out[edge:-edge] ** 2)))

    db = 20 * np.log10(rms_out / max(rms_in, 1e-20))
    assert db < -120.0, f"aliasing: out-of-band leak only {db:.1f} dB below in-band"
