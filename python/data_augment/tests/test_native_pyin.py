"""Compare native pyin against librosa.pyin on 100 synthetic signals.

All outputs (f0, voiced_flag, voiced_prob) must match within float precision.
Each case is seeded so failures are reproducible; the seed picks signal type,
frequencies, sample rate, frame length, and noise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import librosa
import numpy as np
import pytest
from data_augment import _native_pyin

if TYPE_CHECKING:
    from numpy.typing import NDArray

N_CASES = 100
SIGNAL_KINDS = ("silence", "sine", "chirp", "two-tone", "noisy-sine", "am-sine")
SAMPLE_RATES = (16000, 22050, 44100)
FRAME_LENGTHS = (1024, 2048, 4096)


def _make_signal(seed: int) -> tuple[NDArray[np.float64], int, float, float, int, int]:
    rng = np.random.default_rng(seed)
    kind = SIGNAL_KINDS[seed % len(SIGNAL_KINDS)]
    sr = int(rng.choice(SAMPLE_RATES))
    frame_length = int(rng.choice(FRAME_LENGTHS))
    hop_length = frame_length // 4
    fmin = float(rng.uniform(50.0, 120.0))
    fmax = float(rng.uniform(500.0, min(2000.0, sr / 2 - 10)))
    duration = float(rng.uniform(0.25, 0.6))
    n = int(duration * sr)
    t = np.arange(n) / sr

    if kind == "silence":
        sig = np.zeros(n, dtype=np.float64)
    elif kind == "sine":
        f = rng.uniform(fmin * 1.2, fmax * 0.5)
        sig = 0.5 * np.sin(2 * np.pi * f * t)
    elif kind == "chirp":
        f0, f1 = rng.uniform(fmin * 1.2, fmax * 0.5, 2)
        freq = f0 + (f1 - f0) * t / duration
        sig = 0.5 * np.sin(2 * np.pi * np.cumsum(freq) / sr)
    elif kind == "two-tone":
        f1, f2 = rng.uniform(fmin * 1.2, fmax * 0.5, 2)
        sig = 0.3 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)
    elif kind == "noisy-sine":
        f = rng.uniform(fmin * 1.2, fmax * 0.5)
        sig = 0.5 * np.sin(2 * np.pi * f * t) + rng.uniform(0.01, 0.1) * rng.standard_normal(n)
    elif kind == "am-sine":  # amplitude-modulated sine
        carrier = rng.uniform(fmin * 1.2, fmax * 0.5)
        mod = rng.uniform(2.0, 8.0)
        sig = (0.3 + 0.2 * np.sin(2 * np.pi * mod * t)) * np.sin(2 * np.pi * carrier * t)
    else:  # pragma: no cover
        msg = f"unknown kind {kind}"
        raise AssertionError(msg)

    return sig.astype(np.float64), sr, fmin, fmax, frame_length, hop_length


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("seed", range(N_CASES))
def test_native_matches_librosa(seed: int) -> None:
    sig, sr, fmin, fmax, frame_length, hop_length = _make_signal(seed)

    f0_lib, voiced_lib, prob_lib = librosa.pyin(
        sig,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    config = _native_pyin.PyinConfig(frame_length=frame_length, hop_length=hop_length)
    f0_nat, voiced_nat, prob_nat = _native_pyin.pyin(sig, sr=sr, fmin=fmin, fmax=fmax, config=config)

    np.testing.assert_allclose(prob_nat, prob_lib, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(voiced_nat, voiced_lib)
    np.testing.assert_allclose(f0_nat[voiced_lib], f0_lib[voiced_lib], rtol=0, atol=1e-12)
