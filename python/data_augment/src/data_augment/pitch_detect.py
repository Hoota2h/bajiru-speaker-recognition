"""Pitch detection for PSOLA — estimates f0 per frame and marks pitch periods.

Primary backend is a native pure-numpy/scipy port of pyin. librosa is kept as an
optional reference backend (useful for regression checks against upstream). Librosa
doesn't really play well with some systems and is quite heavy to install,
so, I decided to make this instead of just calling through librosa.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Literal

import numpy as np

from data_augment import _native_pyin

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray


def _try_import_librosa() -> ModuleType | None:
    """Return the ``librosa`` module if installed, else ``None``.

    Silent — the native backend is the default and preferred path; missing
    librosa is not a problem unless the caller explicitly requests
    ``backend="librosa"``.
    """
    try:
        return importlib.import_module("librosa")
    except ImportError:  # pragma: no cover — librosa is optional
        return None


_LIBROSA = _try_import_librosa()


Backend = Literal["native", "librosa"]


def _pyin_native(
    signal: NDArray[np.float64],
    sample_rate: int,
    fmin: float,
    fmax: float,
    frame_length: int,
    hop_length: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    config = _native_pyin.PyinConfig(frame_length=frame_length, hop_length=hop_length)
    f0, voiced_flag, _ = _native_pyin.pyin(signal, fmin=fmin, fmax=fmax, sr=sample_rate, config=config)
    return f0, voiced_flag


def _pyin_librosa(
    signal: NDArray[np.float64],
    sample_rate: int,
    fmin: float,
    fmax: float,
    frame_length: int,
    hop_length: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    if _LIBROSA is None:
        msg = "librosa backend requested but librosa is not installed"
        raise RuntimeError(msg)
    f0, voiced_flag, _ = _LIBROSA.pyin(
        signal,
        sr=sample_rate,
        fmin=fmin,
        fmax=fmax,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return f0, voiced_flag


def _fallback_marks(n_samples: int, fallback_period: float) -> NDArray[np.intp]:
    """Evenly-spaced pitch marks across ``n_samples`` — used when pyin can't run."""
    if n_samples <= 0 or fallback_period <= 0:
        return np.array([], dtype=np.intp)
    positions = np.arange(0, n_samples, fallback_period)
    return np.unique(np.round(positions).astype(np.intp))


def detect_pitch_periods(
    signal: NDArray[np.float64],
    sample_rate: int,
    fmin: float = 50.0,
    fmax: float = 600.0,
    backend: Backend = "native",
) -> NDArray[np.intp]:
    """Find pitch period mark locations (sample indices) in the signal.

    Args:
        signal: 1-D audio signal.
        sample_rate: Sample rate of the signal.
        fmin: Minimum expected f0 (Hz). 50 Hz covers deep male voice.
        fmax: Maximum expected f0 (Hz). 600 Hz covers high female/child voice.
        backend: "native" (pure numpy/scipy, default) or "librosa" (reference).

    Returns:
        Sorted array of sample indices marking pitch period boundaries.

    """
    # Frame length must fit at least 2 periods of fmin; round to power of 2.
    min_frame = int(2.0 * sample_rate / fmin) + 1
    frame_length = 1
    while frame_length < min_frame:
        frame_length *= 2
    hop_length = frame_length // 4

    fallback_period = sample_rate / 100.0

    # Signal too short to form a single pyin frame — fall back to evenly-spaced
    # marks at the fallback period. Pitch-shift becomes a near-identity on such
    # clips; stretch still gets marks to work with.
    if len(signal) < frame_length:
        return _fallback_marks(len(signal), fallback_period)

    if backend == "native":
        f0, voiced_flag = _pyin_native(signal, sample_rate, fmin, fmax, frame_length, hop_length)
    elif backend == "librosa":
        f0, voiced_flag = _pyin_librosa(signal, sample_rate, fmin, fmax, frame_length, hop_length)
    else:
        msg = f"Unknown backend: {backend!r}"
        raise ValueError(msg)

    return _peak_aligned_marks(signal, f0, voiced_flag, hop_length, sample_rate, fallback_period)


def _period_at(
    sample_idx: int,
    f0: NDArray[np.float64],
    voiced_flag: NDArray[np.bool_],
    hop_length: int,
    sample_rate: float,
    fallback_period: float,
) -> float:
    """Pyin-estimated local period (samples) at ``sample_idx``, or fallback."""
    frame_idx = min(int(sample_idx // hop_length), len(f0) - 1)
    freq = f0[frame_idx]
    if voiced_flag[frame_idx] and not np.isnan(freq) and freq > 0:
        return float(sample_rate / freq)
    return fallback_period


_SEARCH_LOW = 0.9
_SEARCH_HIGH = 1.1
_MIN_SEED_SAMPLES = 2


def _peak_aligned_marks(
    signal: NDArray[np.float64],
    f0: NDArray[np.float64],
    voiced_flag: NDArray[np.bool_],
    hop_length: int,
    sample_rate: float,
    fallback_period: float,
) -> NDArray[np.intp]:
    """Walk forward at pyin's estimated period, snapping each mark to the local amplitude peak.

    Mirrors Sanna Wager's reference TD-PSOLA mark finder: from the previous mark,
    search a ±10% window around the expected next period and take ``argmax(|signal|)``.
    Puts marks at time-domain peaks (approx. glottal closures for voiced speech)
    so grains capture consistent phase across the signal.
    """
    # Seed: pick the strongest peak within the first estimated period.
    p0 = _period_at(0, f0, voiced_flag, hop_length, sample_rate, fallback_period)
    first_end = min(int(p0 * _SEARCH_HIGH), len(signal))
    if first_end < _MIN_SEED_SAMPLES:
        return _fallback_marks(len(signal), fallback_period)
    marks: list[int] = [int(np.argmax(np.abs(signal[:first_end])))]

    while True:
        prev = marks[-1]
        period = _period_at(prev, f0, voiced_flag, hop_length, sample_rate, fallback_period)
        low = prev + max(1, int(period * _SEARCH_LOW))
        high = min(prev + int(period * _SEARCH_HIGH), len(signal))
        if low >= high:
            break
        marks.append(low + int(np.argmax(np.abs(signal[low:high]))))

    return np.array(marks, dtype=np.intp)
