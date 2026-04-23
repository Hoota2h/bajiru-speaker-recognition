"""TD-PSOLA (Time-Domain Pitch Synchronous Overlap-Add).

See README.md for algorithm notes and failure modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MIN_MARKS = 2
_FALLBACK_PERIOD = 256
_MIN_PERIOD = 4


def _local_period(pitch_marks: NDArray[np.intp], i: int) -> int:
    if i + 1 < len(pitch_marks):
        return max(int(pitch_marks[i + 1] - pitch_marks[i]), _MIN_PERIOD)
    if i > 0:
        return max(int(pitch_marks[i] - pitch_marks[i - 1]), _MIN_PERIOD)
    return _FALLBACK_PERIOD


def _place_peaks(
    signal: NDArray[np.float64],
    input_peaks: NDArray[np.intp],
    new_peaks: NDArray[np.intp],
    analysis_positions: NDArray[np.float64],
    out_len: int,
) -> NDArray[np.float64]:
    """Overlap-add triangle-windowed grains at each ``new_peak``.

    For new peak j:
      - analysis source = nearest input peak to ``analysis_positions[j]``
      - window = triangle from ``new_peaks[j-1]`` through ``new_peaks[j]`` to ``new_peaks[j+1]``
      - grain = signal around the analysis source, same half-widths as the window

    Adjacent triangles sum to 1 (COLA) so the output has uniform amplitude.
    """
    output = np.zeros(out_len, dtype=np.float64)
    n_new = len(new_peaks)
    sig_len = len(signal)

    for j in range(n_new):
        new_pos = int(new_peaks[j])
        i = int(np.argmin(np.abs(input_peaks - analysis_positions[j])))
        src = int(input_peaks[i])

        left_dist = new_pos if j == 0 else new_pos - int(new_peaks[j - 1])
        right_dist = (out_len - 1 - new_pos) if j == n_new - 1 else int(new_peaks[j + 1]) - new_pos

        left_dist = min(left_dist, src)
        right_dist = min(right_dist, sig_len - 1 - src)
        if left_dist < 1 or right_dist < 1:
            continue

        up = np.linspace(0.0, 1.0, left_dist + 1)[1:]
        down = np.linspace(1.0, 0.0, right_dist + 1)[1:]
        window = np.concatenate([up, down])
        grain = signal[src - left_dist : src + right_dist]

        out_start = new_pos - left_dist
        n = min(len(window), len(grain), out_len - out_start)
        output[out_start : out_start + n] += window[:n] * grain[:n]
    return output


def _dedupe_sorted(arr: NDArray[np.intp]) -> NDArray[np.intp]:
    """Keep strictly ascending entries."""
    if len(arr) == 0:
        return arr
    mask = np.concatenate([[True], np.diff(arr) > 0])
    result: NDArray[np.intp] = arr[mask]
    return result


def psola_shift(
    signal: NDArray[np.float64],
    pitch_marks: NDArray[np.intp],
    shift_ratio: float,
) -> NDArray[np.float64]:
    """Pitch-shift preserving length. ``shift_ratio > 1`` shifts up, ``< 1`` shifts down."""
    if shift_ratio <= 0 or len(pitch_marks) < _MIN_MARKS:
        return signal.copy()

    out_len = len(signal)
    # New peaks: interpolate ``len(peaks) * ratio`` references through the input
    # peak positions, giving output peaks at the shifted rate.
    n_refs = max(1, int(len(pitch_marks) * shift_ratio))
    refs = np.linspace(0, len(pitch_marks) - 1, n_refs)
    left_idx = np.floor(refs).astype(np.intp)
    right_idx = np.ceil(refs).astype(np.intp)
    weight = refs - left_idx
    positions = pitch_marks[left_idx] * (1 - weight) + pitch_marks[right_idx] * weight
    new_peaks = np.clip(np.round(positions).astype(np.intp), 0, out_len - 1)
    new_peaks = _dedupe_sorted(new_peaks)
    if len(new_peaks) < 1:
        return signal.copy()

    # For pitch shift the output timeline maps to input timeline 1:1 (duration preserved).
    analysis_positions = new_peaks.astype(np.float64)
    return _place_peaks(signal, pitch_marks, new_peaks, analysis_positions, out_len)


def psola_stretch(
    signal: NDArray[np.float64],
    pitch_marks: NDArray[np.intp],
    factor: float,
) -> NDArray[np.float64]:
    """Time-stretch by ``factor`` preserving pitch. Output length = ``round(len(signal) * factor)``."""
    if factor <= 0 or factor == 1.0:
        return signal.copy()

    out_len = max(1, round(len(signal) * factor))
    if len(pitch_marks) < _MIN_MARKS:
        return np.zeros(out_len, dtype=np.float64)

    # Walk output time at the ORIGINAL local period (pitch preserved).
    # Analysis clock runs at 1/factor the rate: in_time = out_pos / factor.
    out_positions: list[float] = []
    analysis_positions: list[float] = []

    out_pos = float(pitch_marks[0])
    while out_pos < out_len:
        in_time = out_pos / factor
        out_positions.append(out_pos)
        analysis_positions.append(in_time)
        i = int(np.argmin(np.abs(pitch_marks - in_time)))
        out_pos += _local_period(pitch_marks, i)

    new_peaks = np.clip(np.round(np.array(out_positions)).astype(np.intp), 0, out_len - 1)
    analysis = np.array(analysis_positions, dtype=np.float64)
    # De-dupe keeping the matching analysis entries.
    mask = np.concatenate([[True], np.diff(new_peaks) > 0])
    new_peaks = new_peaks[mask]
    analysis = analysis[mask]
    if len(new_peaks) < 1:
        return np.zeros(out_len, dtype=np.float64)

    return _place_peaks(signal, pitch_marks, new_peaks, analysis, out_len)
