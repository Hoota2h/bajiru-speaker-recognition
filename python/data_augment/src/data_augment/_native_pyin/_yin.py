"""YIN math: difference function, parabolic interpolation, local minima, trough probabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.stats

from data_augment._native_pyin._frame import autocorrelate

if TYPE_CHECKING:
    from numpy.typing import NDArray


def cumulative_mean_normalized_difference(
    y_frames: NDArray[np.float64],
    min_period: int,
    max_period: int,
) -> NDArray[np.float64]:
    """YIN cumulative mean normalized difference function (eq. 8 in the YIN paper)."""
    acf_frames = autocorrelate(y_frames, max_size=max_period + 1)

    yin_frames = np.square(y_frames)
    np.cumsum(yin_frames, out=yin_frames, axis=-2)

    k = slice(1, max_period + 1)
    yin_frames[..., 0, :] = 0
    yin_frames[..., k, :] = (
        2 * (acf_frames[..., 0:1, :] - acf_frames[..., k, :]) - yin_frames[..., : k.stop - 1, :]
    )

    yin_numerator = yin_frames[..., min_period : max_period + 1, :]
    k_range = np.arange(1, max_period + 1).reshape(-1, 1)
    cumulative_mean = np.cumsum(yin_frames[..., k, :], axis=-2) / k_range
    yin_denominator = cumulative_mean[..., min_period - 1 : max_period, :]

    tiny = np.finfo(yin_denominator.dtype).tiny
    result: NDArray[np.float64] = yin_numerator / (yin_denominator + tiny)
    return result


def parabolic_interpolation(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-column parabolic shift for sub-bin trough refinement."""
    shifts = np.zeros_like(x)
    left = x[..., :-2, :]
    center = x[..., 1:-1, :]
    right = x[..., 2:, :]
    a = right + left - 2 * center
    b = (right - left) / 2

    with np.errstate(divide="ignore", invalid="ignore"):
        interior_shift = np.where(np.abs(b) < np.abs(a), -b / a, 0.0)
    interior_shift = np.where(np.isfinite(interior_shift), interior_shift, 0.0)
    shifts[..., 1:-1, :] = interior_shift
    return shifts


def localmin_1d(x: NDArray[np.float64]) -> NDArray[np.bool_]:
    """x[i] is a local min iff x[i] < x[i-1] AND x[i] <= x[i+1]; x[0] never min."""
    out = np.zeros_like(x, dtype=bool)
    out[1:-1] = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
    out[-1] = x[-1] < x[-2]
    return out


def yin_trough_probs(
    yin_frames: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    beta_probs: NDArray[np.float64],
    boltzmann_parameter: float,
    no_trough_prob: float,
) -> NDArray[np.float64]:
    """For each frame, spread threshold-weighted probability over yin troughs."""
    yin_probs = np.zeros_like(yin_frames)
    for i in range(yin_frames.shape[1]):
        yin_frame = yin_frames[:, i]
        is_trough = localmin_1d(yin_frame)
        is_trough[0] = yin_frame[0] < yin_frame[1]
        trough_index = np.nonzero(is_trough)[0]
        if len(trough_index) == 0:
            continue

        trough_heights = yin_frame[trough_index]
        trough_thresholds = np.less.outer(trough_heights, thresholds[1:])
        trough_positions = np.cumsum(trough_thresholds, axis=0) - 1
        n_troughs = np.count_nonzero(trough_thresholds, axis=0)

        trough_prior = scipy.stats.boltzmann.pmf(trough_positions, boltzmann_parameter, n_troughs)
        trough_prior[~trough_thresholds] = 0
        probs = trough_prior.dot(beta_probs)

        global_min = int(np.argmin(trough_heights))
        n_below = int(np.count_nonzero(~trough_thresholds[global_min, :]))
        probs[global_min] += no_trough_prob * np.sum(beta_probs[:n_below])

        yin_probs[trough_index, i] = probs
    return yin_probs
