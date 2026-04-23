"""Pure numpy/scipy port of librosa.pyin.

Mirrors the algorithm line-for-line so outputs match librosa within float
precision. 1-D only — data_augment never uses multi-channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.stats

from data_augment._native_pyin._frame import frame
from data_augment._native_pyin._viterbi import (
    transition_local_triangle,
    transition_loop_2,
    viterbi_log,
)
from data_augment._native_pyin._yin import (
    cumulative_mean_normalized_difference,
    parabolic_interpolation,
    yin_trough_probs,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["PyinConfig", "pyin"]


@dataclass(frozen=True)
class PyinConfig:
    """Tunable knobs for pyin. Defaults mirror librosa.pyin."""

    frame_length: int = 2048
    hop_length: int | None = None
    n_thresholds: int = 100
    beta_parameters: tuple[float, float] = (2.0, 18.0)
    boltzmann_parameter: float = 2.0
    resolution: float = 0.1
    max_transition_rate: float = 35.92
    switch_prob: float = 0.01
    no_trough_prob: float = 0.01
    fill_na: float | None = float("nan")
    center: bool = True
    pad_mode: Literal[
        "constant",
        "edge",
        "linear_ramp",
        "maximum",
        "mean",
        "median",
        "minimum",
        "reflect",
        "symmetric",
        "wrap",
    ] = "constant"


_DEFAULT_CONFIG = PyinConfig()


def pyin(
    y: NDArray[np.float64],
    *,
    fmin: float,
    fmax: float,
    sr: float,
    config: PyinConfig = _DEFAULT_CONFIG,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Probabilistic YIN — native numpy/scipy port of librosa.pyin.

    Returns (f0, voiced_flag, voiced_prob).
    """
    frame_length = config.frame_length
    hop_length = config.hop_length if config.hop_length is not None else frame_length // 4

    if config.center:
        pad = frame_length // 2
        y = np.pad(y, (pad, pad), mode=config.pad_mode)

    y_frames = frame(y, frame_length=frame_length, hop_length=hop_length)

    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - 1)

    yin_frames = cumulative_mean_normalized_difference(y_frames, min_period, max_period)
    parabolic_shifts = parabolic_interpolation(yin_frames)

    thresholds = np.linspace(0, 1, config.n_thresholds + 1)
    beta_probs: NDArray[np.float64] = np.diff(
        scipy.stats.beta.cdf(thresholds, config.beta_parameters[0], config.beta_parameters[1])
    )

    n_bins_per_semitone = int(np.ceil(1.0 / config.resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    yin_probs = yin_trough_probs(
        yin_frames,
        thresholds,
        beta_probs,
        config.boltzmann_parameter,
        config.no_trough_prob,
    )
    yin_period, frame_index = np.nonzero(yin_probs)
    period_candidates = (min_period + yin_period) + parabolic_shifts[yin_period, frame_index]
    bin_index = 12 * n_bins_per_semitone * np.log2((sr / period_candidates) / fmin)
    bin_index = np.clip(np.round(bin_index), 0, n_pitch_bins).astype(int)

    observation_probs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]), dtype=np.float64)
    observation_probs[bin_index, frame_index] = yin_probs[yin_period, frame_index]

    voiced_prob = np.clip(np.sum(observation_probs[:n_pitch_bins, :], axis=0), 0, 1)
    observation_probs[n_pitch_bins:, :] = (1 - voiced_prob) / n_pitch_bins

    max_semitones_per_frame = round(config.max_transition_rate * 12 * hop_length / sr)
    transition_width = max_semitones_per_frame * n_bins_per_semitone + 1
    local_trans = transition_local_triangle(n_pitch_bins, transition_width)
    transition = np.kron(transition_loop_2(1 - config.switch_prob), local_trans).astype(np.float64)

    p_init = np.full(2 * n_pitch_bins, 1.0 / (2 * n_pitch_bins))
    epsilon = np.finfo(observation_probs.dtype).tiny
    states = viterbi_log(
        np.log(observation_probs + epsilon).T,
        np.log(transition + epsilon),
        np.log(p_init + epsilon),
    )

    freqs = fmin * 2 ** (np.arange(n_pitch_bins) / (12 * n_bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins

    if config.fill_na is not None:
        f0 = f0.astype(np.float64, copy=True)
        f0[~voiced_flag] = config.fill_na

    return f0, voiced_flag, voiced_prob
