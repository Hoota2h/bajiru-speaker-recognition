"""Transition matrices and log-space Viterbi for native pyin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.signal.windows

if TYPE_CHECKING:
    from numpy.typing import NDArray


def transition_local_triangle(n_states: int, width: int) -> NDArray[np.float64]:
    """Triangle-windowed local transition matrix (wrap=False)."""
    transition = np.zeros((n_states, n_states), dtype=np.float64)
    win = scipy.signal.windows.triang(width)
    lpad = (n_states - width) // 2
    rpad = n_states - width - lpad
    trans_row_base = np.concatenate([np.zeros(lpad), win, np.zeros(rpad)])

    for i in range(n_states):
        trans_row = np.roll(trans_row_base, n_states // 2 + i + 1)
        trans_row[min(n_states, i + width // 2 + 1) :] = 0
        trans_row[: max(0, i - width // 2)] = 0
        transition[i] = trans_row

    transition /= transition.sum(axis=1, keepdims=True)
    return transition


def transition_loop_2(prob: float) -> NDArray[np.float64]:
    """2x2 self-loop transition matrix with diagonal `prob`."""
    return np.array([[prob, 1 - prob], [1 - prob, prob]], dtype=np.float64)


def viterbi_log(
    log_prob: NDArray[np.float64],
    log_trans: NDArray[np.float64],
    log_p_init: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Log-space Viterbi. log_prob shape (n_steps, n_states)."""
    n_steps, n_states = log_prob.shape
    value = np.zeros((n_steps, n_states), dtype=np.float64)
    ptr = np.zeros((n_steps, n_states), dtype=np.int64)

    value[0] = log_prob[0] + log_p_init
    for t in range(1, n_steps):
        trans_out = value[t - 1] + log_trans.T
        for j in range(n_states):
            ptr[t, j] = int(np.argmax(trans_out[j]))
            value[t, j] = log_prob[t, j] + trans_out[j, ptr[t, j]]

    state = np.zeros(n_steps, dtype=np.int64)
    state[-1] = int(np.argmax(value[-1]))
    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]
    return state
