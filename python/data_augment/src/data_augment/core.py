"""Core augmentation pipeline — Augmenter class + handler dispatch.

Input forms for ``augment()``:
- Single 1-D array / tensor → list of ``num_combinations`` augmented signals.
- 2-D array / tensor ``(B, N)`` → list of length ``B x num_combinations``.
- list / tuple of 1-D signals → list of length ``len(input) x num_combinations``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from data_augment.augmentations import (
    AugmentContext,
    apply_apf,
    apply_gain,
    apply_noise,
    apply_phase_flip,
    apply_pitch,
    apply_speed,
    apply_stretch,
    expand_config,
)
from data_augment.defaults import DEFAULT_AUGMENTATIONS

if TYPE_CHECKING:
    import torch

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_BATCH_NDIM = 2

# Heuristic: if a 2-D batch's trailing dim is smaller than this, it's almost
# certainly transposed (no one trains on sub-millisecond audio clips).
_MIN_REASONABLE_SAMPLES = 50

type AudioSignal = NDArray[np.float64] | "torch.Tensor"
"""A single 1-D audio signal (numpy array or torch tensor)."""

type AudioData = AudioSignal | Sequence[AudioSignal]
"""A single signal, a batched 2-D array/tensor, or a sequence of 1-D signals."""

type AugmentationSpec = str | tuple[str, dict[str, Any]]
_Handler = Callable[[NDArray[np.float64], dict[str, Any], AugmentContext], NDArray[np.float64]]

_HANDLERS: dict[str, _Handler] = {
    "pitch": apply_pitch,
    "gain": apply_gain,
    "phase_flip": apply_phase_flip,
    "noise": apply_noise,
    "speed": apply_speed,
    "stretch": apply_stretch,
    "apf": apply_apf,
}

# Augmentations whose config is passed through verbatim (never list-expanded).
# APF's ``sections`` value is itself a list of section dicts — structurally a list
# but semantically one mode. Users who want N APF variants chain N ``("apf", {...})``
# entries with different ``sections`` lists (which will cascade into a single filter).
# I struggled to find a more intuitive way to specify "don't expand this config dict" without
# special-casing APF in the handler logic, so here we are.
_NON_EXPANDING: frozenset[str] = frozenset({"apf"})


class Augmenter:
    """Offline audio data augmenter.

    Args:
        sample_rate: Sample rate of input audio.
        oversample_factor: Internal upsampling ratio for PSOLA-based transforms.
        sinc_kernel_size: Tap count for the Kaiser-windowed sinc filter.
        seed: Optional RNG seed. Only ``noise`` actually uses randomness
            (to generate the noise sequence); combinatorial augmentations are
            otherwise fully deterministic.

    """

    def __init__(  # noqa: D107 — args documented in class docstring
        self,
        sample_rate: int = 16000,
        oversample_factor: int = 4,
        sinc_kernel_size: int = 4097,
        seed: int | None = None,
    ) -> None:
        self._seed = seed
        self._sample_rate = sample_rate
        self._oversample_factor = oversample_factor
        self._sinc_kernel_size = sinc_kernel_size

    def _fresh_context(self) -> AugmentContext:
        """Build a fresh ``AugmentContext`` with a newly-seeded RNG.

        Rebuilt per ``augment()`` call so ``seed=X`` produces identical output on
        repeated calls — ``np.random.Generator`` state is otherwise mutable and
        would drift across calls. ``seed=None`` still draws fresh OS entropy.
        """
        return AugmentContext(
            rng=np.random.default_rng(self._seed),
            sample_rate=self._sample_rate,
            oversample_factor=self._oversample_factor,
            sinc_kernel_size=self._sinc_kernel_size,
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def oversample_factor(self) -> int:
        return self._oversample_factor

    @property
    def sinc_kernel_size(self) -> int:
        return self._sinc_kernel_size

    def augment(
        self,
        data: AudioData,
        augmentations: Iterable[AugmentationSpec] | None = None,
    ) -> list[AudioSignal]:
        """Expand the spec into concrete chains and apply each to every input signal.

        Args:
            data: A 1-D signal, a 2-D batch ``(B, N)``, or a sequence of 1-D signals.
            augmentations: Iterable of ``(name, config)`` or bare ``name`` strings.
                If ``None`` (default), uses :data:`DEFAULT_AUGMENTATIONS`.

        Returns:
            Flat list of length ``num_inputs x num_combinations``. Each element is
            a 1-D signal in the same backend type (numpy / torch) as the input.

        """
        specs = self._normalize_specs(augmentations)
        chains = self._build_chains(specs)
        inputs = _as_signal_list(data)
        ctx = self._fresh_context()
        return [self._apply_chain(x, chain, ctx) for x in inputs for chain in chains]

    @staticmethod
    def _normalize_specs(
        augmentations: Iterable[AugmentationSpec] | None,
    ) -> list[tuple[str, dict[str, Any]]]:
        raw = list(DEFAULT_AUGMENTATIONS if augmentations is None else augmentations)
        normalized: list[tuple[str, dict[str, Any]]] = []
        for spec in raw:
            name, config = (spec, {}) if isinstance(spec, str) else spec
            if name not in _HANDLERS:
                msg = f"Unknown augmentation: {name!r}. Supported: {tuple(_HANDLERS)}"
                raise ValueError(msg)
            normalized.append((name, config))
        return normalized

    @staticmethod
    def _build_chains(
        specs: list[tuple[str, dict[str, Any]]],
    ) -> list[list[tuple[str, dict[str, Any]]]]:
        """Cartesian product of concrete ``(name, config)`` pairs across the chain."""
        if not specs:
            return [[]]  # empty chain → one identity variant
        expansions: list[list[tuple[str, dict[str, Any]]]] = []
        for name, cfg_spec in specs:
            if name in _NON_EXPANDING:
                expansions.append([(name, cfg_spec)])
            else:
                expansions.append([(name, cfg) for cfg in expand_config(cfg_spec)])
        return [list(combo) for combo in itertools.product(*expansions)]

    @staticmethod
    def _apply_chain(
        data: AudioSignal,
        chain: list[tuple[str, dict[str, Any]]],
        ctx: AugmentContext,
    ) -> AudioSignal:
        working, is_torch = Augmenter._to_numpy(data)
        target_len = len(working)
        for name, config in chain:
            working = _HANDLERS[name](working, config, ctx)
        # Length-changing augmentations (speed, stretch) are re-aligned to the input
        # length so every output signal is stackable alongside its source. Pad with
        # symmetric zeros if shorter; center-crop if longer.
        if len(working) != target_len:
            working = _fix_length(working, target_len)
        return Augmenter._from_numpy(working, to_torch=is_torch)

    @staticmethod
    def _to_numpy(data: AudioSignal) -> tuple[NDArray[np.float64], bool]:
        """Convert a single 1-D signal to float64 numpy, tracking original type."""
        if _TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            array, is_torch = data.numpy().astype(np.float64), True
        elif isinstance(data, np.ndarray):
            array, is_torch = data.astype(np.float64), False
        else:
            msg = f"Expected numpy.ndarray or torch.Tensor, got {type(data)}"
            raise TypeError(msg)

        if array.ndim != 1:
            msg = (
                f"Augmenter requires each audio sample to be 1-D mono; "
                f"got shape {array.shape} (ndim={array.ndim}). "
                f"For a batch, pass a 2-D array (B, N) or a list of 1-D arrays."
            )
            raise ValueError(msg)

        return array, is_torch

    @staticmethod
    def _from_numpy(data: NDArray[np.float64], *, to_torch: bool) -> AudioSignal:
        if to_torch and _TORCH_AVAILABLE:
            return torch.from_numpy(data)
        return data


def _fix_length(signal: NDArray[np.float64], target_len: int) -> NDArray[np.float64]:
    """Re-align a 1-D signal to ``target_len``: center-crop if longer, symmetric zero-pad if shorter."""
    n = len(signal)
    if n == target_len:
        return signal
    if n > target_len:
        start = (n - target_len) // 2
        return signal[start : start + target_len].copy()
    pad_total = target_len - n
    left = pad_total // 2
    right = pad_total - left
    return np.pad(signal, (left, right))


def _check_2d_orientation(shape: tuple[int, ...]) -> None:
    """Flag obviously-transposed 2-D inputs (``(samples, batch)`` instead of ``(batch, samples)``)."""
    b, n = shape
    if n < _MIN_REASONABLE_SAMPLES <= b:
        msg = (
            f"2-D input shape {shape} looks transposed — expected (batch, samples) "
            f"where the trailing dim is the time axis. Got only {n} samples per clip. "
            f"Did you mean shape ({n}, {b})? If {n} is really your clip length, "
            f"pass a list of 1-D signals to bypass this check."
        )
        raise ValueError(msg)


def _check_uniform_length(signals: list[AudioSignal]) -> None:
    """All batched signals must have the same length — pre-process clips before calling ``augment``.

    Skips elements that aren't 1-D array-like; those will fail later in ``_to_numpy``
    with a clearer type/shape error.
    """
    lengths: set[int] = set()
    for s in signals:
        if hasattr(s, "shape") and getattr(s, "ndim", None) == 1:
            lengths.add(int(s.shape[-1]))
    if len(lengths) > 1:
        msg = (
            f"All input signals must have the same length. Got lengths: {sorted(lengths)}. "
            f"Pre-process to a fixed length (random crop + zero-pad) before calling augment()."
        )
        raise ValueError(msg)


def _as_signal_list(data: AudioData) -> list[AudioSignal]:
    """Flatten any supported input form into an ordered list of 1-D signals.

    Validates: 2-D inputs for `(batch, samples)` orientation; all batched inputs
    for uniform length.
    """
    if isinstance(data, (list, tuple)):
        signals = list(data)
        _check_uniform_length(signals)
        return signals
    if isinstance(data, np.ndarray) and data.ndim == _BATCH_NDIM:
        _check_2d_orientation(data.shape)
        return [cast("AudioSignal", row) for row in data]
    if _TORCH_AVAILABLE and isinstance(data, torch.Tensor) and data.ndim == _BATCH_NDIM:
        _check_2d_orientation(tuple(data.shape))
        return [cast("AudioSignal", row) for row in data]
    return [cast("AudioSignal", data)]
