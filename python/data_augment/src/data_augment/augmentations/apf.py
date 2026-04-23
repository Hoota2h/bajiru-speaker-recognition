"""All-pass filter cascade — deterministic, one output per config.

Each section is either a 1st-order or 2nd-order APF with a user-specified
break/center frequency (and Q for 2nd-order). Coefficients come straight from
the biquad cookbook; stable by construction for any ``0 < f < sr/2`` and
``Q > 0``.

NOT PARTICULARLY USEFUL IF WE THROW AWAY PHASE INFORMATION BUT FOR TIME-DOMAIN STUFFS,
THIS WILL DO NICE THINGS TO THE WAVEFORM WITHOUT DRAMATICALLY CHANGING PERCEPTUAL QUALITY
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.signal

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from data_augment.augmentations._common import AugmentContext

_FIRST_ORDER = 1
_SECOND_ORDER = 2

_ALLOWED_CONFIG_KEYS: frozenset[str] = frozenset({"sections"})

_DEFAULT_SECTIONS: list[dict[str, Any]] = [
    {"order": 2, "freq_hz": 1000.0, "q": 1.0},
    {"order": 2, "freq_hz": 3000.0, "q": 1.0},
]


def _first_order_coefs(freq_hz: float, sr: float) -> tuple[list[float], list[float]]:
    """1st-order APF: break at ``freq_hz``. ``a = (1-tan(πf/sr)) / (1+tan(πf/sr))``."""
    t = math.tan(math.pi * freq_hz / sr)
    a = (1.0 - t) / (1.0 + t)
    return [-a, 1.0], [1.0, -a]


def _second_order_coefs(freq_hz: float, q: float, sr: float) -> tuple[list[float], list[float]]:
    """2nd-order APF (biquad cookbook): center at ``freq_hz``, quality ``q``."""
    w0 = 2.0 * math.pi * freq_hz / sr
    alpha = math.sin(w0) / (2.0 * q)
    c = math.cos(w0)
    b = [1.0 - alpha, -2.0 * c, 1.0 + alpha]
    a = [1.0 + alpha, -2.0 * c, 1.0 - alpha]
    return b, a


def _validate_freq(freq_hz: float, sr: float) -> None:
    if not (0.0 < freq_hz < sr / 2.0):
        msg = f"freq_hz must lie in (0, sr/2)=(0, {sr / 2}); got {freq_hz}"
        raise ValueError(msg)


def _validate_q(q: float) -> None:
    if q <= 0.0:
        msg = f"q must be positive; got {q}"
        raise ValueError(msg)


def _build_section(spec: dict[str, Any], sr: float) -> tuple[list[float], list[float]]:
    order = int(spec["order"])
    freq_hz = float(spec["freq_hz"])
    _validate_freq(freq_hz, sr)
    if order == _FIRST_ORDER:
        return _first_order_coefs(freq_hz, sr)
    if order == _SECOND_ORDER:
        q = float(spec["q"])
        _validate_q(q)
        return _second_order_coefs(freq_hz, q, sr)
    msg = f"section order must be 1 or 2, got {order}"
    raise ValueError(msg)


def apply_apf(
    signal: NDArray[np.float64],
    config: dict[str, Any],
    ctx: AugmentContext,
) -> NDArray[np.float64]:
    """Run the APF cascade defined by ``config['sections']`` (or a sensible default)."""
    unknown = set(config.keys()) - _ALLOWED_CONFIG_KEYS
    if unknown:
        # APF's config is passed through verbatim (not cartesian-expanded), so list-valued
        # extra keys would be silently dropped. FAIL OUT LOUD
        msg = (
            f"Unknown apf config keys: {sorted(unknown)}. "
            f"Supported: {sorted(_ALLOWED_CONFIG_KEYS)}. "
            "To cascade multiple APFs, chain multiple ('apf', {...}) entries."
        )
        raise ValueError(msg)
    sections: list[dict[str, Any]] = config.get("sections", _DEFAULT_SECTIONS)
    sr = float(ctx.sample_rate)

    y = signal.astype(np.float64, copy=True)
    for spec in sections:
        b, a = _build_section(spec, sr)
        y = scipy.signal.lfilter(b, a, y)
    return np.asarray(y, dtype=np.float64)
