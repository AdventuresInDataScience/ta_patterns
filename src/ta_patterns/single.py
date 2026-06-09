"""
ta_patterns.single
==================
Single-bar (1-candle) pattern detectors.

Return type
-----------
Every function returns a **numpy int8 array** of the same length as the
inputs:

    +1   pattern fired and is **bullish** (or non-directional — see below)
    -1   pattern fired and is **bearish**
     0   pattern did not fire

NON-DIRECTIONAL WARNING
-----------------------
``doji``, ``long_legged_doji``, ``rickshaw_man``, and ``high_wave`` are
**shape-only** patterns — they carry no inherent bullish or bearish bias.
They return **+1 when the shape fires, 0 otherwise**.  This is NOT a
bullish signal; it is simply a flag that a potentially significant
indecision bar has occurred.  Interpret in context.  These patterns are
tagged in the ``NON_DIRECTIONAL`` frozenset in the scanner module.

Threshold conventions
---------------------
All ratio thresholds are expressed as fractions of the bar's own
high-low range, so they scale automatically across instruments and
time-frames.  Every threshold is a keyword argument with a sensible
default.
"""
from __future__ import annotations
import numpy as np
from ._core import (
    _to_np,
    body, upper_wick, lower_wick, safe_range,
    body_top, body_bot, midpoint,
    uptrend, downtrend, _EPS,
)

_a = _to_np   # accepts numpy arrays AND pandas Series


def _i(arr: np.ndarray) -> np.ndarray:
    """Cast bool array to int8 (+1 / 0)."""
    return arr.astype(np.int8)


def _ni(arr: np.ndarray) -> np.ndarray:
    """Cast bool array to int8 and negate (0 / -1)."""
    return -arr.astype(np.int8)


# ---------------------------------------------------------------------------
# NON-DIRECTIONAL: shape-only patterns  (+1 when shape fires, 0 otherwise)
# See module-level NON-DIRECTIONAL WARNING above.
# ---------------------------------------------------------------------------

def doji(o, h, l, c, threshold: float = 0.1) -> np.ndarray:
    """
    **NON-DIRECTIONAL** — returns +1 for shape, 0 otherwise.
    Body < *threshold* × total range (default 10 %).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r = safe_range(h, l)
    return _i((h > l) & (body(o, c) / r < threshold))


def long_legged_doji(o, h, l, c,
                     doji_thr: float = 0.1,
                     shadow_thr: float = 0.20) -> np.ndarray:
    """
    **NON-DIRECTIONAL** — returns +1 for shape, 0 otherwise.
    Doji with significant shadows on BOTH sides.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r = safe_range(h, l)
    return _i(
        doji(o, h, l, c, doji_thr).astype(bool)
        & (upper_wick(o, h, c) / r > shadow_thr)
        & (lower_wick(o, l, c) / r > shadow_thr)
    )


def high_wave(o, h, l, c,
              body_thr: float = 0.20,
              shadow_thr: float = 0.20) -> np.ndarray:
    """
    **NON-DIRECTIONAL** — returns +1 for shape, 0 otherwise.
    Small real body with very long upper AND lower shadows.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r = safe_range(h, l)
    return _i(
        (body(o, c) / r < body_thr)
        & (upper_wick(o, h, c) / r > shadow_thr)
        & (lower_wick(o, l, c) / r > shadow_thr)
    )


def rickshaw_man(o, h, l, c,
                 doji_thr: float = 0.1,
                 shadow_thr: float = 0.20,
                 center_thr: float = 0.35) -> np.ndarray:
    """
    **NON-DIRECTIONAL** — returns +1 for shape, 0 otherwise.
    Long-legged doji with the body near the midpoint of the range.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r = safe_range(h, l)
    range_center = (h + l) / 2.0
    return _i(
        long_legged_doji(o, h, l, c, doji_thr, shadow_thr).astype(bool)
        & (np.abs(midpoint(o, c) - range_center) / r < center_thr)
    )


# ---------------------------------------------------------------------------
# Directional doji variants
# ---------------------------------------------------------------------------

def dragonfly_doji(o, h, l, c,
                   doji_thr: float = 0.1,
                   upper_thr: float = 0.05) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Doji with a long lower shadow and no/tiny upper shadow (T-shape).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r = safe_range(h, l)
    return _i(
        doji(o, h, l, c, doji_thr).astype(bool)
        & (upper_wick(o, h, c) / r < upper_thr)
        & (lower_wick(o, l, c) / r > 0.60)
    )


def gravestone_doji(o, h, l, c,
                    doji_thr: float = 0.1,
                    lower_thr: float = 0.05) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Doji with a long upper shadow and no/tiny lower shadow (inverted T).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r = safe_range(h, l)
    return _ni(
        doji(o, h, l, c, doji_thr).astype(bool)
        & (lower_wick(o, l, c) / r < lower_thr)
        & (upper_wick(o, h, c) / r > 0.60)
    )


def northern_doji(o, h, l, c,
                  doji_thr: float = 0.1, trend_n: int = 5) -> np.ndarray:
    """**BEARISH** (-1 / 0). Doji in an uptrend."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    return _ni(doji(o, h, l, c, doji_thr).astype(bool) & uptrend(c, trend_n))


def southern_doji(o, h, l, c,
                  doji_thr: float = 0.1, trend_n: int = 5) -> np.ndarray:
    """**BULLISH** (+1 / 0). Doji in a downtrend."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    return _i(doji(o, h, l, c, doji_thr).astype(bool) & downtrend(c, trend_n))


# ---------------------------------------------------------------------------
# Marubozu family
# ---------------------------------------------------------------------------

def marubozu_white(o, h, l, c, tol: float = 0.05) -> np.ndarray:
    """**BULLISH** (+1 / 0). White candle with no/negligible shadows."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    return _i((c > o) & (body(o, c) / safe_range(h, l) > 1.0 - tol))


def marubozu_black(o, h, l, c, tol: float = 0.05) -> np.ndarray:
    """**BEARISH** (-1 / 0). Black candle with no/negligible shadows."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    return _ni((c < o) & (body(o, c) / safe_range(h, l) > 1.0 - tol))


def closing_marubozu_white(o, h, l, c, tol: float = 0.05) -> np.ndarray:
    """**BULLISH** (+1 / 0). White candle closing at the high."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    return _i((c > o) & (upper_wick(o, h, c) / safe_range(h, l) < tol))


def closing_marubozu_black(o, h, l, c, tol: float = 0.05) -> np.ndarray:
    """**BEARISH** (-1 / 0). Black candle closing at the low."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    return _ni((c < o) & (lower_wick(o, l, c) / safe_range(h, l) < tol))


def opening_marubozu_white(o, h, l, c, tol: float = 0.05) -> np.ndarray:
    """**BULLISH** (+1 / 0). White candle opening at the low."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    return _i((c > o) & (lower_wick(o, l, c) / safe_range(h, l) < tol))


def opening_marubozu_black(o, h, l, c, tol: float = 0.05) -> np.ndarray:
    """**BEARISH** (-1 / 0). Black candle opening at the high."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    return _ni((c < o) & (upper_wick(o, h, c) / safe_range(h, l) < tol))


# ---------------------------------------------------------------------------
# Hammer / star family
# ---------------------------------------------------------------------------

def hammer(o, h, l, c,
           shadow_factor: float = 2.0,
           upper_tol: float = 0.10,
           require_trend: bool = True,
           trend_n: int = 5) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Small body in the upper portion; lower shadow ≥ shadow_factor × body;
    tiny upper shadow; typically in a downtrend.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r   = safe_range(h, l)
    bd  = body(o, c)
    lw  = lower_wick(o, l, c)
    shape = (
        (bd / r < 0.35)
        & (lw >= shadow_factor * bd + _EPS)
        & (upper_wick(o, h, c) / r < upper_tol)
        & (body_top(o, c) > l + 0.55 * r)
    )
    trend_mask = downtrend(c, trend_n) if require_trend else np.ones(len(c), bool)
    return _i(shape & trend_mask)


def hanging_man(o, h, l, c,
                shadow_factor: float = 2.0,
                upper_tol: float = 0.10,
                require_trend: bool = True,
                trend_n: int = 5) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Same shape as hammer but in an uptrend.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r   = safe_range(h, l)
    bd  = body(o, c)
    lw  = lower_wick(o, l, c)
    shape = (
        (bd / r < 0.35)
        & (lw >= shadow_factor * bd + _EPS)
        & (upper_wick(o, h, c) / r < upper_tol)
        & (body_top(o, c) > l + 0.55 * r)
    )
    trend_mask = uptrend(c, trend_n) if require_trend else np.ones(len(c), bool)
    return _ni(shape & trend_mask)


def shooting_star(o, h, l, c,
                  shadow_factor: float = 2.0,
                  lower_tol: float = 0.10,
                  require_trend: bool = True,
                  trend_n: int = 5) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Small body near the bottom; long upper shadow ≥ shadow_factor × body;
    tiny lower shadow; typically in an uptrend.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r   = safe_range(h, l)
    bd  = body(o, c)
    shape = (
        (bd / r < 0.35)
        & (upper_wick(o, h, c) >= shadow_factor * bd + _EPS)
        & (lower_wick(o, l, c) / r < lower_tol)
        & (body_bot(o, c) < l + 0.45 * r)
    )
    trend_mask = uptrend(c, trend_n) if require_trend else np.ones(len(c), bool)
    return _ni(shape & trend_mask)


def takuri_line(o, h, l, c,
                shadow_factor: float = 3.0,
                upper_tol: float = 0.05) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Like a hammer but with an unusually long lower shadow (≥ 3× body).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r   = safe_range(h, l)
    bd  = body(o, c)
    return _i(
        (bd / r < 0.25)
        & (lower_wick(o, l, c) >= shadow_factor * bd + _EPS)
        & (upper_wick(o, h, c) / r < upper_tol)
    )


# ---------------------------------------------------------------------------
# Belt hold
# ---------------------------------------------------------------------------

def belt_hold_bullish(o, h, l, c, tol: float = 0.05) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    White candle opening at/near the session low; closes strongly.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r = safe_range(h, l)
    return _i(
        (c > o)
        & (lower_wick(o, l, c) / r < tol)
        & (body(o, c) / r > 0.55)
    )


def belt_hold_bearish(o, h, l, c, tol: float = 0.05) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Black candle opening at/near the session high; closes weakly.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r = safe_range(h, l)
    return _ni(
        (c < o)
        & (upper_wick(o, h, c) / r < tol)
        & (body(o, c) / r > 0.55)
    )


# ---------------------------------------------------------------------------
# Long / short candle
# ---------------------------------------------------------------------------

def long_white_day(o, h, l, c, body_thr: float = 0.6) -> np.ndarray:
    """**BULLISH** (+1 / 0)."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    return _i((c > o) & (body(o, c) / safe_range(h, l) > body_thr))


def long_black_day(o, h, l, c, body_thr: float = 0.6) -> np.ndarray:
    """**BEARISH** (-1 / 0)."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    return _ni((c < o) & (body(o, c) / safe_range(h, l) > body_thr))


def short_white_candle(o, h, l, c, body_thr: float = 0.30) -> np.ndarray:
    """
    **NON-DIRECTIONAL** — returns +1 for shape, 0 otherwise.
    White candle with a small body (< body_thr of range).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    return _i((c > o) & (body(o, c) / safe_range(h, l) < body_thr))


def short_black_candle(o, h, l, c, body_thr: float = 0.30) -> np.ndarray:
    """
    **NON-DIRECTIONAL** — returns +1 for shape, 0 otherwise.
    Black candle with a small body (< body_thr of range).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    return _i((c < o) & (body(o, c) / safe_range(h, l) < body_thr))


def white_candle(o, c) -> np.ndarray:
    """**BULLISH** (+1 / 0). Any close > open candle."""
    return _i(_a(c) > _a(o))


def black_candle(o, c) -> np.ndarray:
    """**BEARISH** (-1 / 0). Any close < open candle."""
    return _ni(_a(c) < _a(o))


# ---------------------------------------------------------------------------
# Spinning top
# ---------------------------------------------------------------------------

def spinning_top_white(o, h, l, c,
                        body_thr: float = 0.35,
                        shadow_thr: float = 0.15) -> np.ndarray:
    """**BULLISH** (+1 / 0). White candle with small body and bilateral shadows."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r = safe_range(h, l)
    return _i(
        (c > o)
        & (body(o, c) / r < body_thr)
        & (upper_wick(o, h, c) / r > shadow_thr)
        & (lower_wick(o, l, c) / r > shadow_thr)
    )


def spinning_top_black(o, h, l, c,
                        body_thr: float = 0.35,
                        shadow_thr: float = 0.15) -> np.ndarray:
    """**BEARISH** (-1 / 0). Black candle with small body and bilateral shadows."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    r = safe_range(h, l)
    return _ni(
        (c < o)
        & (body(o, c) / r < body_thr)
        & (upper_wick(o, h, c) / r > shadow_thr)
        & (lower_wick(o, l, c) / r > shadow_thr)
    )
