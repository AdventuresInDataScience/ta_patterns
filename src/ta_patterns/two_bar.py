"""
candlesticks.two_bar
====================
Two-bar (2-candle) patterns.

Convention: the pattern *completes* at bar i (the current / second bar).
The returned boolean array has False at index 0 (no prior bar available).

Internal variable naming:
    o1, h1, l1, c1  →  bar i-1  (prior)
    o0, h0, l0, c0  →  bar i    (current)
"""
from __future__ import annotations
import numpy as np
from ._core import (
    _to_np,
    body, upper_wick, lower_wick, safe_range,
    body_top, body_bot, midpoint,
    uptrend, downtrend, _EPS,
)
from .single import doji, marubozu_white, marubozu_black

_a = _to_np   # handles numpy arrays AND pandas Series

def _i(arr): return arr.astype(np.int8)
def _ni(arr): return -arr.astype(np.int8)


def _pad1(cond: np.ndarray) -> np.ndarray:
    """Prepend one 0 to bring a length-(N-1) int8 array back to length N."""
    return np.concatenate([np.zeros(1, dtype=np.int8), cond])


def _gap_up(h1, l0):
    """l0 > h1 : current low is strictly above prior high (true gap up)."""
    return l0 > h1


def _gap_dn(l1, h0):
    """h0 < l1 : current high is strictly below prior low (true gap down)."""
    return h0 < l1


# ---------------------------------------------------------------------------
# Gapping dojis  (need prior bar context, so live here not in single.py)
# ---------------------------------------------------------------------------

def gapping_up_doji(o, h, l, c, doji_thr: float = 0.1) -> np.ndarray:
    """Doji that gaps UP from the prior bar (bearish warning in uptrend)."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    is_d = doji(o, h, l, c, doji_thr).astype(bool)
    return _i(_pad1((is_d[1:] & _gap_up(h[:-1], l[1:])).astype(bool)))


def gapping_down_doji(o, h, l, c, doji_thr: float = 0.1) -> np.ndarray:
    """Doji that gaps DOWN from the prior bar (bullish warning in downtrend)."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    is_d = doji(o, h, l, c, doji_thr).astype(bool)
    return _ni(_pad1((is_d[1:] & _gap_dn(l[:-1], h[1:])).astype(bool)))


# ---------------------------------------------------------------------------
# Engulfing
# ---------------------------------------------------------------------------

def engulfing_bullish(o, h, l, c) -> np.ndarray:
    """
    D1 black; D2 white whose body fully engulfs D1's body.
    Bullish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    cond = (
        (c1 < o1)                            # D1 black
        & (c0 > o0)                           # D2 white
        & (o0 <= c1)                          # D2 opens at/below D1 close
        & (c0 >= o1)                          # D2 closes at/above D1 open
        & (body(o0, c0) > body(o1, c1))       # D2 body larger
    )
    return _i(_pad1(cond.astype(bool)))


def engulfing_bearish(o, h, l, c) -> np.ndarray:
    """
    D1 white; D2 black whose body fully engulfs D1's body.
    Bearish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    cond = (
        (c1 > o1)
        & (c0 < o0)
        & (o0 >= c1)
        & (c0 <= o1)
        & (body(o0, c0) > body(o1, c1))
    )
    return _ni(_pad1(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Harami
# ---------------------------------------------------------------------------

def harami_bullish(o, h, l, c, body_ratio: float = 0.6) -> np.ndarray:
    """
    D1 long black; D2 small white body *inside* D1 body.
    Bullish reversal: sellers losing conviction.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    bd1 = body(o1, c1)
    cond = (
        (c1 < o1) & (c0 > o0)
        & (bd1 > 0)
        & (body(o0, c0) / (bd1 + _EPS) < body_ratio)
        & (body_top(o0, c0) <= body_top(o1, c1))
        & (body_bot(o0, c0) >= body_bot(o1, c1))
    )
    return _i(_pad1(cond.astype(bool)))


def harami_bearish(o, h, l, c, body_ratio: float = 0.6) -> np.ndarray:
    """
    D1 long white; D2 small black body *inside* D1 body.
    Bearish reversal: buyers losing conviction.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    bd1 = body(o1, c1)
    cond = (
        (c1 > o1) & (c0 < o0)
        & (bd1 > 0)
        & (body(o0, c0) / (bd1 + _EPS) < body_ratio)
        & (body_top(o0, c0) <= body_top(o1, c1))
        & (body_bot(o0, c0) >= body_bot(o1, c1))
    )
    return _ni(_pad1(cond.astype(bool)))


def harami_cross_bullish(o, h, l, c,
                          doji_thr: float = 0.1) -> np.ndarray:
    """D1 long black; D2 is a doji whose body is inside D1's body."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, h0, l0, c0 = o[1:], h[1:], l[1:], c[1:]
    cond = (
        (c1 < o1)
        & doji(o0, h0, l0, c0, doji_thr)
        & (body_top(o0, c0) <= body_top(o1, c1))
        & (body_bot(o0, c0) >= body_bot(o1, c1))
    )
    return _i(_pad1(cond.astype(bool)))


def harami_cross_bearish(o, h, l, c,
                          doji_thr: float = 0.1) -> np.ndarray:
    """D1 long white; D2 is a doji whose body is inside D1's body."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, h0, l0, c0 = o[1:], h[1:], l[1:], c[1:]
    cond = (
        (c1 > o1)
        & doji(o0, h0, l0, c0, doji_thr)
        & (body_top(o0, c0) <= body_top(o1, c1))
        & (body_bot(o0, c0) >= body_bot(o1, c1))
    )
    return _ni(_pad1(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Doji star
# ---------------------------------------------------------------------------

def doji_star_bullish(o, h, l, c, doji_thr: float = 0.1) -> np.ndarray:
    """
    D1 black; D2 is a doji that gaps DOWN below D1's body.
    Bullish reversal warning (two-bar version).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, h0, l0, c0 = o[1:], h[1:], l[1:], c[1:]
    cond = (
        (c1 < o1)
        & doji(o0, h0, l0, c0, doji_thr)
        & (body_top(o0, c0) < c1)             # D2 body below D1 close
    )
    return _i(_pad1(cond.astype(bool)))


def doji_star_bearish(o, h, l, c, doji_thr: float = 0.1) -> np.ndarray:
    """
    D1 white; D2 is a doji that gaps UP above D1's body.
    Bearish reversal warning (two-bar version).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, h0, l0, c0 = o[1:], h[1:], l[1:], c[1:]
    cond = (
        (c1 > o1)
        & doji(o0, h0, l0, c0, doji_thr)
        & (body_bot(o0, c0) > c1)             # D2 body above D1 close
    )
    return _ni(_pad1(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Dark cloud cover / Piercing pattern
# ---------------------------------------------------------------------------

def dark_cloud_cover(o, h, l, c,
                      penetration: float = 0.50,
                      d1_body_thr: float = 0.50) -> np.ndarray:
    """
    D1 long white; D2 black opens above D1 high then closes *penetration*
    fraction into D1's body (measured downward from D1's close).

    Parameters
    ----------
    penetration : float, default 0.50
        Fraction of D1's body that D2 must close through.
        0.50 → D2 closes below the midpoint (classic textbook definition).
        0.30 → looser; 0.70 → stricter (more bearish conviction required).
    d1_body_thr : float, default 0.50
        Minimum body/range ratio for D1 to qualify as a 'substantial' candle.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, h1, l1, c1 = o[:-1], h[:-1], l[:-1], c[:-1]
    o0, h0, l0, c0 = o[1:],  h[1:],  l[1:],  c[1:]
    # D2 must close below:  c1 - penetration * body(D1)
    threshold = c1 - penetration * body(o1, c1)
    cond = (
        (c1 > o1)
        & (body(o1, c1) / safe_range(h1, l1) > d1_body_thr)
        & (c0 < o0)
        & (o0 > h1)               # D2 opens above D1 high
        & (c0 < threshold)        # penetrates D1 body by required amount
        & (c0 > o1)               # but stays above D1 open
    )
    return _ni(_pad1(cond.astype(bool)))


def piercing_pattern(o, h, l, c,
                      penetration: float = 0.50,
                      d1_body_thr: float = 0.50) -> np.ndarray:
    """
    D1 long black; D2 white opens below D1 low then closes *penetration*
    fraction into D1's body (measured upward from D1's close).
    Bullish reversal.  Mirror image of dark_cloud_cover.

    Parameters
    ----------
    penetration : float, default 0.50
        0.50 → D2 closes above the midpoint (classic definition).
        0.30 → looser; 0.70 → stricter.
    d1_body_thr : float, default 0.50
        Minimum body/range ratio for D1.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, h1, l1, c1 = o[:-1], h[:-1], l[:-1], c[:-1]
    o0, h0, l0, c0 = o[1:],  h[1:],  l[1:],  c[1:]
    # D2 must close above:  c1 + penetration * body(D1)
    threshold = c1 + penetration * body(o1, c1)
    cond = (
        (c1 < o1)
        & (body(o1, c1) / safe_range(h1, l1) > d1_body_thr)
        & (c0 > o0)
        & (o0 < l1)               # D2 opens below D1 low
        & (c0 > threshold)        # penetrates D1 body by required amount
        & (c0 < o1)               # but stays below D1 open
    )
    return _i(_pad1(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Neck patterns  (bearish continuation)
# ---------------------------------------------------------------------------

def in_neck(o, h, l, c, tol: float = 0.001) -> np.ndarray:
    """
    D1 black; D2 white opens below D1 low and closes very close to D1 close.
    Bearish continuation – buyers barely made it back.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, l1, c1 = o[:-1], l[:-1], c[:-1]
    o0, c0     = o[1:],  c[1:]
    close_tol  = tol * np.abs(c1) + _EPS
    cond = (
        (c1 < o1) & (c0 > o0)
        & (o0 < l1)
        & (np.abs(c0 - c1) <= close_tol)
    )
    return _ni(_pad1(cond.astype(bool)))


def on_neck(o, h, l, c, tol: float = 0.001) -> np.ndarray:
    """
    D1 black; D2 white opens below D1 low and closes at (within tol of) D1 low.
    Bearish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, l1, c1 = o[:-1], l[:-1], c[:-1]
    o0, c0     = o[1:],  c[1:]
    low_tol    = tol * np.abs(l1) + _EPS
    cond = (
        (c1 < o1) & (c0 > o0)
        & (o0 < l1)
        & (np.abs(c0 - l1) <= low_tol)
    )
    return _ni(_pad1(cond.astype(bool)))


def thrusting(o, h, l, c) -> np.ndarray:
    """
    D1 black; D2 white opens below D1 low and closes between D1 low and D1
    midpoint.  Weaker than piercing; bearish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, l1, c1 = o[:-1], l[:-1], c[:-1]
    o0, c0     = o[1:],  c[1:]
    mid1       = midpoint(o1, c1)
    cond = (
        (c1 < o1) & (c0 > o0)
        & (o0 < l1)
        & (c0 > l1)
        & (c0 < mid1)
    )
    return _ni(_pad1(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Kicking
# ---------------------------------------------------------------------------

def kicking_bullish(o, h, l, c, tol: float = 0.05) -> np.ndarray:
    """
    D1 black marubozu followed by a gap-UP white marubozu.
    Very strong bullish signal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    is_bm = marubozu_black(o, h, l, c, tol)
    is_wm = marubozu_white(o, h, l, c, tol)
    cond  = is_bm[:-1] & is_wm[1:] & _gap_up(h[:-1], l[1:])
    return _i(_pad1(cond.astype(bool)))


def kicking_bearish(o, h, l, c, tol: float = 0.05) -> np.ndarray:
    """
    D1 white marubozu followed by a gap-DOWN black marubozu.
    Very strong bearish signal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    is_wm = marubozu_white(o, h, l, c, tol)
    is_bm = marubozu_black(o, h, l, c, tol)
    cond  = is_wm[:-1] & is_bm[1:] & _gap_dn(l[:-1], h[1:])
    return _ni(_pad1(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Tweezers / matching low
# ---------------------------------------------------------------------------

def tweezer_tops(o, h, l, c, tol: float = 0.001) -> np.ndarray:
    """D1 and D2 share the same high (within *tol* of price). Bearish."""
    h = _a(h)
    price_tol = tol * (h[1:] + h[:-1]) / 2.0 + _EPS
    return _ni(_pad1((np.abs(h[1:] - h[:-1]) <= price_tol).astype(bool)))


def tweezer_bottoms(o, h, l, c, tol: float = 0.001) -> np.ndarray:
    """D1 and D2 share the same low (within *tol* of price). Bullish."""
    l = _a(l)
    price_tol = tol * (l[1:] + l[:-1]) / 2.0 + _EPS
    return _i(_pad1((np.abs(l[1:] - l[:-1]) <= price_tol).astype(bool)))


def matching_low(o, h, l, c, tol: float = 0.001) -> np.ndarray:
    """
    Two consecutive black candles with the same close (within *tol*).
    Bullish support signal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    price_tol = tol * (np.abs(c1) + np.abs(c0)) / 2.0 + _EPS
    cond = (
        (c1 < o1) & (c0 < o0)
        & (np.abs(c0 - c1) <= price_tol)
    )
    return _i(_pad1(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Meeting lines
# ---------------------------------------------------------------------------

def meeting_lines_bullish(o, h, l, c, tol: float = 0.001) -> np.ndarray:
    """
    D1 black; D2 white opens with a gap down but closes at D1's close level.
    Bullish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, l1, c1 = o[:-1], l[:-1], c[:-1]
    o0, c0     = o[1:],  c[1:]
    price_tol  = tol * np.abs(c1) + _EPS
    cond = (
        (c1 < o1) & (c0 > o0)
        & (o0 < l1)
        & (np.abs(c0 - c1) <= price_tol)
    )
    return _i(_pad1(cond.astype(bool)))


def meeting_lines_bearish(o, h, l, c, tol: float = 0.001) -> np.ndarray:
    """
    D1 white; D2 black opens with a gap up but closes at D1's close level.
    Bearish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, h1, c1 = o[:-1], h[:-1], c[:-1]
    o0, c0     = o[1:],  c[1:]
    price_tol  = tol * np.abs(c1) + _EPS
    cond = (
        (c1 > o1) & (c0 < o0)
        & (o0 > h1)
        & (np.abs(c0 - c1) <= price_tol)
    )
    return _ni(_pad1(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Separating lines
# ---------------------------------------------------------------------------

def separating_lines_bullish(o, h, l, c, tol: float = 0.001) -> np.ndarray:
    """
    D1 black; D2 white opens at the same price as D1 (within *tol*).
    Bullish continuation – buyers defended the open.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    price_tol = tol * (np.abs(o1) + np.abs(o0)) / 2.0 + _EPS
    cond = (
        (c1 < o1) & (c0 > o0)
        & (np.abs(o0 - o1) <= price_tol)
    )
    return _i(_pad1(cond.astype(bool)))


def separating_lines_bearish(o, h, l, c, tol: float = 0.001) -> np.ndarray:
    """
    D1 white; D2 black opens at the same price as D1 (within *tol*).
    Bearish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    price_tol = tol * (np.abs(o1) + np.abs(o0)) / 2.0 + _EPS
    cond = (
        (c1 > o1) & (c0 < o0)
        & (np.abs(o0 - o1) <= price_tol)
    )
    return _ni(_pad1(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Windows (gaps) and two-black-gapping
# ---------------------------------------------------------------------------

def rising_window(o, h, l, c) -> np.ndarray:
    """Gap UP: current low > prior high.  Bullish continuation."""
    h, l = _a(h), _a(l)
    return _i(_pad1((l[1:] > h[:-1]).astype(bool)))


def falling_window(o, h, l, c) -> np.ndarray:
    """Gap DOWN: current high < prior low.  Bearish continuation."""
    h, l = _a(h), _a(l)
    return _ni(_pad1((h[1:] < l[:-1]).astype(bool)))


def two_black_gapping(o, h, l, c) -> np.ndarray:
    """
    Two consecutive black candles with a gap DOWN between them.
    Bearish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, l1, c1 = o[:-1], l[:-1], c[:-1]
    o0, h0, c0 = o[1:],  h[1:],  c[1:]
    cond = (
        (c1 < o1) & (c0 < o0)
        & _gap_dn(l1, h0)
    )
    return _ni(_pad1(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Homing pigeon
# ---------------------------------------------------------------------------

def homing_pigeon(o, h, l, c) -> np.ndarray:
    """
    Two black candles; D2 body is entirely inside D1 body.
    Bullish reversal (sellers stalling).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    cond = (
        (c1 < o1) & (c0 < o0)
        & (body_top(o0, c0) <= body_top(o1, c1))
        & (body_bot(o0, c0) >= body_bot(o1, c1))
    )
    return _i(_pad1(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Above / below the stomach
# ---------------------------------------------------------------------------

def above_stomach(o, h, l, c) -> np.ndarray:
    """
    D1 black; D2 white opens within D1 body and closes above D1 midpoint.
    Bullish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    mid1 = midpoint(o1, c1)
    cond = (
        (c1 < o1) & (c0 > o0)
        & (o0 >= c1) & (o0 <= o1)            # D2 opens inside D1 body
        & (c0 > mid1)
    )
    return _i(_pad1(cond.astype(bool)))


def below_stomach(o, h, l, c) -> np.ndarray:
    """
    D1 white; D2 black opens within D1 body and closes below D1 midpoint.
    Bearish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    mid1 = midpoint(o1, c1)
    cond = (
        (c1 > o1) & (c0 < o0)
        & (o0 <= c1) & (o0 >= o1)            # D2 opens inside D1 body
        & (c0 < mid1)
    )
    return _ni(_pad1(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Last engulfing  (engulfing with trend context)
# ---------------------------------------------------------------------------

def last_engulfing_top(o, h, l, c, trend_n: int = 5) -> np.ndarray:
    """Bearish engulfing in an uptrend (final gasp / top reversal)."""
    return _ni((engulfing_bearish(o, h, l, c) != 0) & uptrend(np.asarray(c, float), trend_n))


def last_engulfing_bottom(o, h, l, c, trend_n: int = 5) -> np.ndarray:
    """Bullish engulfing in a downtrend (capitulation / bottom reversal)."""
    return _i((engulfing_bullish(o, h, l, c) != 0) & downtrend(np.asarray(c, float), trend_n))


# ---------------------------------------------------------------------------
# Inverted hammer  (2-line: prior bearish context + inverted hammer shape)
# ---------------------------------------------------------------------------

def inverted_hammer(o, h, l, c,
                    shadow_factor: float = 2.0,
                    lower_tol: float = 0.10) -> np.ndarray:
    """
    D1 black (bearish move); D2 has a small body at the bottom of its range
    with a long upper shadow and tiny/no lower shadow.  Bullish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1           = o[:-1], c[:-1]
    o0, h0, l0, c0   = o[1:], h[1:], l[1:], c[1:]
    r0  = safe_range(h0, l0)
    bd0 = body(o0, c0)
    shape = (
        (bd0 / r0 < 0.35)
        & (upper_wick(o0, h0, c0) >= shadow_factor * bd0 + _EPS)
        & (lower_wick(o0, l0, c0) / r0 < lower_tol)
        & (body_bot(o0, c0) < l0 + 0.45 * r0)    # body in lower portion
    )
    return _i(_pad1(((c1 < o1) & shape).astype(bool)))


# ---------------------------------------------------------------------------
# Shooting star  (2-line: confirmed by prior white + gap up)
# ---------------------------------------------------------------------------

def shooting_star_two_line(o, h, l, c,
                           shadow_factor: float = 2.0,
                           lower_tol: float = 0.10) -> np.ndarray:
    """
    D1 white (uptrend context); D2 gaps UP then forms a small body near the
    bottom with a long upper shadow.  Bearish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, h1, c1     = o[:-1], h[:-1], c[:-1]
    o0, h0, l0, c0 = o[1:], h[1:], l[1:], c[1:]
    r0  = safe_range(h0, l0)
    bd0 = body(o0, c0)
    shape = (
        (bd0 / r0 < 0.35)
        & (upper_wick(o0, h0, c0) >= shadow_factor * bd0 + _EPS)
        & (lower_wick(o0, l0, c0) / r0 < lower_tol)
        & (body_bot(o0, c0) < l0 + 0.45 * r0)
    )
    cond = (c1 > o1) & _gap_up(h1, l0) & shape
    return _ni(_pad1(cond.astype(bool)))
