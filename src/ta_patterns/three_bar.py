"""
candlesticks.three_bar
======================
Three-bar (3-candle) patterns.  True at index i when the pattern
completes at that bar's close.  First two indices are always False.

Internal variable naming:
    o2, h2, l2, c2  →  bar i-2  (oldest)
    o1, h1, l1, c1  →  bar i-1  (middle)
    o0, h0, l0, c0  →  bar i    (current / newest)
"""
from __future__ import annotations
import numpy as np
from ._core import (
    _to_np,
    body, upper_wick, lower_wick, safe_range,
    body_top, body_bot, midpoint,
    uptrend, downtrend, _EPS,
)
from .single import doji

_a = _to_np   # handles numpy arrays AND pandas Series

def _i(arr): return arr.astype(np.int8)
def _ni(arr): return -arr.astype(np.int8)


def _pad2(cond: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(2, dtype=np.int8), cond])


def _sl(o, h, l, c):
    """Return (o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0) as sliced arrays."""
    return (
        o[:-2], h[:-2], l[:-2], c[:-2],
        o[1:-1], h[1:-1], l[1:-1], c[1:-1],
        o[2:],   h[2:],   l[2:],   c[2:],
    )


# ---------------------------------------------------------------------------
# Morning / evening star family
# ---------------------------------------------------------------------------

def morning_star(o, h, l, c) -> np.ndarray:
    """
    D1 long black; D2 small body (any colour) below D1; D3 white closing
    above D1 midpoint.  Bullish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    r2 = safe_range(h2, l2)
    cond = (
        (c2 < o2) & (body(o2,c2)/r2 > 0.50)      # D1 long black
        & (body(o1,c1) < 0.50 * body(o2,c2))       # D2 small
        & (body_top(o1,c1) < c2)                    # D2 body below D1 close
        & (c0 > o0)                                  # D3 white
        & (c0 > midpoint(o2,c2))                    # D3 closes above D1 mid
    )
    return _i(_pad2(cond.astype(bool)))


def evening_star(o, h, l, c) -> np.ndarray:
    """
    D1 long white; D2 small body above D1; D3 black closing below D1 midpoint.
    Bearish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    r2 = safe_range(h2, l2)
    cond = (
        (c2 > o2) & (body(o2,c2)/r2 > 0.50)
        & (body(o1,c1) < 0.50 * body(o2,c2))
        & (body_bot(o1,c1) > c2)                    # D2 body above D1 close
        & (c0 < o0)
        & (c0 < midpoint(o2,c2))
    )
    return _ni(_pad2(cond.astype(bool)))


def morning_doji_star(o, h, l, c, doji_thr: float = 0.1) -> np.ndarray:
    """Like morning_star but D2 is specifically a doji."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    r2 = safe_range(h2, l2)
    cond = (
        (c2 < o2) & (body(o2,c2)/r2 > 0.50)
        & doji(o1,h1,l1,c1, doji_thr).astype(bool)
        & (body_top(o1,c1) < c2)
        & (c0 > o0)
        & (c0 > midpoint(o2,c2))
    )
    return _i(_pad2(cond.astype(bool)))


def evening_doji_star(o, h, l, c, doji_thr: float = 0.1) -> np.ndarray:
    """Like evening_star but D2 is specifically a doji."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    r2 = safe_range(h2, l2)
    cond = (
        (c2 > o2) & (body(o2,c2)/r2 > 0.50)
        & doji(o1,h1,l1,c1, doji_thr).astype(bool)
        & (body_bot(o1,c1) > c2)
        & (c0 < o0)
        & (c0 < midpoint(o2,c2))
    )
    return _ni(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Three soldiers / crows
# ---------------------------------------------------------------------------

def three_white_soldiers(o, h, l, c) -> np.ndarray:
    """
    Three consecutive white candles; each opens inside the prior body and
    closes progressively higher.  Bullish continuation / reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    r0 = safe_range(h0, l0)
    cond = (
        (c2>o2) & (c1>o1) & (c0>o0)            # all white
        & (c2<c1) & (c1<c0)                     # rising closes
        & (o1>=o2) & (o1<=c2)                   # D2 opens inside D1
        & (o0>=o1) & (o0<=c1)                   # D3 opens inside D2
        & (upper_wick(o0,h0,c0)/r0 < 0.30)     # small upper shadow on D3
    )
    return _i(_pad2(cond.astype(bool)))


def three_black_crows(o, h, l, c) -> np.ndarray:
    """
    Three consecutive black candles; each opens inside the prior body and
    closes progressively lower.  Bearish continuation / reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    r0 = safe_range(h0, l0)
    cond = (
        (c2<o2) & (c1<o1) & (c0<o0)
        & (c2>c1) & (c1>c0)                     # falling closes
        & (o1>=c2) & (o1<=o2)                   # D2 opens inside D1 body (black: c<o)
        & (o0>=c1) & (o0<=o1)
        & (lower_wick(o0,l0,c0)/r0 < 0.30)
    )
    return _ni(_pad2(cond.astype(bool)))


def identical_three_crows(o, h, l, c, tol: float = 0.001) -> np.ndarray:
    """
    Three black crows where each opens at (within *tol* of) the prior close.
    More bearish than standard three black crows.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    ptol = tol * np.abs(c2) + _EPS
    cond = (
        (c2<o2) & (c1<o1) & (c0<o0)
        & (c2>c1) & (c1>c0)
        & (np.abs(o1-c2) <= ptol)
        & (np.abs(o0-c1) <= ptol)
    )
    return _ni(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Abandoned baby
# ---------------------------------------------------------------------------

def abandoned_baby_bullish(o, h, l, c, doji_thr: float = 0.1) -> np.ndarray:
    """
    D1 black; D2 doji gaps down (no overlap with D1); D3 white gaps up
    (no overlap with D2).  Strongest bullish reversal in the star family.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2<o2)
        & doji(o1,h1,l1,c1, doji_thr).astype(bool)
        & (h1<l2)                               # D2 gaps down from D1
        & (c0>o0)
        & (l0>h1)                               # D3 gaps up from D2
    )
    return _i(_pad2(cond.astype(bool)))


def abandoned_baby_bearish(o, h, l, c, doji_thr: float = 0.1) -> np.ndarray:
    """
    D1 white; D2 doji gaps up; D3 black gaps down.
    Strongest bearish reversal in the star family.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2>o2)
        & doji(o1,h1,l1,c1, doji_thr).astype(bool)
        & (l1>h2)                               # D2 gaps up from D1
        & (c0<o0)
        & (h0<l1)                               # D3 gaps down from D2
    )
    return _ni(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Three inside / outside
# ---------------------------------------------------------------------------

def three_inside_up(o, h, l, c) -> np.ndarray:
    """
    D1 black; D2 white harami inside D1; D3 white closing above D1 open.
    Bullish reversal confirmed.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2<o2) & (c1>o1)
        & (body_top(o1,c1) <= body_top(o2,c2))
        & (body_bot(o1,c1) >= body_bot(o2,c2))
        & (c0>o0) & (c0>o2)                     # D3 closes above D1 open
    )
    return _i(_pad2(cond.astype(bool)))


def three_inside_down(o, h, l, c) -> np.ndarray:
    """
    D1 white; D2 black harami inside D1; D3 black closing below D1 open.
    Bearish reversal confirmed.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2>o2) & (c1<o1)
        & (body_top(o1,c1) <= body_top(o2,c2))
        & (body_bot(o1,c1) >= body_bot(o2,c2))
        & (c0<o0) & (c0<o2)
    )
    return _ni(_pad2(cond.astype(bool)))


def three_outside_up(o, h, l, c) -> np.ndarray:
    """
    D1 black; D2 white bullish engulfing; D3 white closing above D2.
    Bullish reversal confirmed.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2<o2) & (c1>o1)
        & (o1<=c2) & (c1>=o2)                   # D2 engulfs D1
        & (c0>o0) & (c0>c1)
    )
    return _i(_pad2(cond.astype(bool)))


def three_outside_down(o, h, l, c) -> np.ndarray:
    """
    D1 white; D2 black bearish engulfing; D3 black closing below D2.
    Bearish reversal confirmed.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2>o2) & (c1<o1)
        & (o1>=c2) & (c1<=o2)
        & (c0<o0) & (c0<c1)
    )
    return _ni(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Advance block / Deliberation
# ---------------------------------------------------------------------------

def advance_block(o, h, l, c) -> np.ndarray:
    """
    Three white candles with progressively smaller bodies and/or larger upper
    shadows – buyers exhausting.  Bearish warning.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    bd2,bd1,bd0 = body(o2,c2), body(o1,c1), body(o0,c0)
    uw2 = upper_wick(o2,h2,c2)
    uw1 = upper_wick(o1,h1,c1)
    uw0 = upper_wick(o0,h0,c0)
    cond = (
        (c2>o2) & (c1>o1) & (c0>o0)
        & (c2<c1) & (c1<c0)
        & (bd1 < bd2) & (bd0 < bd1)            # shrinking bodies
        & (uw1 > uw2) & (uw0 > uw1)            # growing upper shadows
    )
    return _ni(_pad2(cond.astype(bool)))


def deliberation(o, h, l, c, small_thr: float = 0.35) -> np.ndarray:
    """
    Three white candles; first two strong, third is small/doji-like.
    Bearish hesitation signal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    bd1,bd0 = body(o1,c1), body(o0,c0)
    r0 = safe_range(h0, l0)
    cond = (
        (c2>o2) & (c1>o1) & (c0>o0)
        & (c2<c1) & (c1<c0)
        & (bd1 > 0)
        & (bd0/r0 < small_thr)                  # D3 small body
        & (bd0 < 0.5*bd1)
    )
    return _ni(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Two crows / upside-gap two crows
# ---------------------------------------------------------------------------

def two_crows(o, h, l, c) -> np.ndarray:
    """
    D1 long white; D2 black gaps up into 'star' position; D3 black opens
    inside D2 body and closes inside D1 body.  Bearish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2>o2) & (body(o2,c2)/safe_range(h2,l2) > 0.50)
        & (c1<o1) & (o1>c2)                     # D2 gaps up
        & (c0<o0)
        & (o0>=c1) & (o0<=o1)                   # D3 opens inside D2
        & (c0>o2) & (c0<c2)                     # D3 closes inside D1 body
    )
    return _ni(_pad2(cond.astype(bool)))


def upside_gap_two_crows(o, h, l, c) -> np.ndarray:
    """
    D1 long white; D2 black gaps up above D1 body; D3 black engulfs D2
    but stays above D1 close.  Bearish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2>o2)
        & (c1<o1) & (o1>c2) & (c1>c2)          # D2 gaps up, stays above D1 close
        & (c0<o0)
        & (o0>=c1) & (o0<=o1)                   # D3 opens inside D2
        & (body(o0,c0) > body(o1,c1))           # D3 engulfs D2
        & (c0>c2)                               # D3 still above D1 close
    )
    return _ni(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Tasuki gaps
# ---------------------------------------------------------------------------

def downside_tasuki_gap(o, h, l, c) -> np.ndarray:
    """
    D1 & D2 black with a gap down; D3 white opens inside D2 and closes in
    the gap but does NOT close it.  Bearish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2<o2) & (c1<o1)
        & (h1<l2)                               # gap down between D1 and D2
        & (c0>o0)
        & (o0>=c1) & (o0<=o1)                   # D3 opens inside D2
        & (c0>h1)                               # D3 closes above D2 high (in gap)
        & (c0<l2)                               # gap remains open
    )
    return _ni(_pad2(cond.astype(bool)))


def upside_tasuki_gap(o, h, l, c) -> np.ndarray:
    """
    D1 & D2 white with a gap up; D3 black opens inside D2 and closes in
    the gap but does NOT close it.  Bullish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2>o2) & (c1>o1)
        & (l1>h2)                               # gap up
        & (c0<o0)
        & (o0>=o1) & (o0<=c1)                   # D3 opens inside D2
        & (c0<l1)                               # D3 closes below D2 low (in gap)
        & (c0>h2)                               # gap remains open
    )
    return _i(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Gap three methods  (continuation patterns)
# ---------------------------------------------------------------------------

def downside_gap_three_methods(o, h, l, c) -> np.ndarray:
    """
    D1 black; D2 black (gap down); D3 white closes inside D1 body.
    Bearish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2<o2) & (c1<o1)
        & (h1<l2)                               # gap down
        & (c0>o0)
        & (c0>body_bot(o2,c2)) & (c0<body_top(o2,c2))
    )
    return _ni(_pad2(cond.astype(bool)))


def upside_gap_three_methods(o, h, l, c) -> np.ndarray:
    """
    D1 white; D2 white (gap up); D3 black closes inside D1 body.
    Bullish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2>o2) & (c1>o1)
        & (l1>h2)
        & (c0<o0)
        & (c0<body_top(o2,c2)) & (c0>body_bot(o2,c2))
    )
    return _i(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Side-by-side white lines
# ---------------------------------------------------------------------------

def side_by_side_white_lines_bullish(o, h, l, c,
                                      size_tol: float = 0.20) -> np.ndarray:
    """
    D1 white; D2 & D3 are similar-sized white candles both opening with
    a gap up above D1.  Bullish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    bd1,bd0 = body(o1,c1), body(o0,c0)
    cond = (
        (c2>o2)
        & (c1>o1) & (l1>h2)                     # D2 white, gaps up
        & (c0>o0)
        & (np.abs(o0-o1)/(o1+_EPS) < size_tol) # similar opens
        & (np.abs(bd0-bd1)/(bd1+_EPS) < size_tol)
    )
    return _i(_pad2(cond.astype(bool)))


def side_by_side_white_lines_bearish(o, h, l, c,
                                      size_tol: float = 0.20) -> np.ndarray:
    """
    D1 black; D2 & D3 are similar-sized white candles with a gap down
    from D1.  Bearish continuation (counter-intuitive).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    bd1,bd0 = body(o1,c1), body(o0,c0)
    cond = (
        (c2<o2)
        & (c1>o1) & (h1<l2)                     # D2 white, gaps down
        & (c0>o0)
        & (np.abs(o0-o1)/(o1+_EPS) < size_tol)
        & (np.abs(bd0-bd1)/(bd1+_EPS) < size_tol)
    )
    return _ni(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Three stars in the south
# ---------------------------------------------------------------------------

def three_stars_south(o, h, l, c) -> np.ndarray:
    """
    Three black candles: each with a lower high and lower low; the third is
    a small near-marubozu.  Bullish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    r0 = safe_range(h0, l0)
    cond = (
        (c2<o2) & (c1<o1) & (c0<o0)
        & (h1<h2) & (h0<h1)                     # lower highs
        & (l1<l2) & (l0<l1)                     # lower lows
        & (body(o2,c2) > body(o1,c1)*1.2)       # D1 larger than D2
        & (body(o0,c0)/r0 > 0.80)               # D3 near-marubozu
    )
    return _i(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Stick sandwich
# ---------------------------------------------------------------------------

def stick_sandwich(o, h, l, c, tol: float = 0.001) -> np.ndarray:
    """
    D1 black; D2 white (higher close); D3 black with same close as D1.
    Bullish reversal (price found support at the matching closes).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    ptol = tol * np.abs(c2) + _EPS
    cond = (
        (c2<o2) & (c1>o1) & (c1>c2)
        & (c0<o0)
        & (np.abs(c0-c2) <= ptol)
    )
    return _i(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Unique three-river bottom
# ---------------------------------------------------------------------------

def unique_three_river_bottom(o, h, l, c) -> np.ndarray:
    """
    D1 long black; D2 black with lower low and long lower shadow;
    D3 small white closing inside D2 body.  Bullish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    r1 = safe_range(h1, l1)
    r2 = safe_range(h2, l2)
    cond = (
        (c2<o2) & (body(o2,c2)/r2 > 0.50)      # D1 long black
        & (c1<o1) & (l1<l2)                     # D2 black, lower low
        & (lower_wick(o1,l1,c1)/r1 > 0.40)     # D2 long lower shadow
        & (c0>o0)                               # D3 white
        & (c0<o1) & (c0>c1)                    # D3 inside D2 body
    )
    return _i(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Tri-star
# ---------------------------------------------------------------------------

def tri_star_bullish(o, h, l, c,
                     doji_thr: float = 0.1, trend_n: int = 5) -> np.ndarray:
    """Three consecutive dojis in a downtrend.  Bullish reversal."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    dn = downtrend(c, trend_n)[2:]
    cond = (
        doji(o2,h2,l2,c2,doji_thr).astype(bool)
        & doji(o1,h1,l1,c1, doji_thr).astype(bool).astype(bool)
        & doji(o0,h0,l0,c0, doji_thr).astype(bool).astype(bool)
        & dn
    )
    return _i(_pad2(cond.astype(bool)))


def tri_star_bearish(o, h, l, c,
                     doji_thr: float = 0.1, trend_n: int = 5) -> np.ndarray:
    """Three consecutive dojis in an uptrend.  Bearish reversal."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    up = uptrend(c, trend_n)[2:]
    cond = (
        doji(o2,h2,l2,c2,doji_thr).astype(bool)
        & doji(o1,h1,l1,c1, doji_thr).astype(bool).astype(bool)
        & doji(o0,h0,l0,c0, doji_thr).astype(bool).astype(bool)
        & up
    )
    return _ni(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Hikkake  (false breakout of an inside bar)
# ---------------------------------------------------------------------------

def hikkake_bullish(o, h, l, c) -> np.ndarray:
    """
    D1 any; D2 inside bar (high < D1 high, low > D1 low);
    D3 closes ABOVE D1 high – false downside breakout resolves bullishly.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (h1 < h2) & (l1 > l2)                  # D2 is an inside bar
        & (c0 > h2)                             # D3 closes above D1 high
    )
    return _i(_pad2(cond.astype(bool)))


def hikkake_bearish(o, h, l, c) -> np.ndarray:
    """
    D1 any; D2 inside bar; D3 closes BELOW D1 low – false upside breakout
    resolves bearishly.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (h1 < h2) & (l1 > l2)
        & (c0 < l2)
    )
    return _ni(_pad2(cond.astype(bool)))


# ---------------------------------------------------------------------------
# Collapsing doji star
# ---------------------------------------------------------------------------

def collapse_doji_star(o, h, l, c, doji_thr: float = 0.1) -> np.ndarray:
    """
    D1 black; D2 doji near D1 close level; D3 black closes below D1 close.
    Bearish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2,h2,l2,c2, o1,h1,l1,c1, o0,h0,l0,c0 = _sl(o,h,l,c)
    cond = (
        (c2<o2)
        & doji(o1,h1,l1,c1, doji_thr).astype(bool)
        & (c0<o0) & (c0<c2)
    )
    return _ni(_pad2(cond.astype(bool)))
