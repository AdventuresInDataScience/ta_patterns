"""
candlesticks.multi_bar
======================
Four- and five-bar patterns.  True at index i when the pattern completes
at that bar's close.  The first (n-1) indices are always False.

Internal convention (bars indexed newest-first from the right):
    bars suffixed 4=oldest ... 0=current (newest)
"""
from __future__ import annotations
import numpy as np
from ._core import (
    _to_np,
    body, upper_wick, lower_wick, safe_range,
    body_top, body_bot, midpoint, _EPS,
)
from .single import marubozu_black, marubozu_white

_a = _to_np   # handles numpy arrays AND pandas Series

def _i(arr): return arr.astype(np.int8)
def _ni(arr): return -arr.astype(np.int8)


def _padn(cond: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([np.zeros(n, dtype=np.int8), cond])


# ---------------------------------------------------------------------------
# Four-bar patterns
# ---------------------------------------------------------------------------

def three_line_strike_bullish(o, h, l, c) -> np.ndarray:
    """
    Three white soldiers (D1-D3) followed by a single black candle (D4) that
    opens above D3 and closes at-or-below D1's open.
    Textbook says this is paradoxically a *bullish continuation* signal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o3,c3 = o[:-3], c[:-3]   # D1 oldest
    o2,c2 = o[1:-2], c[1:-2]
    o1,c1 = o[2:-1], c[2:-1]
    o0,c0 = o[3:],  c[3:]    # D4 current
    cond = (
        (c3>o3) & (c2>o2) & (c1>o1)            # D1-D3 white
        & (c3<c2) & (c2<c1)                     # rising closes
        & (c0<o0)                               # D4 black
        & (o0>c1)                               # D4 opens above D3
        & (c0<=o3)                              # D4 closes at/below D1 open
    )
    return _i(_padn(cond.astype(bool), 3))


def three_line_strike_bearish(o, h, l, c) -> np.ndarray:
    """
    Three black crows (D1-D3) followed by a white candle (D4) that
    closes at-or-above D1's open.  Bearish continuation signal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o3,c3 = o[:-3], c[:-3]
    o2,c2 = o[1:-2], c[1:-2]
    o1,c1 = o[2:-1], c[2:-1]
    o0,c0 = o[3:],  c[3:]
    cond = (
        (c3<o3) & (c2<o2) & (c1<o1)
        & (c3>c2) & (c2>c1)
        & (c0>o0)
        & (o0<c1)
        & (c0>=o3)
    )
    return _ni(_padn(cond.astype(bool), 3))


def concealing_baby_swallow(o, h, l, c, tol: float = 0.05) -> np.ndarray:
    """
    D1 & D2 black marubozus; D3 black with a gap-down open and upper shadow
    that reaches back into D2; D4 black that fully engulfs D3.
    Bullish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    is_bm = marubozu_black(o, h, l, c, tol)

    o3,h3,l3,c3 = o[:-3], h[:-3], l[:-3], c[:-3]
    o2,h2,l2,c2 = o[1:-2], h[1:-2], l[1:-2], c[1:-2]
    o1,h1,l1,c1 = o[2:-1], h[2:-1], l[2:-1], c[2:-1]
    o0,h0,l0,c0 = o[3:],  h[3:],  l[3:],  c[3:]

    cond = (
        is_bm[:-3] & is_bm[1:-2]               # D1 & D2 black marubozu
        & (c1<o1) & (o1<c2)                    # D3 black, gaps down
        & (upper_wick(o1,h1,c1) > 0)           # D3 has upper shadow
        & (h1>=c2)                              # shadow reaches into D2
        & (c0<o0)                               # D4 black
        & (o0>o1) & (c0<c1)                    # D4 engulfs D3
    )
    return _i(_padn(cond.astype(bool), 3))


# ---------------------------------------------------------------------------
# Five-bar patterns
# ---------------------------------------------------------------------------

def rising_three_methods(o, h, l, c) -> np.ndarray:
    """
    D1 long white; D2-D4 three small blacks staying within D1's range;
    D5 long white closing above D1's high.  Bullish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o4,h4,l4,c4 = o[:-4], h[:-4], l[:-4], c[:-4]  # D1
    o3,h3,l3,c3 = o[1:-3], h[1:-3], l[1:-3], c[1:-3]
    o2,h2,l2,c2 = o[2:-2], h[2:-2], l[2:-2], c[2:-2]
    o1,h1,l1,c1 = o[3:-1], h[3:-1], l[3:-1], c[3:-1]
    o0,h0,l0,c0 = o[4:],  h[4:],  l[4:],  c[4:]   # D5

    r4 = safe_range(h4, l4)
    r0 = safe_range(h0, l0)

    cond = (
        (c4>o4) & (body(o4,c4)/r4 > 0.50)      # D1 long white
        & (c3<o3) & (c2<o2) & (c1<o1)          # D2-D4 black
        & (h3<h4) & (h2<h4) & (h1<h4)          # within D1 range (high)
        & (l3>l4) & (l2>l4) & (l1>l4)          # within D1 range (low)
        & (c0>o0) & (body(o0,c0)/r0 > 0.50)    # D5 long white
        & (c0>h4)                               # D5 closes above D1 high
    )
    return _i(_padn(cond.astype(bool), 4))


def falling_three_methods(o, h, l, c) -> np.ndarray:
    """
    D1 long black; D2-D4 three small whites within D1's range;
    D5 long black closing below D1's low.  Bearish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o4,h4,l4,c4 = o[:-4], h[:-4], l[:-4], c[:-4]
    o3,h3,l3,c3 = o[1:-3], h[1:-3], l[1:-3], c[1:-3]
    o2,h2,l2,c2 = o[2:-2], h[2:-2], l[2:-2], c[2:-2]
    o1,h1,l1,c1 = o[3:-1], h[3:-1], l[3:-1], c[3:-1]
    o0,h0,l0,c0 = o[4:],  h[4:],  l[4:],  c[4:]

    r4 = safe_range(h4, l4)
    r0 = safe_range(h0, l0)

    cond = (
        (c4<o4) & (body(o4,c4)/r4 > 0.50)
        & (c3>o3) & (c2>o2) & (c1>o1)
        & (h3<h4) & (h2<h4) & (h1<h4)
        & (l3>l4) & (l2>l4) & (l1>l4)
        & (c0<o0) & (body(o0,c0)/r0 > 0.50)
        & (c0<l4)
    )
    return _ni(_padn(cond.astype(bool), 4))


def mat_hold(o, h, l, c) -> np.ndarray:
    """
    D1 long white; D2 small black (gap up); D3 & D4 small, staying above
    D1's close; D5 long white closing at a new high vs D2-D4.
    Bullish continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o4,h4,l4,c4 = o[:-4], h[:-4], l[:-4], c[:-4]
    o3,h3,l3,c3 = o[1:-3], h[1:-3], l[1:-3], c[1:-3]
    o2,h2,l2,c2 = o[2:-2], h[2:-2], l[2:-2], c[2:-2]
    o1,h1,l1,c1 = o[3:-1], h[3:-1], l[3:-1], c[3:-1]
    o0,h0,l0,c0 = o[4:],  h[4:],  l[4:],  c[4:]

    r4 = safe_range(h4, l4)
    max_h_mid = np.maximum(np.maximum(h3, h2), h1)

    cond = (
        (c4>o4) & (body(o4,c4)/r4 > 0.50)      # D1 long white
        & (c3<o3) & (o3>c4)                     # D2 black, gap up
        & (l3>l4) & (l2>c4) & (l1>c4)          # D2-D4 stay above D1 close
        & (c0>o0)                               # D5 white
        & (c0>max_h_mid)                        # D5 new high vs D2-D4
    )
    return _i(_padn(cond.astype(bool), 4))


def breakaway_bullish(o, h, l, c) -> np.ndarray:
    """
    D1 long black; D2 black (gap down from D1); D3 & D4 small/indecisive;
    D5 long white closing back into the gap (above D2 open, below D1 close).
    Bullish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o4,h4,l4,c4 = o[:-4], h[:-4], l[:-4], c[:-4]
    o3,h3,l3,c3 = o[1:-3], h[1:-3], l[1:-3], c[1:-3]
    o2,h2,l2,c2 = o[2:-2], h[2:-2], l[2:-2], c[2:-2]
    o1,h1,l1,c1 = o[3:-1], h[3:-1], l[3:-1], c[3:-1]
    o0,h0,l0,c0 = o[4:],  h[4:],  l[4:],  c[4:]

    r4 = safe_range(h4, l4)

    cond = (
        (c4<o4) & (body(o4,c4)/r4 > 0.50)      # D1 long black
        & (c3<o3) & (h3<l4)                     # D2 black, gaps down
        & (body(o2,c2) < body(o4,c4)*0.50)      # D3 small
        & (body(o1,c1) < body(o4,c4)*0.50)      # D4 small
        & (c0>o0)                               # D5 white
        & (c0>o3) & (c0<c4)                    # closes into gap
    )
    return _i(_padn(cond.astype(bool), 4))


def breakaway_bearish(o, h, l, c) -> np.ndarray:
    """
    D1 long white; D2 white (gap up); D3 & D4 small; D5 long black
    closing back into the gap.  Bearish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o4,h4,l4,c4 = o[:-4], h[:-4], l[:-4], c[:-4]
    o3,h3,l3,c3 = o[1:-3], h[1:-3], l[1:-3], c[1:-3]
    o2,h2,l2,c2 = o[2:-2], h[2:-2], l[2:-2], c[2:-2]
    o1,h1,l1,c1 = o[3:-1], h[3:-1], l[3:-1], c[3:-1]
    o0,h0,l0,c0 = o[4:],  h[4:],  l[4:],  c[4:]

    r4 = safe_range(h4, l4)

    cond = (
        (c4>o4) & (body(o4,c4)/r4 > 0.50)
        & (c3>o3) & (l3>h4)                     # D2 white, gaps up
        & (body(o2,c2) < body(o4,c4)*0.50)
        & (body(o1,c1) < body(o4,c4)*0.50)
        & (c0<o0)
        & (c0<o3) & (c0>c4)
    )
    return _ni(_padn(cond.astype(bool), 4))


def ladder_bottom(o, h, l, c) -> np.ndarray:
    """
    D1-D3 three successively lower black candles; D4 black with upper shadow
    (buyers trying); D5 white opening above D4 close.  Bullish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o4,h4,l4,c4 = o[:-4], h[:-4], l[:-4], c[:-4]
    o3,h3,l3,c3 = o[1:-3], h[1:-3], l[1:-3], c[1:-3]
    o2,h2,l2,c2 = o[2:-2], h[2:-2], l[2:-2], c[2:-2]
    o1,h1,l1,c1 = o[3:-1], h[3:-1], l[3:-1], c[3:-1]
    o0,h0,l0,c0 = o[4:],  h[4:],  l[4:],  c[4:]

    r1 = safe_range(h1, l1)

    cond = (
        (c4<o4) & (c3<o3) & (c2<o2)            # D1-D3 black
        & (c3<c4) & (c2<c3)                     # descending closes
        & (c1<o1)                               # D4 black
        & (upper_wick(o1,h1,c1)/r1 > 0.10)     # D4 has upper shadow
        & (c0>o0) & (o0>c1)                    # D5 white, opens above D4
    )
    return _i(_padn(cond.astype(bool), 4))


# ---------------------------------------------------------------------------
# New price lines  (N consecutive new closes in one direction)
# ---------------------------------------------------------------------------

def _consec_up(c: np.ndarray, n: int) -> np.ndarray:
    """True at i if the last n closes are each strictly higher than the one before."""
    c = _a(c)
    N = len(c)
    out = np.zeros(N, dtype=bool)
    if n < 2 or n > N:
        return out
    dc_pos = (np.diff(c) > 0).astype(np.int8)       # length N-1
    cs = np.concatenate([[0], np.cumsum(dc_pos)])    # length N
    # out[j] = True iff cs[j] - cs[j-n] == n  (all n steps positive)
    j = np.arange(n, N)
    out[j] = (cs[j] - cs[j - n]) == n
    return out.astype(np.int8)


def _consec_dn(c: np.ndarray, n: int) -> np.ndarray:
    c = _a(c)
    N = len(c)
    out = np.zeros(N, dtype=bool)
    if n < 2 or n > N:
        return out
    dc_neg = (np.diff(c) < 0).astype(np.int8)
    cs = np.concatenate([[0], np.cumsum(dc_neg)])
    j = np.arange(n, N)
    out[j] = (cs[j] - cs[j - n]) == n
    return -out.astype(np.int8)


def new_price_lines_up(c, n: int = 8) -> np.ndarray:
    """True at i if the last *n* closes all made consecutive new highs."""
    return _consec_up(c, n)


def new_price_lines_down(c, n: int = 8) -> np.ndarray:
    """True at i if the last *n* closes all made consecutive new lows."""
    return _consec_dn(c, n)


def eight_new_price_lines(c)   -> np.ndarray: return _consec_up(c, 8)
def ten_new_price_lines(c)     -> np.ndarray: return _consec_up(c, 10)
def twelve_new_price_lines(c)  -> np.ndarray: return _consec_up(c, 12)
def thirteen_new_price_lines(c) -> np.ndarray: return _consec_up(c, 13)
