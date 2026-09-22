"""
ta_patterns.chart_patterns.short
=================================
Short bar patterns (2–10 bars).

Return type
-----------
All functions return **int8** arrays: +1 bullish, -1 bearish, 0 not fired.
Non-directional patterns return +1 when the shape fires.

Reimports
---------
Several patterns here were already implemented in the candlestick module.
Those are reimported directly to avoid code duplication:
  hook_reversal_bottom / top     ← single.hammer / hanging_man shape
  double_key_reversal_*          ← two_bar.engulfing with key-day confirmation
  fakey_bullish / bearish        ← three_bar.hikkake_bullish / bearish
  turn_key_bullish / bearish     ← reimport from candlestick multi_bar
  two_step_bullish / bearish     ← reimport from candlestick multi_bar
"""
from __future__ import annotations
import numpy as np
from .._core import _to_np, uptrend, downtrend, _EPS
from numpy.lib.stride_tricks import sliding_window_view
from ._windows import trailing_windows, RangeAgg

_a = _to_np


def _i(arr):  return arr.astype(np.int8)
def _ni(arr): return -arr.astype(np.int8)


def _combine(bull: np.ndarray, bear: np.ndarray) -> np.ndarray:
    """
    Merge a bullish half (+1/0) and a bearish half (-1/0) into one
    signed int8 array: +1 bullish fired, -1 bearish fired, 0 neither.

    The two halves are mutually exclusive in practice (a top and a bottom
    cannot complete on the same bar), but if they ever coincide the result
    is clipped to 0 so the combined signal never reports a spurious value.
    """
    out = bull.astype(np.int8) + bear.astype(np.int8)
    return np.clip(out, -1, 1).astype(np.int8)


# ---------------------------------------------------------------------------
# Helper: pad result arrays to correct length
# ---------------------------------------------------------------------------

def _pad(cond: np.ndarray, orig_n: int, n_prepend: int) -> np.ndarray:
    result = np.zeros(orig_n, dtype=np.int8)
    safe   = min(len(cond), orig_n - n_prepend)
    if safe > 0:
        result[n_prepend : n_prepend + safe] = cond[:safe]
    return result


# ---------------------------------------------------------------------------
# 2B pattern  (top: second push fails; bottom: second dip holds)
# ---------------------------------------------------------------------------

def two_b_top(o, h, l, c, lookback: int = 5, tol: float = 0.01) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Price makes a new high, then the very next close is below the prior
    swing high within *lookback* bars, signalling a failed breakout.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback + 1, N):
        # Find highest high in lookback window (excluding bar i-1 and i)
        window_h = h[i - lookback - 1 : i - 1]
        if len(window_h) == 0:
            continue
        prior_high = window_h.max()
        # Bar i-1 must set a new high vs window
        if h[i - 1] > prior_high * (1 + tol):
            # Bar i must close back below prior_high
            if c[i] < prior_high:
                result[i] = -1
    return result


def two_b_bottom(o, h, l, c, lookback: int = 5, tol: float = 0.01) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Price makes a new low, then the very next close recovers above the
    prior swing low, signalling a failed breakdown.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback + 1, N):
        window_l = l[i - lookback - 1 : i - 1]
        if len(window_l) == 0:
            continue
        prior_low = window_l.min()
        if l[i - 1] < prior_low * (1 - tol):
            if c[i] > prior_low:
                result[i] = 1
    return result


# ---------------------------------------------------------------------------
# 1-2-3 trend change  (three-pivot reversal)
# ---------------------------------------------------------------------------

def _one_two_three(c, main_idx, sec_idx, main_px, sec_px, N, window,
                   is_bottom):
    """Vectorised core for the 1-2-3 reversal pair.

    The original filtered the whole pivot array with a boolean mask on every
    bar, which is O(N*P).  Both pivot arrays are sorted, so the in-window slice
    and the "first opposite pivot after point 1" lookup are ``searchsorted``
    gathers evaluated for all bars at once.
    """
    result = np.zeros(N, dtype=np.int8)
    if window >= N or len(main_idx) < 2 or len(sec_idx) == 0:
        return result

    i = np.arange(window, N)
    a = np.searchsorted(main_idx, i - window, side='left')
    b = np.searchsorted(main_idx, i, side='left')
    have = (b - a) >= 2

    j1 = np.clip(b - 2, 0, len(main_idx) - 1)
    j3 = np.clip(b - 1, 0, len(main_idx) - 1)
    p1, p3 = main_idx[j1], main_idx[j3]

    # point 2 = first opposite pivot strictly after point 1, still in window
    bh = np.searchsorted(sec_idx, i, side='left')
    k = np.searchsorted(sec_idx, p1, side='right')
    have &= k < bh
    kk = np.clip(k, 0, len(sec_idx) - 1)
    p2 = sec_idx[kk]

    fire = have & (p2 < p3)
    # point 3 must retrace less far than point 1
    fire &= (main_px[p3] > main_px[p1]) if is_bottom else (main_px[p3] < main_px[p1])
    # breakout through point 2
    fire &= (c[window:] > sec_px[p2]) if is_bottom else (c[window:] < sec_px[p2])

    result[window:] = np.where(fire, 1 if is_bottom else -1, 0)
    return result


def one_two_three_bottom(o, h, l, c, window: int = 20,
                          pivot_n: int = 3, tol: float = 0.01) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Downtrend followed by: point-1 (low), point-2 (bounce high),
    point-3 (higher low > point-1), then price breaks above point-2.
    Signal fires at the point-2 breakout bar.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    from ._core import pit_pivot_highs, pit_pivot_lows
    ph_mask = pit_pivot_highs(h, pivot_n)
    pl_mask = pit_pivot_lows(l, pivot_n)
    N = len(c)
    ph_idx = np.where(ph_mask)[0]
    pl_idx = np.where(pl_mask)[0]
    return _one_two_three(c, pl_idx, ph_idx, l, h, N, window, is_bottom=True)


def one_two_three_top(o, h, l, c, window: int = 20,
                       pivot_n: int = 3, tol: float = 0.01) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Uptrend followed by: point-1 (high), point-2 (pullback low),
    point-3 (lower high < point-1), then price breaks below point-2.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    from ._core import pit_pivot_highs, pit_pivot_lows
    ph_mask = pit_pivot_highs(h, pivot_n)
    pl_mask = pit_pivot_lows(l, pivot_n)
    N = len(c)
    ph_idx = np.where(ph_mask)[0]
    pl_idx = np.where(pl_mask)[0]
    return _one_two_three(c, ph_idx, pl_idx, h, l, N, window, is_bottom=False)


# ---------------------------------------------------------------------------
# Two-close reversal  (2-close)
# ---------------------------------------------------------------------------

def two_close_bullish(o, h, l, c) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    D1 is a down bar (c1 < o1).  D2 closes above D1's open.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    cond = (c1 < o1) & (c0 > o1)
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = 1
    return result


def two_close_bearish(o, h, l, c) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    D1 is an up bar (c1 > o1).  D2 closes below D1's open.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    cond = (c1 > o1) & (c0 < o1)
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = -1
    return result


# ---------------------------------------------------------------------------
# 2-dance, 2-did, 2-tall  (Bulkowski's short-term setups)
# ---------------------------------------------------------------------------

def two_dance_bullish(o, h, l, c, tol: float = 0.005) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Two bars with similar closes (within tol), second bar white.
    Price consolidates at support before continuing up.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    c1, c0 = c[:-1], c[1:]
    o0 = o[1:]
    price_tol = tol * (np.abs(c1) + np.abs(c0)) / 2.0 + _EPS
    cond = (np.abs(c0 - c1) <= price_tol) & (c0 > o0)
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = 1
    return result


def two_dance_bearish(o, h, l, c, tol: float = 0.005) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Two bars with similar closes, second bar black — consolidation at resistance.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    c1, c0 = c[:-1], c[1:]
    o0 = o[1:]
    price_tol = tol * (np.abs(c1) + np.abs(c0)) / 2.0 + _EPS
    cond = (np.abs(c0 - c1) <= price_tol) & (c0 < o0)
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = -1
    return result


def two_did_bullish(o, h, l, c, tol: float = 0.005) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    D2 closes above D1 close (D1 was a down bar) — buyers absorb selling.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    c0      = c[1:]
    cond = (c1 < o1) & (c0 > c1)
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = 1
    return result


def two_did_bearish(o, h, l, c, tol: float = 0.005) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    D2 closes below D1 close (D1 was an up bar) — sellers absorb buying.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    c0      = c[1:]
    cond = (c1 > o1) & (c0 < c1)
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = -1
    return result


def two_tall_bullish(o, h, l, c, body_factor: float = 1.5) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    D2 is a tall (large body) white bar, significantly larger than D1 body.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    bd1 = np.abs(c1 - o1)
    bd0 = np.abs(c0 - o0)
    cond = (c0 > o0) & (bd0 > body_factor * bd1 + _EPS)
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = 1
    return result


def two_tall_bearish(o, h, l, c, body_factor: float = 1.5) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    D2 is a tall black bar, significantly larger than D1 body.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, c1 = o[:-1], c[:-1]
    o0, c0 = o[1:],  c[1:]
    bd1 = np.abs(c1 - o1)
    bd0 = np.abs(c0 - o0)
    cond = (c0 < o0) & (bd0 > body_factor * bd1 + _EPS)
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = -1
    return result


# ---------------------------------------------------------------------------
# 3-bar reversal
# ---------------------------------------------------------------------------

def three_bar_bullish(o, h, l, c) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Three-bar pattern: down bar, inside or narrow bar, up bar closing above D1 open.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2, h2, l2, c2 = o[:-2], h[:-2], l[:-2], c[:-2]
    o1, h1, l1, c1 = o[1:-1], h[1:-1], l[1:-1], c[1:-1]
    o0, h0, l0, c0 = o[2:],   h[2:],   l[2:],   c[2:]
    cond = (
        (c2 < o2)                          # D1 down
        & (h1 <= h2) & (l1 >= l2)          # D2 inside or equal range
        & (c0 > o0) & (c0 > o2)            # D3 up, closes above D1 open
    )
    result = np.zeros(len(c), dtype=np.int8)
    result[2:][cond] = 1
    return result


def three_bar_bearish(o, h, l, c) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Three-bar pattern: up bar, inside bar, down bar closing below D1 open.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2, h2, l2, c2 = o[:-2], h[:-2], l[:-2], c[:-2]
    o1, h1, l1, c1 = o[1:-1], h[1:-1], l[1:-1], c[1:-1]
    o0, h0, l0, c0 = o[2:],   h[2:],   l[2:],   c[2:]
    cond = (
        (c2 > o2)
        & (h1 <= h2) & (l1 >= l2)
        & (c0 < o0) & (c0 < o2)
    )
    result = np.zeros(len(c), dtype=np.int8)
    result[2:][cond] = -1
    return result


# ---------------------------------------------------------------------------
# 3DC  (3-day compression / narrow range squeeze)
# ---------------------------------------------------------------------------

def three_day_compression(o, h, l, c, factor: float = 0.5) -> np.ndarray:
    """
    **NON-DIRECTIONAL** (+1 / 0).
    Each of the three bars has a total range less than *factor* × the
    range of the bar 3 bars prior.  Signals a volatility squeeze.

    NON-DIRECTIONAL WARNING: returns +1 for the shape — provides no
    inherent price direction.  Use with other context.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    rng  = h - l
    N    = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(3, N):
        ref = rng[i - 3]
        if rng[i-2] < factor*ref and rng[i-1] < factor*ref and rng[i] < factor*ref:
            result[i] = 1
    return result


# ---------------------------------------------------------------------------
# 3L-R  (3-leg reversal)
# ---------------------------------------------------------------------------

def three_lr_bullish(o, h, l, c, window: int = 15) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Three consecutive lower lows (3 legs down) followed by a higher close.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(3, N):
        if (l[i-3] > l[i-2] > l[i-1]) and c[i] > c[i-1]:
            result[i] = 1
    return result


def three_lr_bearish(o, h, l, c, window: int = 15) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Three consecutive higher highs (3 legs up) followed by a lower close.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(3, N):
        if (h[i-3] < h[i-2] < h[i-1]) and c[i] < c[i-1]:
            result[i] = -1
    return result


def three_lr_inverted_bullish(o, h, l, c) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Inverted 3L-R: three higher lows (hammering the floor) + up close.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(3, N):
        if (l[i-3] < l[i-2] < l[i-1]) and c[i] > c[i-1]:
            result[i] = 1
    return result


def three_lr_inverted_bearish(o, h, l, c) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Inverted 3L-R: three lower highs (ceiling pressing down) + down close.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(3, N):
        if (h[i-3] > h[i-2] > h[i-1]) and c[i] < c[i-1]:
            result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Gap2H  (gap-to-high and gap-to-low)
# ---------------------------------------------------------------------------

def gap2h(o, h, l, c) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Current bar opens with a gap up above the prior two bars' highs.
    Signals strong momentum continuation.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(2, N):
        if o[i] > max(h[i-1], h[i-2]):
            result[i] = 1
    return result


def gap2h_inverted(o, h, l, c) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Current bar opens with a gap down below the prior two bars' lows.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(2, N):
        if o[i] < min(l[i-1], l[i-2]):
            result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Key reversal  (wide-range day closing opposite to prior trend)
# ---------------------------------------------------------------------------

def key_reversal_bottom(o, h, l, c, lookback: int = 5) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Bar makes a new *n*-bar low intraday but closes above the prior close.
    Classic upside key reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback, N):
        prior_low = l[i - lookback : i].min()
        if l[i] < prior_low and c[i] > c[i-1]:
            result[i] = 1
    return result


def key_reversal_top(o, h, l, c, lookback: int = 5) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Bar makes a new *n*-bar high intraday but closes below the prior close.
    Classic downside key reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback, N):
        prior_high = h[i - lookback : i].max()
        if h[i] > prior_high and c[i] < c[i-1]:
            result[i] = -1
    return result


def key_reversal_v2_bullish(o, h, l, c, lookback: int = 5) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    V2: new *n*-bar low AND closes above prior bar's high (stronger).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback, N):
        prior_low = l[i - lookback : i].min()
        if l[i] < prior_low and c[i] > h[i-1]:
            result[i] = 1
    return result


def key_reversal_v2_bearish(o, h, l, c, lookback: int = 5) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    V2: new *n*-bar high AND closes below prior bar's low (stronger).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback, N):
        prior_high = h[i - lookback : i].max()
        if h[i] > prior_high and c[i] < l[i-1]:
            result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Hook reversal
# ---------------------------------------------------------------------------

def hook_reversal_bottom(o, h, l, c) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    D1: down bar (c1 < c0_prior).  D2: opens inside D1 range, closes above D1 open.
    Reimplemented here; semantically equivalent to the candlestick harami + confirm.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, h1, l1, c1 = o[:-1], h[:-1], l[:-1], c[:-1]
    o0, h0, l0, c0 = o[1:],  h[1:],  l[1:],  c[1:]
    cond = (
        (c1 < o1)                               # D1 down
        & (o0 > l1) & (o0 < h1)                 # D2 opens inside D1 range
        & (c0 > o1)                              # D2 closes above D1 open
    )
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = 1
    return result


def hook_reversal_top(o, h, l, c) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    D1 up bar; D2 opens inside D1 range and closes below D1 open.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, h1, l1, c1 = o[:-1], h[:-1], l[:-1], c[:-1]
    o0, h0, l0, c0 = o[1:],  h[1:],  l[1:],  c[1:]
    cond = (
        (c1 > o1)
        & (o0 > l1) & (o0 < h1)
        & (c0 < o1)
    )
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = -1
    return result


# ---------------------------------------------------------------------------
# Inside day / Outside day
# ---------------------------------------------------------------------------

def inside_day(o, h, l, c) -> np.ndarray:
    """
    **NON-DIRECTIONAL** (+1 / 0).
    Current bar's high-low range is entirely inside the prior bar's range.

    NON-DIRECTIONAL WARNING: returns +1 for the shape only.
    """
    h, l = _a(h), _a(l)
    result = np.zeros(len(h), dtype=np.int8)
    result[1:][(h[1:] < h[:-1]) & (l[1:] > l[:-1])] = 1
    return result


def outside_day_bullish(o, h, l, c) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Outside day (range engulfs prior) with a bullish close (c > prior high).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    cond = (h[1:] > h[:-1]) & (l[1:] < l[:-1]) & (c[1:] > h[:-1])
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = 1
    return result


def outside_day_bearish(o, h, l, c) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Outside day with a bearish close (c < prior low).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    cond = (h[1:] > h[:-1]) & (l[1:] < l[:-1]) & (c[1:] < l[:-1])
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = -1
    return result


# ---------------------------------------------------------------------------
# Closing price reversal
# ---------------------------------------------------------------------------

def closing_price_reversal_bottom(o, h, l, c, lookback: int = 5) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Downtrend (new *n*-bar low), but closes higher than it opened.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback, N):
        if l[i] < l[i-lookback:i].min() and c[i] > o[i]:
            result[i] = 1
    return result


def closing_price_reversal_top(o, h, l, c, lookback: int = 5) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Uptrend (new *n*-bar high), but closes lower than it opened.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback, N):
        if h[i] > h[i-lookback:i].max() and c[i] < o[i]:
            result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Open-close reversal
# ---------------------------------------------------------------------------

def open_close_reversal_bottom(o, h, l, c) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    D1 closes near its low; D2 opens lower but closes above D1 close.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, h1, l1, c1 = o[:-1], h[:-1], l[:-1], c[:-1]
    o0, c0 = o[1:], c[1:]
    r1 = np.where(h1 > l1, h1 - l1, _EPS)
    cond = (
        ((c1 - l1) / r1 < 0.3)   # D1 closes near low
        & (o0 < c1)                # D2 opens below D1 close
        & (c0 > c1)                # D2 closes above D1 close
    )
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = 1
    return result


def open_close_reversal_top(o, h, l, c) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    D1 closes near its high; D2 opens higher but closes below D1 close.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, h1, l1, c1 = o[:-1], h[:-1], l[:-1], c[:-1]
    o0, c0 = o[1:], c[1:]
    r1 = np.where(h1 > l1, h1 - l1, _EPS)
    cond = (
        ((h1 - c1) / r1 < 0.3)
        & (o0 > c1)
        & (c0 < c1)
    )
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = -1
    return result


# ---------------------------------------------------------------------------
# One-day reversal
# ---------------------------------------------------------------------------

def one_day_reversal_bottom(o, h, l, c, lookback: int = 10) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Single bar that makes a new *n*-bar low but closes above the midpoint
    of its own range (strong recovery within the session).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback, N):
        if l[i] < l[i-lookback:i].min() and c[i] > (h[i] + l[i]) / 2.0:
            result[i] = 1
    return result


def one_day_reversal_top(o, h, l, c, lookback: int = 10) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Single bar: new *n*-bar high but closes below the midpoint of its range.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback, N):
        if h[i] > h[i-lookback:i].max() and c[i] < (h[i] + l[i]) / 2.0:
            result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Pivot point reversal
# ---------------------------------------------------------------------------

def pivot_point_reversal_bottom(o, h, l, c) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    D1 is a down bar; D2 gaps up (opens above D1 high) and closes higher.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, h1, l1, c1 = o[:-1], h[:-1], l[:-1], c[:-1]
    o0, c0 = o[1:], c[1:]
    cond = (c1 < o1) & (o0 > h1) & (c0 > o0)
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = 1
    return result


def pivot_point_reversal_top(o, h, l, c) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    D1 is an up bar; D2 gaps down (opens below D1 low) and closes lower.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o1, h1, l1, c1 = o[:-1], h[:-1], l[:-1], c[:-1]
    o0, c0 = o[1:], c[1:]
    cond = (c1 > o1) & (o0 < l1) & (c0 < o0)
    result = np.zeros(len(c), dtype=np.int8)
    result[1:][cond] = -1
    return result


# ---------------------------------------------------------------------------
# Double key reversal  (reimport wrapper from candlestick)
# ---------------------------------------------------------------------------

def double_key_reversal_bullish(o, h, l, c) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Reimports from candlestick module.  3-bar: D1 down, D2 key-reversal up,
    D3 confirms by closing above D2 close.
    """
    from .. import three_inside_up as _tiu   # structural equivalent
    # Own implementation: D1 black, D2 key reversal up (new low + close > D1 high),
    # D3 white closing above D2 close
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2, h2, l2, c2 = o[:-2], h[:-2], l[:-2], c[:-2]
    o1, h1, l1, c1 = o[1:-1], h[1:-1], l[1:-1], c[1:-1]
    o0, h0, l0, c0 = o[2:],   h[2:],   l[2:],   c[2:]
    cond = (
        (c2 < o2)           # D1 down
        & (l1 < l2)          # D2 new low
        & (c1 > h2)          # D2 key reversal — closes above D1 high
        & (c0 > c1)          # D3 confirms
    )
    result = np.zeros(len(c), dtype=np.int8)
    result[2:][cond] = 1
    return result


def double_key_reversal_bearish(o, h, l, c) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    3-bar: D1 up, D2 key reversal down (new high + close < D1 low), D3 confirms.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o2, h2, l2, c2 = o[:-2], h[:-2], l[:-2], c[:-2]
    o1, h1, l1, c1 = o[1:-1], h[1:-1], l[1:-1], c[1:-1]
    o0, h0, l0, c0 = o[2:],   h[2:],   l[2:],   c[2:]
    cond = (
        (c2 > o2)
        & (h1 > h2)
        & (c1 < l2)
        & (c0 < c1)
    )
    result = np.zeros(len(c), dtype=np.int8)
    result[2:][cond] = -1
    return result


# ---------------------------------------------------------------------------
# Fakey  (inside bar false breakout — reimport from candlestick)
# ---------------------------------------------------------------------------

def fakey_bullish(o, h, l, c) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).  Reimports semantics from candlestick hikkake_bullish.
    Inside bar followed by false downside break that reverses upward.
    """
    from .. import hikkake_bullish
    return hikkake_bullish(o, h, l, c)


def fakey_bearish(o, h, l, c) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).  Reimports semantics from candlestick hikkake_bearish.
    Inside bar followed by false upside break that reverses downward.
    """
    from .. import hikkake_bearish
    return hikkake_bearish(o, h, l, c)


# ---------------------------------------------------------------------------
# Narrow range 4  (NR4) and Narrow range 7  (NR7)
# ---------------------------------------------------------------------------

def narrow_range_4(o, h, l, c) -> np.ndarray:
    """
    **NON-DIRECTIONAL** (+1 / 0).
    Current bar has the narrowest high-low range of the last 4 bars.
    Signals a volatility contraction — breakout expected.

    NON-DIRECTIONAL WARNING: returns +1 for the shape only.
    """
    h, l = _a(h), _a(l)
    rng = h - l
    N = len(rng)
    result = np.zeros(N, dtype=np.int8)
    for i in range(3, N):
        if rng[i] == rng[i-3:i+1].min():
            result[i] = 1
    return result


def narrow_range_7(o, h, l, c) -> np.ndarray:
    """
    **NON-DIRECTIONAL** (+1 / 0).
    Current bar has the narrowest range of the last 7 bars.
    Stronger squeeze signal than NR4.

    NON-DIRECTIONAL WARNING: returns +1 for the shape only.
    """
    h, l = _a(h), _a(l)
    rng = h - l
    N = len(rng)
    result = np.zeros(N, dtype=np.int8)
    for i in range(6, N):
        if rng[i] == rng[i-6:i+1].min():
            result[i] = 1
    return result


# ---------------------------------------------------------------------------
# Wide ranging day
# ---------------------------------------------------------------------------

def wide_ranging_day_up(o, h, l, c, factor: float = 2.0,
                         lookback: int = 10) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Bar with a very wide range (> factor × average) closing in its upper half.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    rng = h - l
    N   = len(c)
    result = np.zeros(N, dtype=np.int8)
    if lookback >= N:
        return result
    avg_rng = trailing_windows(rng, lookback).mean(axis=1)
    fire = (rng[lookback:] > factor * avg_rng) & \
           (c[lookback:] > (h[lookback:] + l[lookback:]) / 2.0)
    result[lookback:] = np.where(fire, 1, 0)
    return result


def wide_ranging_day_down(o, h, l, c, factor: float = 2.0,
                           lookback: int = 10) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Wide-range bar closing in its lower half.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    rng = h - l
    N   = len(c)
    result = np.zeros(N, dtype=np.int8)
    if lookback >= N:
        return result
    avg_rng = trailing_windows(rng, lookback).mean(axis=1)
    fire = (rng[lookback:] > factor * avg_rng) & \
           (c[lookback:] < (h[lookback:] + l[lookback:]) / 2.0)
    result[lookback:] = np.where(fire, -1, 0)
    return result


# ---------------------------------------------------------------------------
# Weekly reversal  (requires weekly-bar logic; applied to any timeframe)
# ---------------------------------------------------------------------------

def weekly_reversal_upside(o, h, l, c, lookback: int = 5) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Bar makes a new *lookback*-bar low intraday but closes above the prior
    bar's close AND above the current open.  Applied on any timeframe.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback, N):
        if (l[i] < l[i-lookback:i].min()
                and c[i] > c[i-1]
                and c[i] > o[i]):
            result[i] = 1
    return result


def weekly_reversal_downside(o, h, l, c, lookback: int = 5) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    New *lookback*-bar high intraday but closes below prior close and open.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback, N):
        if (h[i] > h[i-lookback:i].max()
                and c[i] < c[i-1]
                and c[i] < o[i]):
            result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Turn-key  (4-bar pattern — reimport from candlestick multi_bar)
# ---------------------------------------------------------------------------

def turn_key_bullish(o, h, l, c) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    3 black candles (lower lows) then a wide white bar closing above D1 open.
    Reimports from candlestick multi_bar.three_line_strike_bullish semantics.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o3, c3 = o[:-3], c[:-3]
    o2, c2 = o[1:-2], c[1:-2]
    o1, c1 = o[2:-1], c[2:-1]
    o0, c0 = o[3:],   c[3:]
    cond = (
        (c3 < o3) & (c2 < o2) & (c1 < o1)
        & (c3 > c2) & (c2 > c1)
        & (c0 > o0)
        & (c0 > o3)
    )
    result = np.zeros(len(c), dtype=np.int8)
    result[3:][cond] = 1
    return result


def turn_key_bearish(o, h, l, c) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    3 white candles then a wide black bar closing below D1 open.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    o3, c3 = o[:-3], c[:-3]
    o2, c2 = o[1:-2], c[1:-2]
    o1, c1 = o[2:-1], c[2:-1]
    o0, c0 = o[3:],   c[3:]
    cond = (
        (c3 > o3) & (c2 > o2) & (c1 > o1)
        & (c3 < c2) & (c2 < c1)
        & (c0 < o0)
        & (c0 < o3)
    )
    result = np.zeros(len(c), dtype=np.int8)
    result[3:][cond] = -1
    return result


# ---------------------------------------------------------------------------
# 2-step reversal  (5-bar pattern)
# ---------------------------------------------------------------------------

def two_step_bullish(o, h, l, c) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    5-bar pattern: big down move (D1), partial bounce (D2-D4), resumption up (D5).
    Similar to breakaway_bullish from candlestick module.
    """
    from .. import breakaway_bullish
    return breakaway_bullish(o, h, l, c)


def two_step_bearish(o, h, l, c) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    5-bar pattern: big up move (D1), partial pullback (D2-D4), resumption down (D5).
    """
    from .. import breakaway_bearish
    return breakaway_bearish(o, h, l, c)


# ---------------------------------------------------------------------------
# Shark-32  (non-directional volatility setup)
# ---------------------------------------------------------------------------

def shark_32(o, h, l, c) -> np.ndarray:
    """
    **NON-DIRECTIONAL** (+1 / 0).
    Bar 3 is an inside bar of bar 2, which is an inside bar of bar 1.
    Three consecutive inside bars signal extreme compression.

    NON-DIRECTIONAL WARNING: direction of breakout is not predicted.
    """
    h, l = _a(h), _a(l)
    N = len(h)
    result = np.zeros(N, dtype=np.int8)
    for i in range(2, N):
        if (h[i] < h[i-1] < h[i-2]) and (l[i] > l[i-1] > l[i-2]):
            result[i] = 1
    return result


# ---------------------------------------------------------------------------
# V-pivot  (sharp single-bar reversal)
# ---------------------------------------------------------------------------

def v_pivot(o, h, l, c, lookback: int = 10, body_pct: float = 0.02) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Bar makes a sharp new low (new *lookback*-bar low) with a small body
    — a V-reversal spike. High recovery from intraday low.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback, N):
        if (l[i] < l[i-lookback:i].min()
                and (c[i] - l[i]) / (h[i] - l[i] + _EPS) > 0.7):
            result[i] = 1
    return result


def inverted_v_pivot(o, h, l, c, lookback: int = 10) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Bar makes a sharp new high (new *lookback*-bar high) then collapses —
    a V-top spike with strong rejection from intraday high.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback, N):
        if (h[i] > h[i-lookback:i].max()
                and (h[i] - c[i]) / (h[i] - l[i] + _EPS) > 0.7):
            result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Carl V  (Bulkowski's specific pattern: V-shaped price structure)
# ---------------------------------------------------------------------------

def carl_v_bullish(o, h, l, c, window: int = 10) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Price drops sharply, then recovers to (or above) the starting level
    within *window* bars — a completed V recovery.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(window, N):
        start_price = c[i - window]
        min_price   = c[i - window : i].min()
        if min_price < start_price * 0.97 and c[i] >= start_price * 0.99:
            result[i] = 1
    return result


def carl_v_bearish(o, h, l, c, window: int = 10) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Price rises sharply, then collapses back to (or below) the starting
    level within *window* bars — an inverted V.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(window, N):
        start_price = c[i - window]
        max_price   = c[i - window : i].max()
        if max_price > start_price * 1.03 and c[i] <= start_price * 1.01:
            result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Diving board  (bullish) / Pothole  (bearish)
# ---------------------------------------------------------------------------

def diving_board(o, h, l, c, lookback: int = 10,
                  drop_pct: float = 0.03) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Price dips sharply for 1-3 bars (diving board shape), then recovers.
    Signal fires on the recovery bar.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback + 2, N):
        ref = c[i - 3]
        trough = min(l[i-2], l[i-1])
        if trough < ref * (1 - drop_pct) and c[i] > ref * 0.99:
            result[i] = 1
    return result


def pothole(o, h, l, c, lookback: int = 10, drop_pct: float = 0.03) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    A pothole is a brief downward retrace (a "pothole in the road") that digs
    below a recent flat base of support for 1-3 bars and then recovers, with
    price closing back at or above the base.  Per Bulkowski it is a bullish
    continuation pattern within an upward trend, not a bearish one.  The
    signal fires on the recovery bar.

    Parameters
    ----------
    lookback : bars of flat base used as the reference level.
    drop_pct : minimum depth of the dip below the base, as a fraction.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(lookback + 2, N):
        ref    = c[i - 3]
        trough = min(l[i - 2], l[i - 1])
        if trough < ref * (1 - drop_pct) and c[i] >= ref * 0.99:
            result[i] = 1
    return result


def double_key_reversal(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish, -1 bearish."""
    return _combine(double_key_reversal_bullish(o, h, l, c, **kw),
                    double_key_reversal_bearish(o, h, l, c, **kw))


def one_two_three(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bottom, -1 top."""
    return _combine(one_two_three_bottom(o, h, l, c, **kw),
                    one_two_three_top(o, h, l, c, **kw))


def two_b(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bottom, -1 top."""
    return _combine(two_b_bottom(o, h, l, c, **kw),
                    two_b_top(o, h, l, c, **kw))


def two_close(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish, -1 bearish."""
    return _combine(two_close_bullish(o, h, l, c, **kw),
                    two_close_bearish(o, h, l, c, **kw))


def two_dance(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish setup, -1 bearish setup."""
    return _combine(two_dance_bullish(o, h, l, c, **kw),
                    two_dance_bearish(o, h, l, c, **kw))


def two_did(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish, -1 bearish."""
    return _combine(two_did_bullish(o, h, l, c, **kw),
                    two_did_bearish(o, h, l, c, **kw))


def two_tall(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish, -1 bearish."""
    return _combine(two_tall_bullish(o, h, l, c, **kw),
                    two_tall_bearish(o, h, l, c, **kw))


def three_bar(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish, -1 bearish."""
    return _combine(three_bar_bullish(o, h, l, c, **kw),
                    three_bar_bearish(o, h, l, c, **kw))


def three_lr(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish, -1 bearish."""
    return _combine(three_lr_bullish(o, h, l, c, **kw),
                    three_lr_bearish(o, h, l, c, **kw))


def closing_price_reversal(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bottom, -1 top."""
    return _combine(closing_price_reversal_bottom(o, h, l, c, **kw),
                    closing_price_reversal_top(o, h, l, c, **kw))


def open_close_reversal(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bottom, -1 top."""
    return _combine(open_close_reversal_bottom(o, h, l, c, **kw),
                    open_close_reversal_top(o, h, l, c, **kw))


def one_day_reversal(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bottom, -1 top."""
    return _combine(one_day_reversal_bottom(o, h, l, c, **kw),
                    one_day_reversal_top(o, h, l, c, **kw))


def pivot_point_reversal(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bottom, -1 top."""
    return _combine(pivot_point_reversal_bottom(o, h, l, c, **kw),
                    pivot_point_reversal_top(o, h, l, c, **kw))


def gap2h_combined(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish gap2h, -1 bearish inverted."""
    return _combine(gap2h(o, h, l, c, **kw),
                    gap2h_inverted(o, h, l, c, **kw))


def turn_key(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish, -1 bearish."""
    return _combine(turn_key_bullish(o, h, l, c, **kw),
                    turn_key_bearish(o, h, l, c, **kw))


def two_step(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish, -1 bearish."""
    return _combine(two_step_bullish(o, h, l, c, **kw),
                    two_step_bearish(o, h, l, c, **kw))


def hook_reversal(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bottom, -1 top."""
    return _combine(hook_reversal_bottom(o, h, l, c, **kw),
                    hook_reversal_top(o, h, l, c, **kw))


def weekly_reversal(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 upside, -1 downside."""
    return _combine(weekly_reversal_upside(o, h, l, c, **kw),
                    weekly_reversal_downside(o, h, l, c, **kw))


def wide_ranging_day(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 upside reversal, -1 downside reversal."""
    return _combine(wide_ranging_day_up(o, h, l, c, **kw),
                    wide_ranging_day_down(o, h, l, c, **kw))


def outside_day(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish outside day, -1 bearish."""
    return _combine(outside_day_bullish(o, h, l, c, **kw),
                    outside_day_bearish(o, h, l, c, **kw))


def fakey(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish, -1 bearish."""
    return _combine(fakey_bullish(o, h, l, c, **kw),
                    fakey_bearish(o, h, l, c, **kw))


def carl_v(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish V, -1 bearish inverted-V."""
    return _combine(carl_v_bullish(o, h, l, c, **kw),
                    carl_v_bearish(o, h, l, c, **kw))


def three_dc(o, h, l, c) -> np.ndarray:
    """Alias for :func:`three_day_compression`."""
    return three_day_compression(o, h, l, c)


def key_reversal(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bottom, -1 top."""
    return _combine(key_reversal_bottom(o, h, l, c, **kw),
                    key_reversal_top(o, h, l, c, **kw))


def key_reversal_v2(o, h, l, c, **kw) -> np.ndarray:
    """**BIDIRECTIONAL**: +1 bullish v2, -1 bearish v2."""
    return _combine(key_reversal_v2_bullish(o, h, l, c, **kw),
                    key_reversal_v2_bearish(o, h, l, c, **kw))

# ---------------------------------------------------------------------------
# Horn top / Horn bottom  (3-bar)
# ---------------------------------------------------------------------------

def horn_top(o, h, l, c, tol: float = 0.02) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Two spike-highs of similar height separated by a smaller middle bar.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(2, N):
        if (abs(h[i] - h[i-2]) / (max(h[i], h[i-2]) + _EPS) < tol
                and h[i-1] < min(h[i], h[i-2]) * 0.99
                and c[i] < c[i-1]):
            result[i] = -1
    return result


def horn_bottom(o, h, l, c, tol: float = 0.02) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Two spike-lows of similar depth separated by a higher middle bar.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(2, N):
        if (abs(l[i] - l[i-2]) / (max(abs(l[i]), abs(l[i-2])) + _EPS) < tol
                and l[i-1] > max(l[i], l[i-2]) * 1.01
                and c[i] > c[i-1]):
            result[i] = 1
    return result


# ---------------------------------------------------------------------------
# Pipe top / Pipe bottom  (2-bar)
# ---------------------------------------------------------------------------

def pipe_top(o, h, l, c, height_tol: float = 0.02) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Two adjacent candles with similar tall highs — exhaustion at the top.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    from .._core import atr as _atr
    atr14 = _atr(h, l, c, 14)
    result = np.zeros(N, dtype=np.int8)
    for i in range(1, N):
        r1 = h[i-1] - l[i-1]
        r2 = h[i]   - l[i]
        if (atr14[i] > 0
                and r1 > atr14[i] * 1.2 and r2 > atr14[i] * 1.2
                and abs(h[i] - h[i-1]) / (h[i] + _EPS) < height_tol
                and c[i] < o[i]):
            result[i] = -1
    return result


def pipe_bottom(o, h, l, c, height_tol: float = 0.02) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Two adjacent candles with similar deep lows — capitulation at the bottom.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    from .._core import atr as _atr
    atr14 = _atr(h, l, c, 14)
    result = np.zeros(N, dtype=np.int8)
    for i in range(1, N):
        r1 = h[i-1] - l[i-1]
        r2 = h[i]   - l[i]
        if (atr14[i] > 0
                and r1 > atr14[i] * 1.2 and r2 > atr14[i] * 1.2
                and abs(l[i] - l[i-1]) / (abs(l[i]) + _EPS) < height_tol
                and c[i] > o[i]):
            result[i] = 1
    return result


# ---------------------------------------------------------------------------
# Vertical run up / down
# ---------------------------------------------------------------------------

def vertical_run_up(o, h, l, c, n: int = 5, close_pct: float = 0.75) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    *n* consecutive bars each closing in the upper *close_pct* fraction of
    their range — nearly vertical price climb.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    r = h - l
    strong = (c - l) / np.where(r > _EPS, r, 1.0) >= close_pct
    cs = np.concatenate([[0], np.cumsum(strong.astype(int))])
    result = np.zeros(N, dtype=np.int8)
    for i in range(n, N):
        if cs[i+1] - cs[i-n+1] == n:
            result[i] = 1
    return result


def vertical_run_down(o, h, l, c, n: int = 5, close_pct: float = 0.75) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    *n* consecutive bars each closing in the lower *close_pct* of their range.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    r = h - l
    strong = (h - c) / np.where(r > _EPS, r, 1.0) >= close_pct
    cs = np.concatenate([[0], np.cumsum(strong.astype(int))])
    result = np.zeros(N, dtype=np.int8)
    for i in range(n, N):
        if cs[i+1] - cs[i-n+1] == n:
            result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Spikes / tails  (single bar, unusual shadow)
# ---------------------------------------------------------------------------

def spikes(o, h, l, c, shadow_factor: float = 2.5) -> np.ndarray:
    """
    **BIDIRECTIONAL** (+1 lower spike / -1 upper spike / 0 none).
    Upper spike (bearish): upper wick ≥ shadow_factor × body.
    Lower spike (bullish): lower wick ≥ shadow_factor × body.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    body   = np.abs(c - o) + _EPS
    uw     = h - np.maximum(o, c)
    lw     = np.minimum(o, c) - l
    result = np.zeros(len(c), dtype=np.int8)
    result[lw >= shadow_factor * body] =  1
    result[uw >= shadow_factor * body] = -1
    return result


# ---------------------------------------------------------------------------
# Failure swing  (price-based momentum divergence)
# ---------------------------------------------------------------------------

def failure_swing(o, h, l, c, n: int = 14) -> np.ndarray:
    """
    **BIDIRECTIONAL** (+1 bullish / -1 bearish / 0 none).
    New intrabar extreme but close reverses through bar midpoint —
    a price-level approximation of the RSI failure swing.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(n, N):
        mid = (h[i] + l[i]) / 2.0
        if l[i] < l[i-n:i].min() and c[i] > mid:
            result[i] =  1
        elif h[i] > h[i-n:i].max() and c[i] < mid:
            result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Elevator stop  (price stalls at a level)
# ---------------------------------------------------------------------------

def elevator_stop(o, h, l, c, window: int = 5, tol: float = 0.005) -> np.ndarray:
    """
    **NON-DIRECTIONAL** (+1 when shape fires).
    Last *window* closes all within *tol* of each other — temporary
    equilibrium at a support/resistance level.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    if window >= N:
        return result
    W = sliding_window_view(np.ascontiguousarray(c), window)   # rows end at i
    band = (W.max(axis=1) - W.min(axis=1)) / (W.mean(axis=1) + _EPS)
    result[window:] = np.where(band[1:] < tol, 1, 0)
    return result


# ---------------------------------------------------------------------------
# Cloud bank  (bullish support cluster)
# ---------------------------------------------------------------------------

def cloud_bank(o, h, l, c, window: int = 20, tol: float = 0.03) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Recent closes cluster in a tight band (< *tol* range/mean) —
    a dense support zone that price is bouncing from.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    if window >= N:
        return result
    W = trailing_windows(c, window)
    wmin = W.min(axis=1)
    band = (W.max(axis=1) - wmin) / (W.mean(axis=1) + _EPS)
    fire = (band < tol) & (c[window:] >= wmin * 0.99)
    result[window:] = np.where(fire, 1, 0)
    return result


# ---------------------------------------------------------------------------
# Flat base  (tight consolidation before breakout)
# ---------------------------------------------------------------------------

def flat_base(o, h, l, c, window: int = 20,
               max_range_pct: float = 0.12, trend_n: int = 20) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Price consolidates in a narrow range (< *max_range_pct* of mean) over
    *window* bars while in an uptrend — bullish continuation setup.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    prior_up = uptrend(c, trend_n)
    start = window + trend_n
    if start >= N:
        return result

    Wh = trailing_windows(h, window)
    Wl = trailing_windows(l, window)
    Wc = trailing_windows(c, window)
    band = (Wh.max(axis=1) - Wl.min(axis=1)) / (Wc.mean(axis=1) + _EPS)

    k = np.arange(window, N) - window            # row index == i - window
    fire = (band < max_range_pct) & prior_up[k] & (k >= trend_n)
    result[window:] = np.where(fire, 1, 0)
    return result


# ---------------------------------------------------------------------------
# ABC correction  (simple 3-leg pullback in an uptrend)
# ---------------------------------------------------------------------------

def abc_correction(o, h, l, c, n: int = 5,
                    min_drop: float = 0.03, max_retrace: float = 0.70) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Three-leg pullback within an uptrend: A (high) → B (low) → C (partial
    bounce below A) → D (entry, higher low than B).  Detected using recent
    pivot extremes.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    from ._core import pivot_highs, pivot_lows, pivot_info
    ph   = pivot_highs(h, n)
    pl   = pivot_lows(l, n)
    t_ph, p_ph = pivot_info(h, ph, n)
    t_pl, p_pl = pivot_info(l, pl, n)
    for ia in range(len(t_ph)):
        tA, pA = t_ph[ia], p_ph[ia]
        B_cands = [(t_pl[j], p_pl[j]) for j in range(len(t_pl)) if t_pl[j] > tA]
        if not B_cands: continue
        tB, pB = B_cands[0]
        if (pA - pB) / (pA + _EPS) < min_drop: continue
        C_cands = [(t_ph[j], p_ph[j]) for j in range(len(t_ph))
                   if t_ph[j] > tB and p_ph[j] < pA]
        if not C_cands: continue
        tC, pC = C_cands[0]
        if (pC - pB) / (pA - pB + _EPS) > max_retrace: continue
        D_cands = [(t_pl[j], p_pl[j]) for j in range(len(t_pl))
                   if t_pl[j] > tC and p_pl[j] > pB]
        if not D_cands: continue
        tD = D_cands[0][0]
        if tD < N and result[tD] == 0:
            result[tD] = 1
    return result


# ---------------------------------------------------------------------------
# Dead-cat bounce  (bearish continuation / bullish inverted)
# ---------------------------------------------------------------------------

def dead_cat_bounce(o, h, l, c, drop_pct: float = 0.10,
                     bounce_pct: float = 0.50, window: int = 15) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    After a sharp decline (≥ *drop_pct* over *window* bars), a weak bounce
    retraces ≤ *bounce_pct* of the drop — expect resumption of the decline.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(window * 2, N):
        drop_end   = i - window // 2
        drop_start = drop_end - window
        if drop_start < 0: continue
        start_p = c[drop_start]
        end_p   = c[drop_end:drop_end + window//2].min()
        if (start_p - end_p) / (start_p + _EPS) < drop_pct: continue
        bounce = (c[i] - end_p) / (start_p - end_p + _EPS)
        if 0.05 < bounce < bounce_pct:
            result[i] = -1
    return result


def dead_cat_bounce_inv(o, h, l, c, rise_pct: float = 0.10,
                         pullback_pct: float = 0.50,
                         window: int = 15) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Sharp rise + shallow pullback — continuation rally expected.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    for i in range(window * 2, N):
        rise_end   = i - window // 2
        rise_start = rise_end - window
        if rise_start < 0: continue
        start_p = c[rise_start]
        end_p   = c[rise_end:rise_end + window//2].max()
        if (end_p - start_p) / (start_p + _EPS) < rise_pct: continue
        pullback = (end_p - c[i]) / (end_p - start_p + _EPS)
        if 0.05 < pullback < pullback_pct:
            result[i] = 1
    return result
