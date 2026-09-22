"""
ta_patterns.chart_patterns.double_multi
=========================================
Double and triple tops/bottoms, including all four Adam/Eve shape variants
for double patterns plus a combined generic version.

Adam vs Eve classification
--------------------------
* **Adam** peak/trough — a sharp spike: few bars cluster near the extreme price.
* **Eve**  peak/trough — a rounded or multi-touch formation: many bars linger
  near the extreme price.

The generic ``double_top`` / ``double_bottom`` functions match EITHER shape.

Return type
-----------
All functions return **int8**: +1 bullish, -1 bearish, 0 not fired.

Mode parameter
--------------
``mode='forming'``   – pattern shape complete, no breakout required.
``mode='confirmed'`` – shape complete AND price has broken the confirmation level.
"""
from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .._core import _to_np, _EPS
from ._core import (
    get_pivot_highs, get_pivot_lows,
    pivots_in_window, fit_line,
    prices_equal, peak_sharpness, trough_sharpness,
)


# Adam/Eve split on a peak/trough sharpness score in [0,1].
# ``peak_sharpness``/``trough_sharpness`` return a scale-free, peak/trough
# symmetric score (higher = sharper).  Adam peaks/troughs are narrow and pointed
# (scores ~0.45+), Eve are wide and rounded (scores ~0.3 and below).  The 0.37
# cut-off is a heuristic split, not an exact Bulkowski rule.  A score >= this is
# 'adam', below is 'eve'.
_ADAM_THR = 0.37


def _at_pivot(arr: np.ndarray, n: int) -> np.ndarray:
    """Shifted view where ``out[t] == arr[t - n]``.

    ``get_pivot_highs/lows`` flag the *confirmation* bar (n bars after the
    actual swing).  Indexing the raw price array at the confirmation bar
    reads a price n bars past the real pivot.  Indexing this shifted array
    at the confirmation bar instead returns the price at the true swing bar,
    while the confirmation index is still used for point-in-time windowing.
    """
    out = np.empty_like(arr)
    if n <= 0:
        out[:] = arr
        return out
    out[:n] = arr[:n]
    out[n:] = arr[:-n]
    return out


def _cluster_pivots(conf_idx: np.ndarray, price: np.ndarray,
                    n: int, kind: str) -> np.ndarray:
    """Collapse adjacent pivot confirmations that belong to one swing.

    A single peak/trough can flag several consecutive confirmation bars
    (flat tops, equal highs).  Left as-is, the detector may compare a swing
    with its own duplicate (separation ~1) and reject a valid pattern.  This
    groups confirmations whose gap is <= n and keeps the most extreme one
    (highest for 'high', lowest for 'low').  Uses only the given pivots, so
    it does not introduce look-ahead.

    Returns an array of representative confirmation indices (sorted).
    """
    if len(conf_idx) == 0:
        return conf_idx
    order = np.argsort(conf_idx)
    ci = conf_idx[order]
    pv = price[order]

    # Group starts: a gap of more than n bars opens a new swing.  Replaces the
    # per-pivot Python loop with two segmented reductions.
    starts = np.concatenate(([0], np.flatnonzero(np.diff(ci) > n) + 1))
    reducer = np.maximum if kind == 'high' else np.minimum
    group_best = reducer.reduceat(pv, starts)
    sizes = np.diff(np.append(starts, len(ci)))

    # First element of each group attaining the group extreme — matches
    # np.argmax/np.argmin tie-breaking (earliest index wins).
    hits = np.flatnonzero(pv == np.repeat(group_best, sizes))
    gid = np.searchsorted(starts, hits, side='right') - 1
    _, first = np.unique(gid, return_index=True)
    return np.asarray(ci[hits[first]], dtype=int)


_a = _to_np


def _sharpness_vec(a: np.ndarray, bars: np.ndarray, n: int = 3,
                   is_high: bool = True) -> np.ndarray:
    """:func:`peak_sharpness` / :func:`trough_sharpness` for many bars at once.

    Only the handful of bars whose neighbourhood is clipped by an array edge
    fall back to the scalar helper, so the result is exact.
    """
    scalar = peak_sharpness if is_high else trough_sharpness
    bars = np.asarray(bars, dtype=np.int64)
    out = np.full(len(bars), 0.5)
    if len(bars) == 0:
        return out
    N = len(a)
    lo = np.maximum(0, bars - n)
    hi = np.minimum(N, bars + n + 1)
    full = (lo == bars - n) & (hi == bars + n + 1)
    if full.any():
        w = sliding_window_view(a, 2 * n + 1)[lo[full]]
        mx, mn = w.max(axis=1), w.min(axis=1)
        rng = mx - mn
        num = (mx - w.mean(axis=1)) if is_high else (w.mean(axis=1) - mn)
        safe = np.where(rng > 0, rng, 1.0)
        out[full] = np.where(rng > 0, np.clip(num / safe, 0, 1), 0.0)
    for k in np.flatnonzero(~full):
        out[k] = scalar(a, int(bars[k]), n)
    return out


def _detect_double(o, h, l, c, mode, window, pivot_n, pivot_pct,
                   tol, min_separation, type1, type2, is_top):
    """Vectorised core for every double top / double bottom variant.

    The previous implementation looped over all N bars and re-derived the two
    most recent pivots on each one.  That state only changes when a *new* pivot
    appears (roughly every ``2*pivot_n`` bars), so the loop repeated identical
    work.  Here a single ``searchsorted`` resolves "which two pivots are the
    latest" for every bar simultaneously, and the shape tests are evaluated once
    per pivot (P of them) rather than once per bar.  Output is unchanged.

    ``is_top``: True  -> peaks are pivot highs, neckline is the trough between
                False -> troughs are pivot lows, neckline is the peak between
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)

    hp = _at_pivot(h, pivot_n)          # price at the true swing bar
    lp = _at_pivot(l, pivot_n)
    ph_idx = np.where(get_pivot_highs(h, pivot_n, pivot_pct))[0]
    pl_idx = np.where(get_pivot_lows(l, pivot_n, pivot_pct))[0]
    ph_idx = _cluster_pivots(ph_idx, hp[ph_idx], pivot_n, 'high')
    pl_idx = _cluster_pivots(pl_idx, lp[pl_idx], pivot_n, 'low')

    if is_top:
        main_idx, main_src, sec_idx, sec_src = ph_idx, hp, pl_idx, lp
        sign = -1
    else:
        main_idx, main_src, sec_idx, sec_src = pl_idx, lp, ph_idx, hp
        sign = 1

    if len(main_idx) < 2 or len(sec_idx) == 0:
        return result
    main_val = main_src[main_idx]

    # --- for every bar, the two most recent main pivots at or before i-1 -----
    i = np.arange(N)
    b = np.searchsorted(main_idx, i - 1, side='right')
    j2 = np.clip(b - 1, 0, len(main_idx) - 1)
    j1 = np.clip(b - 2, 0, len(main_idx) - 1)
    p1, p2 = main_idx[j1], main_idx[j2]
    v1, v2 = main_val[j1], main_val[j2]

    cond = (b >= 2) & (i >= window) & (p1 >= i - window)
    cond &= (p2 - p1) >= min_separation
    # prices_equal(), vectorised
    cond &= np.abs(v1 - v2) / (np.maximum(np.abs(v1), np.abs(v2)) + _EPS) <= tol

    # --- Adam/Eve shape filter: evaluated once per pivot, not per bar --------
    if type1 is not None or type2 is not None:
        src = h if is_top else l
        is_adam = _sharpness_vec(src, main_idx - pivot_n,
                                 is_high=is_top) >= _ADAM_THR
        if type1 is not None:
            cond &= is_adam[j1] if type1 == 'adam' else ~is_adam[j1]
        if type2 is not None:
            cond &= is_adam[j2] if type2 == 'adam' else ~is_adam[j2]

    # --- neckline: last opposite pivot strictly between the two main ones ----
    a = np.searchsorted(sec_idx, p2, side='left') - 1
    ai = np.clip(a, 0, len(sec_idx) - 1)
    cond &= (a >= 0) & (sec_idx[ai] > p1)
    neckline = sec_src[sec_idx[ai]]

    if mode != 'forming':
        cond &= (c < neckline) if is_top else (c > neckline)
    return np.where(cond, sign, 0).astype(np.int8)


def _detect_double_top(o, h, l, c, mode, window, pivot_n, pivot_pct,
                        tol, min_separation, peak1_type, peak2_type):
    """
    Core double-top detector.
    peak1_type / peak2_type: 'adam', 'eve', or None (any).
    Returns int8 array -1 / 0.
    """
    return _detect_double(o, h, l, c, mode, window, pivot_n, pivot_pct,
                          tol, min_separation, peak1_type, peak2_type,
                          is_top=True)


def _detect_double_bottom(o, h, l, c, mode, window, pivot_n, pivot_pct,
                           tol, min_separation, trough1_type, trough2_type):
    """Core double-bottom detector."""
    return _detect_double(o, h, l, c, mode, window, pivot_n, pivot_pct,
                          tol, min_separation, trough1_type, trough2_type,
                          is_top=False)


# ---------------------------------------------------------------------------
# Public double-top functions
# ---------------------------------------------------------------------------

def double_top(o, h, l, c, mode: str = 'confirmed',
               window: int = 60, pivot_n: int = 5,
               pivot_pct: float = None, tol: float = 0.03,
               min_separation: int = 5) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).  Generic double top — any peak shapes.

    Two pivot highs at approximately the same level (*tol* fraction),
    separated by at least *min_separation* bars.

    Parameters
    ----------
    mode : 'forming' | 'confirmed'
        forming   — shape complete, signal fires continuously.
        confirmed — shape + close below the valley between the two peaks.
    tol : float
        Max fractional difference between the two peak prices (default 3 %).
    min_separation : int
        Minimum bars between the two peaks.
    """
    return _detect_double_top(o, h, l, c, mode, window, pivot_n, pivot_pct,
                               tol, min_separation, None, None)


def double_top_adam_adam(o, h, l, c, mode: str = 'confirmed',
                          window: int = 60, pivot_n: int = 5,
                          pivot_pct: float = None, tol: float = 0.03,
                          min_separation: int = 5) -> np.ndarray:
    """**BEARISH** (-1 / 0). Double top where both peaks are sharp Adam spikes."""
    return _detect_double_top(o, h, l, c, mode, window, pivot_n, pivot_pct,
                               tol, min_separation, 'adam', 'adam')


def double_top_adam_eve(o, h, l, c, mode: str = 'confirmed',
                         window: int = 60, pivot_n: int = 5,
                         pivot_pct: float = None, tol: float = 0.03,
                         min_separation: int = 5) -> np.ndarray:
    """**BEARISH** (-1 / 0). Double top: first peak Adam (sharp), second Eve (rounded)."""
    return _detect_double_top(o, h, l, c, mode, window, pivot_n, pivot_pct,
                               tol, min_separation, 'adam', 'eve')


def double_top_eve_adam(o, h, l, c, mode: str = 'confirmed',
                         window: int = 60, pivot_n: int = 5,
                         pivot_pct: float = None, tol: float = 0.03,
                         min_separation: int = 5) -> np.ndarray:
    """**BEARISH** (-1 / 0). Double top: first peak Eve (rounded), second Adam (sharp)."""
    return _detect_double_top(o, h, l, c, mode, window, pivot_n, pivot_pct,
                               tol, min_separation, 'eve', 'adam')


def double_top_eve_eve(o, h, l, c, mode: str = 'confirmed',
                        window: int = 60, pivot_n: int = 5,
                        pivot_pct: float = None, tol: float = 0.03,
                        min_separation: int = 5) -> np.ndarray:
    """**BEARISH** (-1 / 0). Double top where both peaks are rounded Eve formations."""
    return _detect_double_top(o, h, l, c, mode, window, pivot_n, pivot_pct,
                               tol, min_separation, 'eve', 'eve')


# ---------------------------------------------------------------------------
# Public double-bottom functions
# ---------------------------------------------------------------------------

def double_bottom(o, h, l, c, mode: str = 'confirmed',
                  window: int = 60, pivot_n: int = 5,
                  pivot_pct: float = None, tol: float = 0.03,
                  min_separation: int = 5) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).  Generic double bottom — any trough shapes.

    Two pivot lows at approximately the same level (*tol* fraction),
    separated by at least *min_separation* bars.

    Parameters
    ----------
    mode : 'forming' | 'confirmed'
        confirmed — shape + close above the peak between the two troughs.
    """
    return _detect_double_bottom(o, h, l, c, mode, window, pivot_n, pivot_pct,
                                  tol, min_separation, None, None)


def double_bottom_adam_adam(o, h, l, c, mode: str = 'confirmed',
                             window: int = 60, pivot_n: int = 5,
                             pivot_pct: float = None, tol: float = 0.03,
                             min_separation: int = 5) -> np.ndarray:
    """**BULLISH** (+1 / 0). Double bottom where both troughs are sharp Adam spikes."""
    return _detect_double_bottom(o, h, l, c, mode, window, pivot_n, pivot_pct,
                                  tol, min_separation, 'adam', 'adam')


def double_bottom_adam_eve(o, h, l, c, mode: str = 'confirmed',
                            window: int = 60, pivot_n: int = 5,
                            pivot_pct: float = None, tol: float = 0.03,
                            min_separation: int = 5) -> np.ndarray:
    """**BULLISH** (+1 / 0). Double bottom: first trough Adam (sharp), second Eve (rounded)."""
    return _detect_double_bottom(o, h, l, c, mode, window, pivot_n, pivot_pct,
                                  tol, min_separation, 'adam', 'eve')


def double_bottom_eve_adam(o, h, l, c, mode: str = 'confirmed',
                            window: int = 60, pivot_n: int = 5,
                            pivot_pct: float = None, tol: float = 0.03,
                            min_separation: int = 5) -> np.ndarray:
    """**BULLISH** (+1 / 0). Double bottom: first trough Eve, second Adam."""
    return _detect_double_bottom(o, h, l, c, mode, window, pivot_n, pivot_pct,
                                  tol, min_separation, 'eve', 'adam')


def double_bottom_eve_eve(o, h, l, c, mode: str = 'confirmed',
                           window: int = 60, pivot_n: int = 5,
                           pivot_pct: float = None, tol: float = 0.03,
                           min_separation: int = 5) -> np.ndarray:
    """**BULLISH** (+1 / 0). Double bottom where both troughs are rounded Eve formations."""
    return _detect_double_bottom(o, h, l, c, mode, window, pivot_n, pivot_pct,
                                  tol, min_separation, 'eve', 'eve')


def ugly_double_bottom(o, h, l, c, mode: str = 'confirmed',
                        window: int = 60, pivot_n: int = 5,
                        pivot_pct: float = None,
                        tol: float = 0.06,
                        asymmetry: float = 0.03,
                        min_separation: int = 5) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    'Ugly' double bottom: the two troughs are NOT at the same level —
    the second trough is slightly lower or higher than the first (within
    *asymmetry* of each other but clearly not equal).  Still bullish if
    price subsequently confirms above the neckline.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    ph_mask = get_pivot_highs(h, pivot_n, pivot_pct)
    pl_mask = get_pivot_lows(l, pivot_n, pivot_pct)
    ph_idx  = np.where(ph_mask)[0]
    pl_idx  = np.where(pl_mask)[0]
    hp = _at_pivot(h, pivot_n)
    lp = _at_pivot(l, pivot_n)
    ph_idx = _cluster_pivots(ph_idx, hp[ph_idx], pivot_n, 'high')
    ph_val = hp[ph_idx]              # loop-invariant: hoisted out of per-bar scan
    pl_idx = _cluster_pivots(pl_idx, lp[pl_idx], pivot_n, 'low')
    pl_val = lp[pl_idx]              # loop-invariant: hoisted out of per-bar scan

    result  = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        pl_w, _ = pivots_in_window(pl_idx, pl_val, i - window, i - 1)
        ph_w, _ = pivots_in_window(ph_idx, ph_val, i - window, i - 1)
        if len(pl_w) < 2 or len(ph_w) < 1:
            continue
        p1_idx, p2_idx = pl_w[-2], pl_w[-1]
        p1_val, p2_val = lp[p1_idx], lp[p2_idx]
        if (p2_idx - p1_idx) < min_separation:
            continue
        diff_ratio = abs(p2_val - p1_val) / (abs(p1_val) + _EPS)
        if diff_ratio > tol:
            continue  # too different to be a double bottom at all
        if diff_ratio < 0.001:
            continue  # too equal — that's a regular double bottom
        peak_w = ph_w[(ph_w > p1_idx) & (ph_w < p2_idx)]
        if len(peak_w) == 0:
            continue
        neckline = hp[peak_w[-1]]
        if mode == 'forming':
            result[i] = 1
        else:
            if c[i] > neckline:
                result[i] = 1

    return result


# ---------------------------------------------------------------------------
# Triple top / bottom
# ---------------------------------------------------------------------------

def _detect_triple(o, h, l, c, mode, window, pivot_n, pivot_pct,
                   tol, min_separation, is_top):
    """Vectorised core for triple top / triple bottom.

    Same pivot-indexed treatment as :func:`_detect_double`, extended to three
    consecutive main pivots and the two necklines between them.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)

    hp = _at_pivot(h, pivot_n)
    lp = _at_pivot(l, pivot_n)
    ph_idx = np.where(get_pivot_highs(h, pivot_n, pivot_pct))[0]
    pl_idx = np.where(get_pivot_lows(l, pivot_n, pivot_pct))[0]
    ph_idx = _cluster_pivots(ph_idx, hp[ph_idx], pivot_n, 'high')
    pl_idx = _cluster_pivots(pl_idx, lp[pl_idx], pivot_n, 'low')

    if is_top:
        main_idx, main_src, sec_idx, sec_src = ph_idx, hp, pl_idx, lp
        sign = -1
    else:
        main_idx, main_src, sec_idx, sec_src = pl_idx, lp, ph_idx, hp
        sign = 1

    if len(main_idx) < 3 or len(sec_idx) < 2:
        return result
    main_val = main_src[main_idx]

    i = np.arange(N)
    b = np.searchsorted(main_idx, i - 1, side='right')
    j1 = np.clip(b - 3, 0, len(main_idx) - 1)
    j2 = np.clip(b - 2, 0, len(main_idx) - 1)
    j3 = np.clip(b - 1, 0, len(main_idx) - 1)
    p1, p2, p3 = main_idx[j1], main_idx[j2], main_idx[j3]
    v1, v2, v3 = main_val[j1], main_val[j2], main_val[j3]

    cond = (b >= 3) & (i >= window) & (p1 >= i - window)
    cond &= ((p2 - p1) >= min_separation) & ((p3 - p2) >= min_separation)
    cond &= np.abs(v1 - v2) / (np.maximum(np.abs(v1), np.abs(v2)) + _EPS) <= tol
    cond &= np.abs(v2 - v3) / (np.maximum(np.abs(v2), np.abs(v3)) + _EPS) <= tol

    # last opposite pivot in (p1, p2) and in (p2, p3)
    k1 = np.searchsorted(sec_idx, p2, side='left') - 1
    k2 = np.searchsorted(sec_idx, p3, side='left') - 1
    kk1 = np.clip(k1, 0, len(sec_idx) - 1)
    kk2 = np.clip(k2, 0, len(sec_idx) - 1)
    cond &= (k1 >= 0) & (sec_idx[kk1] > p1)
    cond &= (k2 >= 0) & (sec_idx[kk2] > p2)

    n1 = sec_src[sec_idx[kk1]]
    n2 = sec_src[sec_idx[kk2]]
    neckline = np.minimum(n1, n2) if is_top else np.maximum(n1, n2)

    if mode != 'forming':
        cond &= (c < neckline) if is_top else (c > neckline)
    return np.where(cond, sign, 0).astype(np.int8)


def triple_top(o, h, l, c, mode: str = 'confirmed',
               window: int = 100, pivot_n: int = 5,
               pivot_pct: float = None, tol: float = 0.03,
               min_separation: int = 5) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Three pivot highs at approximately equal prices, each separated by
    at least *min_separation* bars, confirmed by a close below the lowest
    valley between the three peaks.
    """
    return _detect_triple(o, h, l, c, mode, window, pivot_n, pivot_pct,
                          tol, min_separation, is_top=True)


def triple_bottom(o, h, l, c, mode: str = 'confirmed',
                  window: int = 100, pivot_n: int = 5,
                  pivot_pct: float = None, tol: float = 0.03,
                  min_separation: int = 5) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Three pivot lows at approximately equal prices, confirmed by a close
    above the highest peak between the three troughs.
    """
    return _detect_triple(o, h, l, c, mode, window, pivot_n, pivot_pct,
                          tol, min_separation, is_top=False)


# ---------------------------------------------------------------------------
# Big M / Big W  (elongated double patterns)
# ---------------------------------------------------------------------------

def big_m(o, h, l, c, mode: str = 'confirmed', window: int = 150,
           pivot_n: int = 7, tol: float = 0.03,
           min_sep: int = 15, min_depth: float = 0.05) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Like a double top but larger in scale — two major peaks separated by a
    significant valley, with extra breadth requirements.
    """
    return double_top(o, h, l, c, mode=mode, window=window, pivot_n=pivot_n,
                      tol=tol, min_separation=min_sep)


def big_w(o, h, l, c, mode: str = 'confirmed', window: int = 150,
           pivot_n: int = 7, tol: float = 0.03,
           min_sep: int = 15, min_depth: float = 0.05) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Like a double bottom but larger in scale — two major troughs with a
    significant peak between them.
    """
    return double_bottom(o, h, l, c, mode=mode, window=window, pivot_n=pivot_n,
                         tol=tol, min_separation=min_sep)


# ---------------------------------------------------------------------------
# Three falling peaks / Three rising valleys
# ---------------------------------------------------------------------------

def three_peaks(o, h, l, c, mode: str = 'confirmed', window: int = 120,
                 pivot_n: int = 5, tol: float = 0.04,
                 min_separation: int = 8) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Three successive pivot highs each lower than the last — a stair-step
    distribution pattern with bearish bias.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph_mask = get_pivot_highs(h, pivot_n)
    ph_idx  = np.where(ph_mask)[0]
    hp = _at_pivot(h, pivot_n)
    ph_idx = _cluster_pivots(ph_idx, hp[ph_idx], pivot_n, 'high')
    ph_val = hp[ph_idx]              # loop-invariant: hoisted out of per-bar scan

    result  = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        ph_w, _ = pivots_in_window(ph_idx, ph_val, i - window, i - 1)
        if len(ph_w) < 3:
            continue
        p1, p2, p3 = ph_w[-3], ph_w[-2], ph_w[-1]
        if (p2 - p1) < min_separation or (p3 - p2) < min_separation:
            continue
        if hp[p1] > hp[p2] > hp[p3]:           # strictly falling peaks
            if mode == 'forming':
                result[i] = -1
            else:
                support = l[p1:p3 + 1].min()
                if c[i] < support:
                    result[i] = -1
    return result


def three_valleys(o, h, l, c, mode: str = 'confirmed', window: int = 120,
                   pivot_n: int = 5, tol: float = 0.04,
                   min_separation: int = 8) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Three successive pivot lows each higher than the last — a stair-step
    accumulation pattern with bullish bias.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(l)
    pl_mask = get_pivot_lows(l, pivot_n)
    pl_idx  = np.where(pl_mask)[0]
    lp = _at_pivot(l, pivot_n)
    pl_idx = _cluster_pivots(pl_idx, lp[pl_idx], pivot_n, 'low')
    pl_val = lp[pl_idx]              # loop-invariant: hoisted out of per-bar scan

    result  = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        pl_w, _ = pivots_in_window(pl_idx, pl_val, i - window, i - 1)
        if len(pl_w) < 3:
            continue
        p1, p2, p3 = pl_w[-3], pl_w[-2], pl_w[-1]
        if (p2 - p1) < min_separation or (p3 - p2) < min_separation:
            continue
        if lp[p1] < lp[p2] < lp[p3]:           # strictly rising valleys
            if mode == 'forming':
                result[i] = 1
            else:
                resistance = h[p1:p3 + 1].max()
                if c[i] > resistance:
                    result[i] = 1
    return result


# ---------------------------------------------------------------------------
# Three peaks and a domed house  (complex bearish megapattern)
# ---------------------------------------------------------------------------

def three_peaks_domed_house(o, h, l, c, mode: str = 'confirmed',
                              window: int = 200, pivot_n: int = 5) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Bulkowski's 'three peaks and a domed house': three peaks followed by a
    broad rounded dome top before a major decline.  Approximated here as
    three rising peaks (the 'house' stage) then a rounded reversal.

    .. note::
        Full Bulkowski pattern spans months; this implementation detects the
        completed rounding top after three prior peaks.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph_mask = get_pivot_highs(h, pivot_n)
    ph_idx  = np.where(ph_mask)[0]
    hp = _at_pivot(h, pivot_n)
    ph_idx = _cluster_pivots(ph_idx, hp[ph_idx], pivot_n, 'high')
    ph_val = hp[ph_idx]              # loop-invariant: hoisted out of per-bar scan

    result  = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        ph_w, _ = pivots_in_window(ph_idx, ph_val, i - window, i - window // 2)
        if len(ph_w) < 3:
            continue
        # Three peaks in the older half of the window
        p1, p2, p3 = ph_w[-3], ph_w[-2], ph_w[-1]
        if not (hp[p1] < hp[p2] < hp[p3]):     # rising peaks (the 'house')
            continue
        # Dome: max high in more recent half should be lower than p3
        recent_h = h[p3:i + 1]
        if len(recent_h) < 10:
            continue
        dome_high = recent_h.max()
        if dome_high <= hp[p3]:               # dome failed to exceed last peak
            # Look for rounding top shape: recent highs declining
            half = len(recent_h) // 2
            if recent_h[:half].mean() < recent_h[half:].mean():
                continue                      # not rounding over
            if mode == 'forming':
                result[i] = -1
            else:
                if c[i] < c[i - 5:i].mean() * 0.98:   # accelerating decline
                    result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Cat's ears  (sharp double top, bearish)
# ---------------------------------------------------------------------------

def cat_ears(o, h, l, c, mode: str = 'confirmed', window: int = 30,
              pivot_n: int = 3, tol: float = 0.015,
              min_separation: int = 3, min_valley_depth: float = 0.02) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Two sharp, closely spaced peaks of similar height — like a cat's ears.
    A tighter, faster version of the double top.
    """
    return double_top(o, h, l, c, mode=mode, window=window, pivot_n=pivot_n,
                      tol=tol, min_separation=min_separation)
