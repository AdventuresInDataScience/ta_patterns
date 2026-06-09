"""
ta_patterns.chart_patterns._core
=================================
Shared utilities for chart pattern detection.

NO-LOOKAHEAD PIVOT DETECTION
-----------------------------
pivot_highs(h, n)[t] = True  →  bar (t − n) is a confirmed pivot high.
Only h[0..t] is used; no look-forward bias.

All chart-pattern sub-modules import from here.
"""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

# Re-export parent helpers so sub-modules can import from one place
from .._core import _to_np, uptrend, downtrend, _EPS, atr as atr
from .._core import roll_mean, avg_body

_a = _to_np          # short alias used throughout chart_patterns


# ---------------------------------------------------------------------------
# Pivot detection  (delayed; no lookahead)
# ---------------------------------------------------------------------------

def pivot_highs(h, n: int = 5, pct: float | None = None) -> np.ndarray:
    """
    ``out[t]`` = True means a pivot high confirmed at t (actual pivot at t−n).
    Uses only h[0..t].  Optional *pct* applies a ZigZag-style minimum-swing
    filter on top of the bar-count neighbourhood.
    """
    h  = _a(h)
    N  = len(h)
    out = np.zeros(N, dtype=bool)
    if N < 2 * n + 1:
        return out

    hp = np.concatenate([np.full(n, -np.inf), h, np.full(n, -np.inf)])
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        wins   = sliding_window_view(hp, 2 * n + 1)
        is_ph  = h >= wins.max(axis=1)
    except AttributeError:
        is_ph = np.array([h[j] >= hp[j:j + 2*n + 1].max() for j in range(N)])

    out[n:] = is_ph[:N - n]

    if pct is not None:
        out = _pct_filter(out, h, pct, is_high=True)
    return out


def pivot_lows(l, n: int = 5, pct: float | None = None) -> np.ndarray:
    """Mirror of :func:`pivot_highs` for lows."""
    l  = _a(l)
    N  = len(l)
    out = np.zeros(N, dtype=bool)
    if N < 2 * n + 1:
        return out

    lp = np.concatenate([np.full(n, np.inf), l, np.full(n, np.inf)])
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        wins   = sliding_window_view(lp, 2 * n + 1)
        is_pl  = l <= wins.min(axis=1)
    except AttributeError:
        is_pl = np.array([l[j] <= lp[j:j + 2*n + 1].min() for j in range(N)])

    out[n:] = is_pl[:N - n]

    if pct is not None:
        out = _pct_filter(out, l, pct, is_high=False)
    return out


def _pct_filter(pivot_bool, prices, pct, is_high):
    idxs = np.where(pivot_bool)[0]
    if len(idxs) < 2:
        return pivot_bool
    filtered       = np.zeros_like(pivot_bool)
    filtered[idxs[0]] = True
    last_p         = prices[idxs[0]]
    for idx in idxs[1:]:
        p = prices[idx]
        if abs(p - last_p) / (abs(last_p) + _EPS) >= pct:
            filtered[idx] = True
            last_p = p
    return filtered


def pivot_info(prices, pivot_bool, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return *(confirm_times, prices_at_pivot)* for all True entries.

    ``prices``     – h array (for highs) or l array (for lows)
    ``pivot_bool`` – output of :func:`pivot_highs` / :func:`pivot_lows`
    ``n``          – same n used to compute pivot_bool
    """
    prices     = _a(prices)
    N          = len(prices)
    t_confirm  = np.where(pivot_bool)[0]
    t_actual   = (t_confirm - n).clip(0, N - 1)
    p_at_pivot = prices[t_actual]
    return t_confirm, p_at_pivot


# ---------------------------------------------------------------------------
# Backward-compatible helpers used by double_multi.py
# ---------------------------------------------------------------------------

def get_pivot_highs(h: np.ndarray, pivot_n: int = 5,
                    pivot_pct: float | None = None) -> np.ndarray:
    """Boolean mask: True at each confirmed pivot-high confirmation bar."""
    return pivot_highs(_a(h), pivot_n, pivot_pct)


def get_pivot_lows(l: np.ndarray, pivot_n: int = 5,
                   pivot_pct: float | None = None) -> np.ndarray:
    """Boolean mask: True at each confirmed pivot-low confirmation bar."""
    return pivot_lows(_a(l), pivot_n, pivot_pct)


def pit_pivot_highs(h: np.ndarray, n: int = 5) -> np.ndarray:
    """
    Point-in-time pivot highs — same as pivot_highs; kept for compatibility
    with short.py that imports this name.
    """
    return pivot_highs(h, n)


def pit_pivot_lows(l: np.ndarray, n: int = 5) -> np.ndarray:
    return pivot_lows(l, n)


def pivots_in_window(pivot_idx: np.ndarray, pivot_val: np.ndarray,
                     start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, values) of pivots whose index is in [start, end].

    ``pivot_idx`` comes from ``np.where`` (and clustering preserves order), so
    it is sorted ascending and the in-window pivots form a contiguous slice
    located with ``np.searchsorted`` in O(log P) — versus the old boolean mask
    that was O(P) on every call and, inside per-bar loops, O(N·P) overall.
    """
    a = np.searchsorted(pivot_idx, start, side="left")
    b = np.searchsorted(pivot_idx, end, side="right")
    return pivot_idx[a:b], pivot_val[a:b]


def last_n_pivots(pivot_idx, pivot_val, n_pivots):
    if len(pivot_idx) < n_pivots:
        return np.array([]), np.array([])
    return pivot_idx[-n_pivots:], pivot_val[-n_pivots:]


def peak_sharpness(h: np.ndarray, peak_bar: int, n: int = 3) -> float:
    """
    How 'sharp' (Adam-like) vs 'rounded' (Eve-like) a peak is.
    Returns a value in [0, 1]: 1 = very sharp, 0 = very rounded.

    Scale-free: normalised by the local high-low *range* so the score does
    not depend on absolute price level and is directly comparable with
    :func:`trough_sharpness`.  A sharp spike leaves most neighbours far below
    the peak (mean near the low end of the range -> score near 1); a rounded
    top keeps neighbours near the peak (mean near the high end -> score ~0).
    """
    N = len(h)
    lo = max(0, peak_bar - n)
    hi = min(N, peak_bar + n + 1)
    window = h[lo:hi]
    if len(window) < 3:
        return 0.5
    mx = window.max()
    mn = window.min()
    rng = mx - mn
    if rng <= 0:
        return 0.0
    return float(np.clip((mx - window.mean()) / rng, 0, 1))


def trough_sharpness(l: np.ndarray, trough_bar: int, n: int = 3) -> float:
    """Mirror of :func:`peak_sharpness` for troughs (same scale-free basis)."""
    N = len(l)
    lo = max(0, trough_bar - n)
    hi = min(N, trough_bar + n + 1)
    window = l[lo:hi]
    if len(window) < 3:
        return 0.5
    mx = window.max()
    mn = window.min()
    rng = mx - mn
    if rng <= 0:
        return 0.0
    return float(np.clip((window.mean() - mn) / rng, 0, 1))


def prices_equal(a: float, b: float, tol: float = 0.02) -> bool:
    return abs(a - b) / (max(abs(a), abs(b)) + _EPS) <= tol


def pivot_indices_values(prices, pivot_bool):
    prices = _a(prices)
    idx    = np.where(pivot_bool)[0]
    return idx, prices[idx]


# ---------------------------------------------------------------------------
# Trendline fitting
# ---------------------------------------------------------------------------

def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit y = slope × x + intercept.  Returns (slope, intercept).
    (Also computes r² internally but does not return it for compatibility
    with double_multi.py which expects a 2-tuple.)
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 2:
        return 0.0, float(y.mean()) if n else 0.0
    mx, my  = x.mean(), y.mean()
    ss_xx   = ((x - mx) ** 2).sum()
    if ss_xx < _EPS:
        return 0.0, my
    slope     = ((x - mx) * (y - my)).sum() / ss_xx
    intercept = my - slope * mx
    return slope, intercept


def fit_line_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Like fit_line but also returns r² as the third element."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 2:
        return 0.0, float(y.mean()) if n else 0.0, 0.0
    mx, my    = x.mean(), y.mean()
    ss_xx     = ((x - mx) ** 2).sum()
    if ss_xx < _EPS:
        return 0.0, my, 0.0
    slope     = ((x - mx) * (y - my)).sum() / ss_xx
    intercept = my - slope * mx
    y_pred    = slope * x + intercept
    ss_res    = ((y - y_pred) ** 2).sum()
    ss_tot    = ((y - my) ** 2).sum()
    r2        = 1.0 - ss_res / ss_tot if ss_tot > _EPS else 1.0
    return slope, intercept, r2


def line_val(slope: float, intercept: float, t: float) -> float:
    """Evaluate trendline at bar *t*."""
    return slope * t + intercept


def line_value(slope: float, intercept: float, x: float) -> float:
    """Alias for :func:`line_val` (backward compat)."""
    return slope * x + intercept


def lines_converging(s_hi: float, s_lo: float,
                     i_hi: float, i_lo: float,
                     t_start: float) -> bool:
    """
    True if upper (hi) and lower (lo) trendlines are converging for t > t_start.
    Converging means the gap (hi − lo) is shrinking → s_lo > s_hi.
    """
    return s_lo > s_hi


def apex_bar(s_hi, i_hi, s_lo, i_lo):
    """Bar where two trendlines would intersect; None if parallel."""
    denom = s_lo - s_hi
    if abs(denom) < _EPS:
        return None
    return (i_hi - i_lo) / denom


# ---------------------------------------------------------------------------
# Fibonacci helpers
# ---------------------------------------------------------------------------

def in_range(val: float, target: float, tol: float = 0.05) -> bool:
    """True if val is within *tol* fraction of target."""
    return abs(val / (target + _EPS) - 1.0) <= tol


def in_band(val: float, lo: float, hi: float) -> bool:
    return lo <= val <= hi


def fib_retracement(swing_start: float, swing_end: float,
                     ratio: float) -> float:
    return swing_end + ratio * (swing_start - swing_end)


def fib_extension(a: float, b: float, c: float, ratio: float) -> float:
    return c - ratio * (b - a)


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------

def roll_max(a: np.ndarray, n: int) -> np.ndarray:
    a   = _a(a)
    N   = len(a)
    out = np.full(N, np.nan)
    if n > N:
        return out
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        out[n - 1:] = sliding_window_view(a, n).max(axis=1)
    except AttributeError:
        for i in range(n - 1, N):
            out[i] = a[i - n + 1:i + 1].max()
    return out


def roll_min(a: np.ndarray, n: int) -> np.ndarray:
    a   = _a(a)
    N   = len(a)
    out = np.full(N, np.nan)
    if n > N:
        return out
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        out[n - 1:] = sliding_window_view(a, n).min(axis=1)
    except AttributeError:
        for i in range(n - 1, N):
            out[i] = a[i - n + 1:i + 1].min()
    return out


# ---------------------------------------------------------------------------
# Misc helpers expected by sub-modules
# ---------------------------------------------------------------------------

def is_flat(slope: float, intercept: float, values: np.ndarray,
            indices: np.ndarray, tol: float = 0.02) -> bool:
    predicted = slope * indices + intercept
    dev = np.abs(values - predicted) / (np.abs(predicted) + _EPS)
    return bool(dev.max() < tol)


def touches_line(prices: np.ndarray, indices: np.ndarray,
                 slope: float, intercept: float, tol: float = 0.01) -> np.ndarray:
    predicted = slope * indices + intercept
    return np.abs(prices - predicted) / (np.abs(predicted) + _EPS) < tol


def count_line_touches(prices: np.ndarray, indices: np.ndarray,
                       slope: float, intercept: float, tol: float = 0.01) -> int:
    return int(touches_line(prices, indices, slope, intercept, tol).sum())


def arrays_equal(a: np.ndarray, b: np.ndarray, tol: float = 0.02) -> np.ndarray:
    return np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + _EPS) <= tol


def _fill_signal(N: int) -> np.ndarray:
    """Return a zeroed int8 result array of length N."""
    return np.zeros(N, dtype=np.int8)
