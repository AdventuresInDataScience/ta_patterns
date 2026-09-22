"""
ta_patterns.chart_patterns.volume
===================================
Volume-based patterns.  All functions require a ``v`` (volume) array in
addition to the standard ``o, h, l, c``.

Return type
-----------
All functions return **int8**: +1 bullish, -1 bearish, 0 not fired.
Non-directional patterns return +1 when the shape fires.

Implementation note
-------------------
Every detector here is fully vectorised over bars.  The rolling windows are
materialised once with ``sliding_window_view`` (a stride trick — no copy) and
the per-window statistics are obtained with cumulative sums along the window
axis, so the cost is one pass over the data instead of one Python iteration
per bar.  Outputs are bit-identical to the previous per-bar implementations.
"""
from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .._core import _to_np, _EPS
from ._core import _fill_signal

_a = _to_np


def _run_fractions(d: np.ndarray, split: np.ndarray, positive: bool):
    """Fraction of ``d[m, :split[m]]`` and ``d[m, split[m]:]`` satisfying a sign
    test, for every row *m* at once.

    ``d`` is the row-wise first difference of the rolling windows.  Cumulative
    sums along the window axis turn the two variable-length slices into O(1)
    lookups, which is what removes the per-bar Python loop.
    """
    W = d.shape[1]
    z = np.zeros((d.shape[0], 1), dtype=np.int32)
    ge = np.concatenate((z, np.cumsum(d >= 0, axis=1, dtype=np.int32)), axis=1)
    le = np.concatenate((z, np.cumsum(d <= 0, axis=1, dtype=np.int32)), axis=1)
    rows = np.arange(d.shape[0])
    first_cnt = (ge if positive else le)[rows, split]
    second_src = (le if positive else ge)
    second_cnt = second_src[:, W] - second_src[rows, split]
    n1 = np.maximum(split, 1)
    n2 = np.maximum(W - split, 1)
    return first_cnt / n1, second_cnt / n2


# ---------------------------------------------------------------------------
# Dome-shaped volume  (climax / distribution)
# ---------------------------------------------------------------------------

def dome_shaped_volume(o, h, l, c, v,
                        window: int = 20,
                        peak_pct: float = 0.70) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Volume forms an inverted U over *window* bars: rises to a climax peak,
    then declines.  The peak must be at least *peak_pct* of the window from
    either end (i.e. not at the edges).  Signals a distribution / topping phase.
    """
    v = _a(v)
    N = len(v)
    result = np.zeros(N, dtype=np.int8)
    if window + 1 > N:
        return result

    W = sliding_window_view(v, window + 1)
    peak_pos = W.argmax(axis=1)
    edge = int(window * (1 - peak_pct))
    inside = (peak_pos > edge) & (peak_pos < window - edge)
    if not inside.any():
        return result

    d = np.diff(W, axis=1)
    rise, fall = _run_fractions(d, peak_pos, positive=True)
    result[window:] = np.where(inside & (rise > 0.6) & (fall > 0.6), -1, 0)
    return result


# ---------------------------------------------------------------------------
# U-shaped volume  (accumulation / base)
# ---------------------------------------------------------------------------

def u_shaped_volume(o, h, l, c, v,
                     window: int = 20,
                     trough_pct: float = 0.70) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Volume forms a U over *window* bars: declines to a trough then recovers.
    Signals a base-building / accumulation phase.
    """
    v = _a(v)
    N = len(v)
    result = np.zeros(N, dtype=np.int8)
    if window + 1 > N:
        return result

    W = sliding_window_view(v, window + 1)
    trough_pos = W.argmin(axis=1)
    edge = int(window * (1 - trough_pct))
    inside = (trough_pos > edge) & (trough_pos < window - edge)
    if not inside.any():
        return result

    d = np.diff(W, axis=1)
    drop, recover = _run_fractions(d, trough_pos, positive=False)
    result[window:] = np.where(inside & (drop > 0.6) & (recover > 0.6), 1, 0)
    return result


# ---------------------------------------------------------------------------
# Volume trend  (rising / falling)
# ---------------------------------------------------------------------------

def _volume_trend(v, window, min_slope_pct, sign):
    """Shared engine for rising/falling volume trend.

    The per-bar ``np.polyfit(x, seg, 1)`` it replaces ran a full least-squares
    SVD for what is, on a fixed integer x-grid, a closed-form slope:
    ``slope = Σ(x-x̄)·y / Σ(x-x̄)²``.  The numerator for every window at once is
    a single matrix-vector product.
    """
    v = _a(v)
    N = len(v)
    result = np.zeros(N, dtype=np.int8)
    n = window + 1
    if n > N:
        return result

    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    ss_xx = (xc ** 2).sum()

    W = sliding_window_view(v, n)
    slope = (W @ xc) / ss_xx

    # Non-finite bars are zeroed before the prefix sum.  A running cumsum would
    # otherwise carry one NaN forward into *every* later window's mean, not just
    # the windows that contain it.  Windows that do contain a non-finite bar are
    # dropped by `ok` below, so the substituted zero never reaches a result.
    finite = np.isfinite(v)
    z0 = np.zeros(1)
    cs = np.concatenate((z0, np.cumsum(np.where(finite, np.abs(v), 0.0))))
    mean_v = (cs[n:] - cs[:-n]) / n

    ok = sliding_window_view(finite, n).all(axis=1)
    hit = ok & (sign * slope / (mean_v + _EPS) > min_slope_pct)
    result[window:] = np.where(hit, sign, 0)
    return result


def rising_volume_trend(o, h, l, c, v,
                         window: int = 20,
                         min_slope_pct: float = 0.003) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Volume has been rising consistently over *window* bars.
    A rising volume trend confirms upward price moves.
    *min_slope_pct* is the minimum slope as a fraction of mean volume.
    """
    return _volume_trend(v, window, min_slope_pct, 1)


def falling_volume_trend(o, h, l, c, v,
                          window: int = 20,
                          min_slope_pct: float = 0.003) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Volume has been declining consistently over *window* bars.
    Falling volume during a rally warns of weak buying conviction.
    """
    return _volume_trend(v, window, min_slope_pct, -1)


# ---------------------------------------------------------------------------
# Volume breakout day
# ---------------------------------------------------------------------------

def volume_breakout_day(o, h, l, c, v,
                         lookback: int = 20,
                         vol_factor: float = 2.0) -> np.ndarray:
    """
    **NON-DIRECTIONAL** (+1 / 0).
    Current bar's volume exceeds *vol_factor* × the average volume of the
    prior *lookback* bars, AND the bar closes more than 1 % above or below
    its open.  Signals a high-conviction directional move.

    NON-DIRECTIONAL WARNING: the +1 flag fires regardless of which direction
    the breakout goes.  Check the individual candle (white/black) for direction.
    """
    o, h, l, c, v = (_a(x) for x in (o, h, l, c, v))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    if lookback >= N:
        return result

    # mean of the *prior* lookback bars, i.e. windows ending at i-1
    avg_vol = sliding_window_view(v[:-1], lookback).mean(axis=1)
    body = np.abs(c[lookback:] - o[lookback:]) / (np.abs(o[lookback:]) + _EPS)
    hit = (avg_vol >= _EPS) & (v[lookback:] > vol_factor * avg_vol) & (body > 0.01)
    result[lookback:] = np.where(hit, 1, 0)
    return result
