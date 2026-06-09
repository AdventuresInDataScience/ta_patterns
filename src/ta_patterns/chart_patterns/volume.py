"""
ta_patterns.chart_patterns.volume
===================================
Volume-based patterns.  All functions require a ``v`` (volume) array in
addition to the standard ``o, h, l, c``.

Return type
-----------
All functions return **int8**: +1 bullish, -1 bearish, 0 not fired.
Non-directional patterns return +1 when the shape fires.
"""
from __future__ import annotations
import numpy as np
from .._core import _to_np, _EPS
from ._core import _fill_signal

_a = _to_np


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
    for i in range(window, N):
        seg = v[i - window : i + 1]
        peak_pos = int(np.argmax(seg))
        # Peak must be in the middle portion
        edge = int(window * (1 - peak_pct))
        if edge < peak_pos < window - edge:
            # Check rising half before peak and falling half after
            first_half  = seg[: peak_pos + 1]
            second_half = seg[peak_pos :]
            if (np.diff(first_half) >= 0).mean() > 0.6 and \
               (np.diff(second_half) <= 0).mean() > 0.6:
                result[i] = -1
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
    for i in range(window, N):
        seg = v[i - window : i + 1]
        trough_pos = int(np.argmin(seg))
        edge = int(window * (1 - trough_pct))
        if edge < trough_pos < window - edge:
            first_half  = seg[: trough_pos + 1]
            second_half = seg[trough_pos :]
            if (np.diff(first_half) <= 0).mean() > 0.6 and \
               (np.diff(second_half) >= 0).mean() > 0.6:
                result[i] = 1
    return result


# ---------------------------------------------------------------------------
# Rising volume trend
# ---------------------------------------------------------------------------

def rising_volume_trend(o, h, l, c, v,
                         window: int = 20,
                         min_slope_pct: float = 0.003) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Volume has been rising consistently over *window* bars.
    A rising volume trend confirms upward price moves.
    *min_slope_pct* is the minimum slope as a fraction of mean volume.
    """
    v = _a(v)
    N = len(v)
    result = np.zeros(N, dtype=np.int8)
    x = np.arange(window + 1, dtype=float)
    for i in range(window, N):
        seg = v[i - window : i + 1]
        if np.any(np.isnan(seg)):
            continue
        slope, _ = np.polyfit(x, seg, 1)
        mean_v = np.mean(np.abs(seg)) + _EPS
        if slope / mean_v > min_slope_pct:
            result[i] = 1
    return result


def falling_volume_trend(o, h, l, c, v,
                          window: int = 20,
                          min_slope_pct: float = 0.003) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Volume has been declining consistently over *window* bars.
    Falling volume during a rally warns of weak buying conviction.
    """
    v = _a(v)
    N = len(v)
    result = np.zeros(N, dtype=np.int8)
    x = np.arange(window + 1, dtype=float)
    for i in range(window, N):
        seg = v[i - window : i + 1]
        if np.any(np.isnan(seg)):
            continue
        slope, _ = np.polyfit(x, seg, 1)
        mean_v = np.mean(np.abs(seg)) + _EPS
        if -slope / mean_v > min_slope_pct:
            result[i] = -1
    return result


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
    for i in range(lookback, N):
        avg_vol = v[i - lookback : i].mean()
        if avg_vol < _EPS:
            continue
        if (v[i] > vol_factor * avg_vol
                and abs(c[i] - o[i]) / (abs(o[i]) + _EPS) > 0.01):
            result[i] = 1
    return result
