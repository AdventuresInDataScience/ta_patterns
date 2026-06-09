"""
ta_patterns.chart_patterns.busted
===================================
Busted chart pattern detectors.

A 'busted' pattern is one that broke out in the expected direction but
quickly reversed — creating a trap for traders who entered on the breakout.
The false breakout itself becomes a signal in the *opposite* direction.

All functions return int8: +1 (bullish reversal), -1 (bearish reversal), 0.
"""
from __future__ import annotations
import numpy as np
from ._core import _a, _EPS
from .classic import (
    ascending_triangle, descending_triangle, symmetrical_triangle,
    rectangle_top, rectangle_bottom,
)
from .double_multi import double_top, double_bottom, triple_top, triple_bottom
from .classic import hs_top, hs_bottom


def _bust(base_signal: np.ndarray, c: np.ndarray,
          reversal_bars: int, reversal_pct: float,
          direction: int) -> np.ndarray:
    """
    Given a base pattern signal (in *direction*), detect when the breakout
    fails and price reverses by *reversal_pct* within *reversal_bars* bars.
    Returns a signal in the opposite direction.

    Point-in-time: the bust is confirmed at the *first* bar where the
    reversal threshold is met, using only the bars from the breakout up to
    and including that bar.  No data beyond the fire bar is consulted, so a
    later extreme cannot retroactively create or move the signal.
    """
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    signal_bars = np.where(base_signal != 0)[0]
    for t in signal_bars:
        last = min(t + reversal_bars, N - 1)
        if direction == 1:                  # was bullish → look for reversal down
            peak = c[t]                     # running peak since breakout
            for k in range(t + 1, last + 1):
                peak = max(peak, c[k])
                if (peak - c[k]) / (peak + _EPS) >= reversal_pct:
                    result[k] = -1          # opposite direction
                    break
        else:                               # was bearish → look for reversal up
            trough = c[t]                   # running trough since breakout
            for k in range(t + 1, last + 1):
                trough = min(trough, c[k])
                if (c[k] - trough) / (abs(trough) + _EPS) >= reversal_pct:
                    result[k] = 1
                    break
    return result


def _bust_pattern(base_fn, o, h, l, c, direction,
                   reversal_bars=10, reversal_pct=0.05, **kw):
    base = base_fn(o, h, l, c, **kw)
    return _bust(base, _a(c), reversal_bars, reversal_pct, direction)


# ---------------------------------------------------------------------------
# Busted ascending triangle  (was bullish → bust is bearish)
# ---------------------------------------------------------------------------

def busted_asc_triangle(o, h, l, c,
                         reversal_bars: int = 10,
                         reversal_pct: float = 0.05,
                         **kw) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Ascending triangle breaks out upward but quickly reverses back below
    resistance — bears trapped the bulls.
    """
    return _bust_pattern(ascending_triangle, o, h, l, c, 1,
                          reversal_bars, reversal_pct, **kw)


def busted_desc_triangle(o, h, l, c,
                          reversal_bars: int = 10,
                          reversal_pct: float = 0.05,
                          **kw) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Descending triangle breaks down but reverses upward — bear trap.
    """
    return _bust_pattern(descending_triangle, o, h, l, c, -1,
                          reversal_bars, reversal_pct, **kw)


def busted_double_bottom(o, h, l, c,
                          reversal_bars: int = 10,
                          reversal_pct: float = 0.05,
                          **kw) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Double bottom breaks out bullishly then collapses — bull trap.
    """
    return _bust_pattern(double_bottom, o, h, l, c, 1,
                          reversal_bars, reversal_pct, **kw)


def busted_double_top(o, h, l, c,
                       reversal_bars: int = 10,
                       reversal_pct: float = 0.05,
                       **kw) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Double top breaks down then reverses upward — bear trap.
    """
    return _bust_pattern(double_top, o, h, l, c, -1,
                          reversal_bars, reversal_pct, **kw)


def busted_hs_bottom(o, h, l, c,
                      reversal_bars: int = 10,
                      reversal_pct: float = 0.05,
                      **kw) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Inverse H&S breaks above neckline then collapses — bull trap.
    """
    return _bust_pattern(hs_bottom, o, h, l, c, 1,
                          reversal_bars, reversal_pct, **kw)


def busted_hs_top(o, h, l, c,
                   reversal_bars: int = 10,
                   reversal_pct: float = 0.05,
                   **kw) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    H&S top breaks below neckline then reverses — bear trap.
    """
    return _bust_pattern(hs_top, o, h, l, c, -1,
                          reversal_bars, reversal_pct, **kw)


def busted_rectangle(o, h, l, c,
                      reversal_bars: int = 10,
                      reversal_pct: float = 0.05,
                      **kw) -> np.ndarray:
    """
    **BIDIRECTIONAL** (+1 or -1).
    Rectangle breakout fails and reverses — direction depends on which way
    the original breakout went.
    """
    bull_bust = _bust_pattern(rectangle_bottom, o, h, l, c, 1,
                               reversal_bars, reversal_pct, **kw)
    bear_bust = _bust_pattern(rectangle_top, o, h, l, c, -1,
                               reversal_bars, reversal_pct, **kw)
    r = bull_bust.astype(np.int8) + bear_bust.astype(np.int8)
    return r.clip(-1, 1).astype(np.int8)


def busted_sym_triangle(o, h, l, c,
                         reversal_bars: int = 10,
                         reversal_pct: float = 0.05,
                         **kw) -> np.ndarray:
    """
    **BIDIRECTIONAL** (+1 or -1).
    Symmetrical triangle breakout reverses — direction is opposite to
    the initial breakout.
    """
    base = symmetrical_triangle(o, h, l, c, **kw)
    c_arr = _a(c)
    N = len(c_arr)
    result = np.zeros(N, dtype=np.int8)
    for t in np.where(base != 0)[0]:
        if t + reversal_bars >= N:
            continue
        orig = int(base[t])
        future = c_arr[t + 1: t + reversal_bars + 1]
        if orig == 1:
            if (future.max() - future.min()) / (future.max() + _EPS) >= reversal_pct:
                result[min(t + future.argmin() + 1, N - 1)] = -1
        elif orig == -1:
            if (future.max() - future.min()) / (abs(future.min()) + _EPS) >= reversal_pct:
                result[min(t + future.argmax() + 1, N - 1)] = 1
    return result


def busted_triple_bottom(o, h, l, c,
                          reversal_bars: int = 10,
                          reversal_pct: float = 0.05,
                          **kw) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Triple bottom breaks out then collapses.
    """
    return _bust_pattern(triple_bottom, o, h, l, c, 1,
                          reversal_bars, reversal_pct, **kw)


def busted_triple_top(o, h, l, c,
                       reversal_bars: int = 10,
                       reversal_pct: float = 0.05,
                       **kw) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Triple top breaks down then reverses.
    """
    return _bust_pattern(triple_top, o, h, l, c, -1,
                          reversal_bars, reversal_pct, **kw)
