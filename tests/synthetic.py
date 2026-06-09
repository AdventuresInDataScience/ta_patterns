"""
Synthetic OHLC builders shared across the test-suite.

These are deliberately plain functions (no pytest fixtures) so the textbook
constructions can be reused freely and the suite stays runnable with or
without pytest.  Every builder returns ``(o, h, l, c)`` float arrays unless
noted; helpers add tiny symmetric shadows so candles are clean by default.
"""
from __future__ import annotations
import numpy as np


def ramp(a, b, n):
    """Linear segment of ``n`` bars going from just past ``a`` to ``b``."""
    return list(np.linspace(a, b, n + 1))[1:]


def ohlc(close, pad=0.1):
    """Turn a close-price path into clean OHLC (open=close, tiny shadows)."""
    c = np.asarray(close, dtype=float)
    o = c.copy()
    h = c + pad
    l = c - pad
    return o, h, l, c


def random_ohlc(seed, n=400, drift=0.0, regime="rw"):
    """Reproducible random OHLCV for invariant / no-look-ahead sweeps."""
    rng = np.random.default_rng(seed)
    if regime == "rw":
        steps = rng.normal(drift, 1, n)
    elif regime == "up":
        steps = rng.normal(0.15, 1, n)
    elif regime == "down":
        steps = rng.normal(-0.15, 1, n)
    elif regime == "vol":
        steps = rng.normal(0, 2.2, n)
    elif regime == "cycle":
        steps = np.diff(np.r_[0, 18 * np.sin(np.linspace(0, 14 * np.pi, n))]) \
            + rng.normal(0, 0.5, n)
    else:
        raise ValueError(regime)
    c = np.cumsum(steps) + 100
    o = c + rng.normal(0, 0.4, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.5, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.5, n))
    v = np.abs(rng.normal(1e6, 3e5, n))
    return o, h, l, c, v


# --------------------------------------------------------------------------
# Double top / bottom with selectable Adam (sharp) / Eve (rounded) extremes
# --------------------------------------------------------------------------

def _peak_seg(kind, start, ext, end):
    """Rise start->ext then fall ext->end; 'adam' = pointed, 'eve' = rounded."""
    if kind == "adam":
        return ramp(start, ext, 4) + ramp(ext, end, 4)
    return ramp(start, ext, 6) + [ext, ext, ext] + ramp(ext, end, 6)


def _trough_seg(kind, start, ext, end):
    """Fall start->ext then rise ext->end."""
    if kind == "adam":
        return ramp(start, ext, 4) + ramp(ext, end, 4)
    return ramp(start, ext, 6) + [ext, ext, ext] + ramp(ext, end, 6)


def double_top(kind1="adam", kind2="adam"):
    """Two equal ~120 peaks over a ~108 valley, then a close below the valley."""
    s = [95.0] + ramp(95, 100, 40)
    s += _peak_seg(kind1, 100, 120, 108)
    s += ramp(108, 108, 3)
    s += _peak_seg(kind2, 108, 120, 108)
    s += ramp(108, 103, 12)            # break below neckline (~108)
    return ohlc(s)


def double_bottom(kind1="adam", kind2="adam"):
    """Two equal ~80 troughs under a ~92 peak, then a close above the peak."""
    s = [105.0] + ramp(105, 100, 40)
    s += _trough_seg(kind1, 100, 80, 92)
    s += ramp(92, 92, 3)
    s += _trough_seg(kind2, 92, 80, 92)
    s += ramp(92, 97, 12)              # break above neckline (~92)
    return ohlc(s)


def triple_top():
    s = [95.0] + ramp(95, 120, 8) + ramp(120, 108, 5) + ramp(108, 120, 5) \
        + ramp(120, 108, 5) + ramp(108, 120, 5) + ramp(120, 105, 12)
    return ohlc(s)


def triple_bottom():
    s = [125.0] + ramp(125, 80, 8) + ramp(80, 92, 5) + ramp(92, 80, 5) \
        + ramp(80, 92, 5) + ramp(92, 80, 5) + ramp(80, 95, 12)
    return ohlc(s)


def three_falling_peaks():
    s = [95.0] + ramp(95, 120, 6) + ramp(120, 100, 5) + ramp(100, 114, 5) \
        + ramp(114, 96, 5) + ramp(96, 108, 5) + ramp(108, 88, 10)
    return ohlc(s)


def three_rising_valleys():
    s = [125.0] + ramp(125, 80, 6) + ramp(80, 100, 5) + ramp(100, 86, 5) \
        + ramp(86, 106, 5) + ramp(106, 92, 5) + ramp(92, 114, 10)
    return ohlc(s)


def ugly_double_bottom():
    s = [110.0] + ramp(110, 100, 30) + ramp(100, 80, 5) + ramp(80, 92, 5) \
        + ramp(92, 82, 5) + ramp(82, 97, 12)        # second trough ~82 (not equal)
    return ohlc(s)


def big_m():
    s = [95.0] + ramp(95, 120, 20) + ramp(120, 108, 18) + ramp(108, 120, 20) \
        + ramp(120, 103, 22)
    return ohlc(s)


def big_w():
    s = [125.0] + ramp(125, 80, 20) + ramp(80, 92, 18) + ramp(92, 80, 20) \
        + ramp(80, 97, 22)
    return ohlc(s)


def cat_ears():
    s = [95.0] + ramp(95, 100, 18) + ramp(100, 120, 3) + ramp(120, 110, 3) \
        + ramp(110, 120, 3) + ramp(120, 104, 8)
    return ohlc(s)


def domed_house():
    s = [95.0] + ramp(95, 110, 8) + ramp(110, 100, 5) + ramp(100, 116, 8) \
        + ramp(116, 106, 5) + ramp(106, 122, 8) + ramp(122, 112, 5)
    s += ramp(112, 120, 6) + ramp(120, 118, 4) + ramp(118, 119, 4) \
        + ramp(119, 110, 6) + ramp(110, 100, 10)
    return ohlc(s)


# --------------------------------------------------------------------------
# Harmonic XABCD builders (clean alternating pivots)
# --------------------------------------------------------------------------

def harmonic_series(pivots, gap=8):
    """Connect successive pivot prices with linear ramps ``gap`` bars apart."""
    seq = [pivots[0]]
    for p in pivots[1:]:
        seq += list(np.linspace(seq[-1], p, gap + 1))[1:]
    return ohlc(seq, pad=0.05)


# Textbook pivot prices (Carney/Bulkowski ratios) — last point = post-D move.
GARTLEY_BULL = [100, 110, 103.82, 107.64, 102.14, 106.0]
GARTLEY_BEAR = [110, 100, 106.18, 102.36, 107.86, 104.0]
BAT_BULL     = [100, 110, 105.6, 108.32, 101.14, 105.0]
BAT_BEAR     = [110, 100, 104.4, 101.68, 108.86, 105.0]
BUTTERFLY_BULL = [100, 110, 102.14, 106.07, 97.28, 101.0]
BUTTERFLY_BEAR = [110, 100, 107.86, 103.93, 112.72, 109.0]
CRAB_BULL    = [100, 110, 105, 109.25, 94.80, 100.0]
CRAB_BEAR    = [110, 100, 105, 100.75, 115.20, 110.0]


# --------------------------------------------------------------------------
# Classic-geometry and busted builders
# --------------------------------------------------------------------------

def _box(reps=4, lo=100, hi=110, step=5):
    s = []
    for _ in range(reps):
        s += ramp(lo, hi, step) + ramp(hi, lo, step)
    return s


def rectangle_top():
    return ohlc(ramp(130, 100, 30) + _box() + ramp(100, 90, 8), pad=0.2)


def rectangle_bottom():
    return ohlc(ramp(70, 100, 30) + _box() + ramp(110, 120, 8), pad=0.2)


def flag_high_tight():
    s = [100.0] + ramp(100, 150, 10) + [149, 148, 150, 149, 148, 150, 149] \
        + ramp(150, 162, 4)
    return ohlc(s, pad=0.2)


def ascending_triangle_breakout():
    s = [90.0] + ramp(90, 120, 5) + ramp(120, 103, 4) + ramp(103, 120, 5) \
        + ramp(120, 108, 4) + ramp(108, 120, 5) + ramp(120, 113, 4) \
        + ramp(113, 120, 4) + ramp(120, 128, 4)
    return ohlc(s, pad=0.2)


def busted_asc_triangle():
    s = [90.0] + ramp(90, 120, 5) + ramp(120, 103, 4) + ramp(103, 120, 5) \
        + ramp(120, 108, 4) + ramp(108, 120, 5) + ramp(120, 113, 4) \
        + ramp(113, 120, 4) + ramp(120, 127, 3) + ramp(127, 116, 5)
    return ohlc(s, pad=0.2)


def busted_desc_triangle():
    s = [130.0] + ramp(130, 100, 5) + ramp(100, 117, 4) + ramp(117, 100, 5) \
        + ramp(100, 112, 4) + ramp(112, 100, 5) + ramp(100, 107, 4) \
        + ramp(107, 100, 4) + ramp(100, 93, 3) + ramp(93, 104, 5)
    return ohlc(s, pad=0.2)


def busted_rectangle():
    s = ramp(80, 100, 12) + _box() + ramp(110, 117, 3) + ramp(117, 104, 5)
    return ohlc(s, pad=0.2)


def fired(arr):
    """True if a detector produced any non-zero signal."""
    return bool(np.any(np.asarray(arr) != 0))
