"""
ta_patterns._core
=================
Shared numeric helpers.  All operate on 1-D numpy arrays.
"""
from __future__ import annotations
import numpy as np

_EPS: float = 1e-10          # guards division by near-zero range


# ---------------------------------------------------------------------------
# Input normalisation  (accept numpy arrays, pandas Series, or any array-like)
# ---------------------------------------------------------------------------

def _to_np(x) -> np.ndarray:
    """
    Convert *x* to a float64 numpy array with minimal copying.

    Accepts:
    - numpy ndarray          → view (or copy if dtype conversion needed)
    - pandas Series          → uses .to_numpy() which strips the index
    - pandas DataFrame col   → same
    - any other array-like   → np.asarray fallback
    """
    if hasattr(x, "to_numpy"):          # pandas Series / DataFrame column
        return x.to_numpy(dtype=float, na_value=np.nan)
    return np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------

def roll_mean(a: np.ndarray, n: int) -> np.ndarray:
    """Unweighted rolling mean; first (n-1) values are NaN."""
    a = np.asarray(a, float)
    out = np.full(len(a), np.nan)
    if n > len(a):
        return out
    cs = np.cumsum(a)
    out[n - 1] = cs[n - 1] / n
    if len(a) > n:
        out[n:] = (cs[n:] - cs[:-n]) / n
    return out


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray,
        period: int = 14) -> np.ndarray:
    """Average True Range over *period* bars."""
    h, l, c = (np.asarray(x, float) for x in (h, l, c))
    pc = np.empty_like(c)
    pc[0] = c[0]
    pc[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return roll_mean(tr, period)


def avg_body(o: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """Rolling average of candle body sizes."""
    return roll_mean(np.abs(np.asarray(c, float) - np.asarray(o, float)), period)


# ---------------------------------------------------------------------------
# Single-bar geometry
# ---------------------------------------------------------------------------

def body(o, c):
    return np.abs(c - o)

def upper_wick(o, h, c):
    return h - np.maximum(o, c)

def lower_wick(o, l, c):
    return np.minimum(o, c) - l

def total_range(h, l):
    return h - l

def safe_range(h, l):
    """high - low, floored at _EPS to avoid division by zero."""
    return np.where(h > l, h - l, _EPS)

def body_top(o, c):
    return np.maximum(o, c)

def body_bot(o, c):
    return np.minimum(o, c)

def midpoint(o, c):
    return (o + c) / 2.0


# ---------------------------------------------------------------------------
# Trend helpers  (simple close-over-close proxies, no external indicators)
# ---------------------------------------------------------------------------

def uptrend(c: np.ndarray, n: int = 5) -> np.ndarray:
    """True at i if close[i] > close[i-n]  (price rose over the past n bars)."""
    c = np.asarray(c, float)
    out = np.zeros(len(c), dtype=bool)
    if n < len(c):
        out[n:] = c[n:] > c[:-n]
    return out


def downtrend(c: np.ndarray, n: int = 5) -> np.ndarray:
    """True at i if close[i] < close[i-n]  (price fell over the past n bars)."""
    c = np.asarray(c, float)
    out = np.zeros(len(c), dtype=bool)
    if n < len(c):
        out[n:] = c[n:] < c[:-n]
    return out
