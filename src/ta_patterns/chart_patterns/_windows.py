"""
ta_patterns.chart_patterns._windows
=====================================
Shared vectorisation primitives for detectors that were written as per-bar
Python loops.

Two recurring shapes in this library:

1. *Trailing fixed windows* — ``a[i-window:i]`` for every bar.  A
   ``sliding_window_view`` gives all of them as one strided view with no copy,
   so a whole column of per-window statistics is a single reduction.

2. *Variable pivot ranges* — ``max(p[a_i:b_i])`` where ``a_i``/``b_i`` come from
   ``searchsorted`` and therefore differ per bar.  A sparse table answers every
   such query in O(1) after an O(P log P) build, which turns an O(N·P) loop into
   two vectorised gathers.

Both are exact: they compute the same quantities the loops did, not
approximations of them.
"""
from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def trailing_windows(a: np.ndarray, window: int) -> np.ndarray:
    """View of ``a[i-window:i]`` for every ``i`` in ``[window, len(a))``.

    Row ``k`` of the result is the window ending just before bar
    ``k + window``.  No data is copied.
    """
    a = np.ascontiguousarray(a, dtype=float)
    if window < 1 or len(a) <= window:
        return np.empty((0, max(window, 0)), dtype=float)
    return sliding_window_view(a[:-1], window)


class RangeAgg:
    """Sparse table for O(1) range max/min over a fixed array.

    Built once per detector; every bar's query is then a vectorised gather.
    ``query(a, b)`` takes arrays of half-open bounds and returns the aggregate
    of ``x[a:b]`` for each pair.  Empty ranges (``b <= a``) yield the identity
    value, so callers should mask them out explicitly.
    """

    __slots__ = ("_levels", "_log", "_n", "_op", "_identity")

    def __init__(self, x: np.ndarray, kind: str = "max"):
        x = np.asarray(x, dtype=float)
        self._n = n = len(x)
        self._op = np.maximum if kind == "max" else np.minimum
        self._identity = -np.inf if kind == "max" else np.inf
        if n == 0:
            self._levels, self._log = [], np.zeros(1, dtype=np.int64)
            return
        levels = [x]
        k = 1
        while (1 << k) <= n:
            prev = levels[-1]
            span = 1 << (k - 1)
            levels.append(self._op(prev[: n - (1 << k) + 1],
                                   prev[span: n - span + 1]))
            k += 1
        self._levels = levels
        # log[m] = floor(log2(m)) for m in [0, n]
        self._log = np.zeros(n + 1, dtype=np.int64)
        if n >= 2:
            self._log[2:] = np.floor(np.log2(np.arange(2, n + 1))).astype(np.int64)

    def query(self, a, b):
        """Aggregate of ``x[a:b]`` for each element of the ``a``/``b`` arrays."""
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        out = np.full(a.shape, self._identity, dtype=float)
        if self._n == 0:
            return out
        length = b - a
        ok = length > 0
        if not ok.any():
            return out
        aa = a[ok]
        ll = length[ok]
        k = self._log[ll]
        span = (1 << k)
        # two overlapping power-of-two blocks cover [a, b) exactly
        left = np.empty(len(aa))
        right = np.empty(len(aa))
        for kv in np.unique(k):
            m = k == kv
            lvl = self._levels[kv]
            left[m] = lvl[aa[m]]
            right[m] = lvl[aa[m] + ll[m] - span[m]]
        out[ok] = self._op(left, right)
        return out


def searchsorted_bounds(t: np.ndarray, lo, hi, hi_side: str = "left"):
    """Vectorised ``[a, b)`` bounds of the pivots whose times fall in a window.

    ``t`` must be sorted ascending (it comes from ``np.where``).  ``lo``/``hi``
    are per-bar arrays, so this resolves every bar's slice in one call rather
    than one ``searchsorted`` pair per loop iteration.
    """
    a = np.searchsorted(t, lo, side="left")
    b = np.searchsorted(t, hi, side=hi_side)
    return a, b
