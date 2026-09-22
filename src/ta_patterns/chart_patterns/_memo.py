"""
ta_patterns.chart_patterns._memo
==================================
A small content-addressed cache for derived features that many detectors
recompute independently.

Why
---
Each chart-pattern detector calls ``pivot_highs`` / ``pivot_lows`` (and often
``_sliding_trendlines``) on the same OHLC arrays with the same parameters.
Across a full ``scan_all_patterns`` run that is ~200 identical recomputations
of the same result — measured at 498 ms versus 2.5 ms for a single call on a
20 000-bar series.

How
---
Keys are the **contents** of the input arrays (a BLAKE2b digest), not their
identity, so the cache is immune to in-place mutation, buffer reuse after a
free, and callers passing copies or pandas Series.  Hashing a 20 000-bar
float64 array costs about 0.1 ms against the 1.2 ms it saves, and the saving
multiplies by every detector in the scan.

Cached values are handed back as **copies**, so a caller mutating a returned
array cannot corrupt another detector's view.  Copying a boolean mask is
microseconds against milliseconds of recomputation.

The cache is bounded (LRU) on both entry count (``TA_PATTERNS_CACHE_SIZE``,
default 64) and total payload bytes (``TA_PATTERNS_CACHE_BYTES``, default
64 MB), so a long-running process holding very long series cannot grow without
limit.  It can be disabled entirely with ``TA_PATTERNS_NO_CACHE=1``, inspected
with :func:`cache_info`, and cleared with :func:`clear_cache`; both are
re-exported from the top-level ``ta_patterns`` package.
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict

import numpy as np

_MAXSIZE = int(os.environ.get("TA_PATTERNS_CACHE_SIZE", "64"))
_MAXBYTES = int(os.environ.get("TA_PATTERNS_CACHE_BYTES", str(64 * 1024 * 1024)))
_ENABLED = os.environ.get("TA_PATTERNS_NO_CACHE", "") not in ("1", "true", "True")

_store: "OrderedDict[tuple, object]" = OrderedDict()
_bytes = 0
_lock = threading.Lock()


def array_key(a) -> tuple | None:
    """Content digest of *a*, or None if it cannot be hashed cheaply."""
    try:
        arr = np.ascontiguousarray(a)
        if arr.dtype == object:
            return None
        digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).digest()
    except (TypeError, ValueError, AttributeError):
        return None
    return (arr.shape, arr.dtype.str, digest)


def _copy(value):
    if isinstance(value, tuple):
        return tuple(v.copy() if isinstance(v, np.ndarray) else v for v in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _sizeof(value) -> int:
    if isinstance(value, tuple):
        return sum(_sizeof(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.nbytes
    return 0


def memoized(key: tuple | None, compute):
    """Return ``compute()``, caching on *key*.  A None key bypasses the cache."""
    global _bytes
    if not _ENABLED or key is None:
        return compute()
    with _lock:
        hit = _store.get(key)
        if hit is not None:
            _store.move_to_end(key)
            return _copy(hit)
    value = compute()
    with _lock:
        # An entry count alone does not bound memory: one entry is a few bytes on
        # a 100-bar series and megabytes on a 100 000-bar one.  Evict on whichever
        # of the two ceilings binds first.
        if key in _store:
            _bytes -= _sizeof(_store[key])
        _store[key] = value
        _store.move_to_end(key)
        _bytes += _sizeof(value)
        while _store and (len(_store) > _MAXSIZE or _bytes > _MAXBYTES):
            _, evicted = _store.popitem(last=False)
            _bytes -= _sizeof(evicted)
    return _copy(value)


def clear_cache() -> None:
    """Drop every cached feature.  Mainly useful in tests and benchmarks."""
    global _bytes
    with _lock:
        _store.clear()
        _bytes = 0


def cache_info() -> dict:
    with _lock:
        return {"entries": len(_store), "maxsize": _MAXSIZE,
                "nbytes": _bytes, "maxbytes": _MAXBYTES, "enabled": _ENABLED}
