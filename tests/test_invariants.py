"""Structural invariants that must hold for every detector on any input.

Catches: wrong dtype, wrong length, out-of-range values, and — most
importantly — a signal whose sign contradicts its catalogue classification
(e.g. a 'bullish' pattern emitting -1, the class of bug that the inverted
pothole detector once had).
"""
import numpy as np
import ta_patterns as tap
import ta_patterns.scanner as sc
from ta_patterns import chart_patterns as cp
from tests.synthetic import random_ohlc

SEEDS = range(8)


def _allowed(name):
    s = set()
    if name in sc.BULLISH or name in cp.BULLISH:
        s |= {0, 1}
    if name in sc.BEARISH or name in cp.BEARISH:
        s |= {0, -1}
    if name in sc.NON_DIRECTIONAL or name in cp.NON_DIRECTIONAL:
        s |= {0, 1}
    if name in cp.BIDIRECTIONAL:
        s |= {0, 1, -1}
    return s


def _scan(seed):
    o, h, l, c, v = random_ohlc(seed, drift=(0.0 if seed % 3 else 0.15))
    return o, h, l, c, v, tap.scan_all_patterns(o, h, l, c, v=v)


def test_dtype_is_int8():
    for seed in SEEDS:
        *_, sigs = _scan(seed)
        for k, a in sigs.items():
            assert a.dtype == np.int8, (k, a.dtype)


def test_length_matches_input():
    for seed in SEEDS:
        o, h, l, c, v, sigs = _scan(seed)
        for k, a in sigs.items():
            assert len(a) == len(c), k


def test_values_in_range():
    for seed in SEEDS:
        *_, sigs = _scan(seed)
        for k, a in sigs.items():
            assert set(np.unique(a).tolist()) <= {-1, 0, 1}, k


def test_sign_matches_classification():
    for seed in SEEDS:
        *_, sigs = _scan(seed)
        for key, a in sigs.items():
            name = key[3:] if key.startswith("cp_") else key
            allowed = _allowed(name)
            assert allowed, f"{name} is uncategorised"
            assert set(np.unique(a).tolist()) <= allowed, (key, np.unique(a))


def test_no_crash_on_short_and_flat_series():
    # tiny and degenerate inputs must not raise
    for c in (np.full(60, 100.0), np.arange(60, dtype=float)):
        o = c.copy(); h = c + 0.1; l = c - 0.1; v = np.ones(60)
        sigs = tap.scan_all_patterns(o, h, l, c, v=v)
        assert len(sigs) == 300
