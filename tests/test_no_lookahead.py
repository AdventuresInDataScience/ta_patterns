"""Point-in-time guarantee: a detector's value at bar t must be identical
whether computed on the full series or on the prefix ending at t.  This is
the property a lookahead leak (e.g. judging a reversal with future data)
would violate.
"""
import ta_patterns as tap
from tests.synthetic import random_ohlc

SEEDS = range(5)
CUTS = (170, 220, 270, 300)


def test_no_lookahead_full_catalog():
    violations = []
    for seed in SEEDS:
        o, h, l, c, v = random_ohlc(seed, n=320)
        full = tap.scan_all_patterns(o, h, l, c, v=v)
        for T in CUTS:
            pre = tap.scan_all_patterns(o[:T], h[:T], l[:T], c[:T], v=v[:T])
            for k in full:
                if int(full[k][T - 1]) != int(pre[k][T - 1]):
                    violations.append((k, seed, T))
    assert not violations, violations[:20]
