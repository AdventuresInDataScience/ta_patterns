"""Point-in-time guarantee: a detector's value at bar t must be identical
whether computed on the full series or on any prefix that contains bar t.
This is the property a lookahead leak (e.g. judging a reversal with future
data) would violate.

Every bar of each prefix is compared, not only its last. A detector that
reads past its own bar does not always change the last bar of a prefix: a
negative index that wraps round to the END of the array (``c[-1]``) changes
an early bar's value instead, and only on the bar where the index first goes
negative. The flag and pennant detectors did exactly that on their first
evaluated bar until 1.2.2, and a last-bar-only comparison never saw it.
"""
import numpy as np

import ta_patterns as tap
from ta_patterns import chart_patterns as cp
from tests.synthetic import random_ohlc

SEEDS = range(5)
# Early cuts too: a leak on a detector's first evaluated bars only shows when
# the prefix and the full series differ there.
CUTS = (60, 120, 170, 220, 270, 300)


def test_no_lookahead_full_catalog():
    violations = []
    for seed in SEEDS:
        o, h, l, c, v = random_ohlc(seed, n=320)
        full = tap.scan_all_patterns(o, h, l, c, v=v)
        for T in CUTS:
            pre = tap.scan_all_patterns(o[:T], h[:T], l[:T], c[:T], v=v[:T])
            for k in full:
                differ = np.flatnonzero(full[k][:T].astype(int) != pre[k].astype(int))
                if len(differ):
                    violations.append((k, seed, T, int(differ[0])))
    assert not violations, violations[:20]


def _walk(seed, n=400):
    """A geometric random walk with 2% daily moves: flagpoles of 5% are common."""
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = np.r_[c[0], c[:-1]]
    return o, np.maximum(o, c) * 1.005, np.minimum(o, c) * 0.995, c


def test_flags_and_pennants_do_not_read_the_last_close():
    """Regression for 1.2.1: ``_flagpole`` measured the pole from ``c[-1]``, the
    last close of the whole series, on each detector's first evaluated bar
    (bar 40 with the defaults). On seed 3, flag_bear flagged bar 40 with 400
    bars of history and not with the first 150 of the same bars."""
    detectors = (cp.flag_bull, cp.flag_bear, cp.pennant_bull, cp.pennant_bear)
    violations = []
    for seed in range(20):
        o, h, l, c = _walk(seed)
        for detector in detectors:
            full = detector(o, h, l, c)
            for T in (60, 150):
                pre = detector(o[:T], h[:T], l[:T], c[:T])
                differ = np.flatnonzero(full[:T] != pre)
                if len(differ):
                    violations.append((detector.__name__, seed, T, int(differ[0])))
    assert not violations, violations[:20]
