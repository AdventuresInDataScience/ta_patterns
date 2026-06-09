"""Catalog integrity: counts, classification reconciliation, no duplicates."""
import ta_patterns as tap
from ta_patterns import chart_patterns as cp
import ta_patterns.scanner as sc
from tests.synthetic import random_ohlc


def test_total_pattern_count():
    assert len(tap.list_all_patterns()) == 300


def test_module_split():
    candles = tap.list_all_patterns(module="candles")
    chart = tap.list_all_patterns(module="chart")
    assert len(candles) == 106
    assert len(chart) == 194


def test_scan_counts_with_and_without_volume():
    o, h, l, c, v = random_ohlc(0)
    assert len(tap.scan_all_patterns(o, h, l, c)) == 295          # OHLC only
    assert len(tap.scan_all_patterns(o, h, l, c, v=v)) == 300     # + volume


def test_chart_direction_partition():
    # bullish / bearish / bidirectional / non-directional must sum to the whole
    total = len(cp.CHART_PATTERNS)
    parts = len(cp.BULLISH) + len(cp.BEARISH) + len(cp.BIDIRECTIONAL) \
        + len(cp.NON_DIRECTIONAL)
    assert parts == total == 194


def test_candle_direction_partition():
    total = len(sc.PATTERNS)
    parts = len(sc.BULLISH) + len(sc.BEARISH) + len(sc.NON_DIRECTIONAL)
    assert parts == total == 106


def test_bullish_and_bearish_disjoint():
    assert not (sc.BULLISH & sc.BEARISH)
    assert not (cp.BULLISH & cp.BEARISH)


def test_list_directions_are_subsets():
    allnames = set(tap.list_all_patterns())
    for d in ("bullish", "bearish", "non_directional"):
        assert set(tap.list_all_patterns(direction=d)) <= allnames
