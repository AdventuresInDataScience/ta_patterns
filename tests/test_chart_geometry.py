"""Classic-geometry breakouts (rectangle, flag, triangle) and busted patterns.

The flag test pins the breakout bug: ``c[i] > flag_hi`` was once impossible
because ``flag_hi`` included bar i's own high (h[i] >= c[i]).
"""
import ta_patterns.chart_patterns as cp
from tests import synthetic as syn


def test_rectangle_top():
    o, h, l, c = syn.rectangle_top()
    assert syn.fired(cp.rectangle_top(o, h, l, c, window=40, pivot_n=2,
                                      flat_tol=0.03))


def test_rectangle_bottom():
    o, h, l, c = syn.rectangle_bottom()
    assert syn.fired(cp.rectangle_bottom(o, h, l, c, window=40, pivot_n=2,
                                         flat_tol=0.03))


def test_flag_high_tight():
    o, h, l, c = syn.flag_high_tight()
    assert syn.fired(cp.flag_high_tight(o, h, l, c, pole_bars=10, window=7,
                                        min_pole=0.40, max_retrace=0.25))


def test_ascending_triangle_breakout():
    o, h, l, c = syn.ascending_triangle_breakout()
    assert syn.fired(cp.ascending_triangle(o, h, l, c, window=30, pivot_n=2,
                                           flat_tol=0.03))


def test_busted_asc_triangle():
    o, h, l, c = syn.busted_asc_triangle()
    assert syn.fired(cp.busted_asc_triangle(o, h, l, c, window=30, pivot_n=2,
                                            reversal_bars=12, reversal_pct=0.04,
                                            flat_tol=0.03))


def test_busted_desc_triangle():
    o, h, l, c = syn.busted_desc_triangle()
    assert syn.fired(cp.busted_desc_triangle(o, h, l, c, window=30, pivot_n=2,
                                             reversal_bars=12, reversal_pct=0.04,
                                             flat_tol=0.03))


def test_busted_rectangle():
    o, h, l, c = syn.busted_rectangle()
    assert syn.fired(cp.busted_rectangle(o, h, l, c, window=45, pivot_n=2,
                                         reversal_bars=12, reversal_pct=0.04,
                                         flat_tol=0.03))
