"""Double / triple / multi family, including the full Adam-Eve variant matrix
with negative controls (the variants that were dead before the float-vs-string,
off-by-pivot_n, clustering, and sharpness-symmetry fixes).
"""
import ta_patterns.chart_patterns as cp
from tests import synthetic as syn

DT_KW = dict(window=60, pivot_n=2, tol=0.06, min_separation=6)


def _dt(fn, k1, k2):
    o, h, l, c = syn.double_top(k1, k2)
    return syn.fired(fn(o, h, l, c, **DT_KW))


def _db(fn, k1, k2):
    o, h, l, c = syn.double_bottom(k1, k2)
    return syn.fired(fn(o, h, l, c, **DT_KW))


# ---- double top Adam/Eve matrix ------------------------------------------

def test_double_top_plain():
    assert _dt(cp.double_top, "adam", "adam")


def test_double_top_adam_adam():
    assert _dt(cp.double_top_adam_adam, "adam", "adam")


def test_double_top_eve_eve():
    assert _dt(cp.double_top_eve_eve, "eve", "eve")


def test_double_top_adam_eve():
    assert _dt(cp.double_top_adam_eve, "adam", "eve")


def test_double_top_eve_adam():
    assert _dt(cp.double_top_eve_adam, "eve", "adam")


def test_double_top_adam_adam_rejects_rounded():
    assert not _dt(cp.double_top_adam_adam, "eve", "eve")


def test_double_top_eve_eve_rejects_pointed():
    assert not _dt(cp.double_top_eve_eve, "adam", "adam")


# ---- double bottom Adam/Eve matrix ---------------------------------------

def test_double_bottom_plain():
    assert _db(cp.double_bottom, "adam", "adam")


def test_double_bottom_adam_adam():
    assert _db(cp.double_bottom_adam_adam, "adam", "adam")


def test_double_bottom_eve_eve():
    assert _db(cp.double_bottom_eve_eve, "eve", "eve")


def test_double_bottom_adam_eve():
    assert _db(cp.double_bottom_adam_eve, "adam", "eve")


def test_double_bottom_eve_adam():
    assert _db(cp.double_bottom_eve_adam, "eve", "adam")


def test_double_bottom_adam_adam_rejects_rounded():
    assert not _db(cp.double_bottom_adam_adam, "eve", "eve")


# ---- triple / big / three-peaks / etc. -----------------------------------

def test_triple_top():
    o, h, l, c = syn.triple_top()
    assert syn.fired(cp.triple_top(o, h, l, c, window=len(c) - 3, pivot_n=2,
                                   tol=0.05, min_separation=4))


def test_triple_bottom():
    o, h, l, c = syn.triple_bottom()
    assert syn.fired(cp.triple_bottom(o, h, l, c, window=len(c) - 3, pivot_n=2,
                                      tol=0.05, min_separation=4))


def test_three_peaks():
    o, h, l, c = syn.three_falling_peaks()
    assert syn.fired(cp.three_peaks(o, h, l, c, window=len(c) - 3, pivot_n=2,
                                    min_separation=4))


def test_three_valleys():
    o, h, l, c = syn.three_rising_valleys()
    assert syn.fired(cp.three_valleys(o, h, l, c, window=len(c) - 3, pivot_n=2,
                                      min_separation=4))


def test_big_m():
    o, h, l, c = syn.big_m()
    assert syn.fired(cp.big_m(o, h, l, c, window=len(c) - 3, pivot_n=3,
                              tol=0.05, min_sep=15))


def test_big_w():
    o, h, l, c = syn.big_w()
    assert syn.fired(cp.big_w(o, h, l, c, window=len(c) - 3, pivot_n=3,
                              tol=0.05, min_sep=15))


def test_cat_ears():
    o, h, l, c = syn.cat_ears()
    assert syn.fired(cp.cat_ears(o, h, l, c, window=25, pivot_n=2,
                                 tol=0.03, min_separation=3))


def test_ugly_double_bottom():
    o, h, l, c = syn.ugly_double_bottom()
    assert syn.fired(cp.ugly_double_bottom(o, h, l, c, window=50, pivot_n=2,
                                           tol=0.06, min_separation=5))


def test_three_peaks_domed_house():
    o, h, l, c = syn.domed_house()
    assert syn.fired(cp.three_peaks_domed_house(o, h, l, c,
                                                window=len(c) - 3, pivot_n=2))
