"""Harmonic patterns on textbook Carney/Bulkowski-ratio constructions.

These are the tests that pin down the XABCD skeleton bug: a correctly-formed
bullish harmonic is X=low, A=high, B=low, C=high, D=low (alternating), and
must fire +1 at the D low; the bearish mirror fires -1 at the D high.
"""
import numpy as np
import ta_patterns.chart_patterns as cp
from tests import synthetic as syn


def _run(fn, pivots, tol=0.08):
    o, h, l, c = syn.harmonic_series(pivots)
    sig = fn(o, h, l, c, window=20, pivot_n=2, fib_tol=tol)
    nz = np.where(sig != 0)[0]
    return sig, nz


def test_gartley_bull():
    sig, nz = _run(cp.gartley_bull, syn.GARTLEY_BULL)
    assert len(nz) and sig[nz[-1]] == 1


def test_gartley_bear():
    sig, nz = _run(cp.gartley_bear, syn.GARTLEY_BEAR)
    assert len(nz) and sig[nz[-1]] == -1


def test_bat_bull():
    sig, nz = _run(cp.bat_bull, syn.BAT_BULL, tol=0.12)
    assert len(nz) and sig[nz[-1]] == 1


def test_bat_bear():
    sig, nz = _run(cp.bat_bear, syn.BAT_BEAR, tol=0.12)
    assert len(nz) and sig[nz[-1]] == -1


def test_butterfly_bull():
    sig, nz = _run(cp.butterfly_bull, syn.BUTTERFLY_BULL, tol=0.12)
    assert len(nz) and sig[nz[-1]] == 1


def test_butterfly_bear():
    sig, nz = _run(cp.butterfly_bear, syn.BUTTERFLY_BEAR, tol=0.12)
    assert len(nz) and sig[nz[-1]] == -1


def test_crab_bull():
    sig, nz = _run(cp.crab_bull, syn.CRAB_BULL, tol=0.12)
    assert len(nz) and sig[nz[-1]] == 1


def test_crab_bear():
    sig, nz = _run(cp.crab_bear, syn.CRAB_BEAR, tol=0.12)
    assert len(nz) and sig[nz[-1]] == -1


def test_abcd_bull():
    # 4-point ABCD: A=high, B=low, C=high, D=low; CD == AB exactly
    o, h, l, c = syn.harmonic_series([110, 103.82, 107.64, 101.46, 105.0])
    sig = cp.abcd_bull(o, h, l, c, window=20, pivot_n=2, fib_tol=0.08)
    assert syn.fired(sig)
