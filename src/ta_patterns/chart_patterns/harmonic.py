"""
ta_patterns.chart_patterns.harmonic
=====================================
Fibonacci harmonic patterns and Wolfe waves.

All functions accept (o, h, l, c) arrays and return **int8** (+1/−1/0).

Harmonic pattern logic
-----------------------
Each pattern identifies 4–5 swing points (XABCD) using pivot detection,
then checks that the retracement/extension ratios between legs match the
Fibonacci ratios that define each pattern.

Pivots are found with the same delayed no-lookahead approach used throughout
the library; the signal fires at the *D* point (the potential reversal zone).

Tolerance parameter *fib_tol* (default 0.05) allows ±5 % deviation on all
ratio checks — tighter values yield fewer but higher-quality signals.

Elliott Wave patterns are intentionally NOT implemented.  They require
subjective multi-level wave labelling that does not reduce to a
deterministic OHLC rule.  This is a deliberate design decision, not an
accidental omission.
"""
from __future__ import annotations
import numpy as np
from ._core import (
    _a, _EPS, pivot_highs, pivot_lows, pivot_info, in_range, in_band,
)


def _piv(h, l, n, pct):
    ph = pivot_highs(h, n, pct)
    pl = pivot_lows(l, n, pct)
    t_ph, p_ph = pivot_info(h, ph, n)
    t_pl, p_pl = pivot_info(l, pl, n)
    return t_ph, p_ph, t_pl, p_pl


def _ratio(a, b):
    """Safe ratio b / a."""
    return abs(b) / (abs(a) + _EPS)


def _last_before(t_arr, t_ref, window):
    """Index of the most recent pivot strictly before ``t_ref`` and within
    ``window`` bars of it, else -1.

    Equivalent to ``max([(t, p) ... if t < t_ref and t_ref - t < window],
    key=time)`` — but because ``t_arr`` is sorted ascending the answer is just
    the pivot immediately preceding ``t_ref`` (when it falls inside the window),
    found with ``searchsorted`` in O(log P) instead of an O(P) scan per pivot.
    """
    b = int(np.searchsorted(t_arr, t_ref, "left"))
    if b == 0:
        return -1
    k = b - 1
    return k if (t_ref - t_arr[k]) < window else -1


# ---------------------------------------------------------------------------
# AB=CD
# ---------------------------------------------------------------------------

def abcd_bull(o, h, l, c,
               mode: str = 'confirmed',
               window: int = 100,
               pivot_n: int = 5,
               pivot_pct: float | None = None,
               fib_tol: float = 0.05) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    AB=CD bullish: ABCD structure where CD ≈ AB in length.
    Entry at D (price expected to rise).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    t_ph, p_ph, t_pl, p_pl = _piv(h, l, pivot_n, pivot_pct)
    result = np.zeros(N, dtype=np.int8)

    # Bullish ABCD: A=high, B=low, C=high (< A), D=low (entry)
    for iD in range(len(t_pl)):
        tD, pD = t_pl[iD], p_pl[iD]
        if tD < window:
            continue
        # C: most recent high before D
        kC = _last_before(t_ph, tD, window)
        if kC < 0:
            continue
        tC, pC = t_ph[kC], p_ph[kC]
        # B: most recent low before C
        kB = _last_before(t_pl, tC, window)
        if kB < 0:
            continue
        tB, pB = t_pl[kB], p_pl[kB]
        # A: most recent high before B
        kA = _last_before(t_ph, tB, window)
        if kA < 0:
            continue
        tA, pA = t_ph[kA], p_ph[kA]

        AB = pA - pB
        CD = pC - pD
        if AB <= 0 or CD <= 0:
            continue
        # CD should equal AB (ratio 1.0, tolerance fib_tol)
        bc_retrace = _ratio(AB, pC - pB)
        cd_ratio   = _ratio(AB, CD)
        if (in_band(bc_retrace, 0.382, 0.886) and
                in_range(cd_ratio, 1.0, fib_tol)):
            t_sig = min(tD, N - 1)
            if result[t_sig] == 0:
                if mode == 'forming':
                    result[t_sig] = 1
                elif c[t_sig] > pD:
                    result[t_sig] = 1
    return result


def abcd_bear(o, h, l, c,
               mode: str = 'confirmed',
               window: int = 100,
               pivot_n: int = 5,
               pivot_pct: float | None = None,
               fib_tol: float = 0.05) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    AB=CD bearish: entry at D (price expected to fall).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    t_ph, p_ph, t_pl, p_pl = _piv(h, l, pivot_n, pivot_pct)
    result = np.zeros(N, dtype=np.int8)

    for iD in range(len(t_ph)):
        tD, pD = t_ph[iD], p_ph[iD]
        if tD < window:
            continue
        kC = _last_before(t_pl, tD, window)
        if kC < 0:
            continue
        tC, pC = t_pl[kC], p_pl[kC]
        kB = _last_before(t_ph, tC, window)
        if kB < 0:
            continue
        tB, pB = t_ph[kB], p_ph[kB]
        kA = _last_before(t_pl, tB, window)
        if kA < 0:
            continue
        tA, pA = t_pl[kA], p_pl[kA]

        AB = pB - pA
        CD = pD - pC
        if AB <= 0 or CD <= 0:
            continue
        bc_retrace = _ratio(AB, pB - pC)
        cd_ratio   = _ratio(AB, CD)
        if (in_band(bc_retrace, 0.382, 0.886) and
                in_range(cd_ratio, 1.0, fib_tol)):
            t_sig = min(tD, N - 1)
            if result[t_sig] == 0:
                if mode == 'forming':
                    result[t_sig] = -1
                elif c[t_sig] < pD:
                    result[t_sig] = -1
    return result


# ---------------------------------------------------------------------------
# Generic XABCD scanner  (used by Gartley, Bat, Butterfly, Crab)
# ---------------------------------------------------------------------------

def _xabcd_bull(h, l, c, t_ph, p_ph, t_pl, p_pl, window, N, mode,
                ab_xa, bc_ab_lo, bc_ab_hi, cd_bc_lo, cd_bc_hi, ad_xa,
                fib_tol):
    """Generic bullish XABCD (X=L, A=H, B=L, C=H, D=L entry).

    Correct alternating harmonic skeleton: the pattern starts at a valley X,
    rallies to peak A, retraces to low B, bounces to high C, then dips to the
    D low (the potential reversal zone) where an upward reversal is expected.
    """
    result = np.zeros(N, dtype=np.int8)
    for iD in range(len(t_pl)):
        tD, pD = t_pl[iD], p_pl[iD]
        if tD < window:
            continue
        kC = _last_before(t_ph, tD, window)
        if kC < 0:
            continue
        tC, pC = t_ph[kC], p_ph[kC]
        kB = _last_before(t_pl, tC, window)
        if kB < 0:
            continue
        tB, pB = t_pl[kB], p_pl[kB]
        kA = _last_before(t_ph, tB, window)
        if kA < 0:
            continue
        tA, pA = t_ph[kA], p_ph[kA]
        kX = _last_before(t_pl, tA, window)
        if kX < 0:
            continue
        tX, pX = t_pl[kX], p_pl[kX]

        XA = pA - pX
        AB = pA - pB
        BC = pC - pB
        CD = pC - pD
        AD = pA - pD
        if XA <= 0 or AB <= 0 or BC <= 0 or CD <= 0:
            continue

        r_AB_XA = _ratio(XA, AB)
        r_BC_AB = _ratio(AB, BC)
        r_CD_BC = _ratio(BC, CD)
        r_AD_XA = _ratio(XA, AD)

        if (in_range(r_AB_XA, ab_xa, fib_tol) and
                in_band(r_BC_AB, bc_ab_lo, bc_ab_hi) and
                in_band(r_CD_BC, cd_bc_lo, cd_bc_hi) and
                in_range(r_AD_XA, ad_xa, fib_tol)):
            t_sig = min(tD, N - 1)
            if result[t_sig] == 0:
                if mode == 'forming':
                    result[t_sig] = 1
                elif c[t_sig] > pD:
                    result[t_sig] = 1
    return result


def _xabcd_bear(h, l, c, t_ph, p_ph, t_pl, p_pl, window, N, mode,
                ab_xa, bc_ab_lo, bc_ab_hi, cd_bc_lo, cd_bc_hi, ad_xa,
                fib_tol):
    """Generic bearish XABCD (X=H, A=L, B=H, C=L, D=H entry).

    Mirror of the bullish case: starts at peak X, drops to valley A, rallies
    to high B, dips to low C, then rises to the D high (the potential
    reversal zone) where a downward reversal is expected.
    """
    result = np.zeros(N, dtype=np.int8)
    for iD in range(len(t_ph)):
        tD, pD = t_ph[iD], p_ph[iD]
        if tD < window:
            continue
        kC = _last_before(t_pl, tD, window)
        if kC < 0:
            continue
        tC, pC = t_pl[kC], p_pl[kC]
        kB = _last_before(t_ph, tC, window)
        if kB < 0:
            continue
        tB, pB = t_ph[kB], p_ph[kB]
        kA = _last_before(t_pl, tB, window)
        if kA < 0:
            continue
        tA, pA = t_pl[kA], p_pl[kA]
        kX = _last_before(t_ph, tA, window)
        if kX < 0:
            continue
        tX, pX = t_ph[kX], p_ph[kX]

        XA = pX - pA
        AB = pB - pA
        BC = pB - pC
        CD = pD - pC
        AD = pD - pA
        if XA <= 0 or AB <= 0 or BC <= 0 or CD <= 0:
            continue

        r_AB_XA = _ratio(XA, AB)
        r_BC_AB = _ratio(AB, BC)
        r_CD_BC = _ratio(BC, CD)
        r_AD_XA = _ratio(XA, AD)

        if (in_range(r_AB_XA, ab_xa, fib_tol) and
                in_band(r_BC_AB, bc_ab_lo, bc_ab_hi) and
                in_band(r_CD_BC, cd_bc_lo, cd_bc_hi) and
                in_range(r_AD_XA, ad_xa, fib_tol)):
            t_sig = min(tD, N - 1)
            if result[t_sig] == 0:
                if mode == 'forming':
                    result[t_sig] = -1
                elif c[t_sig] < pD:
                    result[t_sig] = -1
    return result


# ---------------------------------------------------------------------------
# Gartley
# ---------------------------------------------------------------------------

def gartley_bull(o, h, l, c,
                  mode: str = 'confirmed',
                  window: int = 100,
                  pivot_n: int = 5,
                  pivot_pct: float | None = None,
                  fib_tol: float = 0.05) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Gartley 222: AB/XA≈0.618, BC/AB∈[0.382,0.886],
    CD/BC∈[1.272,1.618], AD/XA≈0.786.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    t_ph, p_ph, t_pl, p_pl = _piv(h, l, pivot_n, pivot_pct)
    return _xabcd_bull(h, l, c, t_ph, p_ph, t_pl, p_pl, window, N, mode,
                       0.618, 0.382, 0.886, 1.272, 1.618, 0.786, fib_tol)


def gartley_bear(o, h, l, c,
                  mode: str = 'confirmed',
                  window: int = 100,
                  pivot_n: int = 5,
                  pivot_pct: float | None = None,
                  fib_tol: float = 0.05) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Bearish Gartley: same ratios, inverted structure.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    t_ph, p_ph, t_pl, p_pl = _piv(h, l, pivot_n, pivot_pct)
    return _xabcd_bear(h, l, c, t_ph, p_ph, t_pl, p_pl, window, N, mode,
                       0.618, 0.382, 0.886, 1.272, 1.618, 0.786, fib_tol)


# ---------------------------------------------------------------------------
# Bat
# ---------------------------------------------------------------------------

def bat_bull(o, h, l, c,
              mode: str = 'confirmed',
              window: int = 100,
              pivot_n: int = 5,
              pivot_pct: float | None = None,
              fib_tol: float = 0.05) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Bat: AB/XA∈[0.382,0.50], BC/AB∈[0.382,0.886],
    CD/BC∈[1.618,2.618], AD/XA≈0.886.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    t_ph, p_ph, t_pl, p_pl = _piv(h, l, pivot_n, pivot_pct)
    # Use midpoint of AB range 0.382–0.50 ≈ 0.441
    return _xabcd_bull(h, l, c, t_ph, p_ph, t_pl, p_pl, window, N, mode,
                       0.441, 0.382, 0.886, 1.618, 2.618, 0.886, fib_tol)


def bat_bear(o, h, l, c,
              mode: str = 'confirmed',
              window: int = 100,
              pivot_n: int = 5,
              pivot_pct: float | None = None,
              fib_tol: float = 0.05) -> np.ndarray:
    """**BEARISH** (-1 / 0). Bearish bat."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    t_ph, p_ph, t_pl, p_pl = _piv(h, l, pivot_n, pivot_pct)
    return _xabcd_bear(h, l, c, t_ph, p_ph, t_pl, p_pl, window, N, mode,
                       0.441, 0.382, 0.886, 1.618, 2.618, 0.886, fib_tol)


# ---------------------------------------------------------------------------
# Butterfly
# ---------------------------------------------------------------------------

def butterfly_bull(o, h, l, c,
                    mode: str = 'confirmed',
                    window: int = 100,
                    pivot_n: int = 5,
                    pivot_pct: float | None = None,
                    fib_tol: float = 0.06) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Butterfly: AB/XA≈0.786, BC/AB∈[0.382,0.886],
    CD/BC∈[1.618,2.240], AD/XA∈[1.272,1.618].
    D extends beyond X.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    t_ph, p_ph, t_pl, p_pl = _piv(h, l, pivot_n, pivot_pct)
    return _xabcd_bull(h, l, c, t_ph, p_ph, t_pl, p_pl, window, N, mode,
                       0.786, 0.382, 0.886, 1.618, 2.240, 1.272, fib_tol)


def butterfly_bear(o, h, l, c,
                    mode: str = 'confirmed',
                    window: int = 100,
                    pivot_n: int = 5,
                    pivot_pct: float | None = None,
                    fib_tol: float = 0.06) -> np.ndarray:
    """**BEARISH** (-1 / 0). Bearish butterfly."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    t_ph, p_ph, t_pl, p_pl = _piv(h, l, pivot_n, pivot_pct)
    return _xabcd_bear(h, l, c, t_ph, p_ph, t_pl, p_pl, window, N, mode,
                       0.786, 0.382, 0.886, 1.618, 2.240, 1.272, fib_tol)


# ---------------------------------------------------------------------------
# Crab
# ---------------------------------------------------------------------------

def crab_bull(o, h, l, c,
               mode: str = 'confirmed',
               window: int = 100,
               pivot_n: int = 5,
               pivot_pct: float | None = None,
               fib_tol: float = 0.06) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Crab: AB/XA∈[0.382,0.618], BC/AB∈[0.382,0.886],
    CD/BC∈[2.240,3.618], AD/XA≈1.618.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    t_ph, p_ph, t_pl, p_pl = _piv(h, l, pivot_n, pivot_pct)
    return _xabcd_bull(h, l, c, t_ph, p_ph, t_pl, p_pl, window, N, mode,
                       0.500, 0.382, 0.886, 2.240, 3.618, 1.618, fib_tol)


def crab_bear(o, h, l, c,
               mode: str = 'confirmed',
               window: int = 100,
               pivot_n: int = 5,
               pivot_pct: float | None = None,
               fib_tol: float = 0.06) -> np.ndarray:
    """**BEARISH** (-1 / 0). Bearish crab."""
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    t_ph, p_ph, t_pl, p_pl = _piv(h, l, pivot_n, pivot_pct)
    return _xabcd_bear(h, l, c, t_ph, p_ph, t_pl, p_pl, window, N, mode,
                       0.500, 0.382, 0.886, 2.240, 3.618, 1.618, fib_tol)


# ---------------------------------------------------------------------------
# Wolfe Wave
# ---------------------------------------------------------------------------

def wolfe_wave_bull(o, h, l, c,
                     mode: str = 'confirmed',
                     window: int = 60,
                     pivot_n: int = 4,
                     pivot_pct: float | None = None) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Bullish Wolfe Wave: 5-pivot pattern where points 1,3,5 are lows and
    2,4 are highs.  Point 5 undershoots the 1-3 trendline; entry at point
    5 when price crosses above it.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    t_ph, p_ph, t_pl, p_pl = _piv(h, l, pivot_n, pivot_pct)
    result = np.zeros(N, dtype=np.int8)

    for i5 in range(len(t_pl)):
        t5, p5 = t_pl[i5], p_pl[i5]
        if t5 < window:
            continue
        # Find points in time order: 1(L), 2(H), 3(L), 4(H), 5(L)
        k4 = _last_before(t_ph, t5, window)
        if k4 < 0:
            continue
        t4, p4 = t_ph[k4], p_ph[k4]
        k3 = _last_before(t_pl, t4, window)
        if k3 < 0:
            continue
        t3, p3 = t_pl[k3], p_pl[k3]
        k2 = _last_before(t_ph, t3, window)
        if k2 < 0:
            continue
        t2, p2 = t_ph[k2], p_ph[k2]
        k1 = _last_before(t_pl, t2, window)
        if k1 < 0:
            continue
        t1, p1 = t_pl[k1], p_pl[k1]

        # Point 3 must be above point 1
        if p3 <= p1:
            continue
        # Point 4 must be below point 2 (inside channel)
        if p4 >= p2:
            continue
        # Point 5 must be below the 1-3 trendline
        from ._core import fit_line_r2 as fit_line, line_val
        s13, i13, _ = fit_line(np.array([t1, t3], float),
                                np.array([p1, p3], float))
        level_at_5 = line_val(s13, i13, t5)
        if p5 >= level_at_5:
            continue
        # Signal: price crosses back above the 1-3 line
        t_sig = min(t5, N - 1)
        if result[t_sig] == 0:
            if mode == 'forming':
                result[t_sig] = 1
            elif c[t_sig] > level_at_5:
                result[t_sig] = 1
    return result


def wolfe_wave_bear(o, h, l, c,
                     mode: str = 'confirmed',
                     window: int = 60,
                     pivot_n: int = 4,
                     pivot_pct: float | None = None) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Bearish Wolfe Wave: 1,3,5 are highs and 2,4 are lows.  Point 5
    overshoots the 1-3 trendline; entry when price falls back below it.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    t_ph, p_ph, t_pl, p_pl = _piv(h, l, pivot_n, pivot_pct)
    result = np.zeros(N, dtype=np.int8)

    for i5 in range(len(t_ph)):
        t5, p5 = t_ph[i5], p_ph[i5]
        if t5 < window:
            continue
        k4 = _last_before(t_pl, t5, window)
        if k4 < 0:
            continue
        t4, p4 = t_pl[k4], p_pl[k4]
        k3 = _last_before(t_ph, t4, window)
        if k3 < 0:
            continue
        t3, p3 = t_ph[k3], p_ph[k3]
        k2 = _last_before(t_pl, t3, window)
        if k2 < 0:
            continue
        t2, p2 = t_pl[k2], p_pl[k2]
        k1 = _last_before(t_ph, t2, window)
        if k1 < 0:
            continue
        t1, p1 = t_ph[k1], p_ph[k1]

        if p3 >= p1:
            continue
        if p4 <= p2:
            continue
        from ._core import fit_line_r2 as fit_line, line_val
        s13, i13, _ = fit_line(np.array([t1, t3], float),
                                np.array([p1, p3], float))
        level_at_5 = line_val(s13, i13, t5)
        if p5 <= level_at_5:
            continue
        t_sig = min(t5, N - 1)
        if result[t_sig] == 0:
            if mode == 'forming':
                result[t_sig] = -1
            elif c[t_sig] < level_at_5:
                result[t_sig] = -1
    return result
