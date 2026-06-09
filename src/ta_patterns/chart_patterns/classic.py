"""
ta_patterns.chart_patterns.classic
====================================
Large geometric chart patterns detected via pivot points and trendline fitting.

All functions
-------------
- Accept (o, h, l, c) numpy/pandas arrays.
- Return **int8**: +1 bullish, -1 bearish, 0 no signal.
- Support ``mode='forming'`` (last pivot confirmed) or
  ``mode='confirmed'`` (price breaks key level; default).
- Accept ``window``, ``pivot_n``, ``pivot_pct`` to tune sensitivity.

Elliott Wave patterns are intentionally NOT implemented.  They require
subjective multi-level wave labelling that does not reduce to a
deterministic OHLC rule.  This is a deliberate design decision, not
an omission.
"""
from __future__ import annotations
import numpy as np
from ._core import (
    _a, _EPS, pivot_highs, pivot_lows, pivot_info,
    fit_line_r2 as fit_line, line_val, lines_converging, apex_bar,
    roll_max, roll_min, atr, uptrend, downtrend,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ph(h, n, pct): return pivot_highs(h, n, pct)
def _pl(l, n, pct): return pivot_lows(l, n, pct)


def _scan(h, l, c, ph, pl, n):
    """Extract confirmation times and prices for highs and lows."""
    t_ph, p_ph = pivot_info(h, ph, n)
    t_pl, p_pl = pivot_info(l, pl, n)
    return t_ph, p_ph, t_pl, p_pl


def _first_break_above(c, from_bar, level, limit):
    """First bar >= from_bar where c > level; -1 if not found."""
    for t in range(from_bar, min(from_bar + limit, len(c))):
        if c[t] > level:
            return t
    return -1


def _first_break_below(c, from_bar, level, limit):
    for t in range(from_bar, min(from_bar + limit, len(c))):
        if c[t] < level:
            return t
    return -1


# ---------------------------------------------------------------------------
# Head and shoulders  (top: bearish; bottom: bullish)
# ---------------------------------------------------------------------------

def hs_top(o, h, l, c,
           mode: str = 'confirmed',
           window: int = 150,
           pivot_n: int = 5,
           pivot_pct: float | None = None,
           shoulder_tol: float = 0.05,
           min_separation: int = 8) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Classic head-and-shoulders top: left shoulder (LS), head (H, highest),
    right shoulder (RS ≈ LS height), neckline through the two troughs.
    Confirmed when price closes below the neckline.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i3 in range(len(t_ph)):            # RS confirmation
        tRS, pRS = t_ph[i3], p_ph[i3]
        # Need H and LS to the left
        left_ph = [(t_ph[j], p_ph[j]) for j in range(i3)
                   if t_ph[j] >= tRS - window]
        if len(left_ph) < 2:
            continue
        # Head: highest peak left of RS
        h_idx = max(range(len(left_ph)), key=lambda k: left_ph[k][1])
        tH, pH = left_ph[h_idx]
        if pH <= pRS:
            continue                        # head must be highest
        # LS: the peak left of H
        ls_cands = [(t, p) for t, p in left_ph[:h_idx]
                    if tH - t >= min_separation]
        if not ls_cands:
            continue
        tLS, pLS = ls_cands[-1]             # most recent valid LS
        if tH - tLS < min_separation or tRS - tH < min_separation:
            continue
        # Shoulder symmetry
        if abs(pLS - pRS) / (max(pLS, pRS) + _EPS) > shoulder_tol:
            continue
        # Neckline through lows between LS-H and H-RS
        lv1 = [(t_pl[j], p_pl[j]) for j in range(len(t_pl))
               if tLS < t_pl[j] < tH]
        lv2 = [(t_pl[j], p_pl[j]) for j in range(len(t_pl))
               if tH < t_pl[j] < tRS]
        if not lv1 or not lv2:
            continue
        tLV1, pLV1 = max(lv1, key=lambda x: x[0])
        tLV2, pLV2 = max(lv2, key=lambda x: x[0])
        nk_slope, nk_int, _ = fit_line(
            np.array([tLV1, tLV2], float), np.array([pLV1, pLV2], float))
        if mode == 'forming':
            result[tRS] = -1
        else:
            for t in range(tRS + 1, min(tRS + window, N)):
                nkl = line_val(nk_slope, nk_int, t)
                if c[t] < nkl:
                    if result[t] == 0:
                        result[t] = -1
                    break
    return result


def hs_bottom(o, h, l, c,
              mode: str = 'confirmed',
              window: int = 150,
              pivot_n: int = 5,
              pivot_pct: float | None = None,
              shoulder_tol: float = 0.05,
              min_separation: int = 8) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Inverse head-and-shoulders: two shoulders above the head (lowest trough).
    Confirmed when price closes above the neckline.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(l)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i3 in range(len(t_pl)):
        tRS, pRS = t_pl[i3], p_pl[i3]
        left_pl = [(t_pl[j], p_pl[j]) for j in range(i3)
                   if t_pl[j] >= tRS - window]
        if len(left_pl) < 2:
            continue
        h_idx = min(range(len(left_pl)), key=lambda k: left_pl[k][1])
        tH, pH = left_pl[h_idx]
        if pH >= pRS:
            continue
        ls_cands = [(t, p) for t, p in left_pl[:h_idx]
                    if tH - t >= min_separation]
        if not ls_cands:
            continue
        tLS, pLS = ls_cands[-1]
        if tH - tLS < min_separation or tRS - tH < min_separation:
            continue
        if abs(pLS - pRS) / (max(abs(pLS), abs(pRS)) + _EPS) > shoulder_tol:
            continue
        pk1 = [(t_ph[j], p_ph[j]) for j in range(len(t_ph))
               if tLS < t_ph[j] < tH]
        pk2 = [(t_ph[j], p_ph[j]) for j in range(len(t_ph))
               if tH < t_ph[j] < tRS]
        if not pk1 or not pk2:
            continue
        tP1, pP1 = max(pk1, key=lambda x: x[0])
        tP2, pP2 = max(pk2, key=lambda x: x[0])
        nk_slope, nk_int, _ = fit_line(
            np.array([tP1, tP2], float), np.array([pP1, pP2], float))
        if mode == 'forming':
            result[tRS] = 1
        else:
            for t in range(tRS + 1, min(tRS + window, N)):
                nkl = line_val(nk_slope, nk_int, t)
                if c[t] > nkl:
                    if result[t] == 0:
                        result[t] = 1
                    break
    return result


def complex_hs_top(o, h, l, c, **kw) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Complex H&S top: multiple shoulders on one or both sides.
    Detected as two or more H&S patterns sharing the same head.
    """
    kw.setdefault('shoulder_tol', 0.08)
    kw.setdefault('window', 200)
    return hs_top(o, h, l, c, **kw)


def complex_hs_bottom(o, h, l, c, **kw) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Complex inverse H&S: multiple shoulders, looser symmetry requirement.
    """
    kw.setdefault('shoulder_tol', 0.08)
    kw.setdefault('window', 200)
    return hs_bottom(o, h, l, c, **kw)


# ---------------------------------------------------------------------------
# Triangles
# ---------------------------------------------------------------------------

def _triangle_lines(t_ph, p_ph, t_pl, p_pl, i_end, window, min_pivots=2):
    """
    Fit upper trendline through recent pivot highs and lower through lows.
    Returns (s_hi, i_hi, s_lo, i_lo, ok) where ok = enough pivots found.
    """
    ph_w = [(t_ph[j], p_ph[j]) for j in range(len(t_ph))
            if i_end - window <= t_ph[j] <= i_end]
    pl_w = [(t_pl[j], p_pl[j]) for j in range(len(t_pl))
            if i_end - window <= t_pl[j] <= i_end]
    if len(ph_w) < min_pivots or len(pl_w) < min_pivots:
        return 0, 0, 0, 0, False
    x_h = np.array([p[0] for p in ph_w], float)
    y_h = np.array([p[1] for p in ph_w], float)
    x_l = np.array([p[0] for p in pl_w], float)
    y_l = np.array([p[1] for p in pl_w], float)
    s_hi, i_hi, r2_hi = fit_line(x_h, y_h)
    s_lo, i_lo, r2_lo = fit_line(x_l, y_l)
    if r2_hi < 0.5 or r2_lo < 0.5:
        return 0, 0, 0, 0, False
    return s_hi, i_hi, s_lo, i_lo, True


def ascending_triangle(o, h, l, c,
                        mode: str = 'confirmed',
                        window: int = 100,
                        pivot_n: int = 5,
                        pivot_pct: float | None = None,
                        flat_tol: float = 0.02) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Flat upper resistance + rising lower support.
    Confirmed on close above resistance.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if abs(s_hi) > flat_tol:            # upper line not flat
            continue
        if s_lo <= 0:                        # lower line not rising
            continue
        if not lines_converging(s_hi, s_lo, i_hi, i_lo, i - window):
            continue
        resistance = line_val(s_hi, i_hi, i)
        if mode == 'forming':
            result[i] = 1
        elif c[i] > resistance:
            result[i] = 1
    return result


def descending_triangle(o, h, l, c,
                         mode: str = 'confirmed',
                         window: int = 100,
                         pivot_n: int = 5,
                         pivot_pct: float | None = None,
                         flat_tol: float = 0.02) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Flat lower support + descending upper resistance.
    Confirmed on close below support.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if abs(s_lo) > flat_tol:
            continue
        if s_hi >= 0:
            continue
        if not lines_converging(s_hi, s_lo, i_hi, i_lo, i - window):
            continue
        support = line_val(s_lo, i_lo, i)
        if mode == 'forming':
            result[i] = -1
        elif c[i] < support:
            result[i] = -1
    return result


def symmetrical_triangle(o, h, l, c,
                          mode: str = 'confirmed',
                          window: int = 100,
                          pivot_n: int = 5,
                          pivot_pct: float | None = None) -> np.ndarray:
    """
    **BIDIRECTIONAL** (+1 bullish breakout / -1 bearish / 0 none).
    Both trendlines converging; direction determined by confirmed breakout.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if s_hi >= 0 or s_lo <= 0:          # need descending hi, ascending lo
            continue
        if not lines_converging(s_hi, s_lo, i_hi, i_lo, i - window):
            continue
        if mode == 'forming':
            result[i] = 1                    # non-directional forming signal
        else:
            res = line_val(s_hi, i_hi, i)
            sup = line_val(s_lo, i_lo, i)
            if c[i] > res:
                result[i] = 1
            elif c[i] < sup:
                result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Broadening formations
# ---------------------------------------------------------------------------

def broadening_top(o, h, l, c,
                   mode: str = 'confirmed',
                   window: int = 80,
                   pivot_n: int = 5,
                   pivot_pct: float | None = None) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Expanding price action: both trendlines diverging (upper rising, lower
    falling).  Bearish bias.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if s_hi <= 0 or s_lo >= 0:          # upper rising, lower falling
            continue
        sup = line_val(s_lo, i_lo, i)
        if mode == 'forming':
            result[i] = -1
        elif c[i] < sup:
            result[i] = -1
    return result


def broadening_bottom(o, h, l, c,
                       mode: str = 'confirmed',
                       window: int = 80,
                       pivot_n: int = 5,
                       pivot_pct: float | None = None) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Expanding price action at a low — broader range with a bullish bias.
    Confirmed when price closes above the upper trendline.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if s_hi <= 0 or s_lo >= 0:
            continue
        res = line_val(s_hi, i_hi, i)
        if mode == 'forming':
            result[i] = 1
        elif c[i] > res:
            result[i] = 1
    return result


def broadening_wedge_asc(o, h, l, c,
                          mode: str = 'confirmed',
                          window: int = 80,
                          pivot_n: int = 5,
                          pivot_pct: float | None = None) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Both trendlines rising but diverging (lower rising faster than upper).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if s_hi <= 0 or s_lo <= 0:          # both must rise
            continue
        if s_lo <= s_hi:                    # lower must rise faster (diverging)
            continue
        sup = line_val(s_lo, i_lo, i)
        if mode == 'forming':
            result[i] = -1
        elif c[i] < sup:
            result[i] = -1
    return result


def broadening_wedge_desc(o, h, l, c,
                           mode: str = 'confirmed',
                           window: int = 80,
                           pivot_n: int = 5,
                           pivot_pct: float | None = None) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Both trendlines falling but diverging (upper falling faster).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if s_hi >= 0 or s_lo >= 0:
            continue
        if s_hi >= s_lo:                    # upper must fall faster
            continue
        res = line_val(s_hi, i_hi, i)
        if mode == 'forming':
            result[i] = 1
        elif c[i] > res:
            result[i] = 1
    return result


def right_angle_broadening_asc(o, h, l, c,
                                 mode: str = 'confirmed',
                                 window: int = 80,
                                 pivot_n: int = 5,
                                 pivot_pct: float | None = None,
                                 flat_tol: float = 0.015) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Flat upper resistance + falling lower support (right-angled, ascending
    bias from lows while highs are flat → bearish resolution).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if abs(s_hi) > flat_tol or s_lo >= 0:
            continue
        sup = line_val(s_lo, i_lo, i)
        if mode == 'forming':
            result[i] = -1
        elif c[i] < sup:
            result[i] = -1
    return result


def right_angle_broadening_desc(o, h, l, c,
                                  mode: str = 'confirmed',
                                  window: int = 80,
                                  pivot_n: int = 5,
                                  pivot_pct: float | None = None,
                                  flat_tol: float = 0.015) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Flat lower support + rising upper resistance.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if s_hi <= 0 or abs(s_lo) > flat_tol:
            continue
        res = line_val(s_hi, i_hi, i)
        if mode == 'forming':
            result[i] = 1
        elif c[i] > res:
            result[i] = 1
    return result


# ---------------------------------------------------------------------------
# Wedges
# ---------------------------------------------------------------------------

def rising_wedge(o, h, l, c,
                  mode: str = 'confirmed',
                  window: int = 80,
                  pivot_n: int = 5,
                  pivot_pct: float | None = None) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Both trendlines rising but converging (lower rises faster → support
    approaching resistance).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if s_hi <= 0 or s_lo <= 0:
            continue
        if not lines_converging(s_hi, s_lo, i_hi, i_lo, i - window):
            continue                         # s_lo > s_hi for convergence
        sup = line_val(s_lo, i_lo, i)
        if mode == 'forming':
            result[i] = -1
        elif c[i] < sup:
            result[i] = -1
    return result


def falling_wedge(o, h, l, c,
                   mode: str = 'confirmed',
                   window: int = 80,
                   pivot_n: int = 5,
                   pivot_pct: float | None = None) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Both trendlines falling but converging (upper falls faster).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if s_hi >= 0 or s_lo >= 0:
            continue
        if not lines_converging(s_hi, s_lo, i_hi, i_lo, i - window):
            continue                         # s_hi < s_lo for convergence
        res = line_val(s_hi, i_hi, i)
        if mode == 'forming':
            result[i] = 1
        elif c[i] > res:
            result[i] = 1
    return result


# ---------------------------------------------------------------------------
# Rectangle top / bottom
# ---------------------------------------------------------------------------

def rectangle_top(o, h, l, c,
                   mode: str = 'confirmed',
                   window: int = 80,
                   pivot_n: int = 4,
                   pivot_pct: float | None = None,
                   flat_tol: float = 0.025) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Price oscillates between flat support and flat resistance; breakout
    expected downward (bearish context = downtrend before the rectangle).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    dt = downtrend(c, 20)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if abs(s_hi) > flat_tol or abs(s_lo) > flat_tol:
            continue
        sup = line_val(s_lo, i_lo, i)
        if mode == 'forming' and dt[i]:
            result[i] = -1
        elif mode == 'confirmed' and c[i] < sup:
            result[i] = -1
    return result


def rectangle_bottom(o, h, l, c,
                      mode: str = 'confirmed',
                      window: int = 80,
                      pivot_n: int = 4,
                      pivot_pct: float | None = None,
                      flat_tol: float = 0.025) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Flat support and resistance; upside breakout after an uptrend context.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    ut = uptrend(c, 20)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if abs(s_hi) > flat_tol or abs(s_lo) > flat_tol:
            continue
        res = line_val(s_hi, i_hi, i)
        if mode == 'forming' and ut[i]:
            result[i] = 1
        elif mode == 'confirmed' and c[i] > res:
            result[i] = 1
    return result


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

def channel_asc(o, h, l, c,
                 mode: str = 'confirmed',
                 window: int = 80,
                 pivot_n: int = 4,
                 pivot_pct: float | None = None,
                 parallel_tol: float = 0.25) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Price trends upward in a parallel channel; bearish when it breaks below
    the lower channel line (support failure).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if s_hi <= 0 or s_lo <= 0:
            continue
        if abs(s_hi - s_lo) / (abs(s_lo) + _EPS) > parallel_tol:
            continue                         # slopes not parallel enough
        sup = line_val(s_lo, i_lo, i)
        if mode == 'forming':
            result[i] = -1
        elif c[i] < sup:
            result[i] = -1
    return result


def channel_desc(o, h, l, c,
                  mode: str = 'confirmed',
                  window: int = 80,
                  pivot_n: int = 4,
                  pivot_pct: float | None = None,
                  parallel_tol: float = 0.25) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Price trends downward in a parallel channel; bullish breakout above
    the upper channel line.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        if s_hi >= 0 or s_lo >= 0:
            continue
        if abs(s_hi - s_lo) / (abs(s_lo) + _EPS) > parallel_tol:
            continue
        res = line_val(s_hi, i_hi, i)
        if mode == 'forming':
            result[i] = 1
        elif c[i] > res:
            result[i] = 1
    return result


# ---------------------------------------------------------------------------
# Flags and pennants
# ---------------------------------------------------------------------------

def _flagpole(c, i, min_move: float, pole_bars: int):
    """Return (pole_start, direction) if a flagpole exists before bar i."""
    if i < pole_bars:
        return -1, 0
    pole_end   = i - 1
    pole_start = pole_end - pole_bars
    move = (c[pole_end] - c[pole_start]) / (c[pole_start] + _EPS)
    if move >= min_move:
        return pole_start, 1
    if move <= -min_move:
        return pole_start, -1
    return -1, 0


def flag_bull(o, h, l, c,
               mode: str = 'confirmed',
               window: int = 30,
               pole_bars: int = 10,
               min_pole: float = 0.05,
               pivot_n: int = 3,
               pivot_pct: float | None = None,
               max_retrace: float = 0.50) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Sharp upward move (flagpole) followed by a shallow downward parallel
    channel (flag).  Confirmed on breakout above the flag's upper trendline.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(pole_bars + window, N):
        pole_start, direction = _flagpole(c, i - window, min_pole, pole_bars)
        if direction != 1:
            continue
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window, min_pivots=2)
        if not ok:
            continue
        if s_hi >= 0 or s_lo >= 0:          # flag should drift down (both neg)
            continue
        # Retrace should be limited
        flag_drop = c[i] - c[i - window]
        pole_gain = c[i - window] - c[pole_start]
        if abs(flag_drop) / (pole_gain + _EPS) > max_retrace:
            continue
        res = line_val(s_hi, i_hi, i)
        if mode == 'forming':
            result[i] = 1
        elif c[i] > res:
            result[i] = 1
    return result


def flag_bear(o, h, l, c,
               mode: str = 'confirmed',
               window: int = 30,
               pole_bars: int = 10,
               min_pole: float = 0.05,
               pivot_n: int = 3,
               pivot_pct: float | None = None,
               max_retrace: float = 0.50) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Sharp downward move followed by a shallow upward channel.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(pole_bars + window, N):
        pole_start, direction = _flagpole(c, i - window, min_pole, pole_bars)
        if direction != -1:
            continue
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window, min_pivots=2)
        if not ok:
            continue
        if s_hi <= 0 or s_lo <= 0:
            continue
        flag_rise = c[i] - c[i - window]
        pole_drop = c[pole_start] - c[i - window]
        if abs(flag_rise) / (pole_drop + _EPS) > max_retrace:
            continue
        sup = line_val(s_lo, i_lo, i)
        if mode == 'forming':
            result[i] = -1
        elif c[i] < sup:
            result[i] = -1
    return result


def flag_high_tight(o, h, l, c,
                     mode: str = 'confirmed',
                     pole_bars: int = 10,
                     min_pole: float = 0.40,
                     window: int = 20,
                     max_retrace: float = 0.20) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Very sharp rise (≥ 40 % in ~10 bars) then a tight consolidation
    (< 20 % retracement).  One of Bulkowski's best-performing patterns.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    result = np.zeros(N, dtype=np.int8)

    for i in range(pole_bars + window, N):
        pole_end   = i - window
        pole_start = pole_end - pole_bars
        if pole_start < 0:
            continue
        pole_gain = (c[pole_end] - c[pole_start]) / (c[pole_start] + _EPS)
        if pole_gain < min_pole:
            continue
        # Flag: tight range over the consolidation *before* the current bar
        flag_hi = h[pole_end:i].max()
        flag_lo = l[pole_end:i].min()
        retrace = (flag_hi - flag_lo) / (c[pole_end] + _EPS)
        if retrace > max_retrace:
            continue
        if mode == 'forming':
            result[i] = 1
        elif c[i] > flag_hi:
            result[i] = 1
    return result


def pennant_bull(o, h, l, c,
                  mode: str = 'confirmed',
                  pole_bars: int = 10,
                  min_pole: float = 0.05,
                  window: int = 15,
                  pivot_n: int = 3,
                  pivot_pct: float | None = None) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Like a bull flag but the consolidation forms a symmetrical triangle
    (converging trendlines rather than parallel).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(pole_bars + window, N):
        pole_start, direction = _flagpole(c, i - window, min_pole, pole_bars)
        if direction != 1:
            continue
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window, min_pivots=2)
        if not ok:
            continue
        if s_hi >= 0 or s_lo <= 0:          # converging: hi descending, lo ascending
            continue
        res = line_val(s_hi, i_hi, i)
        if mode == 'forming':
            result[i] = 1
        elif c[i] > res:
            result[i] = 1
    return result


def pennant_bear(o, h, l, c,
                  mode: str = 'confirmed',
                  pole_bars: int = 10,
                  min_pole: float = 0.05,
                  window: int = 15,
                  pivot_n: int = 3,
                  pivot_pct: float | None = None) -> np.ndarray:
    """
    **BEARISH** (-1 / 0). Bear pennant: sharp drop then converging triangle.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(pole_bars + window, N):
        pole_start, direction = _flagpole(c, i - window, min_pole, pole_bars)
        if direction != -1:
            continue
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window, min_pivots=2)
        if not ok:
            continue
        if s_hi >= 0 or s_lo <= 0:
            continue
        sup = line_val(s_lo, i_lo, i)
        if mode == 'forming':
            result[i] = -1
        elif c[i] < sup:
            result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Cup with handle
# ---------------------------------------------------------------------------

def cup_with_handle(o, h, l, c,
                     mode: str = 'confirmed',
                     cup_window: int = 65,
                     handle_window: int = 15,
                     pivot_n: int = 5,
                     max_handle_retrace: float = 0.50) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    U-shaped base (cup) followed by a brief pullback (handle).
    Confirmed when price closes above the cup rim.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, None)
    pl = _pl(l, pivot_n, None)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(cup_window + handle_window, N):
        cup_start = i - cup_window - handle_window
        handle_start = i - handle_window

        # Left rim: highest point in cup region
        cup_ph = [(t_ph[j], p_ph[j]) for j in range(len(t_ph))
                  if cup_start <= t_ph[j] < handle_start]
        if not cup_ph:
            continue
        left_rim_price = max(p for _, p in cup_ph)

        # Cup bottom: lowest point in cup
        cup_pl = [(t_pl[j], p_pl[j]) for j in range(len(t_pl))
                  if cup_start <= t_pl[j] < handle_start]
        if not cup_pl:
            continue
        cup_bottom = min(p for _, p in cup_pl)
        cup_depth  = left_rim_price - cup_bottom
        if cup_depth / (left_rim_price + _EPS) < 0.05:
            continue                         # cup too shallow

        # Right rim: price should recover near left rim
        right_area = c[handle_start:i]
        if len(right_area) == 0 or right_area.max() < left_rim_price * 0.95:
            continue

        # Handle: limited pullback
        handle_drop = (right_area.max() - c[i]) / (cup_depth + _EPS)
        if handle_drop > max_handle_retrace:
            continue

        if mode == 'forming':
            result[i] = 1
        elif c[i] > left_rim_price:
            result[i] = 1
    return result


def inverted_cup_with_handle(o, h, l, c,
                               mode: str = 'confirmed',
                               cup_window: int = 65,
                               handle_window: int = 15,
                               pivot_n: int = 5,
                               max_handle_retrace: float = 0.50) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Inverted U-shaped top followed by a brief bounce (handle).
    Confirmed when price breaks below the cup base level.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, None)
    pl = _pl(l, pivot_n, None)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(cup_window + handle_window, N):
        cup_start    = i - cup_window - handle_window
        handle_start = i - handle_window

        cup_pl = [(t_pl[j], p_pl[j]) for j in range(len(t_pl))
                  if cup_start <= t_pl[j] < handle_start]
        if not cup_pl:
            continue
        left_rim_price = min(p for _, p in cup_pl)

        cup_ph = [(t_ph[j], p_ph[j]) for j in range(len(t_ph))
                  if cup_start <= t_ph[j] < handle_start]
        if not cup_ph:
            continue
        cup_top   = max(p for _, p in cup_ph)
        cup_depth = cup_top - left_rim_price
        if cup_depth / (cup_top + _EPS) < 0.05:
            continue

        right_area = c[handle_start:i]
        if len(right_area) == 0 or right_area.min() > left_rim_price * 1.05:
            continue

        handle_rise = (c[i] - right_area.min()) / (cup_depth + _EPS)
        if handle_rise > max_handle_retrace:
            continue

        if mode == 'forming':
            result[i] = -1
        elif c[i] < left_rim_price:
            result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Rounding patterns
# ---------------------------------------------------------------------------

def rounding_bottom(o, h, l, c,
                     mode: str = 'confirmed',
                     window: int = 40,
                     min_depth: float = 0.05) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Gradual U-shaped price curve over *window* bars.  Detected by checking
    that the middle third of the window has lower closes than the outer thirds.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    third = window // 3

    for i in range(window, N):
        w = c[i - window:i]
        left_mean  = w[:third].mean()
        mid_mean   = w[third:2 * third].mean()
        right_mean = w[2 * third:].mean()
        depth = (min(left_mean, right_mean) - mid_mean) / (left_mean + _EPS)
        if mid_mean < min(left_mean, right_mean) and depth >= min_depth:
            rim = max(left_mean, right_mean)
            if mode == 'forming':
                result[i] = 1
            elif c[i] > rim:
                result[i] = 1
    return result


def rounding_top(o, h, l, c,
                  mode: str = 'confirmed',
                  window: int = 40,
                  min_depth: float = 0.05) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Gradual inverted-U price curve.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    third = window // 3

    for i in range(window, N):
        w = c[i - window:i]
        left_mean  = w[:third].mean()
        mid_mean   = w[third:2 * third].mean()
        right_mean = w[2 * third:].mean()
        height = (mid_mean - max(left_mean, right_mean)) / (mid_mean + _EPS)
        if mid_mean > max(left_mean, right_mean) and height >= min_depth:
            neckline = min(left_mean, right_mean)
            if mode == 'forming':
                result[i] = -1
            elif c[i] < neckline:
                result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Diamond patterns
# ---------------------------------------------------------------------------

def diamond_top(o, h, l, c,
                 mode: str = 'confirmed',
                 window: int = 60,
                 pivot_n: int = 4,
                 pivot_pct: float | None = None) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Broadening formation followed by a symmetrical triangle — creates a
    diamond shape.  First half expands, second half contracts.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)
    half = window // 2

    for i in range(window, N):
        # First half: broadening (diverging lines)
        s_hi1, i_hi1, s_lo1, i_lo1, ok1 = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i - half, half)
        # Second half: contracting (converging lines)
        s_hi2, i_hi2, s_lo2, i_lo2, ok2 = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, half)
        if not (ok1 and ok2):
            continue
        broadening = (s_hi1 > 0 and s_lo1 < 0)   # first half expanding
        contracting = lines_converging(s_hi2, s_lo2, i_hi2, i_lo2, i - half)
        if not (broadening and contracting):
            continue
        sup = line_val(s_lo2, i_lo2, i)
        if mode == 'forming':
            result[i] = -1
        elif c[i] < sup:
            result[i] = -1
    return result


def diamond_bottom(o, h, l, c,
                    mode: str = 'confirmed',
                    window: int = 60,
                    pivot_n: int = 4,
                    pivot_pct: float | None = None) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Diamond bottom: broadening then contracting at a price low.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)
    half = window // 2

    for i in range(window, N):
        s_hi1, i_hi1, s_lo1, i_lo1, ok1 = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i - half, half)
        s_hi2, i_hi2, s_lo2, i_lo2, ok2 = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, half)
        if not (ok1 and ok2):
            continue
        broadening = (s_hi1 > 0 and s_lo1 < 0)
        contracting = lines_converging(s_hi2, s_lo2, i_hi2, i_lo2, i - half)
        if not (broadening and contracting):
            continue
        res = line_val(s_hi2, i_hi2, i)
        if mode == 'forming':
            result[i] = 1
        elif c[i] > res:
            result[i] = 1
    return result


# ---------------------------------------------------------------------------
# Bump-and-run reversal
# ---------------------------------------------------------------------------

def bump_and_run_top(o, h, l, c,
                      mode: str = 'confirmed',
                      lead_window: int = 30,
                      bump_window: int = 15,
                      bump_factor: float = 2.0,
                      pivot_n: int = 4,
                      pivot_pct: float | None = None) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Gradual lead-in trendline followed by a steep 'bump' (manic buying),
    then price collapses back through the lead-in line.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)
    total = lead_window + bump_window

    for i in range(total, N):
        # Lead-in trendline (older portion)
        lead_pl = [(t_pl[j], p_pl[j]) for j in range(len(t_pl))
                   if i - total <= t_pl[j] < i - bump_window]
        if len(lead_pl) < 2:
            continue
        x_l = np.array([p[0] for p in lead_pl], float)
        y_l = np.array([p[1] for p in lead_pl], float)
        s_lead, i_lead, r2 = fit_line(x_l, y_l)
        if r2 < 0.6 or s_lead <= 0:        # lead-in must rise
            continue
        # Bump: slope in bump window must be steeper
        bump_pl = [(t_pl[j], p_pl[j]) for j in range(len(t_pl))
                   if i - bump_window <= t_pl[j] <= i]
        if len(bump_pl) < 2:
            continue
        xb = np.array([p[0] for p in bump_pl], float)
        yb = np.array([p[1] for p in bump_pl], float)
        s_bump, _, _ = fit_line(xb, yb)
        if s_bump < bump_factor * s_lead:   # bump not steep enough
            continue
        lead_level = line_val(s_lead, i_lead, i)
        if mode == 'forming':
            result[i] = -1
        elif c[i] < lead_level:
            result[i] = -1
    return result


def bump_and_run_bottom(o, h, l, c,
                         mode: str = 'confirmed',
                         lead_window: int = 30,
                         bump_window: int = 15,
                         bump_factor: float = 2.0,
                         pivot_n: int = 4,
                         pivot_pct: float | None = None) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Downward lead-in, steep downward bump, then recovery through lead-in.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)
    total = lead_window + bump_window

    for i in range(total, N):
        lead_ph = [(t_ph[j], p_ph[j]) for j in range(len(t_ph))
                   if i - total <= t_ph[j] < i - bump_window]
        if len(lead_ph) < 2:
            continue
        x_h = np.array([p[0] for p in lead_ph], float)
        y_h = np.array([p[1] for p in lead_ph], float)
        s_lead, i_lead, r2 = fit_line(x_h, y_h)
        if r2 < 0.6 or s_lead >= 0:
            continue
        bump_ph = [(t_ph[j], p_ph[j]) for j in range(len(t_ph))
                   if i - bump_window <= t_ph[j] <= i]
        if len(bump_ph) < 2:
            continue
        xb = np.array([p[0] for p in bump_ph], float)
        yb = np.array([p[1] for p in bump_ph], float)
        s_bump, _, _ = fit_line(xb, yb)
        if s_bump > bump_factor * s_lead:
            continue
        lead_level = line_val(s_lead, i_lead, i)
        if mode == 'forming':
            result[i] = 1
        elif c[i] > lead_level:
            result[i] = 1
    return result


# ---------------------------------------------------------------------------
# Scallops
# ---------------------------------------------------------------------------

def _scallop_shape(prices, window):
    """Return True if prices form a J-shape (scallop): decline then recovery."""
    if len(prices) < window:
        return False
    third = window // 3
    left  = prices[:third].mean()
    mid   = prices[third:2*third].mean()
    right = prices[2*third:].mean()
    return mid < left and right > mid and right >= left * 0.95


def scallop_asc(o, h, l, c,
                 mode: str = 'confirmed',
                 window: int = 25) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Repeated J-shaped recovery (scallop) in a rising trend.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    ut = uptrend(c, window)
    for i in range(window, N):
        w = c[i - window:i]
        if _scallop_shape(w, window) and ut[i]:
            if mode == 'forming':
                result[i] = 1
            elif c[i] > w.max():
                result[i] = 1
    return result


def scallop_asc_inv(o, h, l, c,
                     mode: str = 'confirmed',
                     window: int = 25) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Inverted ascending scallop — ∩-shape in a rising trend signals a top.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    ut = uptrend(c, window)
    for i in range(window, N):
        w = c[i - window:i]
        if _scallop_shape(-w, window) and ut[i]:  # inverted
            if mode == 'forming':
                result[i] = -1
            elif c[i] < w.min():
                result[i] = -1
    return result


def scallop_desc(o, h, l, c,
                  mode: str = 'confirmed',
                  window: int = 25) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Inverted J-shape (decay) in a falling trend.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    dt = downtrend(c, window)
    for i in range(window, N):
        w = c[i - window:i]
        if _scallop_shape(-w, window) and dt[i]:
            if mode == 'forming':
                result[i] = -1
            elif c[i] < w.min():
                result[i] = -1
    return result


def scallop_desc_inv(o, h, l, c,
                      mode: str = 'confirmed',
                      window: int = 25) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    J-shaped recovery in a falling trend — potential bottom reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    dt = downtrend(c, window)
    for i in range(window, N):
        w = c[i - window:i]
        if _scallop_shape(w, window) and dt[i]:
            if mode == 'forming':
                result[i] = 1
            elif c[i] > w.max():
                result[i] = 1
    return result


# ---------------------------------------------------------------------------
# Island reversals
# ---------------------------------------------------------------------------

def island_top(o, h, l, c,
                mode: str = 'confirmed',
                max_island_bars: int = 10) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Price gaps up into an island, trades there, then gaps down below the
    entry gap — leaving an isolated price island.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    result = np.zeros(N, dtype=np.int8)
    for i in range(max_island_bars + 1, N):
        # Find gap-up entry into island
        for k in range(1, max_island_bars):
            entry = i - k
            if entry < 1:
                break
            if l[entry] > h[entry - 1]:      # gap up into island
                # Exit: gap down out of island  (current bar)
                if h[i] < l[i - 1] and h[i] < l[entry]:
                    if mode == 'forming' or c[i] < l[entry]:
                        result[i] = -1
                break
    return result


def island_bottom(o, h, l, c,
                   mode: str = 'confirmed',
                   max_island_bars: int = 10) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Price gaps down into island, then gaps up out — bullish reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    result = np.zeros(N, dtype=np.int8)
    for i in range(max_island_bars + 1, N):
        for k in range(1, max_island_bars):
            entry = i - k
            if entry < 1:
                break
            if h[entry] < l[entry - 1]:      # gap down into island
                if l[i] > h[i - 1] and l[i] > h[entry]:
                    if mode == 'forming' or c[i] > h[entry]:
                        result[i] = 1
                break
    return result


def long_island(o, h, l, c,
                 mode: str = 'confirmed',
                 min_island_bars: int = 3,
                 max_island_bars: int = 20) -> np.ndarray:
    """
    **BIDIRECTIONAL** (+1 or -1).
    Like island reversal but spanning several days.  Direction depends on
    whether price exited via a gap up (+1) or gap down (-1).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    result = np.zeros(N, dtype=np.int8)
    for i in range(max_island_bars + 1, N):
        for k in range(min_island_bars, max_island_bars):
            entry = i - k
            if entry < 1:
                break
            gap_up_in   = l[entry] > h[entry - 1]
            gap_down_in = h[entry] < l[entry - 1]
            if gap_up_in:
                # Island entered via gap up; exit via gap down = bearish
                if h[i] < l[i - 1]:
                    result[i] = -1
                break
            if gap_down_in:
                # Island entered via gap down; exit via gap up = bullish
                if l[i] > h[i - 1]:
                    result[i] = 1
                break
    return result


# ---------------------------------------------------------------------------
# Measured moves
# ---------------------------------------------------------------------------

def measured_move_up(o, h, l, c,
                      mode: str = 'confirmed',
                      window: int = 60,
                      pivot_n: int = 5,
                      pivot_pct: float | None = None,
                      leg_tol: float = 0.20) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Two upward legs of approximately equal length separated by a correction.
    Signal at the start of the second leg (or at completion).
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for j in range(2, len(t_ph)):
        tA, pA = t_pl[-1] if t_pl.size else (0, 0), 0
        # Find three-point structure: low1, high1, low2, high2
        for jh in range(1, len(t_ph)):
            tH1, pH1 = t_ph[jh - 1], p_ph[jh - 1]
            tH2, pH2 = t_ph[jh],     p_ph[jh]
            # Low between them
            lows_between = [(t_pl[k], p_pl[k]) for k in range(len(t_pl))
                            if tH1 < t_pl[k] < tH2]
            if not lows_between:
                continue
            tL2, pL2 = min(lows_between, key=lambda x: x[1])
            # Low before H1
            lows_before = [(t_pl[k], p_pl[k]) for k in range(len(t_pl))
                           if t_pl[k] < tH1]
            if not lows_before:
                continue
            tL1, pL1 = max(lows_before, key=lambda x: x[0])
            leg1 = pH1 - pL1
            leg2 = pH2 - pL2
            if leg1 <= 0 or leg2 <= 0:
                continue
            if abs(leg2 - leg1) / (leg1 + _EPS) < leg_tol:
                t_signal = min(tH2, N - 1)
                if result[t_signal] == 0:
                    if mode == 'forming':
                        result[t_signal] = 1
                    elif c[t_signal] > pH1:
                        result[t_signal] = 1
    return result


def measured_move_down(o, h, l, c,
                        mode: str = 'confirmed',
                        window: int = 60,
                        pivot_n: int = 5,
                        pivot_pct: float | None = None,
                        leg_tol: float = 0.20) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Two downward legs of approximately equal size separated by a bounce.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, pivot_pct)
    pl = _pl(l, pivot_n, pivot_pct)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for jl in range(1, len(t_pl)):
        tL1, pL1 = t_pl[jl - 1], p_pl[jl - 1]
        tL2, pL2 = t_pl[jl],     p_pl[jl]
        highs_between = [(t_ph[k], p_ph[k]) for k in range(len(t_ph))
                         if tL1 < t_ph[k] < tL2]
        if not highs_between:
            continue
        tH2, pH2 = max(highs_between, key=lambda x: x[1])
        highs_before = [(t_ph[k], p_ph[k]) for k in range(len(t_ph))
                        if t_ph[k] < tL1]
        if not highs_before:
            continue
        tH1, pH1 = max(highs_before, key=lambda x: x[0])
        leg1 = pH1 - pL1
        leg2 = pH2 - pL2
        if leg1 <= 0 or leg2 <= 0:
            continue
        if abs(leg2 - leg1) / (leg1 + _EPS) < leg_tol:
            t_signal = min(tL2, N - 1)
            if result[t_signal] == 0:
                if mode == 'forming':
                    result[t_signal] = -1
                elif c[t_signal] < pH2:
                    result[t_signal] = -1
    return result


# ---------------------------------------------------------------------------
# V patterns
# ---------------------------------------------------------------------------

def v_bottom(o, h, l, c,
              mode: str = 'confirmed',
              window: int = 10,
              min_drop: float = 0.05) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Sharp decline then equally sharp recovery — a V-shape.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    half = window // 2
    for i in range(window, N):
        left  = c[i - window:i - half]
        right = c[i - half:i]
        if len(left) == 0 or len(right) == 0:
            continue
        drop    = (left[0] - left.min()) / (left[0] + _EPS)
        recover = (right[-1] - right.min()) / (left[0] + _EPS)
        if drop >= min_drop and recover >= min_drop * 0.8:
            if mode == 'forming':
                result[i] = 1
            elif c[i] >= left[0] * 0.98:
                result[i] = 1
    return result


def v_top(o, h, l, c,
           mode: str = 'confirmed',
           window: int = 10,
           min_rise: float = 0.05) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Sharp rise then sharp fall — inverted V.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    half = window // 2
    for i in range(window, N):
        left  = c[i - window:i - half]
        right = c[i - half:i]
        if len(left) == 0 or len(right) == 0:
            continue
        rise  = (left.max() - left[0]) / (left[0] + _EPS)
        fall  = (right.max() - right[-1]) / (left[0] + _EPS)
        if rise >= min_rise and fall >= min_rise * 0.8:
            if mode == 'forming':
                result[i] = -1
            elif c[i] <= left[0] * 1.02:
                result[i] = -1
    return result


def v_bottom_extended(o, h, l, c,
                       mode: str = 'confirmed',
                       window: int = 20,
                       extension: float = 1.10) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Like a V-bottom but the right side overshoots the starting price —
    extended recovery signals continued buying.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    half = window // 2
    for i in range(window, N):
        left  = c[i - window:i - half]
        right = c[i - half:i]
        if len(left) == 0 or len(right) == 0:
            continue
        if left.min() < left[0] * 0.97 and c[i] > left[0] * extension:
            if mode == 'forming':
                result[i] = 1
            elif c[i] > right.mean():
                result[i] = 1
    return result


def v_top_extended(o, h, l, c,
                    mode: str = 'confirmed',
                    window: int = 20,
                    extension: float = 0.90) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Extended inverted V: recovery from initial top extends to new lows.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    result = np.zeros(N, dtype=np.int8)
    half = window // 2
    for i in range(window, N):
        left  = c[i - window:i - half]
        right = c[i - half:i]
        if len(left) == 0 or len(right) == 0:
            continue
        if left.max() > left[0] * 1.03 and c[i] < left[0] * extension:
            if mode == 'forming':
                result[i] = -1
            elif c[i] < right.mean():
                result[i] = -1
    return result


# ---------------------------------------------------------------------------
# Roof / inverted roof  (arc patterns)
# ---------------------------------------------------------------------------

def roof(o, h, l, c, mode: str = 'confirmed', window: int = 30,
          min_height: float = 0.04) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Arched price action: rises to a peak then curves back down.
    Similar to rounding top but over a shorter timeframe.
    """
    return rounding_top(o, h, l, c, mode=mode, window=window,
                        min_depth=min_height)


def inverted_roof(o, h, l, c, mode: str = 'confirmed', window: int = 30,
                   min_depth: float = 0.04) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Inverted arch: dips to a trough then curves back up.
    """
    return rounding_bottom(o, h, l, c, mode=mode, window=window,
                            min_depth=min_depth)


# ---------------------------------------------------------------------------
# Mountain  (sharp rise + sharp fall)
# ---------------------------------------------------------------------------

def mountain(o, h, l, c, mode: str = 'confirmed',
              window: int = 20, min_move: float = 0.06) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Price forms a mountain: rises sharply then falls sharply back to (or
    below) the starting level.
    """
    return v_top(o, h, l, c, mode=mode, window=window, min_rise=min_move)


# ---------------------------------------------------------------------------
# Partial rise / partial decline  (pre-breakout signals)
# ---------------------------------------------------------------------------

def partial_rise(o, h, l, c,
                  mode: str = 'confirmed',
                  window: int = 60,
                  pivot_n: int = 4,
                  partial_pct: float = 0.50) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Within a consolidation pattern (triangle/rectangle), price makes a
    partial rise toward resistance but fails to reach it — often precedes a
    downside breakout.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, None)
    pl = _pl(l, pivot_n, None)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        res = line_val(s_hi, i_hi, i)
        sup = line_val(s_lo, i_lo, i)
        band = res - sup
        if band <= 0:
            continue
        # Partial rise: price rose from near support but turned before resistance
        if (c[i] > sup + 0.20 * band and        # bounced off support
                c[i] < sup + partial_pct * band and  # didn't reach resistance
                c[i] < c[i - 1]):                    # turning down
            result[i] = -1
    return result


def partial_decline(o, h, l, c,
                     mode: str = 'confirmed',
                     window: int = 60,
                     pivot_n: int = 4,
                     partial_pct: float = 0.50) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Within a consolidation, price dips toward support but fails to reach it —
    often precedes an upside breakout.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(h)
    ph = _ph(h, pivot_n, None)
    pl = _pl(l, pivot_n, None)
    t_ph, p_ph, t_pl, p_pl = _scan(h, l, c, ph, pl, pivot_n)
    result = np.zeros(N, dtype=np.int8)

    for i in range(window, N):
        s_hi, i_hi, s_lo, i_lo, ok = _triangle_lines(
            t_ph, p_ph, t_pl, p_pl, i, window)
        if not ok:
            continue
        res = line_val(s_hi, i_hi, i)
        sup = line_val(s_lo, i_lo, i)
        band = res - sup
        if band <= 0:
            continue
        if (c[i] < sup + 0.80 * band and        # fell toward support
                c[i] > sup + (1 - partial_pct) * band and  # didn't reach support
                c[i] > c[i - 1]):                    # turning up
            result[i] = 1
    return result
