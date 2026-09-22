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
from ._memo import array_key, memoized
from ._windows import trailing_windows, RangeAgg
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

    Pivot confirmation times (``t_ph``/``t_pl``) come from ``np.where`` and are
    therefore sorted ascending, so the bars inside the window ``[i_end-window,
    i_end]`` form a contiguous slice located with ``np.searchsorted`` in
    O(log P) — instead of re-scanning every pivot on every bar, which made the
    enclosing per-bar loops O(N²).  The selected pivot set (and its order) is
    identical to the old list-comprehension, so results are unchanged.
    """
    lo_t = i_end - window
    a_h = np.searchsorted(t_ph, lo_t, "left")
    b_h = np.searchsorted(t_ph, i_end, "right")
    a_l = np.searchsorted(t_pl, lo_t, "left")
    b_l = np.searchsorted(t_pl, i_end, "right")
    if (b_h - a_h) < min_pivots or (b_l - a_l) < min_pivots:
        return 0, 0, 0, 0, False
    x_h = t_ph[a_h:b_h].astype(float)
    y_h = p_ph[a_h:b_h].astype(float)
    x_l = t_pl[a_l:b_l].astype(float)
    y_l = p_pl[a_l:b_l].astype(float)
    s_hi, i_hi, r2_hi = fit_line(x_h, y_h)
    s_lo, i_lo, r2_lo = fit_line(x_l, y_l)
    if r2_hi < 0.5 or r2_lo < 0.5:
        return 0, 0, 0, 0, False
    return s_hi, i_hi, s_lo, i_lo, True


def _sliding_fit(t, p, window, N):
    """
    Vectorised sliding-window linear regression of ``p`` (pivot prices) on
    ``t`` (pivot times), evaluated for *every* bar ``i`` in ``[0, N)`` over the
    window ``[i-window, i]``.

    Returns ``(slope, intercept, r2, n)`` — each a length-N array — reproducing
    :func:`fit_line_r2`'s result per bar via prefix sums, so the whole family of
    triangle/wedge/channel detectors can share one O(N) pass instead of calling
    :func:`_triangle_lines` (and ``fit_line_r2``) once per bar, per detector.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    M = len(t)
    bars = np.arange(N)
    if M == 0:
        z = np.zeros(N)
        return z, z.copy(), z.copy(), np.zeros(N, dtype=np.int64)
    # prefix sums, padded with a leading 0 so prefix[k] = sum of first k items
    z0   = np.zeros(1)
    csx  = np.concatenate((z0, np.cumsum(t)))
    csy  = np.concatenate((z0, np.cumsum(p)))
    csxx = np.concatenate((z0, np.cumsum(t * t)))
    csyy = np.concatenate((z0, np.cumsum(p * p)))
    csxy = np.concatenate((z0, np.cumsum(t * p)))
    # window [i-window, i] is a contiguous slice [a:b] of the sorted pivot times
    a = np.searchsorted(t, bars - window, side="left")
    b = np.searchsorted(t, bars, side="right")
    n   = (b - a).astype(float)
    Sx  = csx[b]  - csx[a]
    Sy  = csy[b]  - csy[a]
    Sxx = csxx[b] - csxx[a]
    Syy = csyy[b] - csyy[a]
    Sxy = csxy[b] - csxy[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        n_safe = np.where(n > 0, n, 1.0)
        mx = Sx / n_safe
        my = Sy / n_safe
        ss_xx  = Sxx - Sx * Sx / n_safe          # Σ(x-mx)^2
        ss_xy  = Sxy - Sx * Sy / n_safe          # Σ(x-mx)(y-my)
        ss_tot = Syy - Sy * Sy / n_safe          # Σ(y-my)^2
        good = ss_xx > _EPS
        slope = np.where(good, ss_xy / np.where(good, ss_xx, 1.0), 0.0)
        intercept = my - slope * mx
        ss_res = ss_tot - slope * ss_xy
        r2 = np.where((ss_tot > _EPS) & good, 1.0 - ss_res / ss_tot, 0.0)
        # fit_line_r2: ss_xx < _EPS → r2 = 0; ss_tot < _EPS (but ss_xx ok) → r2 = 1
        r2 = np.where(good & ~(ss_tot > _EPS), 1.0, r2)
    # n < 2 mirrors fit_line_r2's early return: slope 0, intercept mean(y), r2 0
    one_pt = (b - a) < 2
    slope     = np.where(one_pt, 0.0, slope)
    intercept = np.where(one_pt, np.where(n > 0, my, 0.0), intercept)
    r2        = np.where(one_pt, 0.0, r2)
    return slope, intercept, r2, (b - a)


def _sliding_trendlines(t_ph, p_ph, t_pl, p_pl, N, window, min_pivots=2):
    """
    Per-bar upper/lower trendline fits for every bar in ``[0, N)`` — the
    vectorised, shared-across-detectors equivalent of calling
    :func:`_triangle_lines` in a ``for i in range(window, N)`` loop.

    Returns ``(S_HI, I_HI, S_LO, I_LO, OK)`` arrays of length N where ``OK[i]``
    matches the old ``ok`` flag (enough pivots in window *and* both r² ≥ 0.5).

    Memoised on pivot contents: ~22 detectors request the identical fit with the
    default ``window=100, pivot_n=5``, so in a full scan this is computed once.
    """
    keys = [array_key(x) for x in (t_ph, p_ph, t_pl, p_pl)]
    key = None if any(k is None for k in keys) else \
        ("trend", tuple(keys), N, window, min_pivots)
    return memoized(key, lambda: _sliding_trendlines_impl(
        t_ph, p_ph, t_pl, p_pl, N, window, min_pivots))


def _sliding_trendlines_impl(t_ph, p_ph, t_pl, p_pl, N, window, min_pivots=2):
    s_hi, i_hi, r2_hi, n_hi = _sliding_fit(t_ph, p_ph, window, N)
    s_lo, i_lo, r2_lo, n_lo = _sliding_fit(t_pl, p_pl, window, N)
    ok = ((n_hi >= min_pivots) & (n_lo >= min_pivots)
          & (r2_hi >= 0.5) & (r2_lo >= 0.5))
    return s_hi, i_hi, s_lo, i_lo, ok


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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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
    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    result = np.zeros(N, dtype=np.int8)

    for i in range(pole_bars + window, N):
        pole_start, direction = _flagpole(c, i - window, min_pole, pole_bars)
        if direction != 1:
            continue
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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
    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    result = np.zeros(N, dtype=np.int8)

    for i in range(pole_bars + window, N):
        pole_start, direction = _flagpole(c, i - window, min_pole, pole_bars)
        if direction != -1:
            continue
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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
    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    result = np.zeros(N, dtype=np.int8)

    for i in range(pole_bars + window, N):
        pole_start, direction = _flagpole(c, i - window, min_pole, pole_bars)
        if direction != 1:
            continue
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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
    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    result = np.zeros(N, dtype=np.int8)

    for i in range(pole_bars + window, N):
        pole_start, direction = _flagpole(c, i - window, min_pole, pole_bars)
        if direction != -1:
            continue
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    start = cup_window + handle_window
    if start >= N:
        return result
    i = np.arange(start, N)
    cup_start    = i - cup_window - handle_window
    handle_start = i - handle_window

    # Range max/min over the cup's pivot span.  The span bounds move with i, so
    # a sparse table answers all N queries in O(1) each instead of re-slicing
    # and re-reducing the pivot arrays on every bar.
    a_h = np.searchsorted(t_ph, cup_start, "left")
    b_h = np.searchsorted(t_ph, handle_start, "left")
    a_l = np.searchsorted(t_pl, cup_start, "left")
    b_l = np.searchsorted(t_pl, handle_start, "left")

    have = (b_h > a_h) & (b_l > a_l)
    left_rim = RangeAgg(p_ph, "max").query(a_h, b_h)
    cup_bot  = RangeAgg(p_pl, "min").query(a_l, b_l)

    cup_depth = left_rim - cup_bot
    with np.errstate(invalid="ignore"):
        fire = have & (cup_depth / (left_rim + _EPS) >= 0.05)

    # Right area is a fixed-length trailing window of closes: c[i-handle:i]
    # NaN note: a mask AND suppresses the bar, where the loop's `if x < y:
    # continue` fell through on NaN and could fire on corrupt data.  Refusing to
    # signal on a window we cannot evaluate is the intended behaviour.
    right_max = trailing_windows(c, handle_window).max(axis=1)[start - handle_window:]
    fire &= right_max >= left_rim * 0.95
    with np.errstate(invalid="ignore"):
        fire &= (right_max - c[start:]) / (cup_depth + _EPS) <= max_handle_retrace
    if mode != 'forming':
        fire &= c[start:] > left_rim
    result[start:] = np.where(fire, 1, 0)
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

    start = cup_window + handle_window
    if start >= N:
        return result
    i = np.arange(start, N)
    cup_start    = i - cup_window - handle_window
    handle_start = i - handle_window

    a_l = np.searchsorted(t_pl, cup_start, "left")
    b_l = np.searchsorted(t_pl, handle_start, "left")
    a_h = np.searchsorted(t_ph, cup_start, "left")
    b_h = np.searchsorted(t_ph, handle_start, "left")

    have = (b_l > a_l) & (b_h > a_h)
    left_rim = RangeAgg(p_pl, "min").query(a_l, b_l)
    cup_top  = RangeAgg(p_ph, "max").query(a_h, b_h)

    cup_depth = cup_top - left_rim
    with np.errstate(invalid="ignore"):
        fire = have & (cup_depth / (cup_top + _EPS) >= 0.05)

    # NaN note: see cup_with_handle — a NaN window is suppressed, not fired.
    right_min = trailing_windows(c, handle_window).min(axis=1)[start - handle_window:]
    fire &= right_min <= left_rim * 1.05
    with np.errstate(invalid="ignore"):
        fire &= (c[start:] - right_min) / (cup_depth + _EPS) <= max_handle_retrace
    if mode != 'forming':
        fire &= c[start:] < left_rim
    result[start:] = np.where(fire, -1, 0)
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
    W = trailing_windows(c, window)
    if len(W) == 0:
        return result

    left  = W[:, :third].mean(axis=1)
    mid   = W[:, third:2 * third].mean(axis=1)
    right = W[:, 2 * third:].mean(axis=1)

    # np.minimum propagates NaN, where the loop's scalar min() returned whichever
    # argument came first and so gave an order-dependent answer.  Propagating is
    # the well-defined choice: an unevaluable window does not fire.
    outer = np.minimum(left, right)
    depth = (outer - mid) / (left + _EPS)
    fire  = (mid < outer) & (depth >= min_depth)
    if mode != 'forming':
        fire &= c[window:] > np.maximum(left, right)      # close above the rim
    result[window:] = np.where(fire, 1, 0)
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
    W = trailing_windows(c, window)
    if len(W) == 0:
        return result

    left  = W[:, :third].mean(axis=1)
    mid   = W[:, third:2 * third].mean(axis=1)
    right = W[:, 2 * third:].mean(axis=1)

    # See rounding_bottom on the min/max NaN semantics.
    outer  = np.maximum(left, right)
    height = (mid - outer) / (mid + _EPS)
    fire   = (mid > outer) & (height >= min_depth)
    if mode != 'forming':
        fire &= c[window:] < np.minimum(left, right)      # close below neckline
    result[window:] = np.where(fire, -1, 0)
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
    half = window // 2

    # One vectorised pass gives the half-window fit ending at EVERY bar.  The
    # pattern's "first half" fit is that same array shifted back by `half`, so
    # both legs of the diamond come from a single O(N) computation instead of
    # two scalar least-squares fits per bar.
    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, half)

    i = np.arange(N)
    j = i - half                                  # first-half end bar
    valid = i >= window
    ok = np.zeros(N, dtype=bool)
    ok[valid] = OK[valid] & OK[j[valid]]

    s_hi1 = np.where(valid, S_HI[j.clip(0)], 0.0)
    s_lo1 = np.where(valid, S_LO[j.clip(0)], 0.0)

    fire = ok & (s_hi1 > 0) & (s_lo1 < 0) & (S_LO > S_HI)
    if mode != 'forming':
        fire &= c < (S_LO * i + I_LO)             # close below support
    return np.where(fire, -1, 0).astype(np.int8)


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
    half = window // 2

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, half)

    i = np.arange(N)
    j = i - half
    valid = i >= window
    ok = np.zeros(N, dtype=bool)
    ok[valid] = OK[valid] & OK[j[valid]]

    s_hi1 = np.where(valid, S_HI[j.clip(0)], 0.0)
    s_lo1 = np.where(valid, S_LO[j.clip(0)], 0.0)

    fire = ok & (s_hi1 > 0) & (s_lo1 < 0) & (S_LO > S_HI)
    if mode != 'forming':
        fire &= c > (S_HI * i + I_HI)             # close above resistance
    return np.where(fire, 1, 0).astype(np.int8)


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

    if total >= N:
        return result
    # Both fits are fixed-width sliding regressions over the pivot series, so
    # one prefix-sum pass each covers every bar.  The lead-in window
    # [i-total, i-bump_window) is the bump-width fit evaluated `bump_window+1`
    # bars earlier, with width lead_window-1.
    S_LEAD, I_LEAD, R2_LEAD, N_LEAD = _sliding_fit(t_pl, p_pl, lead_window - 1, N)
    S_BUMP, _, _, N_BUMP = _sliding_fit(t_pl, p_pl, bump_window, N)

    i = np.arange(total, N)
    j = i - bump_window - 1                    # lead-in window end bar
    s_lead, i_lead = S_LEAD[j], I_LEAD[j]

    fire = (N_LEAD[j] >= 2) & (R2_LEAD[j] >= 0.6) & (s_lead > 0)
    fire &= (N_BUMP[i] >= 2) & (S_BUMP[i] >= bump_factor * s_lead)
    if mode != 'forming':
        fire &= c[total:] < (s_lead * i + i_lead)      # break back through lead-in
    result[total:] = np.where(fire, -1, 0)
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

    if total >= N:
        return result
    S_LEAD, I_LEAD, R2_LEAD, N_LEAD = _sliding_fit(t_ph, p_ph, lead_window - 1, N)
    S_BUMP, _, _, N_BUMP = _sliding_fit(t_ph, p_ph, bump_window, N)

    i = np.arange(total, N)
    j = i - bump_window - 1
    s_lead, i_lead = S_LEAD[j], I_LEAD[j]

    fire = (N_LEAD[j] >= 2) & (R2_LEAD[j] >= 0.6) & (s_lead < 0)
    fire &= (N_BUMP[i] >= 2) & (S_BUMP[i] <= bump_factor * s_lead)
    if mode != 'forming':
        fire &= c[total:] > (s_lead * i + i_lead)
    result[total:] = np.where(fire, 1, 0)
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


def _scallop_masks(c, window):
    """Vectorised ``_scallop_shape`` over every trailing window at once.

    Returns ``(rows_j, rows_inv, wmax, wmin)`` where ``rows_j[k]`` is the
    J-shape test and ``rows_inv[k]`` the inverted (∩) test for the window
    ending just before bar ``k + window``.  Negating the prices flips each
    comparison, which is how the inverted variants are derived.
    """
    third = window // 3
    W = trailing_windows(c, window)
    if len(W) == 0:
        z = np.zeros(0, dtype=bool)
        return z, z, np.zeros(0), np.zeros(0)
    left  = W[:, :third].mean(axis=1)
    mid   = W[:, third:2 * third].mean(axis=1)
    right = W[:, 2 * third:].mean(axis=1)
    j_shape = (mid < left) & (right > mid) & (right >= left * 0.95)
    inverted = (mid > left) & (right < mid) & (right <= left * 0.95)
    return j_shape, inverted, W.max(axis=1), W.min(axis=1)


def _scallop(c, N, window, trend, inverted, sign, mode):
    """Shared body for all four scallop detectors."""
    result = np.zeros(N, dtype=np.int8)
    j_shape, inv, wmax, wmin = _scallop_masks(c, window)
    if len(j_shape) == 0:
        return result
    fire = (inv if inverted else j_shape) & trend[window:]
    if mode != 'forming':
        fire &= (c[window:] < wmin) if inverted else (c[window:] > wmax)
    result[window:] = np.where(fire, sign, 0)
    return result


def scallop_asc(o, h, l, c,
                 mode: str = 'confirmed',
                 window: int = 25) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    Repeated J-shaped recovery (scallop) in a rising trend.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    return _scallop(c, N, window, uptrend(c, window),
                    inverted=False, sign=1, mode=mode)


def scallop_asc_inv(o, h, l, c,
                     mode: str = 'confirmed',
                     window: int = 25) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Inverted ascending scallop — ∩-shape in a rising trend signals a top.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    return _scallop(c, N, window, uptrend(c, window),
                    inverted=True, sign=-1, mode=mode)


def scallop_desc(o, h, l, c,
                  mode: str = 'confirmed',
                  window: int = 25) -> np.ndarray:
    """
    **BEARISH** (-1 / 0).
    Inverted J-shape (decay) in a falling trend.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    return _scallop(c, N, window, downtrend(c, window),
                    inverted=True, sign=-1, mode=mode)


def scallop_desc_inv(o, h, l, c,
                      mode: str = 'confirmed',
                      window: int = 25) -> np.ndarray:
    """
    **BULLISH** (+1 / 0).
    J-shaped recovery in a falling trend — potential bottom reversal.
    """
    o, h, l, c = (_a(x) for x in (o, h, l, c))
    N = len(c)
    return _scallop(c, N, window, downtrend(c, window),
                    inverted=False, sign=1, mode=mode)


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

    # (Previously wrapped in a redundant ``for j in range(2, len(t_ph))`` loop
    # whose index was never used — it re-ran this identical scan P times, with
    # every pass after the first a no-op thanks to the ``result[...] == 0``
    # guard.  Removing it is output-identical and drops an O(P) factor.)
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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

    S_HI, I_HI, S_LO, I_LO, OK = _sliding_trendlines(
        t_ph, p_ph, t_pl, p_pl, N, window)
    for i in range(window, N):
        if not OK[i]:
            continue
        s_hi, i_hi, s_lo, i_lo = S_HI[i], I_HI[i], S_LO[i], I_LO[i]
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
