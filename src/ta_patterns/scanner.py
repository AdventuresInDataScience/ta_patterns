"""
ta_patterns.scanner
===================
Convenience functions to run all (or a subset of) pattern detectors
over a price series and return labelled results.

Signed conventions
------------------
Every detector and every scan/score helper here uses:
    +1  →  bullish pattern fired
    -1  →  bearish pattern fired
     0  →  no pattern fired

This means a *positive* net_score is net bullish, negative is net bearish.
If you prefer the opposite polarity, multiply the result by -1.  Note that
non-directional shape patterns (e.g. plain doji) return +1 when their shape
fires but are excluded from the net_score helpers, which sum only
directional (bullish/bearish) patterns.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from . import single as _s
from . import two_bar as _t
from . import three_bar as _tb
from . import multi_bar as _m
from ._core import _to_np


def _f(*arrs):
    return tuple(_to_np(a) for a in arrs)


def scan_all(o, h, l, c) -> Dict[str, np.ndarray]:
    """
    Run every pattern detector over aligned OHLC arrays.

    Accepts numpy arrays **or** pandas Series (index is ignored).

    Returns
    -------
    dict  →  {pattern_name: int8 numpy array}
    A non-zero value at index i (+1 bullish / -1 bearish) means the pattern
    completed at bar i (no look-forward bias).
    """
    o, h, l, c = _f(o, h, l, c)
    R: Dict[str, np.ndarray] = {}

    # ── single bar ──────────────────────────────────────────────────────────
    R["white_candle"]              = _s.white_candle(o, c)
    R["black_candle"]              = _s.black_candle(o, c)
    R["doji"]                      = _s.doji(o, h, l, c)
    R["dragonfly_doji"]            = _s.dragonfly_doji(o, h, l, c)
    R["gravestone_doji"]           = _s.gravestone_doji(o, h, l, c)
    R["long_legged_doji"]          = _s.long_legged_doji(o, h, l, c)
    R["northern_doji"]             = _s.northern_doji(o, h, l, c)
    R["southern_doji"]             = _s.southern_doji(o, h, l, c)
    R["rickshaw_man"]              = _s.rickshaw_man(o, h, l, c)
    R["marubozu_white"]            = _s.marubozu_white(o, h, l, c)
    R["marubozu_black"]            = _s.marubozu_black(o, h, l, c)
    R["closing_marubozu_white"]    = _s.closing_marubozu_white(o, h, l, c)
    R["closing_marubozu_black"]    = _s.closing_marubozu_black(o, h, l, c)
    R["opening_marubozu_white"]    = _s.opening_marubozu_white(o, h, l, c)
    R["opening_marubozu_black"]    = _s.opening_marubozu_black(o, h, l, c)
    R["hammer"]                    = _s.hammer(o, h, l, c)
    R["hanging_man"]               = _s.hanging_man(o, h, l, c)
    R["shooting_star"]             = _s.shooting_star(o, h, l, c)
    R["takuri_line"]               = _s.takuri_line(o, h, l, c)
    R["high_wave"]                 = _s.high_wave(o, h, l, c)
    R["belt_hold_bullish"]         = _s.belt_hold_bullish(o, h, l, c)
    R["belt_hold_bearish"]         = _s.belt_hold_bearish(o, h, l, c)
    R["long_white_day"]            = _s.long_white_day(o, h, l, c)
    R["long_black_day"]            = _s.long_black_day(o, h, l, c)
    R["short_white_candle"]        = _s.short_white_candle(o, h, l, c)
    R["short_black_candle"]        = _s.short_black_candle(o, h, l, c)
    R["spinning_top_white"]        = _s.spinning_top_white(o, h, l, c)
    R["spinning_top_black"]        = _s.spinning_top_black(o, h, l, c)

    # ── two bar ─────────────────────────────────────────────────────────────
    R["gapping_up_doji"]           = _t.gapping_up_doji(o, h, l, c)
    R["gapping_down_doji"]         = _t.gapping_down_doji(o, h, l, c)
    R["engulfing_bullish"]         = _t.engulfing_bullish(o, h, l, c)
    R["engulfing_bearish"]         = _t.engulfing_bearish(o, h, l, c)
    R["harami_bullish"]            = _t.harami_bullish(o, h, l, c)
    R["harami_bearish"]            = _t.harami_bearish(o, h, l, c)
    R["harami_cross_bullish"]      = _t.harami_cross_bullish(o, h, l, c)
    R["harami_cross_bearish"]      = _t.harami_cross_bearish(o, h, l, c)
    R["doji_star_bullish"]         = _t.doji_star_bullish(o, h, l, c)
    R["doji_star_bearish"]         = _t.doji_star_bearish(o, h, l, c)
    R["dark_cloud_cover"]          = _t.dark_cloud_cover(o, h, l, c)
    R["piercing_pattern"]          = _t.piercing_pattern(o, h, l, c)
    R["in_neck"]                   = _t.in_neck(o, h, l, c)
    R["on_neck"]                   = _t.on_neck(o, h, l, c)
    R["thrusting"]                 = _t.thrusting(o, h, l, c)
    R["kicking_bullish"]           = _t.kicking_bullish(o, h, l, c)
    R["kicking_bearish"]           = _t.kicking_bearish(o, h, l, c)
    R["tweezer_tops"]              = _t.tweezer_tops(o, h, l, c)
    R["tweezer_bottoms"]           = _t.tweezer_bottoms(o, h, l, c)
    R["matching_low"]              = _t.matching_low(o, h, l, c)
    R["meeting_lines_bullish"]     = _t.meeting_lines_bullish(o, h, l, c)
    R["meeting_lines_bearish"]     = _t.meeting_lines_bearish(o, h, l, c)
    R["separating_lines_bullish"]  = _t.separating_lines_bullish(o, h, l, c)
    R["separating_lines_bearish"]  = _t.separating_lines_bearish(o, h, l, c)
    R["rising_window"]             = _t.rising_window(o, h, l, c)
    R["falling_window"]            = _t.falling_window(o, h, l, c)
    R["two_black_gapping"]         = _t.two_black_gapping(o, h, l, c)
    R["homing_pigeon"]             = _t.homing_pigeon(o, h, l, c)
    R["above_stomach"]             = _t.above_stomach(o, h, l, c)
    R["below_stomach"]             = _t.below_stomach(o, h, l, c)
    R["last_engulfing_top"]        = _t.last_engulfing_top(o, h, l, c)
    R["last_engulfing_bottom"]     = _t.last_engulfing_bottom(o, h, l, c)
    R["inverted_hammer"]           = _t.inverted_hammer(o, h, l, c)
    R["shooting_star_two_line"]    = _t.shooting_star_two_line(o, h, l, c)

    # ── three bar ───────────────────────────────────────────────────────────
    R["morning_star"]                      = _tb.morning_star(o, h, l, c)
    R["evening_star"]                      = _tb.evening_star(o, h, l, c)
    R["morning_doji_star"]                 = _tb.morning_doji_star(o, h, l, c)
    R["evening_doji_star"]                 = _tb.evening_doji_star(o, h, l, c)
    R["three_white_soldiers"]              = _tb.three_white_soldiers(o, h, l, c)
    R["three_black_crows"]                 = _tb.three_black_crows(o, h, l, c)
    R["identical_three_crows"]             = _tb.identical_three_crows(o, h, l, c)
    R["abandoned_baby_bullish"]            = _tb.abandoned_baby_bullish(o, h, l, c)
    R["abandoned_baby_bearish"]            = _tb.abandoned_baby_bearish(o, h, l, c)
    R["three_inside_up"]                   = _tb.three_inside_up(o, h, l, c)
    R["three_inside_down"]                 = _tb.three_inside_down(o, h, l, c)
    R["three_outside_up"]                  = _tb.three_outside_up(o, h, l, c)
    R["three_outside_down"]                = _tb.three_outside_down(o, h, l, c)
    R["advance_block"]                     = _tb.advance_block(o, h, l, c)
    R["deliberation"]                      = _tb.deliberation(o, h, l, c)
    R["two_crows"]                         = _tb.two_crows(o, h, l, c)
    R["upside_gap_two_crows"]              = _tb.upside_gap_two_crows(o, h, l, c)
    R["downside_tasuki_gap"]               = _tb.downside_tasuki_gap(o, h, l, c)
    R["upside_tasuki_gap"]                 = _tb.upside_tasuki_gap(o, h, l, c)
    R["three_stars_south"]                 = _tb.three_stars_south(o, h, l, c)
    R["stick_sandwich"]                    = _tb.stick_sandwich(o, h, l, c)
    R["unique_three_river_bottom"]         = _tb.unique_three_river_bottom(o, h, l, c)
    R["tri_star_bullish"]                  = _tb.tri_star_bullish(o, h, l, c)
    R["tri_star_bearish"]                  = _tb.tri_star_bearish(o, h, l, c)
    R["downside_gap_three_methods"]        = _tb.downside_gap_three_methods(o, h, l, c)
    R["upside_gap_three_methods"]          = _tb.upside_gap_three_methods(o, h, l, c)
    R["side_by_side_white_lines_bullish"]  = _tb.side_by_side_white_lines_bullish(o, h, l, c)
    R["side_by_side_white_lines_bearish"]  = _tb.side_by_side_white_lines_bearish(o, h, l, c)
    R["hikkake_bullish"]                   = _tb.hikkake_bullish(o, h, l, c)
    R["hikkake_bearish"]                   = _tb.hikkake_bearish(o, h, l, c)
    R["collapse_doji_star"]                = _tb.collapse_doji_star(o, h, l, c)

    # ── multi bar ───────────────────────────────────────────────────────────
    R["three_line_strike_bullish"]  = _m.three_line_strike_bullish(o, h, l, c)
    R["three_line_strike_bearish"]  = _m.three_line_strike_bearish(o, h, l, c)
    R["concealing_baby_swallow"]    = _m.concealing_baby_swallow(o, h, l, c)
    R["rising_three_methods"]       = _m.rising_three_methods(o, h, l, c)
    R["falling_three_methods"]      = _m.falling_three_methods(o, h, l, c)
    R["mat_hold"]                   = _m.mat_hold(o, h, l, c)
    R["breakaway_bullish"]          = _m.breakaway_bullish(o, h, l, c)
    R["breakaway_bearish"]          = _m.breakaway_bearish(o, h, l, c)
    R["ladder_bottom"]              = _m.ladder_bottom(o, h, l, c)
    R["eight_new_price_lines"]      = _m.eight_new_price_lines(c)
    R["ten_new_price_lines"]        = _m.ten_new_price_lines(c)
    R["twelve_new_price_lines"]     = _m.twelve_new_price_lines(c)
    R["thirteen_new_price_lines"]   = _m.thirteen_new_price_lines(c)

    # ── length normalisation ────────────────────────────────────────────────
    # For very short inputs (N < pattern width), the padding helpers can
    # produce arrays whose length != N.  We right-align and zero-fill so
    # the returned dict always has every value at exactly length N.
    N = len(o)
    for k in list(R.keys()):
        v = R[k]
        if len(v) == N:
            continue
        fixed = np.zeros(N, dtype=np.int8)
        take  = min(len(v), N)
        if take > 0:
            fixed[N - take:] = v[:take]
        R[k] = fixed

    return R


# ---------------------------------------------------------------------------
# Direction metadata
# ---------------------------------------------------------------------------

BULLISH: frozenset = frozenset({
    "white_candle", "dragonfly_doji", "southern_doji", "hammer",
    "takuri_line", "belt_hold_bullish", "marubozu_white",
    "closing_marubozu_white", "opening_marubozu_white", "long_white_day",
    "spinning_top_white", "gapping_up_doji",
    "engulfing_bullish", "harami_bullish", "harami_cross_bullish",
    "doji_star_bullish", "piercing_pattern", "kicking_bullish",
    "tweezer_bottoms", "matching_low", "meeting_lines_bullish",
    "separating_lines_bullish", "rising_window", "homing_pigeon",
    "above_stomach", "last_engulfing_bottom", "inverted_hammer",
    "morning_star", "morning_doji_star", "three_white_soldiers",
    "abandoned_baby_bullish", "three_inside_up", "three_outside_up",
    "three_stars_south", "stick_sandwich", "unique_three_river_bottom",
    "tri_star_bullish", "upside_gap_three_methods", "upside_tasuki_gap",
    "side_by_side_white_lines_bullish", "hikkake_bullish",
    "three_line_strike_bullish", "concealing_baby_swallow",
    "rising_three_methods", "mat_hold", "breakaway_bullish",
    "ladder_bottom",
    "eight_new_price_lines", "ten_new_price_lines",
    "twelve_new_price_lines", "thirteen_new_price_lines",
})

BEARISH: frozenset = frozenset({
    "black_candle", "gravestone_doji", "northern_doji", "hanging_man",
    "shooting_star", "belt_hold_bearish", "marubozu_black",
    "closing_marubozu_black", "opening_marubozu_black", "long_black_day",
    "spinning_top_black", "gapping_down_doji",
    "engulfing_bearish", "harami_bearish", "harami_cross_bearish",
    "doji_star_bearish", "dark_cloud_cover", "in_neck", "on_neck",
    "thrusting", "kicking_bearish", "tweezer_tops", "meeting_lines_bearish",
    "separating_lines_bearish", "falling_window", "two_black_gapping",
    "below_stomach", "last_engulfing_top", "shooting_star_two_line",
    "evening_star", "evening_doji_star", "three_black_crows",
    "identical_three_crows", "abandoned_baby_bearish", "three_inside_down",
    "three_outside_down", "advance_block", "deliberation", "two_crows",
    "upside_gap_two_crows", "downside_tasuki_gap", "tri_star_bearish",
    "downside_gap_three_methods", "collapse_doji_star",
    "side_by_side_white_lines_bearish", "hikkake_bearish",
    "three_line_strike_bearish", "falling_three_methods", "breakaway_bearish",
})

NON_DIRECTIONAL: frozenset = frozenset({
    "doji", "long_legged_doji", "rickshaw_man", "high_wave",
    "short_white_candle", "short_black_candle",
})

#: Complete set of all recognised pattern names (BULLISH ∪ BEARISH ∪ NON_DIRECTIONAL).
#: Use ``ta.list_patterns()`` for a sorted list, or ``ta.PATTERNS`` directly.
PATTERNS: frozenset = BULLISH | BEARISH | NON_DIRECTIONAL

#: All patterns with a definite directional bias (BULLISH ∪ BEARISH).
DIRECTIONAL: frozenset = BULLISH | BEARISH


# ---------------------------------------------------------------------------
# Filtered scans
# ---------------------------------------------------------------------------

def scan_bullish(o, h, l, c) -> Dict[str, np.ndarray]:
    """Return only bullish-directional pattern signals (int8 arrays)."""
    return {k: v for k, v in scan_all(o, h, l, c).items() if k in BULLISH}


def scan_bearish(o, h, l, c) -> Dict[str, np.ndarray]:
    """Return only bearish-directional pattern signals (int8 arrays)."""
    return {k: v for k, v in scan_all(o, h, l, c).items() if k in BEARISH}



# ---------------------------------------------------------------------------
# Multi-pattern extraction  (the main convenience API)
# ---------------------------------------------------------------------------

def list_patterns(direction: str = "all") -> List[str]:
    """
    Return a sorted list of all available pattern names.

    Parameters
    ----------
    direction : str, default ``'all'``
        ``'all'``     – every pattern (106 total)
        ``'bullish'`` – bullish patterns only
        ``'bearish'`` – bearish patterns only
        ``'non_directional'`` – neutral / shape-only patterns

    Example
    -------
    >>> tap.list_patterns("bullish")[:5]
    ['abandoned_baby_bullish', 'above_stomach', 'belt_hold_bullish', ...]
    """
    _map = {"all": PATTERNS, "bullish": BULLISH,
            "bearish": BEARISH, "non_directional": NON_DIRECTIONAL}
    if direction not in _map:
        raise ValueError(
            f"direction must be one of {list(_map)}, got {direction!r}"
        )
    return sorted(_map[direction])


def _fuzzy_suggest(unknown: set, available: frozenset) -> str:
    """Build a human-readable suggestion string using difflib."""
    import difflib
    lines = []
    for name in sorted(unknown):
        hits = difflib.get_close_matches(name, sorted(available), n=3, cutoff=0.4)
        line = f"  '{name}'"
        if hits:
            line += f"  →  did you mean: {hits}"
        lines.append(line)
    return "\n".join(lines)


def net_score(o, h, l, c) -> np.ndarray:
    """
    Per-bar aggregate candlestick signal score.

    Sums only the **directional** candlestick patterns (BULLISH ∪ BEARISH).
    Non-directional shape patterns (plain ``doji``, ``high_wave``,
    ``rickshaw_man``, ``long_legged_doji``, ``short_white_candle``,
    ``short_black_candle``) are deliberately excluded: they return +1 when
    their shape fires but carry no bullish/bearish bias, so including them
    would skew an otherwise sign-balanced score upward.

    Returns **int16** (range roughly ±100 with the current catalogue).

    Interpretation
    --------------
    Strongly positive  →  many bullish patterns coinciding (high conviction)
    Strongly negative  →  many bearish patterns coinciding
    Near zero          →  conflicting or no directional signals

    Example
    -------
    >>> score = tap.net_score(o, h, l, c)
    >>> score[-1]   # net signal at the most recent bar
    """
    raw   = scan_all(o, h, l, c)
    n     = len(next(iter(raw.values())))
    score = np.zeros(n, dtype=np.int16)
    for name, arr in raw.items():
        if name in DIRECTIONAL:          # BULLISH ∪ BEARISH only
            score += arr.astype(np.int16)
    return score


def batch_patterns(
    o,
    h,
    l,
    c,
    patterns: "str | List[str]" = "all",
    as_frame: bool = True,
) -> "np.ndarray | pd.DataFrame":
    """
    Run a selection of pattern detectors and return the results as a single
    2-D structure — one column per pattern, one row per bar.

    Parameters
    ----------
    o, h, l, c : array-like
        OHLC data.  Accepts numpy arrays **or** pandas Series.
    patterns : str or list of str, default ``'all'``
        Which patterns to include:

        * ``'all'``     – all 106 patterns, sorted alphabetically
        * ``'bullish'`` – bullish subset
        * ``'bearish'`` – bearish subset
        * ``'non_directional'`` – neutral / shape-only subset
        * ``'directional'`` – bullish ∪ bearish subset
        * ``list``      – explicit list, e.g. ``['hammer', 'morning_star']``;
          order is preserved in the output columns.

        Each column is **int8**: ``+1`` bullish fired, ``-1`` bearish fired,
        ``0`` did not fire.

    as_frame : bool, default ``True``
        * ``True``  – return a **pandas DataFrame** with pattern names as
          column headers.  Falls back to numpy with a warning if pandas is
          not installed.
        * ``False`` – return a plain **numpy 2-D array** of shape
          ``(n_bars, n_patterns)``.

    Returns
    -------
    pd.DataFrame  *or*  np.ndarray  depending on ``as_frame``.

    Raises
    ------
    ValueError
        If any string in ``patterns`` is not a recognised pattern name.
        The error message includes fuzzy-matched suggestions.
    TypeError
        If ``patterns`` is not a string keyword or a list.

    Examples
    --------
    All patterns as a signed DataFrame (the typical ML-feature use-case)::

        df = tap.batch_patterns(o, h, l, c)
        # shape: (n_bars, 106)  –  int8 columns, named after each pattern

    Only reversal classics, no trend gate on hammer::

        df = tap.batch_patterns(
            o, h, l, c,
            patterns=['hammer', 'engulfing_bullish', 'morning_star',
                      'three_white_soldiers', 'dark_cloud_cover'],
        )

    Raw numpy matrix (e.g. for direct sklearn input)::

        X = tap.batch_patterns(o, h, l, c, as_frame=False)  # shape (n, 106), int8

    Discovery::

        tap.list_patterns()              # all names
        tap.list_patterns('bullish')     # bullish only
        tap.PATTERNS                     # frozenset of all names
    """
    o, h, l, c = _f(o, h, l, c)

    # ── resolve which patterns to use ───────────────────────────────────
    _kw = {"all": PATTERNS, "directional": DIRECTIONAL,
           "bullish": BULLISH, "bearish": BEARISH,
           "non_directional": NON_DIRECTIONAL}

    if isinstance(patterns, str):
        if patterns not in _kw:
            raise ValueError(
                f"patterns keyword {patterns!r} not recognised. "
                f"Use one of {sorted(_kw)} or pass a list of pattern names."
            )
        names: List[str] = sorted(_kw[patterns])

    elif isinstance(patterns, (list, tuple)):
        names = list(patterns)
        unknown = set(names) - PATTERNS
        if unknown:
            suggestions = _fuzzy_suggest(unknown, PATTERNS)
            raise ValueError(
                f"Unknown pattern name(s):\n{suggestions}\n\n"
                f"Call  tap.list_patterns()  to see all "
                f"{len(PATTERNS)} available names."
            )
    else:
        raise TypeError(
            f"patterns must be a string keyword or a list of pattern names, "
            f"got {type(patterns).__name__!r}"
        )

    if not names:
        matrix = np.empty((len(o), 0), dtype=np.int8)
        if as_frame:
            try:
                import pandas as pd
                return pd.DataFrame(matrix)
            except ImportError:
                pass
        return matrix

    # ── compute all, then select ─────────────────────────────────────────
    raw   = scan_all(o, h, l, c)
    dtype = np.int8

    # stack preserves column order from `names`
    matrix = np.column_stack([raw[name].astype(dtype) for name in names])
    # shape: (n_bars, len(names))

    # ── format output ────────────────────────────────────────────────────
    if as_frame:
        try:
            import pandas as pd
            return pd.DataFrame(matrix, columns=names)
        except ImportError:
            import warnings
            warnings.warn(
                "pandas is not installed; returning numpy array. "
                "Install with: pip install pandas",
                UserWarning,
                stacklevel=2,
            )
    return matrix




def signals_at(o, h, l, c, idx: int) -> List[Tuple[str, str]]:
    """
    Return all patterns active at bar *idx* as ``(name, direction)`` pairs.

    Parameters
    ----------
    idx : int
        Bar index (supports negative indexing, e.g. ``-1`` for last bar).

    Returns
    -------
    list of (pattern_name, 'bullish' | 'bearish' | 'non_directional')
    sorted alphabetically by name.
    """
    all_sigs = scan_all(o, h, l, c)
    result = []
    for name, arr in all_sigs.items():
        if arr[idx]:
            if name in BULLISH:
                direction = "bullish"
            elif name in BEARISH:
                direction = "bearish"
            else:
                direction = "non_directional"
            result.append((name, direction))
    return sorted(result)


def to_dataframe(o, h, l, c):
    """
    Run all patterns and return a pandas DataFrame (one column per pattern).

    Returns a DataFrame of int8 columns (+1/-1/0).  Requires pandas >= 1.3.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is needed for to_dataframe().  pip install pandas"
        ) from exc
    data = scan_all(o, h, l, c)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Unified API: candlestick + chart patterns together
# ---------------------------------------------------------------------------

def scan_all_patterns(o, h, l, c,
                      v=None,
                      mode: str = 'confirmed',
                      window: int = 100,
                      pivot_n: int = 5,
                      pivot_pct=None):
    """
    Run **all** pattern detectors — candlestick (106) and chart (170+) —
    and return a single merged dict.

    Parameters
    ----------
    v    : volume array; if None, volume-based chart patterns are skipped.
    mode : 'confirmed' (default) or 'forming' — applies to chart patterns.

    Returns
    -------
    dict : pattern_name → int8 numpy array (+1 bull / -1 bear / 0 none)
    """
    from . import chart_patterns as _cp

    candle = scan_all(o, h, l, c)
    chart  = _cp.chart_scan_all(o, h, l, c, v=v, mode=mode,
                                  window=window, pivot_n=pivot_n,
                                  pivot_pct=pivot_pct)
    # Chart names that clash with candle names get a 'cp_' prefix
    merged = dict(candle)
    for name, arr in chart.items():
        key = f"cp_{name}" if name in merged else name
        merged[key] = arr
    return merged


def batch_all(
    o,
    h,
    l,
    c,
    v=None,
    candle_patterns: "str | list" = "all",
    chart_patterns:  "str | list" = "all",
    mode: str = "confirmed",
    window: int = 100,
    pivot_n: int = 5,
    pivot_pct=None,
    as_frame: bool = True,
):
    """
    Run both candlestick **and** chart pattern detectors in a single call
    and return a wide matrix — one column per pattern, one row per bar.

    This is the primary entry point when you want every signal in one place
    for downstream ML feature engineering or dashboard display.

    Parameters
    ----------
    o, h, l, c    : OHLC arrays (numpy or pandas).
    v             : volume array; supply to include volume-based chart
                    patterns (dome, U-shape, rising/falling trend, breakout).
                    If None those 5 patterns are skipped.
    candle_patterns : which candlestick patterns to include.
                    Same keywords as :func:`batch_patterns`:
                    ``'all'``, ``'bullish'``, ``'bearish'``,
                    ``'non_directional'``, ``'directional'``,
                    or an explicit list of names.
    chart_patterns  : which chart patterns to include.
                    Same keywords as :func:`chart_patterns.chart_batch`:
                    ``'all'``, ``'bullish'``, ``'bearish'``,
                    ``'bidirectional'``, ``'non_directional'``,
                    ``'directional'``, or an explicit list.
    mode          : ``'confirmed'`` (default) or ``'forming'`` — controls
                    when pivot-based chart patterns fire.
    window        : lookback in bars for chart pattern pivot scanning.
    pivot_n       : neighbourhood half-width for pivot detection.
    pivot_pct     : optional minimum % swing filter for pivot detection.
    as_frame      : True → pandas DataFrame with pattern names as columns.
                    False → numpy int8 array of shape (n_bars, n_patterns).

    Returns
    -------
    pd.DataFrame  *or*  np.ndarray

    Examples
    --------
    All patterns in one DataFrame — 295 from OHLC alone::

        df = tap.batch_all(o, h, l, c)

    With volume patterns (300 total)::

        df = tap.batch_all(o, h, l, c, v=volume)

    Bullish only from both modules::

        df = tap.batch_all(o, h, l, c,
                           candle_patterns='bullish',
                           chart_patterns='bullish')

    Explicit mix::

        df = tap.batch_all(o, h, l, c,
                           candle_patterns=['hammer', 'morning_star'],
                           chart_patterns=['double_bottom', 'hs_bottom'])
    """
    import difflib
    from . import chart_patterns as _cp

    o, h, l, c = _f(o, h, l, c)
    N = len(o)

    # ── candlestick side ──────────────────────────────────────────────────
    candle_mat = batch_patterns(o, h, l, c, patterns=candle_patterns,
                                 as_frame=False)
    _ckw = {"all": PATTERNS, "directional": DIRECTIONAL,
            "bullish": BULLISH, "bearish": BEARISH,
            "non_directional": NON_DIRECTIONAL}
    if isinstance(candle_patterns, str):
        candle_names = sorted(_ckw[candle_patterns])
    else:
        candle_names = list(candle_patterns)

    # ── chart side ───────────────────────────────────────────────────────
    chart_mat = _cp.chart_batch(o, h, l, c, v=v, patterns=chart_patterns,
                                 mode=mode, window=window,
                                 pivot_n=pivot_n, pivot_pct=pivot_pct,
                                 as_frame=False)
    _cckw = {"all": _cp.CHART_PATTERNS, "directional": _cp.DIRECTIONAL,
             "bullish": _cp.BULLISH, "bearish": _cp.BEARISH,
             "bidirectional": _cp.BIDIRECTIONAL,
             "non_directional": _cp.NON_DIRECTIONAL}
    if isinstance(chart_patterns, str):
        # Match what chart_batch actually returned (same filter)
        raw_cp = _cp.chart_scan_all(o, h, l, c, v=v, mode=mode,
                                     window=window, pivot_n=pivot_n,
                                     pivot_pct=pivot_pct)
        if chart_patterns in _cckw:
            chart_names = sorted(k for k in raw_cp
                                  if k in _cckw[chart_patterns])
        else:
            chart_names = sorted(raw_cp)
    else:
        chart_names = list(chart_patterns)

    # Prefix collisions with 'cp_'
    candle_set = set(candle_names)
    chart_cols = [f"cp_{n}" if n in candle_set else n for n in chart_names]

    # ── stack ─────────────────────────────────────────────────────────────
    parts, all_names = [], []
    if candle_mat.size:
        parts.append(candle_mat.astype(np.int8))
        all_names.extend(candle_names)
    if chart_mat.size:
        parts.append(chart_mat.astype(np.int8))
        all_names.extend(chart_cols)

    if not parts:
        matrix = np.empty((N, 0), dtype=np.int8)
    else:
        matrix = np.column_stack(parts) if len(parts) > 1 else parts[0]

    if as_frame:
        try:
            import pandas as pd
            return pd.DataFrame(matrix, columns=all_names)
        except ImportError:
            import warnings
            warnings.warn("pandas not installed; returning numpy array.",
                          stacklevel=2)
    return matrix


def list_all_patterns(
    direction: str = "all",
    module: str = "all",
) -> "list[str]":
    """
    Return a sorted list of pattern names across one or both modules.

    Parameters
    ----------
    direction : ``'all'``, ``'bullish'``, ``'bearish'``, ``'directional'``,
                ``'non_directional'``, ``'bidirectional'`` (chart only).
    module    : ``'all'`` (default) — both modules.
                ``'candles'``       — candlestick patterns only.
                ``'chart'``         — chart patterns only.

    Examples
    --------
    >>> tap.list_all_patterns(direction='bullish')
    >>> tap.list_all_patterns(module='chart', direction='bearish')
    """
    from . import chart_patterns as _cp

    candle_map = {
        "all":             PATTERNS,
        "bullish":         BULLISH,
        "bearish":         BEARISH,
        "directional":     DIRECTIONAL,
        "non_directional": NON_DIRECTIONAL,
    }
    chart_map = {
        "all":             _cp.CHART_PATTERNS,
        "bullish":         _cp.BULLISH,
        "bearish":         _cp.BEARISH,
        "directional":     _cp.DIRECTIONAL,
        "bidirectional":   _cp.BIDIRECTIONAL,
        "non_directional": _cp.NON_DIRECTIONAL,
    }

    if module not in ("all", "candles", "chart"):
        raise ValueError(f"module must be 'all', 'candles', or 'chart'; got {module!r}")
    if direction not in candle_map and direction not in chart_map:
        raise ValueError(f"direction={direction!r} not recognised.")

    names: "set[str]" = set()
    if module in ("all", "candles"):
        names |= candle_map.get(direction, set())
    if module in ("all", "chart"):
        names |= chart_map.get(direction, set())
    return sorted(names)


def net_score_all(o, h, l, c, v=None,
                  mode: str = "confirmed") -> np.ndarray:
    """
    Aggregate signal score across **all** patterns (candlestick + chart).

    Only patterns with a fixed directional bias contribute: candlestick
    BULLISH ∪ BEARISH and chart BULLISH ∪ BEARISH.  Three groups are
    deliberately excluded so the score stays sign-balanced and free of
    double counting:

    * **Non-directional** patterns (shape-only indecision/compression bars)
      carry no bull/bear bias.
    * **Bidirectional** chart patterns (e.g. ``symmetrical_triangle``) have
      no fixed prior bias.
    * **Combined** detectors (``two_b``, ``three_bar``, ``key_reversal`` …)
      are merges of their own ``*_bullish``/``*_bearish`` halves, which are
      already counted individually; summing the combine too would
      double-count the same event.

    Returns an **int16** array of length N.

    Parameters
    ----------
    v    : volume array; supply to include volume-based chart patterns.
    mode : 'confirmed' (default) or 'forming'.
    """
    from . import chart_patterns as _cp

    candle_dir = DIRECTIONAL
    chart_dir  = _cp.BULLISH | _cp.BEARISH

    all_sigs = scan_all_patterns(o, h, l, c, v=v, mode=mode)
    n        = len(next(iter(all_sigs.values())))
    score    = np.zeros(n, dtype=np.int16)
    for key, arr in all_sigs.items():
        name = key[3:] if key.startswith("cp_") else key   # undo collision prefix
        if name in candle_dir or name in chart_dir:
            score += arr.astype(np.int16)
    return score
