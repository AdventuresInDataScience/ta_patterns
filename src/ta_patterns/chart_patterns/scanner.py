"""
ta_patterns.chart_patterns.scanner
=====================================
Scan all chart patterns over a price series.

Usage
-----
>>> from ta_patterns.chart_patterns import chart_scan_all, chart_batch
>>> signals = chart_scan_all(o, h, l, c)        # dict: name → int8 array
>>> df      = chart_batch(o, h, l, c)           # DataFrame, same length as input
>>> df_vol  = chart_batch(o, h, l, c, v=volume) # include volume patterns
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from . import short    as _sh
from . import double_multi as _dm
from . import classic  as _cl
from . import harmonic as _ha
from . import volume   as _vo
from . import busted   as _bu
from ._core import _a


def _f(*arrs):
    return tuple(_a(a) for a in arrs)


# ---------------------------------------------------------------------------
# Direction catalogues
# ---------------------------------------------------------------------------

BULLISH: frozenset = frozenset({
    # short
    "two_b_bottom","two_close_bullish","hook_reversal_bottom","key_reversal_bottom",
    "key_reversal_v2_bullish","double_key_reversal_bullish","one_day_reversal_bottom",
    "open_close_reversal_bottom","pivot_point_reversal_bottom",
    "closing_price_reversal_bottom","one_two_three_bottom","three_lr_bullish",
    "three_lr_inverted_bullish","fakey_bullish","turn_key_bullish","two_step_bullish",
    "gap2h","weekly_reversal_upside","wide_ranging_day_up","outside_day_bullish",
    "v_pivot","diving_board","pipe_bottom","horn_bottom","vertical_run_up",
    "dead_cat_bounce_inv","flat_base","abc_correction","carl_v_bullish",
    "two_dance_bullish","two_did_bullish","two_tall_bullish","three_bar_bullish",
    "pothole",
    # double_multi
    "double_bottom","double_bottom_adam_adam","double_bottom_adam_eve",
    "double_bottom_eve_adam","double_bottom_eve_eve","double_bottom_ugly",
    "triple_bottom","big_w","three_valleys",
    # classic
    "ascending_triangle","broadening_bottom","broadening_wedge_desc",
    "right_angle_broadening_desc","falling_wedge","rectangle_bottom","channel_desc",
    "hs_bottom","complex_hs_bottom","cup_with_handle","rounding_bottom",
    "diamond_bottom","bump_and_run_bottom","flag_bull","flag_high_tight",
    "pennant_bull","scallop_asc","scallop_desc_inv","island_bottom",
    "measured_move_up","v_bottom","v_bottom_extended","inverted_roof",
    "partial_decline",
    # harmonic
    "abcd_bull","gartley_bull","bat_bull","butterfly_bull","crab_bull",
    "wolfe_wave_bull",
    # volume
    "u_shaped_volume","rising_volume_trend",
    # busted
    "busted_double_top","busted_hs_top","busted_desc_triangle",
    "busted_triple_top",
})

BEARISH: frozenset = frozenset({
    # short
    "two_b_top","two_close_bearish","hook_reversal_top","key_reversal_top",
    "key_reversal_v2_bearish","double_key_reversal_bearish","one_day_reversal_top",
    "open_close_reversal_top","pivot_point_reversal_top",
    "closing_price_reversal_top","one_two_three_top","three_lr_bearish",
    "three_lr_inverted_bearish","fakey_bearish","turn_key_bearish","two_step_bearish",
    "gap2h_inverted","weekly_reversal_downside","wide_ranging_day_down",
    "outside_day_bearish","inverted_v_pivot","pipe_top","horn_top",
    "vertical_run_down","dead_cat_bounce","carl_v_bearish",
    "two_dance_bearish","two_did_bearish","two_tall_bearish","three_bar_bearish",
    # double_multi
    "double_top","double_top_adam_adam","double_top_adam_eve",
    "double_top_eve_adam","double_top_eve_eve","triple_top","big_m",
    "three_peaks","three_peaks_domed_house","cat_ears",
    # classic
    "descending_triangle","broadening_top","broadening_wedge_asc",
    "right_angle_broadening_asc","rising_wedge","rectangle_top","channel_asc",
    "hs_top","complex_hs_top","inverted_cup_with_handle","rounding_top",
    "diamond_top","bump_and_run_top","flag_bear","pennant_bear",
    "scallop_asc_inv","scallop_desc","island_top","measured_move_down",
    "v_top","v_top_extended","roof","mountain","partial_rise",
    # harmonic
    "abcd_bear","gartley_bear","bat_bear","butterfly_bear","crab_bear",
    "wolfe_wave_bear",
    # volume
    "dome_shaped_volume","falling_volume_trend",
    # busted
    "busted_double_bottom","busted_hs_bottom","busted_asc_triangle",
    "busted_triple_bottom",
})

BIDIRECTIONAL: frozenset = frozenset({
    "two_b","two_close","hook_reversal","key_reversal","key_reversal_v2",
    "double_key_reversal","one_day_reversal","open_close_reversal",
    "pivot_point_reversal","closing_price_reversal","one_two_three",
    "three_lr","fakey","turn_key","two_step","gap2h_combined",
    "weekly_reversal","wide_ranging_day","outside_day","carl_v",
    "two_dance","two_did","two_tall","three_bar",
    "failure_swing","spikes","long_island","symmetrical_triangle",
    "busted_rectangle","busted_sym_triangle",
})

NON_DIRECTIONAL: frozenset = frozenset({
    "inside_day","narrow_range_4","narrow_range_7","three_day_compression",
    "three_dc","shark_32","elevator_stop","cloud_bank",
    "volume_breakout_day",
})

DIRECTIONAL: frozenset = BULLISH | BEARISH | BIDIRECTIONAL
CHART_PATTERNS: frozenset = DIRECTIONAL | NON_DIRECTIONAL


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def chart_scan_all(o, h, l, c,
                   v: Optional[np.ndarray] = None,
                   mode: str = 'confirmed',
                   window: int = 100,
                   pivot_n: int = 5,
                   pivot_pct: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Run all chart pattern detectors over (o, h, l, c) arrays.

    Parameters
    ----------
    o, h, l, c : array-like (numpy or pandas)
    v          : volume array; if None, volume patterns are skipped.
    mode       : 'confirmed' (default) or 'forming'
    window     : lookback in bars for pivot-based patterns
    pivot_n    : pivot neighbourhood half-width
    pivot_pct  : optional minimum % swing filter for pivots

    Returns
    -------
    dict: pattern_name → int8 numpy array (+1/-1/0)
    """
    o, h, l, c = _f(o, h, l, c)
    N = len(o)
    kw = dict(mode=mode, window=window, pivot_n=pivot_n, pivot_pct=pivot_pct)
    sk = {}                                  # short kwargs (no mode/window/pivot)
    R: Dict[str, np.ndarray] = {}

    # ── short bar patterns ──────────────────────────────────────────────────
    R["two_b_top"]                    = _sh.two_b_top(o,h,l,c)
    R["two_b_bottom"]                 = _sh.two_b_bottom(o,h,l,c)
    R["two_close_bullish"]            = _sh.two_close_bullish(o,h,l,c)
    R["two_close_bearish"]            = _sh.two_close_bearish(o,h,l,c)
    R["two_dance_bullish"]            = _sh.two_dance_bullish(o,h,l,c)
    R["two_dance_bearish"]            = _sh.two_dance_bearish(o,h,l,c)
    R["two_did_bullish"]              = _sh.two_did_bullish(o,h,l,c)
    R["two_did_bearish"]              = _sh.two_did_bearish(o,h,l,c)
    R["two_tall_bullish"]             = _sh.two_tall_bullish(o,h,l,c)
    R["two_tall_bearish"]             = _sh.two_tall_bearish(o,h,l,c)
    R["three_bar_bullish"]            = _sh.three_bar_bullish(o,h,l,c)
    R["three_bar_bearish"]            = _sh.three_bar_bearish(o,h,l,c)
    R["three_day_compression"]        = _sh.three_day_compression(o,h,l,c)
    R["three_dc"]                     = _sh.three_dc(o,h,l,c)
    R["three_lr_bullish"]             = _sh.three_lr_bullish(o,h,l,c)
    R["three_lr_bearish"]             = _sh.three_lr_bearish(o,h,l,c)
    R["three_lr_inverted_bullish"]    = _sh.three_lr_inverted_bullish(o,h,l,c)
    R["three_lr_inverted_bearish"]    = _sh.three_lr_inverted_bearish(o,h,l,c)
    R["hook_reversal_bottom"]         = _sh.hook_reversal_bottom(o,h,l,c)
    R["hook_reversal_top"]            = _sh.hook_reversal_top(o,h,l,c)
    R["key_reversal_bottom"]          = _sh.key_reversal_bottom(o,h,l,c)
    R["key_reversal_top"]             = _sh.key_reversal_top(o,h,l,c)
    R["key_reversal_v2_bullish"]      = _sh.key_reversal_v2_bullish(o,h,l,c)
    R["key_reversal_v2_bearish"]      = _sh.key_reversal_v2_bearish(o,h,l,c)
    R["double_key_reversal_bullish"]  = _sh.double_key_reversal_bullish(o,h,l,c)
    R["double_key_reversal_bearish"]  = _sh.double_key_reversal_bearish(o,h,l,c)
    R["inside_day"]                   = _sh.inside_day(o,h,l,c)
    R["outside_day_bullish"]          = _sh.outside_day_bullish(o,h,l,c)
    R["outside_day_bearish"]          = _sh.outside_day_bearish(o,h,l,c)
    R["open_close_reversal_bottom"]   = _sh.open_close_reversal_bottom(o,h,l,c)
    R["open_close_reversal_top"]      = _sh.open_close_reversal_top(o,h,l,c)
    R["one_day_reversal_bottom"]      = _sh.one_day_reversal_bottom(o,h,l,c)
    R["one_day_reversal_top"]         = _sh.one_day_reversal_top(o,h,l,c)
    R["pivot_point_reversal_bottom"]  = _sh.pivot_point_reversal_bottom(o,h,l,c)
    R["pivot_point_reversal_top"]     = _sh.pivot_point_reversal_top(o,h,l,c)
    R["closing_price_reversal_bottom"]= _sh.closing_price_reversal_bottom(o,h,l,c)
    R["closing_price_reversal_top"]   = _sh.closing_price_reversal_top(o,h,l,c)
    R["one_two_three_bottom"]         = _sh.one_two_three_bottom(o,h,l,c)
    R["one_two_three_top"]            = _sh.one_two_three_top(o,h,l,c)
    R["fakey_bullish"]                = _sh.fakey_bullish(o,h,l,c)
    R["fakey_bearish"]                = _sh.fakey_bearish(o,h,l,c)
    R["turn_key_bullish"]             = _sh.turn_key_bullish(o,h,l,c)
    R["turn_key_bearish"]             = _sh.turn_key_bearish(o,h,l,c)
    R["two_step_bullish"]             = _sh.two_step_bullish(o,h,l,c)
    R["two_step_bearish"]             = _sh.two_step_bearish(o,h,l,c)
    R["gap2h"]                        = _sh.gap2h(o,h,l,c)
    R["gap2h_inverted"]               = _sh.gap2h_inverted(o,h,l,c)
    R["narrow_range_4"]               = _sh.narrow_range_4(o,h,l,c)
    R["narrow_range_7"]               = _sh.narrow_range_7(o,h,l,c)
    R["shark_32"]                     = _sh.shark_32(o,h,l,c)
    R["weekly_reversal_upside"]       = _sh.weekly_reversal_upside(o,h,l,c)
    R["weekly_reversal_downside"]     = _sh.weekly_reversal_downside(o,h,l,c)
    R["wide_ranging_day_up"]          = _sh.wide_ranging_day_up(o,h,l,c)
    R["wide_ranging_day_down"]        = _sh.wide_ranging_day_down(o,h,l,c)
    R["v_pivot"]                      = _sh.v_pivot(o,h,l,c)
    R["inverted_v_pivot"]             = _sh.inverted_v_pivot(o,h,l,c)
    R["diving_board"]                 = _sh.diving_board(o,h,l,c)
    R["pothole"]                      = _sh.pothole(o,h,l,c)
    R["pipe_top"]                     = _sh.pipe_top(o,h,l,c)
    R["pipe_bottom"]                  = _sh.pipe_bottom(o,h,l,c)
    R["horn_top"]                     = _sh.horn_top(o,h,l,c)
    R["horn_bottom"]                  = _sh.horn_bottom(o,h,l,c)
    R["vertical_run_up"]              = _sh.vertical_run_up(o,h,l,c)
    R["vertical_run_down"]            = _sh.vertical_run_down(o,h,l,c)
    R["spikes"]                       = _sh.spikes(o,h,l,c)
    R["failure_swing"]                = _sh.failure_swing(o,h,l,c)
    R["elevator_stop"]                = _sh.elevator_stop(o,h,l,c)
    R["cloud_bank"]                   = _sh.cloud_bank(o,h,l,c)
    R["flat_base"]                    = _sh.flat_base(o,h,l,c)
    R["abc_correction"]               = _sh.abc_correction(o,h,l,c)
    R["dead_cat_bounce"]              = _sh.dead_cat_bounce(o,h,l,c)
    R["dead_cat_bounce_inv"]          = _sh.dead_cat_bounce_inv(o,h,l,c)
    R["carl_v_bullish"]               = _sh.carl_v_bullish(o,h,l,c)
    R["carl_v_bearish"]               = _sh.carl_v_bearish(o,h,l,c)

    # ── combined (bidirectional) short detectors ────────────────────────────
    # Each merges its bullish/bearish halves into one signed series. Kept
    # alongside the split versions so callers can use either representation.
    R["two_b"]                        = _sh.two_b(o,h,l,c)
    R["two_close"]                    = _sh.two_close(o,h,l,c)
    R["two_dance"]                    = _sh.two_dance(o,h,l,c)
    R["two_did"]                      = _sh.two_did(o,h,l,c)
    R["two_tall"]                     = _sh.two_tall(o,h,l,c)
    R["three_bar"]                    = _sh.three_bar(o,h,l,c)
    R["three_lr"]                     = _sh.three_lr(o,h,l,c)
    R["hook_reversal"]                = _sh.hook_reversal(o,h,l,c)
    R["key_reversal"]                 = _sh.key_reversal(o,h,l,c)
    R["key_reversal_v2"]              = _sh.key_reversal_v2(o,h,l,c)
    R["double_key_reversal"]          = _sh.double_key_reversal(o,h,l,c)
    R["outside_day"]                  = _sh.outside_day(o,h,l,c)
    R["open_close_reversal"]          = _sh.open_close_reversal(o,h,l,c)
    R["one_day_reversal"]             = _sh.one_day_reversal(o,h,l,c)
    R["pivot_point_reversal"]         = _sh.pivot_point_reversal(o,h,l,c)
    R["closing_price_reversal"]       = _sh.closing_price_reversal(o,h,l,c)
    R["one_two_three"]                = _sh.one_two_three(o,h,l,c)
    R["fakey"]                        = _sh.fakey(o,h,l,c)
    R["turn_key"]                     = _sh.turn_key(o,h,l,c)
    R["two_step"]                     = _sh.two_step(o,h,l,c)
    R["gap2h_combined"]               = _sh.gap2h_combined(o,h,l,c)
    R["weekly_reversal"]              = _sh.weekly_reversal(o,h,l,c)
    R["wide_ranging_day"]             = _sh.wide_ranging_day(o,h,l,c)
    R["carl_v"]                       = _sh.carl_v(o,h,l,c)

    # ── double / multi ──────────────────────────────────────────────────────
    import inspect as _insp
    def _call_dm(fn, *args, **kw):
        """Call fn, dropping kwargs its signature does not accept."""
        sig = _insp.signature(fn)
        accepted = set(sig.parameters)
        filtered = {k: v for k, v in kw.items() if k in accepted}
        return fn(*args, **filtered)

    for name, fn, dir_ in [
        ("double_top",           _dm.double_top,           'confirmed'),
        ("double_top_adam_adam", _dm.double_top_adam_adam, 'confirmed'),
        ("double_top_adam_eve",  _dm.double_top_adam_eve,  'confirmed'),
        ("double_top_eve_adam",  _dm.double_top_eve_adam,  'confirmed'),
        ("double_top_eve_eve",   _dm.double_top_eve_eve,   'confirmed'),
        ("double_bottom",        _dm.double_bottom,        'confirmed'),
        ("double_bottom_adam_adam", _dm.double_bottom_adam_adam, 'confirmed'),
        ("double_bottom_adam_eve",  _dm.double_bottom_adam_eve,  'confirmed'),
        ("double_bottom_eve_adam",  _dm.double_bottom_eve_adam,  'confirmed'),
        ("double_bottom_eve_eve",   _dm.double_bottom_eve_eve,   'confirmed'),
        ("double_bottom_ugly",   _dm.ugly_double_bottom,   'confirmed'),
        ("triple_top",           _dm.triple_top,           'confirmed'),
        ("triple_bottom",        _dm.triple_bottom,        'confirmed'),
        ("big_m",                _dm.big_m,                'confirmed'),
        ("big_w",                _dm.big_w,                'confirmed'),
        ("three_peaks",          _dm.three_peaks,          'confirmed'),
        ("three_valleys",        _dm.three_valleys,        'confirmed'),
        ("three_peaks_domed_house", _dm.three_peaks_domed_house, 'confirmed'),
        ("cat_ears",             _dm.cat_ears,             'confirmed'),
    ]:
        R[name] = _call_dm(fn, o, h, l, c, mode=mode, window=window,
                         pivot_n=pivot_n, pivot_pct=pivot_pct)

    # ── classic geometric ───────────────────────────────────────────────────
    classic_fns = [
        ("ascending_triangle",          _cl.ascending_triangle),
        ("descending_triangle",         _cl.descending_triangle),
        ("symmetrical_triangle",        _cl.symmetrical_triangle),
        ("broadening_top",              _cl.broadening_top),
        ("broadening_bottom",           _cl.broadening_bottom),
        ("broadening_wedge_asc",        _cl.broadening_wedge_asc),
        ("broadening_wedge_desc",       _cl.broadening_wedge_desc),
        ("right_angle_broadening_asc",  _cl.right_angle_broadening_asc),
        ("right_angle_broadening_desc", _cl.right_angle_broadening_desc),
        ("rising_wedge",                _cl.rising_wedge),
        ("falling_wedge",               _cl.falling_wedge),
        ("rectangle_top",               _cl.rectangle_top),
        ("rectangle_bottom",            _cl.rectangle_bottom),
        ("channel_asc",                 _cl.channel_asc),
        ("channel_desc",                _cl.channel_desc),
        ("hs_top",                      _cl.hs_top),
        ("hs_bottom",                   _cl.hs_bottom),
        ("complex_hs_top",              _cl.complex_hs_top),
        ("complex_hs_bottom",           _cl.complex_hs_bottom),
        ("cup_with_handle",             _cl.cup_with_handle),
        ("inverted_cup_with_handle",    _cl.inverted_cup_with_handle),
        ("rounding_bottom",             _cl.rounding_bottom),
        ("rounding_top",                _cl.rounding_top),
        ("diamond_top",                 _cl.diamond_top),
        ("diamond_bottom",              _cl.diamond_bottom),
        ("bump_and_run_top",            _cl.bump_and_run_top),
        ("bump_and_run_bottom",         _cl.bump_and_run_bottom),
        ("flag_bull",                   _cl.flag_bull),
        ("flag_bear",                   _cl.flag_bear),
        ("flag_high_tight",             _cl.flag_high_tight),
        ("pennant_bull",                _cl.pennant_bull),
        ("pennant_bear",                _cl.pennant_bear),
        ("scallop_asc",                 _cl.scallop_asc),
        ("scallop_asc_inv",             _cl.scallop_asc_inv),
        ("scallop_desc",                _cl.scallop_desc),
        ("scallop_desc_inv",            _cl.scallop_desc_inv),
        ("island_top",                  _cl.island_top),
        ("island_bottom",               _cl.island_bottom),
        ("long_island",                 _cl.long_island),
        ("measured_move_up",            _cl.measured_move_up),
        ("measured_move_down",          _cl.measured_move_down),
        ("v_bottom",                    _cl.v_bottom),
        ("v_top",                       _cl.v_top),
        ("v_bottom_extended",           _cl.v_bottom_extended),
        ("v_top_extended",              _cl.v_top_extended),
        ("roof",                        _cl.roof),
        ("inverted_roof",               _cl.inverted_roof),
        ("mountain",                    _cl.mountain),
        ("partial_rise",                _cl.partial_rise),
        ("partial_decline",             _cl.partial_decline),
    ]
    for name, fn in classic_fns:
        R[name] = _call_dm(fn, o, h, l, c, mode=mode, window=window,
                           pivot_n=pivot_n, pivot_pct=pivot_pct)

    # ── harmonic ────────────────────────────────────────────────────────────
    harm_fns = [
        ("abcd_bull",      _ha.abcd_bull),
        ("abcd_bear",      _ha.abcd_bear),
        ("gartley_bull",   _ha.gartley_bull),
        ("gartley_bear",   _ha.gartley_bear),
        ("bat_bull",       _ha.bat_bull),
        ("bat_bear",       _ha.bat_bear),
        ("butterfly_bull", _ha.butterfly_bull),
        ("butterfly_bear", _ha.butterfly_bear),
        ("crab_bull",      _ha.crab_bull),
        ("crab_bear",      _ha.crab_bear),
        ("wolfe_wave_bull",_ha.wolfe_wave_bull),
        ("wolfe_wave_bear",_ha.wolfe_wave_bear),
    ]
    for name, fn in harm_fns:
        R[name] = _call_dm(fn, o, h, l, c, mode=mode, window=window,
                           pivot_n=pivot_n, pivot_pct=pivot_pct)

    # ── volume (only if v provided) ──────────────────────────────────────────
    if v is not None:
        v_arr = _a(v)
        R["dome_shaped_volume"]   = _vo.dome_shaped_volume(o, h, l, c, v_arr)
        R["u_shaped_volume"]      = _vo.u_shaped_volume(o, h, l, c, v_arr)
        R["rising_volume_trend"]  = _vo.rising_volume_trend(o, h, l, c, v_arr)
        R["falling_volume_trend"] = _vo.falling_volume_trend(o, h, l, c, v_arr)
        R["volume_breakout_day"]  = _vo.volume_breakout_day(o, h, l, c, v_arr)

    # ── busted ──────────────────────────────────────────────────────────────
    bust_fns = [
        ("busted_asc_triangle",  _bu.busted_asc_triangle),
        ("busted_desc_triangle", _bu.busted_desc_triangle),
        ("busted_double_bottom", _bu.busted_double_bottom),
        ("busted_double_top",    _bu.busted_double_top),
        ("busted_hs_bottom",     _bu.busted_hs_bottom),
        ("busted_hs_top",        _bu.busted_hs_top),
        ("busted_rectangle",     _bu.busted_rectangle),
        ("busted_sym_triangle",  _bu.busted_sym_triangle),
        ("busted_triple_bottom", _bu.busted_triple_bottom),
        ("busted_triple_top",    _bu.busted_triple_top),
    ]
    for name, fn in bust_fns:
        R[name] = _call_dm(fn, o, h, l, c, mode=mode, window=window,
                           pivot_n=pivot_n, pivot_pct=pivot_pct)

    # ── length normalisation ────────────────────────────────────────────────
    for k in list(R.keys()):
        v_arr_k = R[k]
        if len(v_arr_k) == N:
            continue
        fixed = np.zeros(N, dtype=np.int8)
        take  = min(len(v_arr_k), N)
        if take > 0:
            fixed[N - take:] = v_arr_k[:take]
        R[k] = fixed

    return R


# ---------------------------------------------------------------------------
# chart_batch  (DataFrame / numpy matrix convenience)
# ---------------------------------------------------------------------------

def chart_batch(o, h, l, c,
                v=None,
                patterns: "str | List[str]" = "all",
                mode: str = 'confirmed',
                window: int = 100,
                pivot_n: int = 5,
                pivot_pct=None,
                as_frame: bool = True):
    """
    Run selected chart patterns and return a 2-D result.

    Parameters
    ----------
    patterns : 'all', 'bullish', 'bearish', 'bidirectional',
               'directional', 'non_directional', or list of names.
    mode     : 'confirmed' or 'forming'.
    as_frame : True → pandas DataFrame; False → numpy int8 array.
    """
    import difflib
    raw = chart_scan_all(o, h, l, c, v=v, mode=mode, window=window,
                          pivot_n=pivot_n, pivot_pct=pivot_pct)

    _kw = {
        "all":             CHART_PATTERNS,
        "directional":     DIRECTIONAL,
        "bullish":         BULLISH,
        "bearish":         BEARISH,
        "bidirectional":   BIDIRECTIONAL,
        "non_directional": NON_DIRECTIONAL,
    }

    if isinstance(patterns, str):
        if patterns not in _kw:
            raise ValueError(
                f"patterns={patterns!r} not recognised. "
                f"Use one of {sorted(_kw)} or a list of names."
            )
        names = sorted(k for k in raw if k in _kw[patterns])
    elif isinstance(patterns, (list, tuple)):
        unknown = [n for n in patterns if n not in raw]
        if unknown:
            suggestions = {u: difflib.get_close_matches(u, sorted(raw), n=3)
                           for u in unknown}
            lines = "\n".join(f"  '{u}' → {s}" for u, s in suggestions.items())
            raise ValueError(f"Unknown pattern(s):\n{lines}\n"
                             f"Call chart_list_patterns() for available names.")
        names = list(patterns)
    else:
        raise TypeError(f"patterns must be str or list, got {type(patterns).__name__!r}")

    if not names:
        matrix = np.empty((len(_a(c)), 0), dtype=np.int8)
        if as_frame:
            try:
                import pandas as pd
                return pd.DataFrame(matrix)
            except ImportError:
                pass
        return matrix

    matrix = np.column_stack([raw[n].astype(np.int8) for n in names])
    if as_frame:
        try:
            import pandas as pd
            return pd.DataFrame(matrix, columns=names)
        except ImportError:
            import warnings
            warnings.warn("pandas not installed; returning numpy array.", stacklevel=2)
    return matrix


def chart_list_patterns(direction: str = "all") -> List[str]:
    """Return a sorted list of all available chart pattern names."""
    _map = {
        "all":             CHART_PATTERNS,
        "directional":     DIRECTIONAL,
        "bullish":         BULLISH,
        "bearish":         BEARISH,
        "bidirectional":   BIDIRECTIONAL,
        "non_directional": NON_DIRECTIONAL,
    }
    if direction not in _map:
        raise ValueError(f"direction must be one of {sorted(_map)}")
    return sorted(_map[direction])
