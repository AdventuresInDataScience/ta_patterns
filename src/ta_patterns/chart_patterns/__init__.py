"""
ta_patterns.chart_patterns
============================
Chart pattern detectors — 194 patterns across 6 categories (short-bar,
double/multi, classic-geometric, harmonic, volume, busted).  Five are
volume-based and only run when a volume array is supplied, so 189 are
available from OHLC alone.

Elliott Wave patterns are intentionally NOT implemented.  They require
subjective multi-level wave labelling that does not reduce to a
deterministic OHLC rule.  This is a deliberate design decision, not
an accidental omission.

All functions return **int8**: +1 bullish, -1 bearish, 0 none.
All functions accept ``mode='forming'`` or ``mode='confirmed'`` (default).

Quick start
-----------
::

    import ta_patterns.chart_patterns as cp

    # Single pattern
    sig = cp.double_top(o, h, l, c)                # int8 array

    # Batch
    df  = cp.chart_batch(o, h, l, c)               # DataFrame, all patterns
    df  = cp.chart_batch(o, h, l, c, patterns='bullish')
    df  = cp.chart_batch(o, h, l, c, v=volume)     # include volume patterns

    # Forming vs confirmed
    sig = cp.hs_top(o, h, l, c, mode='forming')    # fires at right shoulder
    sig = cp.hs_top(o, h, l, c, mode='confirmed')  # fires at neckline break

    # Looser / tighter pivots
    sig = cp.ascending_triangle(o, h, l, c, pivot_n=3, window=60)
    sig = cp.ascending_triangle(o, h, l, c, pivot_n=7, pivot_pct=0.02)
"""
from .short import (
    two_b_top, two_b_bottom, two_b,
    two_close_bullish, two_close_bearish, two_close,
    two_dance_bullish, two_dance_bearish, two_dance,
    two_did_bullish, two_did_bearish, two_did,
    two_tall_bullish, two_tall_bearish, two_tall,
    three_bar_bullish, three_bar_bearish, three_bar,
    three_day_compression, three_dc,
    three_lr_bullish, three_lr_bearish,
    three_lr_inverted_bullish, three_lr_inverted_bearish, three_lr,
    hook_reversal_bottom, hook_reversal_top, hook_reversal,
    key_reversal_bottom, key_reversal_top, key_reversal,
    key_reversal_v2_bullish, key_reversal_v2_bearish, key_reversal_v2,
    double_key_reversal_bullish, double_key_reversal_bearish, double_key_reversal,
    inside_day, outside_day_bullish, outside_day_bearish, outside_day,
    open_close_reversal_bottom, open_close_reversal_top, open_close_reversal,
    one_day_reversal_bottom, one_day_reversal_top, one_day_reversal,
    pivot_point_reversal_bottom, pivot_point_reversal_top, pivot_point_reversal,
    closing_price_reversal_bottom, closing_price_reversal_top, closing_price_reversal,
    one_two_three_bottom, one_two_three_top, one_two_three,
    fakey_bullish, fakey_bearish, fakey,
    turn_key_bullish, turn_key_bearish, turn_key,
    two_step_bullish, two_step_bearish, two_step,
    gap2h, gap2h_inverted, gap2h_combined,
    narrow_range_4, narrow_range_7,
    shark_32, elevator_stop, cloud_bank,
    weekly_reversal_upside, weekly_reversal_downside, weekly_reversal,
    wide_ranging_day_up, wide_ranging_day_down, wide_ranging_day,
    v_pivot, inverted_v_pivot,
    pipe_top, pipe_bottom,
    horn_top, horn_bottom,
    vertical_run_up, vertical_run_down,
    spikes, failure_swing,
    diving_board, pothole,
    flat_base, abc_correction,
    dead_cat_bounce, dead_cat_bounce_inv,
    carl_v_bullish, carl_v_bearish, carl_v,
)

from .double_multi import (
    double_top, double_top_adam_adam, double_top_adam_eve,
    double_top_eve_adam, double_top_eve_eve,
    double_bottom, double_bottom_adam_adam, double_bottom_adam_eve,
    double_bottom_eve_adam, double_bottom_eve_eve,
    ugly_double_bottom,
    triple_top, triple_bottom,
    big_m, big_w,
    three_peaks, three_valleys, three_peaks_domed_house,
    cat_ears,
)

from .classic import (
    ascending_triangle, descending_triangle, symmetrical_triangle,
    broadening_top, broadening_bottom,
    broadening_wedge_asc, broadening_wedge_desc,
    right_angle_broadening_asc, right_angle_broadening_desc,
    rising_wedge, falling_wedge,
    rectangle_top, rectangle_bottom,
    channel_asc, channel_desc,
    hs_top, hs_bottom, complex_hs_top, complex_hs_bottom,
    cup_with_handle, inverted_cup_with_handle,
    rounding_bottom, rounding_top,
    diamond_top, diamond_bottom,
    bump_and_run_top, bump_and_run_bottom,
    flag_bull, flag_bear, flag_high_tight,
    pennant_bull, pennant_bear,
    scallop_asc, scallop_asc_inv, scallop_desc, scallop_desc_inv,
    island_top, island_bottom, long_island,
    measured_move_up, measured_move_down,
    v_bottom, v_top, v_bottom_extended, v_top_extended,
    roof, inverted_roof, mountain,
    partial_rise, partial_decline,
)

from .harmonic import (
    abcd_bull, abcd_bear,
    gartley_bull, gartley_bear,
    bat_bull, bat_bear,
    butterfly_bull, butterfly_bear,
    crab_bull, crab_bear,
    wolfe_wave_bull, wolfe_wave_bear,
)

from .volume import (
    dome_shaped_volume, u_shaped_volume,
    rising_volume_trend, falling_volume_trend,
    volume_breakout_day,
)

from .busted import (
    busted_asc_triangle, busted_desc_triangle,
    busted_double_bottom, busted_double_top,
    busted_hs_bottom, busted_hs_top,
    busted_rectangle, busted_sym_triangle,
    busted_triple_bottom, busted_triple_top,
)

from .scanner import (
    chart_scan_all, chart_batch, chart_list_patterns,
    BULLISH, BEARISH, BIDIRECTIONAL, NON_DIRECTIONAL,
    DIRECTIONAL, CHART_PATTERNS,
)

# Feature cache controls (see ._memo) — pivot/trendline results are memoised
# across detectors within a scan.
from ._memo import clear_cache, cache_info
