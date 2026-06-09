# Pattern catalog

All **300** detectors, grouped by module and direction. Names are the exact
function names importable from `ta_patterns` (candlesticks) or
`ta_patterns.chart_patterns` (chart patterns).

| Module | Bullish | Bearish | Bidirectional | Non-directional | Total |
|---|---|---|---|---|---|
| Candlestick | 51 | 49 | — | 6 | 106 |
| Chart | 79 | 76 | 30 | 9 | 194 |
| **All** | | | | | **300** |

> Bidirectional and non-directional patterns are excluded from the
> `net_score` helpers; see [concepts](concepts.md#scoring-net_score-and-net_score_all).

## Candlesticks (106)
### Candlesticks — bullish (51)

`abandoned_baby_bullish`, `above_stomach`, `belt_hold_bullish`, `breakaway_bullish`, `closing_marubozu_white`, `concealing_baby_swallow`, `doji_star_bullish`, `dragonfly_doji`, `eight_new_price_lines`, `engulfing_bullish`, `gapping_up_doji`, `hammer`, `harami_bullish`, `harami_cross_bullish`, `hikkake_bullish`, `homing_pigeon`, `inverted_hammer`, `kicking_bullish`, `ladder_bottom`, `last_engulfing_bottom`, `long_white_day`, `marubozu_white`, `mat_hold`, `matching_low`, `meeting_lines_bullish`, `morning_doji_star`, `morning_star`, `opening_marubozu_white`, `piercing_pattern`, `rising_three_methods`, `rising_window`, `separating_lines_bullish`, `side_by_side_white_lines_bullish`, `southern_doji`, `spinning_top_white`, `stick_sandwich`, `takuri_line`, `ten_new_price_lines`, `thirteen_new_price_lines`, `three_inside_up`, `three_line_strike_bullish`, `three_outside_up`, `three_stars_south`, `three_white_soldiers`, `tri_star_bullish`, `tweezer_bottoms`, `twelve_new_price_lines`, `unique_three_river_bottom`, `upside_gap_three_methods`, `upside_tasuki_gap`, `white_candle`


### Candlesticks — bearish (49)

`abandoned_baby_bearish`, `advance_block`, `below_stomach`, `belt_hold_bearish`, `black_candle`, `breakaway_bearish`, `closing_marubozu_black`, `collapse_doji_star`, `dark_cloud_cover`, `deliberation`, `doji_star_bearish`, `downside_gap_three_methods`, `downside_tasuki_gap`, `engulfing_bearish`, `evening_doji_star`, `evening_star`, `falling_three_methods`, `falling_window`, `gapping_down_doji`, `gravestone_doji`, `hanging_man`, `harami_bearish`, `harami_cross_bearish`, `hikkake_bearish`, `identical_three_crows`, `in_neck`, `kicking_bearish`, `last_engulfing_top`, `long_black_day`, `marubozu_black`, `meeting_lines_bearish`, `northern_doji`, `on_neck`, `opening_marubozu_black`, `separating_lines_bearish`, `shooting_star`, `shooting_star_two_line`, `side_by_side_white_lines_bearish`, `spinning_top_black`, `three_black_crows`, `three_inside_down`, `three_line_strike_bearish`, `three_outside_down`, `thrusting`, `tri_star_bearish`, `tweezer_tops`, `two_black_gapping`, `two_crows`, `upside_gap_two_crows`


### Candlesticks — non-directional (6)

`doji`, `high_wave`, `long_legged_doji`, `rickshaw_man`, `short_black_candle`, `short_white_candle`



## Chart patterns (194)
### Chart — bullish (79)

`abc_correction`, `abcd_bull`, `ascending_triangle`, `bat_bull`, `big_w`, `broadening_bottom`, `broadening_wedge_desc`, `bump_and_run_bottom`, `busted_desc_triangle`, `busted_double_top`, `busted_hs_top`, `busted_triple_top`, `butterfly_bull`, `carl_v_bullish`, `channel_desc`, `closing_price_reversal_bottom`, `complex_hs_bottom`, `crab_bull`, `cup_with_handle`, `dead_cat_bounce_inv`, `diamond_bottom`, `diving_board`, `double_bottom`, `double_bottom_adam_adam`, `double_bottom_adam_eve`, `double_bottom_eve_adam`, `double_bottom_eve_eve`, `double_bottom_ugly`, `double_key_reversal_bullish`, `fakey_bullish`, `falling_wedge`, `flag_bull`, `flag_high_tight`, `flat_base`, `gap2h`, `gartley_bull`, `hook_reversal_bottom`, `horn_bottom`, `hs_bottom`, `inverted_roof`, `island_bottom`, `key_reversal_bottom`, `key_reversal_v2_bullish`, `measured_move_up`, `one_day_reversal_bottom`, `one_two_three_bottom`, `open_close_reversal_bottom`, `outside_day_bullish`, `partial_decline`, `pennant_bull`, `pipe_bottom`, `pivot_point_reversal_bottom`, `pothole`, `rectangle_bottom`, `right_angle_broadening_desc`, `rising_volume_trend`, `rounding_bottom`, `scallop_asc`, `scallop_desc_inv`, `three_bar_bullish`, `three_lr_bullish`, `three_lr_inverted_bullish`, `three_valleys`, `triple_bottom`, `turn_key_bullish`, `two_b_bottom`, `two_close_bullish`, `two_dance_bullish`, `two_did_bullish`, `two_step_bullish`, `two_tall_bullish`, `u_shaped_volume`, `v_bottom`, `v_bottom_extended`, `v_pivot`, `vertical_run_up`, `weekly_reversal_upside`, `wide_ranging_day_up`, `wolfe_wave_bull`


### Chart — bearish (76)

`abcd_bear`, `bat_bear`, `big_m`, `broadening_top`, `broadening_wedge_asc`, `bump_and_run_top`, `busted_asc_triangle`, `busted_double_bottom`, `busted_hs_bottom`, `busted_triple_bottom`, `butterfly_bear`, `carl_v_bearish`, `cat_ears`, `channel_asc`, `closing_price_reversal_top`, `complex_hs_top`, `crab_bear`, `dead_cat_bounce`, `descending_triangle`, `diamond_top`, `dome_shaped_volume`, `double_key_reversal_bearish`, `double_top`, `double_top_adam_adam`, `double_top_adam_eve`, `double_top_eve_adam`, `double_top_eve_eve`, `fakey_bearish`, `falling_volume_trend`, `flag_bear`, `gap2h_inverted`, `gartley_bear`, `hook_reversal_top`, `horn_top`, `hs_top`, `inverted_cup_with_handle`, `inverted_v_pivot`, `island_top`, `key_reversal_top`, `key_reversal_v2_bearish`, `measured_move_down`, `mountain`, `one_day_reversal_top`, `one_two_three_top`, `open_close_reversal_top`, `outside_day_bearish`, `partial_rise`, `pennant_bear`, `pipe_top`, `pivot_point_reversal_top`, `rectangle_top`, `right_angle_broadening_asc`, `rising_wedge`, `roof`, `rounding_top`, `scallop_asc_inv`, `scallop_desc`, `three_bar_bearish`, `three_lr_bearish`, `three_lr_inverted_bearish`, `three_peaks`, `three_peaks_domed_house`, `triple_top`, `turn_key_bearish`, `two_b_top`, `two_close_bearish`, `two_dance_bearish`, `two_did_bearish`, `two_step_bearish`, `two_tall_bearish`, `v_top`, `v_top_extended`, `vertical_run_down`, `weekly_reversal_downside`, `wide_ranging_day_down`, `wolfe_wave_bear`


### Chart — bidirectional (30)

`busted_rectangle`, `busted_sym_triangle`, `carl_v`, `closing_price_reversal`, `double_key_reversal`, `failure_swing`, `fakey`, `gap2h_combined`, `hook_reversal`, `key_reversal`, `key_reversal_v2`, `long_island`, `one_day_reversal`, `one_two_three`, `open_close_reversal`, `outside_day`, `pivot_point_reversal`, `spikes`, `symmetrical_triangle`, `three_bar`, `three_lr`, `turn_key`, `two_b`, `two_close`, `two_dance`, `two_did`, `two_step`, `two_tall`, `weekly_reversal`, `wide_ranging_day`


### Chart — non-directional (9)

`cloud_bank`, `elevator_stop`, `inside_day`, `narrow_range_4`, `narrow_range_7`, `shark_32`, `three_day_compression`, `three_dc`, `volume_breakout_day`
