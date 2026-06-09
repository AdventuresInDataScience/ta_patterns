"""
ta_patterns
===========
Pure-numpy vectorised library of **300 pattern detectors**:
106 candlestick patterns + 194 chart patterns (geometric, harmonic,
busted, volume).  Five chart patterns are volume-based and only run when a
volume array is supplied, so 295 detectors are available from OHLC alone
and all 300 once volume is passed.

Design goals
------------
* **No look-forward bias** – flag at index i uses only bars 0…i
* **Point-in-time at the close** – flag is set when bar i closes
* **numpy + pandas friendly** – all OHLC inputs may be numpy arrays or
  pandas Series; the index is stripped automatically
* **Pure numpy internals** – fully vectorised, no TA-Lib dependency
* **Signed int8 output** – every detector returns +1/-1/0
* **Configurable thresholds** – all ratio parameters are keyword arguments

Quick start
-----------
::

    import ta_patterns as tap
    import pandas as pd

    df = pd.read_csv("ohlcv.csv")
    o, h, l, c = df.open, df.high, df.low, df.close   # pandas Series OK

    # single pattern → int8 array (+1 bull / -1 bear / 0 none)
    ha = tap.hammer(o, h, l, c)                    # strict (default)
    ha_loose = tap.hammer(o, h, l, c,
                         shadow_factor=1.5,
                         require_trend=False)     # shape only, no trend gate

    # dark cloud / piercing with custom penetration depth
    dc = tap.dark_cloud_cover(o, h, l, c, penetration=0.60)

    # scan candlestick patterns at once
    sigs   = tap.scan_all(o, h, l, c)             # {name: int8 array}
    score  = tap.net_score(o, h, l, c)            # int16 net directional score

    # candlestick + chart patterns together
    sigs   = tap.scan_all_patterns(o, h, l, c)            # OHLC only (295)
    sigs   = tap.scan_all_patterns(o, h, l, c, v=volume)  # incl. volume (300)
    score  = tap.net_score_all(o, h, l, c, v=volume)      # int16 net score

    # pandas DataFrame of every signal (one int8 column per pattern)
    df = tap.batch_all(o, h, l, c, v=volume)

Return values
-------------
All pattern functions return a **numpy int8 array**: +1 fired-and-bullish,
-1 fired-and-bearish, 0 not fired.  Non-directional patterns (plain doji,
high_wave, etc.) have no bull/bear bias and return +1 when the shape fires —
see the NON_DIRECTIONAL frozensets.  The net_score helpers sum only the
directional (bullish/bearish) patterns, so non-directional and
bidirectional patterns do not skew the score.
"""
__version__ = "1.1.1"

# ── single bar ───────────────────────────────────────────────────────────────
from .single import (
    doji, dragonfly_doji, gravestone_doji, long_legged_doji,
    northern_doji, southern_doji, rickshaw_man,
    marubozu_white, marubozu_black,
    closing_marubozu_white, closing_marubozu_black,
    opening_marubozu_white, opening_marubozu_black,
    hammer, hanging_man, shooting_star, takuri_line, high_wave,
    belt_hold_bullish, belt_hold_bearish,
    long_white_day, long_black_day, short_white_candle, short_black_candle,
    white_candle, black_candle,
    spinning_top_white, spinning_top_black,
)

# ── two bar ──────────────────────────────────────────────────────────────────
from .two_bar import (
    gapping_up_doji, gapping_down_doji,
    engulfing_bullish, engulfing_bearish,
    harami_bullish, harami_bearish, harami_cross_bullish, harami_cross_bearish,
    doji_star_bullish, doji_star_bearish,
    dark_cloud_cover, piercing_pattern,
    in_neck, on_neck, thrusting,
    kicking_bullish, kicking_bearish,
    tweezer_tops, tweezer_bottoms, matching_low,
    meeting_lines_bullish, meeting_lines_bearish,
    separating_lines_bullish, separating_lines_bearish,
    rising_window, falling_window, two_black_gapping,
    homing_pigeon, above_stomach, below_stomach,
    last_engulfing_top, last_engulfing_bottom,
    inverted_hammer, shooting_star_two_line,
)

# ── three bar ────────────────────────────────────────────────────────────────
from .three_bar import (
    morning_star, evening_star, morning_doji_star, evening_doji_star,
    three_white_soldiers, three_black_crows, identical_three_crows,
    abandoned_baby_bullish, abandoned_baby_bearish,
    three_inside_up, three_inside_down, three_outside_up, three_outside_down,
    advance_block, deliberation,
    two_crows, upside_gap_two_crows,
    downside_tasuki_gap, upside_tasuki_gap,
    three_stars_south, stick_sandwich, unique_three_river_bottom,
    tri_star_bullish, tri_star_bearish,
    downside_gap_three_methods, upside_gap_three_methods,
    side_by_side_white_lines_bullish, side_by_side_white_lines_bearish,
    hikkake_bullish, hikkake_bearish, collapse_doji_star,
)

# ── multi bar ────────────────────────────────────────────────────────────────
from .multi_bar import (
    three_line_strike_bullish, three_line_strike_bearish,
    concealing_baby_swallow,
    rising_three_methods, falling_three_methods,
    mat_hold, breakaway_bullish, breakaway_bearish, ladder_bottom,
    new_price_lines_up, new_price_lines_down,
    eight_new_price_lines, ten_new_price_lines,
    twelve_new_price_lines, thirteen_new_price_lines,
)

# ── scanner ──────────────────────────────────────────────────────────────────
from .scanner import (
    # candlestick-only (backward compatible)
    scan_all, scan_bullish, scan_bearish,
    net_score, batch_patterns, list_patterns,
    signals_at, to_dataframe,
    BULLISH, BEARISH, NON_DIRECTIONAL, DIRECTIONAL, PATTERNS,
    # unified candlestick + chart patterns API
    scan_all_patterns,
    batch_all,
    list_all_patterns,
    net_score_all,
)

# ── core utilities ────────────────────────────────────────────────────────────
from ._core import atr, avg_body, _to_np
