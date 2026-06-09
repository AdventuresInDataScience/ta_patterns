# Core concepts

## Signal encoding

Every detector returns a NumPy `int8` array the same length as the input:

| Value | Meaning |
|---|---|
| `+1` | pattern fired with a **bullish** bias |
| `-1` | pattern fired with a **bearish** bias |
| `0`  | pattern did not fire on this bar |

A pattern's *classification* (bullish / bearish / bidirectional /
non-directional) constrains which values it can emit:

- **Bullish** patterns emit only `+1` / `0`.
- **Bearish** patterns emit only `-1` / `0`.
- **Non-directional** patterns (e.g. plain `doji`, `high_wave`) describe a
  shape with no directional bias and emit `+1` / `0` when the shape appears.
- **Bidirectional** chart patterns (e.g. `symmetrical_triangle`) can resolve
  either way and may emit `+1`, `-1`, or `0` depending on the break.

This contract is checked for the whole catalog in
`tests/test_invariants.py`.

## Point-in-time (no look-ahead)

A signal at bar *i* is computed using only bars `0..i`. The value a detector
assigns to bar *i* is identical whether you run it on the full series or on
the prefix that ends at *i*. This is what makes the library safe for
backtesting: a pattern can never "see" a future bar that confirms it.

For pattern types that are only meaningful once confirmed (a swing high, a
breakout, a busted reversal), the signal is placed on the **confirmation
bar**, not retroactively on the earlier extreme. `tests/test_no_lookahead.py`
verifies this across every detector.

## Pivots, confirmation delay, and `pivot_n`

Chart patterns are built on swing highs/lows ("pivots"). A pivot at bar *t* is
only *confirmable* once `pivot_n` later bars exist to prove it was a local
extreme — so internally a pivot is flagged at its **confirmation bar**
(`t + pivot_n`), never at *t* itself. This preserves the point-in-time
guarantee.

`pivot_n` (default 5 for most chart detectors) trades sensitivity against
noise: smaller values detect more, smaller swings; larger values require
broader, cleaner swings. Some detectors also accept `pivot_pct` to require a
minimum percentage move for a pivot to count.

```python
cp.double_top(o, h, l, c, pivot_n=3, window=60)   # more, smaller tops
cp.double_top(o, h, l, c, pivot_n=7)              # only broad tops
```

> Implementation note: when a detector needs the *price* at a pivot it reads
> it at the true swing bar (`confirmation − pivot_n`), while still using the
> confirmation bar for windowing — so equal-peak and shape tests use the real
> extreme, not a price several bars late.

## `mode='confirmed'` vs `mode='forming'`

Chart detectors accept a `mode`:

- `confirmed` (default) — the signal fires when the pattern *completes*
  (e.g. a double top confirms on a close below the intervening valley).
- `forming` — the signal fires earlier, as the geometry falls into place
  (e.g. at the second peak), before the confirming break. Useful for alerts;
  riskier as a trade trigger.

## Adam vs Eve extremes

Some double-top/bottom variants distinguish the *shape* of each extreme:

- **Adam** — narrow, pointed, often a one- or two-bar spike.
- **Eve** — wide, rounded.

The library scores each extreme with a scale-free "sharpness" metric
(normalised by the local high-low range, so peaks and troughs are directly
comparable) and labels it Adam or Eve against a threshold. The four
combinations are exposed as `double_top_adam_adam`, `double_top_adam_eve`,
`double_top_eve_adam`, `double_top_eve_eve` (and the `double_bottom_*`
mirror). The Adam/Eve split is a heuristic, not an exact vendor rule.

## Scoring: `net_score` and `net_score_all`

The scoring helpers sum **only directional** patterns (bullish `+1`, bearish
`-1`). Three groups are deliberately excluded so the score stays
sign-balanced and free of double counting:

- non-directional patterns (no bull/bear bias),
- bidirectional chart patterns (no fixed prior bias),
- "combined" detectors such as `two_b` / `key_reversal` that merely merge
  their own `*_bullish` / `*_bearish` halves, which are already counted.

So a lone `doji` scores `0`, not a spurious `+1`.

## Volume patterns

Five chart patterns depend on volume. They run only when a `v` array is
passed, which is why the catalog totals **295 from OHLC alone** and **300 once
volume is supplied**.
