# API reference

All names below are importable from the top-level `ta_patterns` package
unless stated otherwise. Chart-only helpers live under
`ta_patterns.chart_patterns` (aliased `cp` here).

## Conventions

- `o, h, l, c` — open/high/low/close, NumPy arrays or pandas Series.
- `v` — volume (optional; required only for the 5 volume patterns).
- Detector return value — `numpy.int8` array, values in `{-1, 0, 1}`.

---

## Individual detectors

Every pattern is a function `name(o, h, l, c, **kwargs) -> np.ndarray`.
Candlestick detectors take only `o, h, l, c` plus shape thresholds; chart
detectors additionally accept the common chart parameters below. See the
[pattern catalog](patterns_catalog.md) for the full list of names.

### Common candlestick parameters (vary by pattern)

| Parameter | Meaning |
|---|---|
| `penetration` | how deep one body must close into the prior body (e.g. `dark_cloud_cover`, `piercing_pattern`) |
| `shadow_factor` | required shadow-to-body ratio (e.g. `hammer`) |
| `require_trend` | gate the shape on a prior trend (set `False` for shape-only) |
| `tol` | tolerance for "equal"/"marubozu" style comparisons |

### Common chart parameters

| Parameter | Default | Meaning |
|---|---|---|
| `mode` | `'confirmed'` | `'confirmed'` (fires on completion) or `'forming'` |
| `window` | `100` | look-back window (bars) for the pattern geometry |
| `pivot_n` | `5` | bars each side required to confirm a swing pivot |
| `pivot_pct` | `None` | optional minimum % move for a pivot to count |

Harmonic detectors also take `fib_tol` (Fibonacci-ratio tolerance, default
~0.05–0.06). Busted detectors take `reversal_bars` and `reversal_pct`
(how fast and how far the failed breakout must reverse). Flag/pole detectors
take `pole_bars`, `min_pole`, `max_retrace`.

---

## Scanners and aggregation

### `scan_all_patterns(o, h, l, c, v=None, mode='confirmed', window=100, pivot_n=5, pivot_pct=None)`

Run **all** detectors (candlestick + chart) and return a single dict
`{name: int8 array}`. Without `v`, the 5 volume patterns are skipped (295
entries); with `v`, all 300 run. Chart names that collide with candlestick
names are prefixed `cp_`.

### `scan_all(o, h, l, c)` · `scan_bullish(...)` · `scan_bearish(...)`

Candlestick-only scanners returning `{name: int8 array}`. `scan_bullish` /
`scan_bearish` restrict to that direction.

### `chart_scan_all(o, h, l, c, v=None, mode='confirmed', window=100, pivot_n=5, pivot_pct=None)`

*(in `ta_patterns.chart_patterns`)* — chart-only equivalent of
`scan_all_patterns`.

### `net_score_all(o, h, l, c, v=None, mode='confirmed') -> np.ndarray`

Per-bar sum over **directional** patterns only (bullish `+1`, bearish `-1`),
across candlestick and chart. Non-directional, bidirectional, and combined
detectors are excluded (see [concepts](concepts.md#scoring-net_score-and-net_score_all)).

### `net_score(o, h, l, c) -> np.ndarray`

Candlestick-only net directional score.

### `batch_all(o, h, l, c, v=None, candle_patterns='all', chart_patterns='all', ...) -> pandas.DataFrame`

Wide DataFrame, one `int8` column per pattern. `candle_patterns` /
`chart_patterns` accept `'all'`, a direction string, or an explicit list of
names.

### `batch_patterns(...)` · `to_dataframe(o, h, l, c)`

Candlestick-only DataFrame builders. `chart_batch(...)` is the chart-only
counterpart (in `ta_patterns.chart_patterns`), and accepts
`patterns='bullish'|'bearish'|'bidirectional'|...` or a list.

### `signals_at(o, h, l, c, idx) -> list[tuple[str, str]]`

Patterns that fired on bar `idx`, as `(name, direction)` tuples.

---

## Catalog introspection

### `list_all_patterns(direction='all', module='all') -> list[str]`

Sorted pattern names.

- `direction`: `'all'`, `'bullish'`, `'bearish'`, `'directional'`,
  `'non_directional'`, or `'bidirectional'` (chart only).
- `module`: `'all'`, `'candles'`, or `'chart'`.

### `list_patterns(direction='all')`

Candlestick-only name list. `chart_list_patterns(...)` is the chart-only
counterpart.

### Classification sets

Frozensets of names you can test membership against:

- Candlestick (in `ta_patterns.scanner`): `PATTERNS`, `BULLISH`, `BEARISH`,
  `DIRECTIONAL`, `NON_DIRECTIONAL`.
- Chart (in `ta_patterns.chart_patterns`): `CHART_PATTERNS`, `BULLISH`,
  `BEARISH`, `DIRECTIONAL`, `BIDIRECTIONAL`, `NON_DIRECTIONAL`.

---

## Utilities

### `atr(h, l, c, period=14)` · `avg_body(o, c, period=...)`

Helper indicators used internally and exposed for convenience.

---

## Counts

| Group | Count |
|---|---|
| Candlestick total | 106 (51 bullish, 49 bearish, 6 non-directional) |
| Chart total | 194 (79 bullish, 76 bearish, 30 bidirectional, 9 non-directional) |
| **Grand total** | **300** (295 without volume) |
