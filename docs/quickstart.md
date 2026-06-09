# Quickstart

## Inputs

Every detector takes open/high/low/close arrays (`o, h, l, c`), and a few
chart detectors optionally take volume (`v`). Inputs may be NumPy arrays or
pandas Series — a Series index is stripped automatically, and the returned
array is plain NumPy.

```python
import numpy as np
import ta_patterns as tap

o = np.array([10.0, 13.2])
h = np.array([13.1, 13.4])
l = np.array([9.9, 11.3])
c = np.array([13.0, 11.4])

tap.dark_cloud_cover(o, h, l, c)     # array([ 0, -1], dtype=int8)
```

## A single pattern

Each pattern is a top-level function returning an `int8` array:

```python
sig = tap.hammer(o, h, l, c)         # +1 where a hammer fired, else 0
```

Most detectors accept keyword arguments to loosen or tighten the rule:

```python
tap.hammer(o, h, l, c, shadow_factor=1.5, require_trend=False)
tap.dark_cloud_cover(o, h, l, c, penetration=0.60)
```

Chart detectors share a common set of knobs:

```python
import ta_patterns.chart_patterns as cp

cp.double_top(o, h, l, c, mode="confirmed")   # fires on neckline break (default)
cp.double_top(o, h, l, c, mode="forming")     # fires as the shape completes
cp.ascending_triangle(o, h, l, c, pivot_n=3, window=60)
```

## Scanning everything

```python
# candlestick + chart, merged into one dict {name: int8 array}
signals = tap.scan_all_patterns(o, h, l, c, v=volume)

# candlestick only
signals = tap.scan_all(o, h, l, c)
```

When a chart pattern name collides with a candlestick name, the chart entry is
prefixed with `cp_` in the merged dict.

## Scoring and tabulating

```python
# per-bar net directional score (int) across all patterns
score = tap.net_score_all(o, h, l, c, v=volume)

# wide pandas DataFrame, one int8 column per pattern
table = tap.batch_all(o, h, l, c, v=volume)

# which patterns fired on a given bar?
tap.signals_at(o, h, l, c, idx=-1)   # [(name, 'bullish'|'bearish'|...), ...]
```

## Listing the catalog

```python
tap.list_all_patterns()                                  # all 300
tap.list_all_patterns(module="chart", direction="bearish")
tap.list_all_patterns(module="candles", direction="bullish")
```
