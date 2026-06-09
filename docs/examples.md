# Examples

These recipes assume:

```python
import numpy as np
import pandas as pd
import ta_patterns as tap
import ta_patterns.chart_patterns as cp

df = pd.read_csv("ohlcv.csv", parse_dates=["date"]).set_index("date")
o, h, l, c, v = df.open, df.high, df.low, df.close, df.volume
```

## 1. Flag every hammer and inspect the dates

```python
sig = tap.hammer(o, h, l, c)
hammer_days = df.index[sig == 1]
print(hammer_days)
```

## 2. Build a feature matrix for a model

```python
features = tap.batch_all(o, h, l, c, v=v)   # DataFrame, one int8 col per pattern
features.index = df.index
# e.g. drop all-zero columns (patterns that never fired in this series)
features = features.loc[:, (features != 0).any()]
```

## 3. A simple net-score overlay

```python
score = tap.net_score_all(o, h, l, c, v=v)   # +ve = net bullish, -ve = net bearish
df["pattern_score"] = score
bullish_clusters = df[df.pattern_score >= 3]
```

## 4. Confirmed vs forming double tops

```python
confirmed = cp.double_top(o, h, l, c, mode="confirmed")  # on neckline break
forming   = cp.double_top(o, h, l, c, mode="forming")    # at the second peak

# 'forming' fires earlier — useful for an early alert, riskier as a trigger
early_alerts = df.index[forming == -1]
triggers     = df.index[confirmed == -1]
```

## 5. Tuning pivot sensitivity

```python
# detect smaller, more frequent tops
sensitive = cp.double_top(o, h, l, c, pivot_n=3, window=60)

# only broad, well-formed tops
strict = cp.double_top(o, h, l, c, pivot_n=7, window=120)
```

## 6. Adam/Eve double bottoms

```python
# sharp-then-rounded double bottom (Adam first trough, Eve second)
ae = cp.double_bottom_adam_eve(o, h, l, c)
print(df.index[ae == 1])
```

## 7. Harmonic patterns with a looser Fibonacci tolerance

```python
gartley = cp.gartley_bull(o, h, l, c, fib_tol=0.06)   # +1 at the D reversal low
bat     = cp.bat_bull(o, h, l, c, fib_tol=0.08)
```

## 8. What fired today?

```python
todays = tap.signals_at(o, h, l, c, idx=-1)
for name, direction in todays:
    print(f"{name:30s} {direction}")
```

## 9. Restrict a scan to bearish chart patterns

```python
bear_names = tap.list_all_patterns(module="chart", direction="bearish")
table = cp.chart_batch(o, h, l, c, v=v, patterns="bearish")
```

## 10. Point-in-time check (sanity)

```python
# the value at any bar must not depend on later bars
full = tap.scan_all_patterns(o, h, l, c, v=v)
T = 200
prefix = tap.scan_all_patterns(o[:T], h[:T], l[:T], c[:T], v=v[:T])
assert all(int(full[k][T-1]) == int(prefix[k][T-1]) for k in full)
```
