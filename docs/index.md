# ta_patterns documentation

`ta_patterns` is a pure-NumPy, vectorised library of **300 technical-analysis
pattern detectors** — 106 candlestick patterns and 194 chart patterns
(short-bar, double/multi, classic-geometric, harmonic, volume, busted).

Every detector returns a signed `int8` array (`+1` bullish, `-1` bearish,
`0` none) the same length as the input, and every detector is **point-in-time
safe**: the value at bar *i* depends only on bars `0..i`.

## Contents

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Core concepts](concepts.md) — signals, point-in-time, pivots, Adam/Eve,
  scoring
- [API reference](api_reference.md) — every public function and its parameters
- [Pattern catalog](patterns_catalog.md) — all 300 patterns by category and
  direction
- [Examples](examples.md) — worked end-to-end recipes

## At a glance

```python
import ta_patterns as tap
import pandas as pd

df = pd.read_csv("ohlcv.csv")
o, h, l, c, v = df.open, df.high, df.low, df.close, df.volume

# one pattern -> int8 array
sig = tap.hammer(o, h, l, c)

# every pattern at once -> {name: int8 array}
signals = tap.scan_all_patterns(o, h, l, c, v=v)   # 300 with volume

# a net directional score per bar
score = tap.net_score_all(o, h, l, c, v=v)
```

## Design guarantees

| Property | Guarantee |
|---|---|
| Output dtype | `numpy.int8` |
| Output length | equal to the input length |
| Output values | `{-1, 0, 1}` only |
| Look-ahead | none — bar *i* uses only bars `0..i` |
| Inputs | NumPy arrays or pandas Series (index stripped automatically) |
| Dependencies | NumPy only (pandas optional, used by the DataFrame helpers) |

These properties are enforced by the test-suite (`tests/test_invariants.py`,
`tests/test_no_lookahead.py`) across the entire catalog.
