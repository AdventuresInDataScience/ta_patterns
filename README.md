# ta_patterns

Pure-NumPy, fully vectorised library of **300 technical-analysis pattern
detectors** — **106 candlestick** patterns and **194 chart** patterns
(short-bar, double/multi, classic-geometric, harmonic, volume, busted).
No TA-Lib dependency.

Every detector returns a signed `int8` array (`+1` bullish / `-1` bearish /
`0` none) and is **point-in-time safe** — the value at bar *i* depends only
on bars `0..i`, so a column can go straight into a backtest without
look-ahead.

- **Python:** ≥ 3.10
- **Dependencies:** `numpy` (required); `pandas` (optional, for the
  DataFrame helpers)
- **Source:** <https://github.com/AdventuresInDataScience/ta_patterns>

## Install

```bash
pip install ta_patterns
```

With the optional pandas helpers (`batch_all`, `to_dataframe`, `chart_batch`):

```bash
pip install "ta_patterns[pandas]"
```

## Quickstart

```python
import pandas as pd
import ta_patterns as tap
import ta_patterns.chart_patterns as cp

df = pd.read_csv("ohlcv.csv", parse_dates=["date"]).set_index("date")
o, h, l, c, v = df.open, df.high, df.low, df.close, df.volume

sig     = tap.hammer(o, h, l, c)                 # one pattern -> int8 array
signals = tap.scan_all_patterns(o, h, l, c, v=v) # all 300 -> {name: array}
score   = tap.net_score_all(o, h, l, c, v=v)     # net directional score
table   = tap.batch_all(o, h, l, c, v=v)         # wide pandas DataFrame
```

NumPy arrays work equally well; pandas indexes are stripped automatically.
Every threshold that defines a shape is a keyword argument:

```python
tap.hammer(o, h, l, c, shadow_factor=1.5, require_trend=False)
cp.double_top(o, h, l, c, pivot_n=3, tol=0.06, mode="forming")
```

## What the columns look like

`batch_all` returns one `int8` column per pattern, aligned to your bars:

```
1254 bars x 300 patterns in 169 ms
dtypes: {'int8'}
memory: 0.42 MB for 376,200 cells
values: -1: 8,782,  0: 355,821,  1: 11,597
253 of 300 patterns fired at least once; 47 never fired

             close  white_candle  two_did  black_candle  rising_volume_trend  two_tall
date
2026-09-10  326.57             1        1             0                    1         1
2026-09-11  332.27             1        0             0                    1         0
2026-09-14  333.08             0        0            -1                    1         0
2026-09-15  331.34             1        0             0                    1         0
2026-09-16  332.41             0        0            -1                    1         0
```

Roughly 5% of cells are non-zero on daily equity data — about 16 simultaneous
signals per bar across all 300 detectors.

## End-to-end walkthrough

**[`examples/end_to_end.ipynb`](https://github.com/AdventuresInDataScience/ta_patterns/blob/main/examples/end_to_end.ipynb)**
is a vignette on real market data — five years of daily bars for four symbols
via `yfinance`, from a single detector through the full 300-column signal
matrix to end-to-end timings. It is committed **with its outputs**, so it
renders as documentation on GitHub without being run.

To run it yourself:

```bash
git clone https://github.com/AdventuresInDataScience/ta_patterns
cd ta_patterns
pip install -e ".[test]" yfinance jupyter
jupyter lab examples/end_to_end.ipynb
```

## Benchmarks

AAPL daily, 1,254 bars, Python 3.12 / NumPy 2.1, single core:

| Entry point | Time |
|---|---|
| `scan_all` — 106 candlestick patterns | 2 ms |
| `chart_scan_all` — 194 chart patterns | 167 ms |
| `scan_all_patterns` — all 300 | 170 ms |
| `batch_all` — all 300, as a DataFrame | 172 ms |

Scaling of the full 300-detector scan:

| Bars | Time | ms / 1k bars | bars/sec |
|---|---|---|---|
| 250 | 36 ms | 145 | 6,900 |
| 1,254 | 175 ms | 140 | 7,200 |
| 5,000 | 750 ms | 150 | 6,700 |
| 10,000 | 1,810 ms | 181 | 5,500 |

Throughput degrades on long histories because eight detectors
(`abc_correction`, `measured_move_up`/`_down`, `hs_top`/`hs_bottom`,
`busted_hs_*`, `complex_hs_*`) search pivot *pairs and triples*, so their cost
grows with the square of the pivot count — 2.8× to 7.1× worse than linear, and
~26% of the scan at 8,000 bars. Cap their `window` or exclude them by name if
you scan decades of data. The notebook's last section prints this breakdown for
your own machine and data.

### Speedup over earlier releases

1.2.0 rewrote the chart detectors around sliding windows, prefix sums and
pivot-indexed search. Identical inputs, same machine, cold cache, best of
three:

| Bars | 1.1.0 | 1.1.1 | 1.2.x | vs 1.1.1 | vs 1.1.0 |
|---|---|---|---|---|---|
| 500 | 755 ms | 329 ms | 68 ms | 4.8× | 11× |
| 1,254 | 2,640 ms | 865 ms | 171 ms | 5.1× | 15× |
| 2,000 | 5,182 ms | 1,408 ms | 278 ms | 5.1× | 19× |
| 5,000 | 25,792 ms | 3,761 ms | 769 ms | 4.9× | 34× |

The gap over 1.1.0 widens with series length because that version was
quadratic in the bar count across most chart detectors.

## Documentation

Full guide at
[the docs directory](https://github.com/AdventuresInDataScience/ta_patterns/tree/main/docs):

- [Installation](https://github.com/AdventuresInDataScience/ta_patterns/blob/main/docs/installation.md)
- [Quickstart](https://github.com/AdventuresInDataScience/ta_patterns/blob/main/docs/quickstart.md)
- [Core concepts](https://github.com/AdventuresInDataScience/ta_patterns/blob/main/docs/concepts.md)
- [API reference](https://github.com/AdventuresInDataScience/ta_patterns/blob/main/docs/api_reference.md)
- [Pattern catalog](https://github.com/AdventuresInDataScience/ta_patterns/blob/main/docs/patterns_catalog.md)
- [Examples](https://github.com/AdventuresInDataScience/ta_patterns/blob/main/docs/examples.md)

## Design guarantees

- **No look-forward bias** — bar *i* uses only bars `0..i`; pivots use
  delayed confirmation.
- **Point-in-time at the close** — a flag is set on the bar that completes
  the pattern.
- **Signed `int8` output** — `+1` / `-1` / `0`, always the input length.
- **NumPy / pandas friendly** — Series inputs are accepted; the index is
  stripped automatically.

## Notes

- Elliott Wave patterns are intentionally **not** implemented — they require
  subjective multi-level wave labelling that does not reduce to a
  deterministic OHLC rule.
- Five chart patterns are volume-based and run only when a volume array is
  supplied (295 from OHLC alone, 300 with volume).

## Contributing

```bash
git clone https://github.com/AdventuresInDataScience/ta_patterns
cd ta_patterns
pip install -e ".[test,docs]"

pytest               # preferred
python run_tests.py  # zero-dependency fallback
```

The suite enforces the library's guarantees across the whole catalog:
output dtype/length/range, sign-vs-classification, no look-ahead, and
textbook firing cases for every pattern family.

`tools/golden.py check` additionally asserts that no detector's output has
moved on a set of fixed series — run it before and after any change to the
detector internals.

<details>
<summary>Repository layout</summary>

```
.
├── pyproject.toml          # build metadata, deps, pytest config
├── README.md
├── mkdocs.yml              # docs site config
├── run_tests.py            # zero-dependency test runner (pytest fallback)
├── .github/workflows/      # publish.yml — test, build and release on a v* tag
├── src/
│   └── ta_patterns/        # the package (src layout)
│       ├── __init__.py     # public candlestick API + re-exports
│       ├── _core.py        # shared numeric helpers
│       ├── scanner.py      # scan_all_patterns, net_score_all, batch_all, ...
│       ├── single.py two_bar.py three_bar.py multi_bar.py   # 106 candlesticks
│       └── chart_patterns/ # 194 chart patterns (own pivot engine + scanner)
│           ├── _core.py scanner.py
│           ├── _memo.py _windows.py    # feature cache, sliding-window helpers
│           └── short.py double_multi.py classic.py harmonic.py
│               volume.py busted.py
├── tests/                  # pytest suite (also runnable via run_tests.py)
├── examples/               # end_to_end.ipynb — walkthrough on real data
├── tools/                  # golden.py (output regression), bench.py (timings)
└── docs/                   # full documentation (mkdocs)
```

**Why a `chart_patterns/` sub-package?** Candlestick detectors work
bar-by-bar; chart detectors need a pivot/swing engine, their own scanner,
and a separate `_core`. Keeping them as a sub-package isolates that
machinery (and avoids two `_core`/`scanner` modules colliding). The two
families are still scanned together via `scan_all_patterns`.

</details>

## License

MIT
