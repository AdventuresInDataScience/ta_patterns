# ta_patterns

Pure-NumPy, fully vectorised library of **300 technical-analysis pattern
detectors** — **106 candlestick** patterns and **194 chart** patterns
(short-bar, double/multi, classic-geometric, harmonic, volume, busted).
No TA-Lib dependency.

- **Version:** 1.2.0
- **Python:** ≥ 3.10
- **Dependencies:** `numpy` (required); `pandas` (optional, for the
  DataFrame helpers)

Every detector returns a signed `int8` array (`+1` bullish / `-1` bearish /
`0` none) and is **point-in-time safe** — the value at bar *i* depends only
on bars `0..i`.

## Project layout

```
.
├── pyproject.toml          # build metadata, deps, pytest config
├── README.md
├── mkdocs.yml              # docs site config
├── run_tests.py            # zero-dependency test runner (pytest fallback)
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

> **Why a `chart_patterns/` sub-package?** Candlestick detectors work
> bar-by-bar; chart detectors need a pivot/swing engine, their own scanner,
> and a separate `_core`. Keeping them as a sub-package isolates that
> machinery (and avoids two `_core`/`scanner` modules colliding). The two
> families are still scanned together via `scan_all_patterns`.

## Install

```bash
pip install .                  # or: pip install -e ".[test,docs]"
```

## Quickstart

```python
import ta_patterns as tap

sig     = tap.hammer(o, h, l, c)                 # one pattern -> int8 array
signals = tap.scan_all_patterns(o, h, l, c, v=v) # all 300 -> {name: array}
score   = tap.net_score_all(o, h, l, c, v=v)     # net directional score
table   = tap.batch_all(o, h, l, c, v=v)         # wide pandas DataFrame
```

## End-to-end walkthrough

**[`examples/end_to_end.ipynb`](examples/end_to_end.ipynb)** is a vignette on
real market data — five years of daily bars for four symbols via `yfinance`,
from a single detector through the full 300-column signal matrix to end-to-end
timings. It is committed **with its outputs**, so it reads as documentation
without running anything.

```bash
pip install yfinance jupyter
jupyter lab examples/end_to_end.ipynb
```

Downloads are cached under `examples/_data/`, and the notebook carries a
synthetic series generator if you have no network.

### What the columns look like

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

### Benchmarks

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

### Speedup vs. the pre-refactor implementation

Same inputs, same machine, cold cache, best of three:

| Bars | `fb6875e` (original) | current | speedup |
|---|---|---|---|
| 500 | 755 ms | 68 ms | **11×** |
| 1,254 (AAPL) | 2,587 ms | 181 ms | **14×** |
| 2,000 | 5,182 ms | 278 ms | **19×** |
| 5,000 | 25,792 ms | 769 ms | **34×** |

The gap widens with series length because the original was quadratic in the
bar count across most chart detectors.

See [`docs/`](docs/index.md) for the full guide:

- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Core concepts](docs/concepts.md)
- [API reference](docs/api_reference.md)
- [Pattern catalog](docs/patterns_catalog.md)
- [Examples](docs/examples.md)

## Tests

```bash
pytest               # preferred
python run_tests.py  # zero-dependency fallback
```

The suite enforces the library's guarantees across the whole catalog:
output dtype/length/range, sign-vs-classification, no look-ahead, and
textbook firing cases for every pattern family.

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
