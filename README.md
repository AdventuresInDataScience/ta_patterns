# ta_patterns

Pure-NumPy, fully vectorised library of **300 technical-analysis pattern
detectors** — **106 candlestick** patterns and **194 chart** patterns
(short-bar, double/multi, classic-geometric, harmonic, volume, busted).
No TA-Lib dependency.

- **Version:** 1.1.0
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
│           └── short.py double_multi.py classic.py harmonic.py
│               volume.py busted.py
├── tests/                  # pytest suite (also runnable via run_tests.py)
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
