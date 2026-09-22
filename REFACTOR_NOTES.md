# Performance refactor — what changed

**Result: full 292-detector scan at 5,000 bars goes from 5,894 ms to 1,090 ms (5.4×).**
Every detector produces **bit-identical output on finite input**. Four detector
families deliberately changed how they treat NaN — see
[NaN semantics](#nan-semantics) below. The candlestick half of the library was
already fully vectorised and was not touched.

---

## Verification

Nothing here is trusted on inspection. Two independent checks:

| Check | Scope | Result |
|---|---|---|
| `tools/golden.py` | 1,128 outputs — 292 detectors × 3 series shapes (GBM / choppy / trending) × both `mode`s | ALL IDENTICAL |
| Cross-version diff | old vs new package run in separate interpreters, 20,000-bar series (376 outputs) | ALL IDENTICAL |
| Cross-version diff | same, **100,000-bar** series (376 outputs) | ALL IDENTICAL |
| `run_tests.py` | the repo's existing suite | 65 passed, 0 failed |

```bash
python tools/golden.py save     # record baseline
python tools/golden.py check    # assert nothing moved
python tools/bench.py 5000      # per-detector timings
```

Run `golden.py save` on the current source before making any further change,
then `check` after. It is the reason these rewrites were safe.

---

## NaN semantics

The golden files above use finite series. A separate differential pass that
injects NaN — in blocks, and one bar at a time including at exact pivot swing
bars — found four places where the vectorised form does not match the loop.
One was a bug and is fixed; the other three are intentional.

**Fixed.** `_volume_trend` (`rising_volume_trend`, `falling_volume_trend`) took
its per-window mean from a global `np.cumsum`, so a single NaN anywhere carried
forward into *every later window* — `v[0] = nan` took `rising_volume_trend`
from 280 fires to 0 across the whole series, where the loop only lost the
windows that actually overlapped the NaN. Non-finite bars are now zeroed before
the prefix sum, and the window gate is `isfinite`, so a NaN is contained to the
windows spanning it.

Note the same prefix-sum shape in `_sliding_fit` (`classic.py`), which sums
pivot *prices*. It cannot leak the same way only because a NaN bar can never
satisfy the `>=`/`<=` comparisons that qualify a pivot, so a NaN price never
enters `p`. That is load-bearing — if pivot detection ever changes to admit
NaN, those prefix sums need the same treatment.

**Intentional, not restored.** In all three the loop form fired on data it
could not actually evaluate, and the vectorised form refuses to:

| Detector | Loop behaviour on NaN | Now |
|---|---|---|
| `cup_with_handle`, `inverted_cup_with_handle` | `if x < y: continue` falls through, because `nan < y` is False → could fire on a NaN window | mask AND suppresses the bar |
| `rounding_bottom`, `rounding_top` (and `roof`, `inverted_roof`, which delegate) | scalar `min(a, b)`/`max(a, b)` returns whichever argument came first, so the answer depended on argument order | `np.minimum`/`np.maximum` propagate → no fire |

A NaN close means the window's shape is unknown. Returning 0 there is the
defensible reading, and it is now the documented contract: **a detector does not
signal on a window containing a non-finite bar.**

---

## The four changes

### 1. Closed-form regression instead of `np.polyfit` per bar
`volume.py` was calling `np.polyfit` — a full least-squares SVD — once per bar
for a degree-1 slope. On a fixed integer x-grid the slope is closed-form, and
all windows become one matrix–vector product.

### 2. Reuse of the vectorised trendline fit that already existed
`classic.py` contained both `_triangle_lines` (scalar, per-bar) and
`_sliding_trendlines` (prefix-sum, all bars at once). The diamonds and
bump-and-runs still used the scalar one. The "first half" fit is the same
sliding fit shifted back, so one O(N) pass covers both legs.

### 3. Index by pivot, not by bar
The double/triple/1-2-3 engines looped over all N bars, re-deriving the last
two or three pivots each time. That state only changes when a *new* pivot
appears (~every `2·pivot_n` bars), so the loop repeated identical work ~10×.
One `searchsorted` now resolves "which pivots are latest" for every bar at
once, and Adam/Eve sharpness is computed once per pivot instead of once per
bar. **This is what removed the super-linear blow-up.**

### 4. Shared-feature cache (`_memo.py`)
Every chart detector independently recomputed `pivot_highs`/`pivot_lows` on the
same arrays — ~200 redundant computations per scan (498 ms vs 2.5 ms for one).
Now content-hash cached. Keys are array *contents*, not identity, so in-place
mutation or buffer reuse cannot serve a stale result; values are returned as
copies so one detector cannot corrupt another's view.

---

## Measured

Full scan, N=5,000: **5,894 ms → 1,090 ms**. Candlestick portion: 9 ms → 8 ms.

Worst offenders, and note the scaling — the quadratic cliff is gone:

| detector | 5k | 20k | 80k |
|---|---|---|---|
| `diamond_top` | 408 → **3.2** ms | 1,664 → **9.8** | 6,666 → **111** |
| `rising_volume_trend` | 334 → **0.8** ms | 1,362 → **1.7** | 5,452 → **6.1** |
| `double_top_adam_adam` | 195 → **2.2** ms | 824 → **5.4** | 3,003 → **20** |
| `bump_and_run_top` | 128 → **1.6** ms | 527 → **4.9** | 2,190 → **43** |
| `one_two_three` | 101 → **2.2** ms | 563 → **7.7** | 3,960 → **29** |
| `cup_with_handle` | 80 → **9.5** ms | 319 → **6.6** | 1,275 → **24** |
| `triple_top` | 62 → **1.2** ms | 252 → **3.1** | 1,035 → **11** |
| `rounding_bottom` | 61 → **0.7** ms | 237 → **2.3** | 945 → **7.7** |

---

## Files

**New**
- `src/ta_patterns/chart_patterns/_memo.py` — content-hashed LRU for shared features
- `src/ta_patterns/chart_patterns/_windows.py` — `trailing_windows` + `RangeAgg`
  (O(1) range max/min sparse table; unit-tested exhaustively against brute force
  for every possible range up to n=257)
- `tools/golden.py`, `tools/bench.py`, `tools/golden_baseline.json`

**Rewritten**
- `chart_patterns/volume.py` — all 5 detectors
- `chart_patterns/double_multi.py` — `_detect_double` and `_detect_triple`
  replace the four old per-bar engines (~24 public detectors); `_cluster_pivots`
  vectorised with segmented reductions
- `chart_patterns/classic.py` — diamonds, cups, roundings, scallops, bump-and-runs
- `chart_patterns/short.py` — `flat_base`, `cloud_bank`, `elevator_stop`,
  `wide_ranging_day` pair, 1-2-3 pair
- `chart_patterns/_core.py` — pivot memoisation; deleted dead numpy<1.20
  fallback branches (`pyproject.toml` already requires numpy>=1.22, and those
  branches hid O(N) Python loops)

**Public API is unchanged.** Same function names, signatures, defaults, return
dtypes. Nothing to update in calling code.

---

## Cache control

```python
from ta_patterns.chart_patterns._memo import clear_cache, cache_info
```
Or `TA_PATTERNS_NO_CACHE=1` to disable, `TA_PATTERNS_CACHE_SIZE=n` to resize
(default 64 entries, LRU).

---

## What's left, and one caveat

**One thing to keep an eye on.** `bump_and_run` and the triangle family use the
prefix-sum regression in `_sliding_fit`, which is algebraically identical to the
centred formula but less numerically stable at large bar indices (it forms
`Σx² − (Σx)²/n`, which loses precision as x grows). This was the one change I
was not willing to assume was safe, so it was tested directly: output is
identical at 20,000 **and at 100,000 bars**. That covers any realistic daily or
hourly history. If you ever push into millions of bars and see drift, the fix is
to subtract a constant offset from the pivot times before the prefix sums — the
slope is unchanged and the intercept corrects by `slope × offset`.

**Remaining work.** The profile is now a flat tail: no detector exceeds 50 ms
and the top 40 are 70 % of a much smaller total. The stragglers
(`abc_correction`, `measured_move_*`, `three_peaks*`, `complex_hs_*`,
`busted_*`, `v_top`, `two_b`) all fit one of the three recipes above —
mostly recipe 3, indexing by pivot. Expect roughly another 2× if you finish
them, which is real but no longer the difference between usable and not.

Two traps that nearly bit me while doing this, worth carrying forward:

1. **Inverting a `continue` guard flips strictness.** `if x > tol: continue`
   becomes `x <= tol`, not `x < tol`. Several detectors sit exactly on those
   boundaries for flat trendlines.
2. **`if result[t] == 0` guards are order-dependent.** Some detectors only write
   when the slot is still zero (first-write-wins). Naive vectorising silently
   converts that to last-write-wins. None of the functions rewritten here had
   that pattern; `harmonic.py` does, so check it before touching that module.
