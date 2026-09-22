"""
Per-detector benchmark.

    python tools/bench.py [N] [--top K] [--json out.json]

Times every detector on a deterministic series and reports the slowest, the
candlestick/chart split, and the cumulative time share.  Use it to confirm a
change actually helped and to catch performance regressions.
"""
import sys, os, time, json, warnings
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
warnings.filterwarnings("ignore")

import ta_patterns as tap
from ta_patterns.scanner import PATTERNS
from ta_patterns import chart_patterns as _cp
from ta_patterns.chart_patterns._memo import clear_cache


def series(N, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, N)))
    o = c * (1 + rng.normal(0, 0.002, N))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, N)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, N)))
    v = rng.integers(int(1e5), int(1e6), N).astype(float)
    return o, h, l, c, v


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    N = int(args[0]) if args else 5000
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 20
    o, h, l, c, v = series(N)

    rows = []
    for name in sorted(set(PATTERNS) | set(_cp.CHART_PATTERNS)):
        fn = getattr(tap, name, None) or getattr(_cp, name, None)
        if not callable(fn):
            continue
        kind = "candle" if name in PATTERNS else "chart"
        try:
            t0 = time.perf_counter()
            try:
                fn(o, h, l, c)
            except TypeError:
                fn(o, h, l, c, v)
            rows.append((time.perf_counter() - t0, name, kind))
        except Exception as e:
            print(f"  ERR {name}: {type(e).__name__}")

    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    cand = sum(r[0] for r in rows if r[2] == "candle")
    print(f"\nN={N}  {len(rows)} detectors")
    print(f"total {tot*1000:8.1f} ms   candlestick {cand*1000:6.1f} ms   "
          f"chart {(tot-cand)*1000:8.1f} ms\n")
    print(f"--- slowest {top} ---")
    for dt, name, kind in rows[:top]:
        print(f"{dt*1000:9.2f} ms  {dt/tot*100:5.1f}%  {kind:6s} {name}")
    cum = 0.0
    print()
    for i, (dt, _, _) in enumerate(rows, 1):
        cum += dt
        if i in (5, 10, 20, 40, 80, len(rows)):
            print(f"top {i:4d}: {cum/tot*100:5.1f}% of total")

    # scan-level timing, where the shared-feature cache pays off
    clear_cache()
    t0 = time.perf_counter()
    tap.scan_all_patterns(o, h, l, c, v=v)
    print(f"\nscan_all_patterns (cold cache): {(time.perf_counter()-t0)*1000:.0f} ms")

    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w") as f:
            json.dump({n: d for d, n, _ in rows}, f, indent=1, sort_keys=True)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
