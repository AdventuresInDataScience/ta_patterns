"""
Golden-output regression harness.

    python tools/golden.py save    # record baseline from current source
    python tools/golden.py check   # assert current source matches baseline

Every detector is run on several deterministic OHLCV series and in both
``mode='forming'`` and ``mode='confirmed'`` where supported.  Outputs are
hashed; any change in any signal on any bar fails the check.
"""
import sys, os, json, hashlib, inspect, warnings
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
BASELINE = os.path.join(ROOT, "tools", "golden_baseline.json")

warnings.filterwarnings("ignore")

import ta_patterns as tap
from ta_patterns.scanner import PATTERNS
from ta_patterns import chart_patterns as _cp


def make_series(N, seed, kind="gbm"):
    rng = np.random.default_rng(seed)
    if kind == "gbm":
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, N)))
    elif kind == "trend":
        c = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.008, N)))
    else:  # choppy / mean-reverting, exercises pivot-dense paths
        c = 100 + np.cumsum(rng.normal(0, 0.5, N)) * 0.3 + 5 * np.sin(np.arange(N) / 9.0)
    o = c * (1 + rng.normal(0, 0.002, N))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, N)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, N)))
    v = rng.integers(int(1e5), int(1e6), N).astype(float)
    return o, h, l, c, v


CASES = [
    ("gbm2000",    make_series(2000, 0, "gbm")),
    ("chop1500",   make_series(1500, 7, "chop")),
    ("trend3000",  make_series(3000, 3, "trend")),
]


def detectors():
    out = {}
    for name in sorted(set(PATTERNS) | set(_cp.CHART_PATTERNS)):
        fn = getattr(tap, name, None) or getattr(_cp, name, None)
        if callable(fn):
            out[name] = fn
    return out


def _digest(arr):
    a = np.ascontiguousarray(arr)
    return hashlib.blake2b(a.view(np.uint8), digest_size=12).hexdigest()


def collect():
    res = {}
    for name, fn in detectors().items():
        params = inspect.signature(fn).parameters
        modes = ["confirmed", "forming"] if "mode" in params else [None]
        for case, (o, h, l, c, v) in CASES:
            for mode in modes:
                kw = {} if mode is None else {"mode": mode}
                key = f"{name}|{case}|{mode}"
                try:
                    try:
                        out = fn(o, h, l, c, **kw)
                    except TypeError:
                        out = fn(o, h, l, c, v, **kw)
                    res[key] = [_digest(np.asarray(out, np.int8)),
                                int(np.count_nonzero(out))]
                except Exception as e:                    # unchanged failures
                    res[key] = ["ERR:" + type(e).__name__, -1]
    return res


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    cur = collect()
    if cmd == "save":
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w") as f:
            json.dump(cur, f, indent=0, sort_keys=True)
        print(f"saved {len(cur)} golden outputs -> {BASELINE}")
        return 0

    with open(BASELINE) as f:
        base = json.load(f)
    missing = sorted(set(base) - set(cur))
    added = sorted(set(cur) - set(base))
    diff = [k for k in base if k in cur and base[k] != cur[k]]
    print(f"checked {len(cur)} outputs")
    if missing:
        print(f"  MISSING ({len(missing)}): {missing[:8]}")
    if added:
        print(f"  ADDED   ({len(added)}): {added[:8]}")
    if diff:
        print(f"  CHANGED ({len(diff)}):")
        for k in diff[:25]:
            print(f"    {k}\n      baseline={base[k]}  current={cur[k]}")
    if not (missing or added or diff):
        print("  ALL IDENTICAL")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
