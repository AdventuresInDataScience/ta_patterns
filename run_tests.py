#!/usr/bin/env python3
"""Minimal zero-dependency test runner.

Prefer ``pytest`` (``pip install -e .[test] && pytest``).  This fallback
discovers ``tests/test_*.py``, runs every ``test_*`` function, and reports
pass/fail counts — handy in environments where pytest is unavailable.
"""
import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main():
    test_files = sorted((ROOT / "tests").glob("test_*.py"))
    passed = failed = 0
    failures = []
    for f in test_files:
        mod = importlib.import_module(f"tests.{f.stem}")
        for name in dir(mod):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
            except Exception:
                failed += 1
                failures.append(f"{f.stem}::{name}\n{traceback.format_exc()}")
    print(f"\n{passed} passed, {failed} failed")
    for fl in failures:
        print("\n--- FAIL", fl)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
