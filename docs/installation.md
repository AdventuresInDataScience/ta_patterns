# Installation

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.22
- pandas ≥ 1.5 *(optional — only needed for the DataFrame helpers
  `to_dataframe`, `batch_patterns`, `batch_all`, `chart_batch`)*

## From PyPI

```bash
pip install ta_patterns
```

With the optional pandas helpers:

```bash
pip install "ta_patterns[pandas]"
```

## From source

The project uses a `src/` layout and a standard `pyproject.toml`
(setuptools build backend).

```bash
git clone https://github.com/AdventuresInDataScience/ta_patterns
cd ta_patterns
pip install .
```

For development, install in editable mode with the test and docs extras:

```bash
pip install -e ".[test,docs]"
```

Available extras:

| Extra | Pulls in |
|---|---|
| `pandas` | pandas (DataFrame helpers) |
| `test` | pytest, pandas |
| `docs` | mkdocs, mkdocs-material |

## Verifying the install

```python
import ta_patterns as tap
print(tap.__version__)              # 1.2.1
print(len(tap.list_all_patterns())) # 300
```

## Running the tests

With pytest (preferred):

```bash
pytest
```

`pyproject.toml` sets `pythonpath = ["src", "."]`, so pytest imports the
package straight from `src/` without requiring an install, and can resolve
the suite's own `tests.synthetic` helper module.

Without pytest, a zero-dependency fallback runner is included:

```bash
python run_tests.py
```
