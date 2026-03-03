# Contributing

Thanks for your interest in improving `radtree`.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
pre-commit install
```

## Quality checks

Run all checks before opening a pull request:

```bash
ruff check .
ruff format --check .
pytest
python -m build
```

## Pull request guidelines

- Keep changes focused and small.
- Preserve public API compatibility where possible.
- Add or update tests for behavior changes.
- Update `CHANGELOG.md` when user-facing behavior changes.

## Reporting issues

Please include:
- Python version
- platform/OS
- reproducible code snippet
- traceback or screenshot
