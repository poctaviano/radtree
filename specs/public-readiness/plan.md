# Public Readiness Plan

## Strategy
Use a spec-driven delivery flow:
1. Lock scope and acceptance criteria.
2. Apply small implementation slices mapped to task items.
3. Validate each slice with automated checks.

## Technical Decisions

### Packaging
- Adopt `pyproject.toml` with setuptools backend.
- Remove deprecated dependency alias `sklearn` in favor of `scikit-learn`.
- Add missing runtime dependency `pandas`.
- Add `python_requires` and modern project metadata.

### API and Reliability
- Preserve `plot_radial` signature and default semantics.
- Fix runtime breakpoints in current supported dependency stack:
  - numpy/pandas label handling
  - `num_samples=None`
  - `data=` default label column handling
  - `spring=True` branch
  - pandas 3 fillna compatibility
  - `quick_fitted_tree` estimator handling when GridSearch is not used

### Quality Gates
- Lint: `ruff check`
- Format: `ruff format --check`
- Tests: `pytest`
- Build: `python -m build`

### CI
- Matrix: Python 3.10, 3.11, 3.12.
- Jobs: lint/format, tests, build.

### Documentation
- Rewrite README for public portfolio clarity.
- Add contributor and changelog docs.
- Add explicit third-party notices section.

## Risks and Mitigations
- Risk: behavior drift in visualization internals.
  - Mitigation: keep public function signatures unchanged and add regression tests around core call paths.
- Risk: notebook-era assumptions break under newer pandas/sklearn.
  - Mitigation: normalize input handling and test ndarray/DataFrame paths.
