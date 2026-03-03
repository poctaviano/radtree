# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Moved third-party NetworkX license text to `THIRD_PARTY_NOTICES/networkx-bsd-3-clause.txt` to avoid GitHub misclassifying it as a project license.
- Ignored generated `plots/` artifacts in `.gitignore`.

## [0.1.0] - 2026-03-03

### Added
- Spec-driven readiness documents under `specs/public-readiness/`.
- Modern packaging via `pyproject.toml`.
- Third-party notices file.
- CI workflow for lint, format check, tests, and build.
- Pre-commit configuration.
- Pytest coverage for `plot_radial` smoke and deterministic sampling behavior.
- Contributor guide.

### Changed
- License metadata alignment to BSD-3-Clause.
- Runtime dependency declarations updated for modern Python stacks.
- `plot_radial` now returns `(fig, ax)` and is robust across ndarray/DataFrame/Series inputs.
- `quick_fitted_tree` behavior hardened when GridSearch is not enabled.

### Fixed
- `num_samples=None` behavior now works as documented.
- pandas compatibility issues in depth interpolation and label indexing.
- `spring=True` failure due incorrect fixed-node index reference.
- Right-branch edge weight assignment bug in graph construction.
