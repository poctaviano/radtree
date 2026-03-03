# Public Readiness Tasks

## P0 (Required)
- [x] OSS hygiene: align main license metadata and third-party notices.
- [x] Packaging modernization: add `pyproject.toml`, correct dependencies, define supported Python versions.
- [x] Runtime reliability fixes in `plot_radial` and `quick_fitted_tree` while keeping API compatibility.
- [x] Add pytest suite for smoke and regression coverage.
- [x] Add CI workflow for lint, format, tests, and build.
- [x] Add pre-commit configuration.
- [x] Rewrite README for quickstart/API/performance/contribution/license clarity.
- [x] Add `CONTRIBUTING.md` and `CHANGELOG.md`.

## P1 (Strong Improvements)
- [ ] Add type hints and stronger public docstrings.
- [ ] Extract reusable notebook logic into package modules.
- [ ] Add release workflow and badges.

## P2 (Optional)
- [ ] Add opt-in low-risk performance optimization and benchmark guidance.

## Task-to-Spec Traceability
- AC1: packaging modernization + CI matrix + tests.
- AC2/AC3: runtime fixes + smoke tests + README quickstart.
- AC4: OSS hygiene and license consistency.
- AC5: CI workflow.
- AC6: README rewrite.
- AC7: contributor/changelog/pre-commit docs.
