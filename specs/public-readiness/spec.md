# Public Readiness Spec

## Context
`radtree` is an older open-source project that should be suitable for public portfolio use, including interview review in devtools-focused environments.

## Audience
- Python users who want to visualize `sklearn` decision trees in a radial plot.
- Interviewers and maintainers evaluating engineering quality.

## Goals
1. The package installs in a clean virtual environment.
2. The documented quickstart works on supported Python versions.
3. The main public API (`plot_radial`) remains stable.
4. Licensing and third-party attribution are explicit and correct.
5. Automated checks enforce reliability and code quality.

## Non-goals
- No major algorithm rewrite.
- No backward-incompatible API redesign.
- No notebook removal; notebooks remain as usage examples.

## Constraints
- Keep `plot_radial` API stable unless adding strictly backward-compatible behavior.
- Prefer small, safe refactors.

## Acceptance Criteria
1. `pip install -e .` succeeds on Python 3.10, 3.11, and 3.12.
2. `plot_radial` works for the README quickstart path.
3. `num_samples=None` is supported as documented.
4. GitHub recognizes the repository license.
5. CI runs lint, format check, tests, and package build.
6. README contains: what it is, quickstart, API overview, performance guidance, contribution link, and license section.
7. Repository includes `CHANGELOG.md`, `CONTRIBUTING.md`, and pre-commit configuration.

## Done Definition
The repo can be publicly featured when all acceptance criteria pass locally and in CI.
