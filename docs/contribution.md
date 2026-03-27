# Contribution Guide

**Audience:** Contributors

---

## Getting Started

1. Fork the repository or clone it into your `ComfyUI/custom_nodes/` directory.
2. Install development dependencies:
    ```bash
    pip install -e .[dev]
    pre-commit install
    ```
3. Run tests with `pytest` before submitting changes.

## Branch and Commit Standards

- Use feature branches for new features or fixes (e.g., `feature/video-segment-extender`).
- Write clear, descriptive commit messages (see [Conventional Commits](https://www.conventionalcommits.org/)).
- Rebase or merge main before opening a pull request.

## Reviews and PRs

- Open a pull request for all changes.
- Ensure all tests pass and code is linted (Ruff, Mypy).
- Request review from maintainers.
- Address all review comments before merging.

## Tests and Quality

- All new nodes must include unit tests in `tests/`.
- Run `pytest` and ensure 100% pass rate.
- Use `ruff` and `mypy` for linting and static analysis.
- Pre-commit hooks will run automatically on commit.

## Code Conventions

- Follow PEP 8 and project-specific style (see [python-style-guide](../.github/skills/python-style-guide/SKILL.md)).
- Use type hints and docstrings for all public functions/classes.
- Keep functions and modules small and focused.

## Best Practices

- Document new features in [docs/features/](features/).
- Update [modules.md](modules.md) and [usage.md](usage.md) as needed.
- Link all new docs from [SUMMARY.md](SUMMARY.md).
- Avoid duplicating information across files.

---

For more details, see the [setup guide](setup.md) and [architecture overview](architecture.md).
