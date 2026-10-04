# CLAUDE.md

Guidance for Claude Code in this repository. The global rules apply: strict TDD, 100% line/branch/function coverage, uv, Conventional Commits, no commit or push without being asked.

## Project

`apeks` is a pygame-ce simulation: agents walk a tile grid, leave decaying trail marks, seek charging stations when low, and push lower-priority agents aside. Python 3.14, managed with uv.

## Commands

- Setup: `uv sync`, then `git config core.hooksPath .githooks` (installs the pre-commit hook).
- Run: `uv run apeks`
- Tests with coverage gate: `uv run pytest` (branch coverage, fails under 100%)
- Lint and format: `uv run ruff check .` and `uv run ruff format .`; keep formatter changes.

## Layout

- `src/apeks/config.py`: constants only.
- `src/apeks/sim.py`: pure simulation (`Trail`, `Agent`, `World`, `build_nav`, `advance`). No pygame calls.
- `src/apeks/render.py`: drawing onto a `pygame.Surface`.
- `src/apeks/app.py`: `move_camera` and `main`, the window and event loop.
- `tests/`: pytest; `conftest.py` seeds `random` and selects the SDL dummy video driver, so tests run offline and headless.

## Testing rules

- Test first. A bugfix adds a regression test in the same change.
- Keep logic out of `main`. The only code excluded from coverage is `main`, marked `# pragma: no cover` because it opens a window and runs the event loop. Put anything new in a testable function.
- Tests use small fixtures and a seeded `random`, never timing or real windows.
