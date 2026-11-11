# CLAUDE.md

Guidance for Claude Code in this repository. The global rules apply: strict TDD, 100% line/branch/function coverage, uv, Conventional Commits, no commit or push without being asked.

## Project

`apeks` simulates autonomous cargo convoys on a directed sea-lane network to compare classical engines (ACO routing, Boids formation) with LLM-written policies. Python 3.14, managed with uv. See README.md for the story and arms.

## Commands

- Setup: `uv sync`, then `git config core.hooksPath .githooks` (installs the pre-commit hook).
- Window: `uv run apeks [experiments/<arm>.toml]`
- Headless: `uv run apeks-run experiments/*.toml [--llm ollama] [--ticks N] [--logs DIR] [--no-trace] [--snapshot-every N]`; writes `logs/out_<arm>_<stamp>.json` and `logs/trace_<arm>_<stamp>.jsonl`
- Tests with coverage gate: `uv run pytest` (branch coverage, fails under 100%)
- Lint and format: `uv run ruff check .` and `uv run ruff format .`; keep formatter changes.

## Layout

- `src/apeks/config.py`: constants only.
- Pure simulation, no pygame: `geometry`, `ocean`, `network`, `radio`, `policy`, `routing`, `formation`, `llm`, `world`, `scenario`, `experiment`, `runner`, `trace`.
- Engines share one observation API per module; the word "pheromone" stays inside the ACO engine.
- `src/apeks/render.py`: drawing onto a `pygame.Surface`.
- `src/apeks/app.py`: `parse_args`, `change_speed` and `main`, the window and event loop.
- `experiments/`: one TOML config per arm.
- `tests/`: pytest; `conftest.py` seeds `random` and selects the SDL dummy video driver, so tests run offline and headless. LLM calls use `StubClient` or a faked `urlopen`.

## Testing rules

- Test first. A bugfix adds a regression test in the same change.
- Keep logic out of `main`. The only code excluded from coverage is `main`, marked `# pragma: no cover` because it opens a window and runs the event loop. Put anything new in a testable function.
- Tests use small fixtures and a seeded `random`, never timing or real windows.
