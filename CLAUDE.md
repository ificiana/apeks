# CLAUDE.md

Guidance for Claude Code in this repository. The global rules apply: strict TDD, 100% line/branch/function coverage, uv, Conventional Commits, no commit or push without being asked.

## Project

`apeks` simulates autonomous cargo convoys on a directed sea-lane network to compare classical engines (ACO routing, Boids formation) with LLM-written policies. Python 3.14, managed with uv. See README.md for the story and arms.

## Commands

- Setup: `uv sync`, then `git config core.hooksPath .githooks` (installs the pre-commit hook).
- Window: `uv run apeks [experiments/<arm>.toml]`
- Headless: `uv run apeks-run experiments/*.toml [--llm stub|ollama|claude] [--model M] [--ticks N] [--logs DIR] [--no-trace] [--snapshot-every N] [-j N]`; runs in parallel, one process per CPU by default; a `seed` list in a TOML expands to `<arm>_s<seed>` runs; writes `logs/out_<arm>_<stamp>.json` and `logs/trace_<arm>_<stamp>.jsonl`
- Sweeps: run them on `ssh ai_server` (80 cores, repo at `~/dev/apeks`, `uv` and `claude` in `~/.local/bin`) when it is reachable; a full sweep is about 4× faster there than on the 8-core laptop. Make sure its checkout is at the commit under test, run with `export PATH=$HOME/.local/bin:$PATH`, then copy `logs/<run>` back. Fall back to running locally if the server is unreachable. The server's hostname is `ai104`; if `hostname` prints that, you are already on it, so run in place.
- Report sweeps: report N uses `logs/rN`. Set the `seed` list in every `experiments/*_sweep.toml`, then run from the repo root:
  `TIMEFORMAT='WALL %R s'; { time uv run apeks-run experiments/*_sweep.toml --snapshot-every 10 --logs logs/rN > logs/rN/summary.jsonl 2> logs/rN/stderr.log; } 2>> logs/rN/stderr.log` (run `mkdir -p logs/rN` first). This writes `summary.jsonl` (one JSON line per run, from stdout) and `stderr.log` (the runner's stderr, ending with the wall time) next to the `out_*`/`trace_*` files. The report then follows the format of the previous one in `reports/`: descriptive table, paired Wilcoxon against classical with a t-test cross-check, bootstrap 95% CIs, d_z, a Holm correction over 30 comparisons, the combined arm against LLM routing, the 2×2 interaction, radio/policy/LLM use, findings, limits and next steps.
- Tests with coverage gate: `uv run pytest` (branch coverage, fails under 100%)
- Lint and format: `uv run ruff check .` and `uv run ruff format .`; keep formatter changes.

## Layout

- `src/apeks/config.py`: constants only.
- Pure simulation, no pygame: `seeds`, `geometry`, `ocean`, `network`, `mapgen`, `radio`, `hazards`, `cargo`, `hq`, `news`, `policy`, `routing`, `formation`, `llm`, `world`, `scenario`, `experiment`, `runner`, `trace`.
- Only `World` holds the truth (closures, storms, transit times). Convoys, port stations and HQ act on beliefs that reach them by radio, uplink or news. Every random stream comes from `seeds.derive(seed, name)`.
- Engines read their parameters from the observation (`obs.params`), since each convoy holds its own policy version. Engines share one observation API per module; the word "pheromone" stays inside the ACO engine.
- `src/apeks/render.py`: the dashboard (`Camera`, `History`, `layout`, `handle_event`, `draw`) onto a `pygame.Surface`.
- `src/apeks/app.py`: `parse_args`, `start`, `advance` and `main`, the window and event loop.
- `experiments/`: one TOML config per arm.
- `tests/`: pytest; `conftest.py` seeds `random` and selects the SDL dummy video driver, so tests run offline and headless. LLM calls use `StubClient` or a faked `urlopen`.

## Testing rules

- Test first. A bugfix adds a regression test in the same change.
- Keep logic out of `main`. The only code excluded from coverage is `main`, marked `# pragma: no cover` because it opens a window and runs the event loop. Put anything new in a testable function.
- Tests use small fixtures and a seeded `random`, never timing or real windows.
