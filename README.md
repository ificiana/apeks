# apeks

Autonomous cargo convoys on a sea-lane network, used to compare classical swarm
intelligence (Ant Colony Optimisation for routing, Boids for formation) with
policies written by a large language model. The project applies both and
observes what happens; it makes no claim that either is better.

## Story

Convoys of autonomous cargo vessels shuttle between ports. They sail together
for shared sensing and mutual safety. The sea is a directed graph of lanes
between ports and waypoints:

- Currents, waves and a passing storm make a lane's cost depend on its
  direction: `W1>W2` and `W2>W1` are different edges.
- Lanes have a width and a capacity. The strait `W2>W3` is short but single
  file, and convoys sharing an overfull lane slow down.
- Ports have berths. Convoys queue, load or unload, then take the next order.
- Disruptions: a storm (t 1500–2500) and a grounded vessel closing the strait
  (t 3000–4000).

Two decisions are made over and over:

| Module | Classical engine | LLM engine |
| --- | --- | --- |
| routing: which lane to take at a waypoint | ACO | scored routing with an LLM-written policy |
| formation: how each vessel steers in its convoy | Boids | slot formation with an LLM-written policy |

## Physical explainability

Nothing in the simulation needs information a real fleet could not have.

- Every vessel is a radio beacon. Each tick it broadcasts a position report.
  Receivers estimate the sender's relative position from the signal, with
  error growing with distance. Range, loss and bandwidth are limited.
- After finishing a lane, a convoy broadcasts a lane report (lane, transit
  time). Other convoys keep the reports they hear. ACO uses them as pheromone
  that fades with age. The word "pheromone" exists only inside the ACO engine.
- Port stations repeat navigation notices for closed and reopened lanes. A
  convoy out of range that picks a closed lane finds the blockage at the lane
  entrance.
- Route estimates use charted mean currents. Vessels feel the true drift,
  including waves.

## The LLM as a policy author

The LLM never steers a vessel. It acts like a shore operations centre that
rewrites a typed policy when something happens: mission start, a storm forming
or clearing, a closure or reopening.

- The prompt describes each policy field with its type and bounds, the current
  policy, and a situation summary: closures, storms, reported delays by lane,
  lane widths and capacities, deliveries and near misses.
- The reply is constrained to the policy's JSON schema, then validated:
  numbers are clamped, invalid choices and unknown fields are dropped.
- A new policy takes effect after a latency in sim time (`LLM_LATENCY`).
  Requests are capped per run (`LLM_BUDGET`). A failed call keeps the old
  policy.

The routing policy weights estimated lane time, time-to-destination and
reported delay, and can list lanes to avoid. The formation policy picks a
shape (column, line, wedge), the slot spacing and the gains.

## Arms

`experiments/` holds one config per arm. Every module's engine is chosen per
config:

| Config | routing | formation |
| --- | --- | --- |
| `classical.toml` | aco | boids |
| `llm.toml` | llm | llm |
| `llm_routing.toml` | llm | boids |
| `llm_formation.toml` | aco | llm |

Other config keys: `seed`, `ticks`, `llm` (`stub` or `ollama`), `model`, `url`,
`latency`, `budget`.

## Metrics

Each run reports:

- deliveries and mean trip time
- energy (thrust through water)
- port wait and lane congestion (ticks beyond capacity)
- near misses (vessel pairs closing within `NEAR_MISS`)
- blocked-lane discoveries
- radio sent, delivered, lost and overflow counts
- LLM calls, failures and skipped calls

The full JSON adds delivery times, the event log and every LLM prompt trigger,
reply and accepted policy.

## Usage

```sh
uv sync
git config core.hooksPath .githooks

uv run apeks                                # window, classical arm
uv run apeks experiments/llm.toml           # window, any arm; space pauses, up/down speed
uv run apeks-run experiments/*.toml         # headless, one JSON summary line per arm
uv run apeks-run experiments/llm.toml --llm ollama --ticks 3500
```

`--ticks` and `--llm` override every config. Each arm saves two files under
`logs/` (git-ignored; `--logs DIR` to change), stamped with the invocation's
local time:

- `out_<arm>_<YYYYmmdd-HHMMSS>.json`: config, summary, event log, LLM log
- `trace_<arm>_<YYYYmmdd-HHMMSS>.jsonl`: the trace (`--no-trace` skips it)

### Traces

The trace is one JSON record per line, enough to replay or plot a run without
re-simulating it. Every record has `t` and `kind`:

| kind | contents |
| --- | --- |
| `meta` | config, nodes, ports, lanes, currents, storms, scheduled events, convoys (first record) |
| `snapshot` | every vessel's position, velocity, water velocity, drift and energy; every convoy's state, lane, destination, centre and known closures; port queues, closures, metrics, radio counters, both engines' parameters |
| `route_choice` | convoy, node, destination, each option's estimate, cost-to-go and report count, the choice |
| `lane_start`, `lane_end` | convoy, lane, transit time |
| `queued`, `berthed`, `delivered` | convoy, port, trip time |
| `blocked`, `notice_heard` | convoy, lane, closed |
| `near_miss` | the two vessels and their distance |
| `event` | closures, reopenings, storms forming and clearing |
| `policy_request`, `policy_applied` | module, trigger, full prompt, raw reply, accepted policy or error |

`--snapshot-every N` sets the snapshot interval (default 1 tick; about 20 MB per
6000-tick run). Load a trace with `apeks.trace.read_trace(path, kind=None)` or
`pandas.read_json(path, lines=True)`.

### Ollama

The `stub` backend replies `{}`, which leaves the default policy in force, so
runs work offline. For a real model:

```sh
ollama pull lfm3      # or any model that supports structured output
uv run apeks-run experiments/llm.toml --llm ollama
```

Calls go to `POST {url}/api/generate` with the policy's JSON schema as
`format`, thinking off and temperature 0. Set `url` in a config to use a
remote server.

## Layout

| Path | Contents |
| --- | --- |
| `src/apeks/geometry.py` | 2-D vector helpers |
| `src/apeks/ocean.py` | currents, storms, waves; mean and true drift |
| `src/apeks/network.py` | ports, waypoints, directed lanes; transit estimates; Dijkstra time-to-go |
| `src/apeks/radio.py` | broadcast radio with range, loss, bandwidth and position noise |
| `src/apeks/policy.py` | typed, bounded policy fields; JSON schema; validation |
| `src/apeks/routing.py` | ACO and scored routing engines |
| `src/apeks/formation.py` | Boids and slot formation engines |
| `src/apeks/llm.py` | LLM clients and the policy author |
| `src/apeks/world.py` | vessels, convoys, ports, events, metrics, the step loop |
| `src/apeks/scenario.py` | the default map, ocean, events and convoys |
| `src/apeks/experiment.py` | experiment configs and world assembly |
| `src/apeks/runner.py` | headless CLI |
| `src/apeks/trace.py` | JSONL trace writer and reader |
| `src/apeks/render.py`, `app.py` | pygame drawing and the window |

## Development

```sh
uv run pytest            # 100% line and branch coverage required
uv run ruff check .
uv run ruff format .
```

Tests run offline and headless with a seeded `random`. The LLM is a stub or a
faked HTTP call.
