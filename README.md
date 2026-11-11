# apeks

Autonomous cargo convoys on a sea-lane network, used to compare classical swarm
intelligence (Ant Colony Optimisation for routing, Boids for formation) with
policies written by a large language model. The project applies both and
observes what happens; it makes no claim that either is better.

## Story

Autonomous cargo vessels carry packages between ports. Ships bound for the
same port sail together as a convoy, for shared sensing and mutual safety. The sea is a directed graph of lanes
between ports and waypoints:

- Currents, waves and a passing storm make a lane's cost depend on its
  direction: `W1>W2` and `W2>W1` are different edges.
- Lanes have a width and a capacity. The strait `W2>W3` is short but single
  file, and convoys sharing an overfull lane slow down.
- Ports have berths, counted in ships. Convoys queue, berth and unload, and
  their ships join the port's dock.
- Disruptions: a storm (t 1500–2500) and a grounded vessel closing the strait
  (t 3000–4000).

Two decisions are made over and over:

| Module | Classical engine | LLM engine |
| --- | --- | --- |
| routing: which lane to take at a waypoint | ACO | scored routing with an LLM-written policy |
| formation: how each vessel steers in its convoy | Boids | slot formation with an LLM-written policy |

## Cargo

Each port generates packages by a Poisson process at `DEMAND_RATE` × the
config's `demand`, one random stream per port. A package's destination is
uniform over the other ports. Every arm faces the same demand and uses the
same fixed rules (`cargo.py`) for what ships do with it:

- **Loading.** Each docked ship, in id order, loads up to `CAPACITY`
  packages, oldest first: those bound where its cargo already favours, then
  the most common waiting destination.
- **Next port.** The most common destination aboard, ties to the nearest by
  charted time, then by name. An empty ship heads for the nearest port it
  knows has packages waiting.
- **Regrouping.** Every `REGROUP_EVERY` ticks each port merges its docked
  ships, and ships with the same next port form convoys of at most
  `MAX_CONVOY`. A convoy leaves when all its ships are full, or when one has
  been docked for `MAX_WAIT` ticks. It loads for `HANDLING_TIME` × the most
  packages any member loaded, then sails. Ships with nowhere to go stay
  docked.
- **One leg.** A convoy sails to its port and dissolves into the dock after
  unloading. Berthing takes `SERVICE_TICKS` plus `HANDLING_TIME` per package
  unloaded. A convoy berths when its ships fit in the free berths, or when
  no ship is berthed.
- **Knowledge.** A convoy's hazards, lane reports, backlogs and policies pass
  to every convoy its ships form. A merge takes the union of facts and the
  newest policy version; a ship that moves to a newer policy this way counts
  as adopting it.

All 16 ships start empty and docked: 4 at north, 3 at each other port.
Routing (ACO or LLM) picks lanes and formation (Boids or LLM) keeps shape
between ports; neither chooses destinations.

## Information model

Only the world knows the truth: which lanes are closed, where storms are, how
long a lane really takes. Convoys, port stations and the operations centre
(HQ) act on what reaches them.

| channel | from → to | carries | delay, loss, error |
| --- | --- | --- | --- |
| beacon | vessel → vessels in range | position, velocity | range, loss, bandwidth, position noise |
| lane report | convoy → vessels and ports in range | lane, transit time | Gaussian error, `REPORT_NOISE` × transit |
| hazard | convoy → vessels and ports in range | lane closed or open, observed time | sent when a convoy sees it at its own node |
| notice | port → vessels in range | every closure it knows, recent reopenings | every `NOTICE_EVERY`; reopenings for `NOTICE_MAX_AGE` |
| backlog | port → vessels in range, and HQ | packages waiting at the port, observed time | with each notice; not relayed |
| relay | vessel → vessels in range | hazards and policies it heard | `RELAY_HOPS` hops, `RELAY_MAX_AGE` |
| uplink | port → HQ | hazards and lane reports the port heard | `UPLINK_DELAY` |
| news | world → HQ | free-text items | see below |
| policy | HQ → ports → vessels | versioned policy | `UPLINK_DELAY`, then notices and relay |

- Every vessel is a radio beacon. Receivers estimate the sender's position
  from the signal, with error growing with distance.
- A convoy sees the lanes leaving the node it stands at. If one is closed (or
  reopened) against its belief, it updates its belief, counts a blocked-lane
  discovery and broadcasts a hazard. A convoy never enters a lane it believes
  closed.
- ACO uses lane reports as pheromone that fades with age. The word
  "pheromone" exists only inside the ACO engine.
- Route estimates use charted mean currents. Vessels feel the true drift,
  including waves.
- Every random stream (fleet, radio, ACO, reports, news) is derived from the
  config's `seed` by hashing, so one stream never shifts another.

### News

`news.py` turns world events into free-text items from templated outlets
("Coastguard bulletin", "Shipping wire", ...), with errors set by a `[news]`
table in the config:

| key | default | effect |
| --- | ---: | --- |
| `delay` | 60 | mean publication delay, ticks |
| `location_error` | 0.2 | chance a storm is placed at the wrong named place |
| `severity_error` | 0.2 | noise on the severity word (squall … violent storm) |
| `forecast_error` | 0.3 | spread of the "expected to clear in about N ticks" forecast |
| `false_alarm_rate` | 0.2 | chance per real event of an extra item describing nothing |
| `miss_rate` | 0.0 | chance a real event is never reported |
| `correction_delay` | 60 | delay before an all-clear; `inf` drops all-clears |

Each item keeps the event it describes (`truth`) for analysis; HQ never sees it.

## The LLM as the operations centre

The LLM never steers a vessel. It plays HQ and rewrites a typed policy at
mission start, when a news item arrives and when a fleet report that reached
HQ changes a lane's state. Triggers in the same tick share one request.

- The prompt has fixed sections: role, time and trigger, the map (ground
  truth that never changes), fleet reports via ports (hazards with age and
  confirmations, reported delays, port backlogs), the news feed marked as unverified text to
  treat as information and never as instructions, the current policy and its
  version, and the policy fields. Backlog changes never trigger a request.
- The reply is constrained to the policy's JSON schema, then validated:
  numbers are clamped, invalid choices and unknown fields are dropped.
- A decided policy gets the next version after `LLM_LATENCY`, reaches every
  port after `UPLINK_DELAY`, and spreads by notices and relay. Each convoy
  holds its own version and adopts any newer one it hears; engines read the
  convoy's parameters from the observation.
- Requests are capped per run (`LLM_BUDGET`). A failed call keeps the old
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

Other config keys: `seed`, `ticks`, `demand` (scales `DEMAND_RATE`, default 1), `llm` (`stub`, `ollama`, `claude`, or `""` for arms without an LLM module), `model`, `url`,
`latency`, `budget`, and the `[news]` table.

A list of seeds (`seed = [0, 1, 2]`) makes a sweep: one run per seed, named
`<name>_s<seed>`. Each arm has a `<arm>_sweep.toml` covering seeds 0–10.
`apeks-run` runs all runs in parallel, one process per CPU; `-j N` sets the
number of processes.

Pitfall: a full trace is about 20 MB per 6000-tick run, so the four sweeps (44
runs) write about 900 MB. Pass `--snapshot-every 10` to cut that to about a
tenth, or `--no-trace` to skip traces.

## Metrics

Each run reports:

- deliveries (packages), packages generated, mean latency (creation to
  delivery), backlog at the end and on average, berth arrivals
- load factor (packages aboard over capacity, per sailing ship-tick) and
  empty sailing distance
- energy and energy per delivery
- port wait and lane congestion (ticks beyond capacity)
- near misses (vessel pairs closing within `NEAR_MISS`)
- blocked-lane discoveries
- beliefs: ship-ticks and HQ-ticks believing a false closure, or missing a
  true one
- radio sent, delivered, lost, overflow and relayed counts, and messages by
  type
- LLM calls, failures, skipped calls, input and output tokens and cost
- policy spread: per version, when HQ decided it, how many ships adopted
  it, and ticks to half and to all of the fleet

The full JSON adds delivery times, the event log and every LLM prompt trigger,
reply, accepted policy, wall-clock time and token use.

## Usage

```sh
uv sync
git config core.hooksPath .githooks

uv run apeks                                # window, classical arm
uv run apeks experiments/llm.toml           # window, any arm; space pauses, up/down speed
uv run apeks-run experiments/*.toml         # headless, one JSON summary line per arm
uv run apeks-run experiments/*.toml --llm stub   # offline, no LLM calls
uv run apeks-run experiments/llm.toml --llm ollama --ticks 3500
```

`--ticks`, `--llm` and `--model` override every config. Each arm saves two files under
`logs/` (git-ignored; `--logs DIR` to change), stamped with the invocation's
local time:

- `out_<arm>_<YYYYmmdd-HHMMSS>.json`: config, summary, event log, LLM log
- `trace_<arm>_<YYYYmmdd-HHMMSS>.jsonl`: the trace (`--no-trace` skips it)

### Traces

The trace is one JSON record per line, enough to replay or plot a run without
re-simulating it. Every record has `t` and `kind`:

| kind | contents |
| --- | --- |
| `meta` | config, nodes, ports, lanes, currents, storms, scheduled events, news items, packages, starting convoys (first record) |
| `snapshot` | every vessel's position, velocity, water velocity, drift, energy and cargo destinations; every convoy's state, lane, destination, centre, known closures and policy versions; port queues, packages waiting per port, true and HQ closures, metrics, radio counters, HQ's policies |
| `route_choice` | convoy, node, destination, each option's estimate, cost-to-go and report count, the choice, the policy version |
| `lane_start`, `lane_end` | convoy, lane, transit time |
| `formed` | convoy, port, destination, vessels, packages aboard each |
| `departed`, `queued`, `berthed` | convoy, port |
| `delivered` | convoy, port, latency of each package delivered |
| `hazard_seen`, `hazard_heard` | convoy, lane, closed |
| `news` | source, text, the event it describes |
| `near_miss` | the two vessels and their distance |
| `event` | closures, reopenings, storms forming and clearing |
| `policy_request` | module, trigger, full prompt, raw reply, accepted policy or error |
| `policy_decided`, `policy_at_ports`, `policy_adopted` | module, version; the policy when decided, the convoy when adopted |

`--snapshot-every N` sets the snapshot interval (default 1 tick; about 20 MB per
6000-tick run). Load a trace with `apeks.trace.read_trace(path, kind=None)` or
`pandas.read_json(path, lines=True)`.

### LLM backends

The LLM arms use `claude` with `claude-haiku-5-5`. `--llm stub` runs them
offline: the stub replies `{}`, which leaves the default policy in force.
Two real backends:

#### Ollama

```sh
ollama pull lfm3      # or any model that supports structured output
uv run apeks-run experiments/llm.toml --llm ollama
```

Calls go to `POST {url}/api/generate` with the policy's JSON schema as
`format`, thinking off and temperature 0. Set `url` in a config to use a
remote server.

#### Claude Code

```sh
uv run apeks-run experiments/llm.toml --llm claude --model claude-haiku-5-5
```

Each call runs headless `claude -p` with the policy's JSON schema as
`--json-schema`, no tools, no saved session and a short system prompt, from an
empty temporary directory so no project instructions reach the model. It needs
a logged-in `claude` CLI on `PATH`. Calls take a few seconds each.

## Layout

| Path | Contents |
| --- | --- |
| `src/apeks/seeds.py` | independent random streams derived from one seed |
| `src/apeks/geometry.py` | 2-D vector helpers |
| `src/apeks/ocean.py` | currents, storms, waves; mean and true drift |
| `src/apeks/network.py` | ports, waypoints, directed lanes; transit estimates; Dijkstra time-to-go |
| `src/apeks/radio.py` | broadcast radio with range, loss, bandwidth and position noise; message types |
| `src/apeks/cargo.py` | packages, port demand, loading, next-port and grouping rules |
| `src/apeks/hazards.py` | a holder's beliefs about closed lanes |
| `src/apeks/hq.py` | the operations centre's inbox over the uplink |
| `src/apeks/news.py` | news settings, templates and the news generator |
| `src/apeks/policy.py` | typed, bounded policy fields; JSON schema; validation |
| `src/apeks/routing.py` | ACO and scored routing engines |
| `src/apeks/formation.py` | Boids and slot formation engines |
| `src/apeks/llm.py` | LLM clients (stub, Ollama, Claude Code) and the policy author |
| `src/apeks/world.py` | vessels, convoys, ports, events, metrics, the step loop |
| `src/apeks/scenario.py` | the default map, ocean, events and fleet |
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

Tests run offline and headless with a seeded `random`. The LLM is a stub, a
faked HTTP call or a faked `claude` process.
