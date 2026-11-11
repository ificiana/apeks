# apeks: LLM-authored policies against swarm intelligence for autonomous cargo convoys

## Abstract

`apeks` is a discrete-time simulator of autonomous cargo convoys on a directed sea-lane network, built to compare two ways of controlling a fleet. The **classical** controller uses swarm intelligence: Ant Colony Optimisation (ACO) picks lanes, and Boids keeps convoys in formation. The **LLM** controller never steers a ship. A large language model plays the operations centre (HQ): it reads a briefing assembled from what the fleet has reported and from a noisy free-text news feed, and it writes a typed, bounded policy. Deterministic engines then execute that policy on every ship, at every tick. Policies reach ships only over a simulated radio, with delay, loss and limited bandwidth. No agent in the simulation knows anything it could not have seen or heard.

Four arms cross the two modules (routing × formation), each classical or LLM. On a fixed five-port map with a storm and a strait blockade (50 seeds, 200 runs), **LLM routing delivers 2.2× as many packages as ACO**: 388–392 of 456 against 178. It does so at 38% of ACO's latency and 44% of its energy per delivery, better in 50 of 50 seeds (paired effect size d_z 4.6–5.9). **LLM formation brings no cargo benefit and costs safety**: it adds 81 near misses per run over Boids, worse in 45 of 50 seeds. Routing and formation effects are additive for cargo.

On generated maps of 5–250 ports (11 seeds per size, 110 runs), LLM routing beats ACO at every size: 3.4× deliveries at 5 ports, rising to 15× at 250. Classical routing stays at about 57–131 deliveries while demand grows 50×. LLM routing's own share of demand delivered falls from 73% to 4%. Its gains peak at 100 ports, where a fixed budget of 20 LLM calls and radio-queue overflow stop new policies from reaching the whole fleet. It also concentrates traffic: congestion rises to 2–3.4× classical's on maps of 50 ports or more.

The simulator is about 4 800 lines of Python 3.14 with 100% line and branch test coverage. A 250-port, 750-ship, 6000-tick run takes about 6 minutes on one core.

---

## Contents

1. [Problem and research question](#1-problem-and-research-question)
2. [System overview](#2-system-overview)
3. [World model](#3-world-model)
4. [Cargo and logistics](#4-cargo-and-logistics)
5. [Information model](#5-information-model)
6. [Routing engines](#6-routing-engines)
7. [Formation engines](#7-formation-engines)
8. [The LLM as operations centre](#8-the-llm-as-operations-centre)
9. [Implementation](#9-implementation)
10. [Experimental method](#10-experimental-method)
11. [Results on the classic map](#11-results-on-the-classic-map)
12. [Results on generated maps: scaling](#12-results-on-generated-maps-scaling)
13. [Discussion](#13-discussion)
14. [Threats to validity](#14-threats-to-validity)
15. [Future work](#15-future-work)
16. [Reproducing the results](#16-reproducing-the-results)
17. [Appendix: constants, metrics and trace format](#17-appendix-constants-metrics-and-trace-format)

---

## 1. Problem and research question

Fleets of autonomous vessels must make two kinds of recurring decisions:

- **Routing.** At each waypoint, which lane to take next. The decision is global in effect, because every convoy's choice changes congestion for the others, and it must adapt to storms, closures and currents.
- **Formation.** How each ship steers relative to its convoy-mates: a local, continuous, safety-critical control problem.

Classical swarm intelligence handles both with local rules and emergent coordination. ACO reinforces lanes that other agents report as fast. Boids combines separation, alignment and cohesion. These methods are cheap, decentralised and well understood, but they react slowly to rare events and cannot use unstructured information such as a coastguard bulletin.

Large language models read unstructured text, reason about a whole network and can be asked for structured output. They are also slow (seconds per call), costly, non-deterministic and unsafe to put in a real-time control loop.

**Research question.** If an LLM is used only as a *policy author* (rewriting the parameters of deterministic engines at a few event-driven moments, under realistic information constraints), does it outperform swarm-intelligence controllers? How does the answer differ between routing and formation, and between small and large networks?

The project applies both approaches under identical conditions and measures outcomes. It does not assume that either should win.

### Design principles

1. **The LLM never steers.** It writes a policy; deterministic engines act on it every tick. Control latency stays in the simulator, and LLM latency only delays policy changes.
2. **Information has a path.** Every fact an agent acts on arrives through a channel with delay, loss or error. This covers own observation, radio, port stations, the shore uplink and the news.
3. **Truth stays in the world.** Only the `World` object holds true closures, storms and transit times. Convoys, port stations and HQ hold beliefs.
4. **Identical conditions across arms.** Every arm faces the same map, demand, hazards and random streams for a given seed, so comparisons are paired.
5. **Deterministic replay.** One master seed derives every random stream by hashing, so adding a consumer never shifts another's draws. The results do not depend on Python's string-hash seed.

---

## 2. System overview

```
                 ┌──────────────── World (truth) ─────────────────┐
                 │ ocean (currents, storms, waves)                │
                 │ network (ports, waypoints, directed lanes)     │
                 │ closures, demand, vessels, convoys, metrics    │
                 └──────┬────────────────────────────────┬────────┘
                        │ events                         │ observations
                        ▼                                ▼
                   news generator               convoys / port stations
                        │ free text                      │ radio: beacons, lane reports,
                        ▼                                │ hazards, notices, backlogs, relay
                 ┌─────────────┐   uplink (delay)        │
                 │     HQ      │◄─────── port stations ◄─┘
                 │ (LLM author)│
                 └──────┬──────┘
                        │ versioned policy (JSON, schema-checked)
                        ▼
              ports ─► notices ─► convoys ─► relay ─► convoys
                        │
                        ▼
      routing engine (ACO | scored)   formation engine (Boids | slot)
                        │                                │
                        └───────► lane choice / steering ┘
```

Each tick (`World.step`, `DT = 1`) runs these phases in order:

| phase | work |
| --- | --- |
| events | scheduled storms, closures and reopenings; news items published |
| llm | due policy requests are sent and decided policies take effect |
| radio | beacons, lane reports, hazards, port notices and backlogs, relays; delivery with range, loss and bandwidth |
| convoys | arrival at nodes, hazard discovery, route choice via the routing engine |
| ports | unloading, berthing queues, regrouping docked ships into new convoys |
| move | formation steering and physics for every ship |
| safety | near-miss detection, belief accuracy accounting |
| trace | JSONL trace records and snapshots |

### Modules

| module | role |
| --- | --- |
| `seeds` | independent random streams derived from one seed |
| `geometry` | 2-D vectors, segment projection, spatial hash grid |
| `ocean` | currents, storms and waves; mean and true drift |
| `network` | nodes, ports and directed lanes; transit estimates; shortest times |
| `mapgen` | seeded landmass, shore ports, connected lanes and events |
| `scenario` | the fixed five-port classic map |
| `radio` | broadcast radio with range, loss, bandwidth and position noise |
| `hazards` | a holder's beliefs about closed lanes |
| `cargo` | packages, demand, loading, next-port and grouping rules |
| `hq` | the operations centre's inbox over the uplink |
| `news` | templated, error-prone news items |
| `policy` | typed, bounded policy fields; JSON schema; validation |
| `routing` | ACO and scored routing engines |
| `formation` | Boids and slot formation engines |
| `llm` | LLM clients (stub, Ollama, Claude Code) and the policy author |
| `world` | vessels, convoys, ports, metrics, the step loop |
| `experiment`, `runner`, `trace` | configs, parallel headless runner, JSONL traces |
| `controls`, `render`, `app` | interactive pygame dashboard |

The simulation core does not depend on pygame. The dashboard is a separate layer that steps the world on its own thread, so LLM calls never block the window.

---

## 3. World model

### 3.1 Ocean

Ships feel a drift field made of four parts:

- **Base drift**, a constant vector.
- **Currents.** Each is a circle (centre `c`, radius `r`, velocity `v`) whose speed falls linearly to zero at the rim: `v · max(0, 1 − |p − c| / r)`.
- **Storms.** Each is a rotating swirl active during `[start, end)`. The drift at `p` is `perp(normalize(p − c)) · strength · max(0, 1 − |p − c| / r)`.
- **Waves.** An oscillation `A · sin(2πt / T)` along a fixed direction.

`mean_drift` (base + currents + active storms) is what ships *plan* with. `drift` adds the wave term and is what ships *feel*, so plans and outcomes differ.

### 3.2 Network

The network is a directed graph of nodes (ports and waypoints) and lanes:

- A *channel* between two nodes creates two lanes, `a>b` and `b>a`. They share width and capacity, but under currents their costs differ.
- Each lane has a **width** (the navigable corridor) and a **capacity** (convoys at full speed). When `n` convoys share a lane of capacity `k`, each sails at `min(1, k/n)` of its speed, and ticks beyond capacity count as congestion.
- **Transit estimates** integrate mean drift along the lane at 8 sample points. With unit direction `u`, cruise water speed `s` and drift `d`:
  - the vessel crabs against the cross-track drift `|d · ⊥u|`;
  - it gains or loses the along-track drift `d · u`;
  - its ground speed is `√(s² − cross²) + d · u`.

  Any sample where the cross drift reaches `s`, or ground speed is ≤ 0, makes the lane impassable (∞).
- **Shortest times** (`times_to`, `times_from`) run Dijkstra over lane costs. The search uses SciPy's compiled implementation over a CSR matrix, so its distances are the same as a plain-Python Dijkstra (see §9.3).

### 3.3 Maps

**Classic map** (`scenario.py`). Five ports and a handful of waypoints, with a single-file strait `W2>W3` (width 14, capacity 1). Every other lane is width 50–60, capacity 2. A storm runs t = 1500–2500, and the strait is closed t = 3000–4000. Reports 1–7 use it.

**Generated maps** (`mapgen.py`), from `(seed, ports, adjacency)`:

1. **Area.** The map grows as √(ports/5) from 1000 × 700.
2. **Landmass.** A seeded landmass on a grid.
3. **Ports.** Ports sit on the shore. As many waypoints sit in open water, spread by farthest-point sampling: each new point maximises its distance to all points chosen so far.
4. **Lanes.** Each node connects to its `adjacency` nearest nodes in line of sight. Kruskal's algorithm over all line-of-sight pairs then finds the connected parts. Bridging waypoints are added along water paths until the map is one component.
5. **Events.** Every 5 ports add 1 storm, 1 lane closure and 2 currents, all placed and timed from the seed.
6. **Closures never cut the map in two.** Candidate lanes exclude bridges, found with an iterative Tarjan bridge search.
7. **Fleet.** Each port has 4 berths and 3 ships.

| ports | nodes | lanes | ships | storms | closures | currents |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 12 | 16 | 15 | 1 | 1 | 2 |
| 10 | 21 | 36 | 30 | 2 | 2 | 4 |
| 50 | 108 | 189 | 150 | 10 | 10 | 20 |
| 100 | 214 | 383 | 300 | 20 | 20 | 40 |
| 250 | 535 | 951 | 750 | 50 | 50 | 100 |

(Node and lane counts are means over seeds 0–10, adjacency 3.)

### 3.4 Vessels and physics

Each ship has:
- position, ground velocity and water velocity;
- maximum speed `MAX_SPEED = 2` and maximum acceleration `MAX_ACCEL = 0.2` per tick;
- an energy counter and a cargo hold.

Each tick, given the formation engine's desired ground velocity `v*` and the true drift `d`:

```
want    = clamp(v* − d, max_speed)                 # water velocity needed
change  = clamp(want − water_vel, MAX_ACCEL · dt)  # limited thrust
water_vel += change
vel      = water_vel + d
pos     += vel · dt
energy  += |water_vel| · dt
```

Energy is distance pushed *through the water*. A fleet that waits spends less, whatever it delivers, which is why energy per delivery is the efficiency metric.

A convoy arrives when its centre is within `ARRIVE_RADIUS = 30` of a node. A **near miss** is a ship pair that comes closer than `NEAR_MISS = 4`, counted once per close approach.

---

## 4. Cargo and logistics

Logistics rules are fixed and identical across arms. Neither routing nor formation chooses destinations, so differences between arms come only from how ships move between ports.

- **Demand.** Each port generates packages by a Poisson process at `DEMAND_RATE × demand` (0.015 per tick per port at `demand = 1`), with its own random stream `derive(seed, "demand:<port>")`. A package's destination is uniform over the other ports. Queues are unbounded.
- **Ships start** empty and docked, `SHIPS_PER_PORT = 3` at each port. Each holds `CAPACITY = 4` packages.
- **Loading** runs per ship, in ship-id order:
  - The ship picks a target: the most common destination aboard, or the most common waiting destination if it is empty.
  - It loads the oldest packages for the target first.
  - It then loads packages for ports on the way, cheapest detour first, until the hold is full. A port's detour is `t(here → port) + t(port → target) − t(here → target)` in charted time. Unreachable ports are skipped.
- **Next port.** The most common destination aboard. Ties go to the nearest by charted time, then by name. An empty ship heads for the nearest port it *knows* has packages waiting, learned from backlog notices.
- **Regrouping.** Every `REGROUP_EVERY = 10` ticks, each port merges its docked ships. Ships with the same next port form convoys of at most `MAX_CONVOY = 4`.
  - A convoy leaves when all its ships are full, or when one has been docked `MAX_WAIT = 100` ticks.
  - It first loads for `HANDLING_TIME = 5` ticks × the most packages any member loaded.
- **One leg.** A convoy sails to its port, berths and dissolves into the dock after unloading.
  - Berthing takes `SERVICE_TICKS = 40` plus 5 ticks per package unloaded.
  - A convoy berths when its ships fit in the free berths, or when no ship is berthed.
  - If it reaches another port where a member has packages to deliver, it stops there (a *stopover*).
- **Knowledge travels with ships.** A convoy holds hazards, lane reports, backlogs and policy versions. When convoys merge at a port, the new convoy takes the union of facts and the newest policy version. A ship that moves to a newer policy this way counts as adopting it.

---

## 5. Information model

Every channel has a physical path and a failure mode:

| channel | from → to | carries | delay, loss, error |
| --- | --- | --- | --- |
| beacon | vessel → vessels in range | position, velocity | range 260, 5% loss, bandwidth, position noise 3% of distance |
| lane report | convoy → vessels and ports in range | lane, transit time | Gaussian error, sd 10% of transit |
| hazard | convoy → vessels and ports in range | lane closed/open, time observed | sent when a convoy sees it from its own node |
| notice | port → vessels in range | known closures, recent reopenings | every 20 ticks; reopenings repeated for 600 ticks |
| backlog | port → vessels in range, HQ | packages waiting, time observed | with each notice; not relayed |
| relay | vessel → vessels in range | hazards and policies heard | up to 3 hops, hazards younger than 300 ticks |
| uplink | port → HQ | hazards and lane reports the port heard | 30 ticks each way |
| news | world → HQ | free-text items | publication delay, location/severity/forecast errors, false alarms, misses |
| policy | HQ → ports → vessels | versioned policy | LLM latency 20, uplink 30, then notices and relay |

**Radio.** Every vessel is a beacon. Delivery checks range, an independent 5% loss per receiver, and a bandwidth of 96 transmissions per tick per receiver neighbourhood. Excess transmissions are counted as overflow and dropped. Receivers estimate the sender's position with error that grows with distance. Neighbour queries use a spatial hash grid (`geometry.Grid`) with cells about the radio range.

**Discovery.** A convoy observes only the lanes leaving the node it stands at.
- If one is closed, or reopened, against its belief, the convoy updates its belief, counts a *blocked-lane discovery* and broadcasts a hazard.
- It never enters a lane it believes closed.

**Beliefs.** Each tick the world counts ship-ticks spent believing a false closure, or missing a true one, and the same for HQ. This measures how well information flows, separately from what the controllers do with it.

**News.** `news.py` turns world events into items from templated outlets ("Coastguard bulletin", "Shipping wire", ...). Errors are configurable per experiment:

| setting | default | effect |
| --- | ---: | --- |
| `delay` | 60 | mean publication delay (ticks) |
| `location_error` | 0.2 | chance a storm is placed at the wrong named place |
| `severity_error` | 0.2 | noise on the severity word (squall … violent storm) |
| `forecast_error` | 0.3 | spread of "expected to clear in about N ticks" |
| `false_alarm_rate` | 0.2 | chance per real event of an extra item describing nothing |
| `miss_rate` | 0 | chance a real event is never reported |
| `correction_delay` | 60 | delay before an all-clear (`inf` drops all-clears) |

Each item keeps the event it describes (`truth`) for analysis. HQ never sees it.

---

## 6. Routing engines

Both engines read the same `RouteObservation`. It holds:
- the current time, node and destination;
- one `RouteOption` per lane leaving the node;
- the policy parameters *this convoy* holds (`obs.params`).

Each option carries the lane's charted transit estimate `e`, the charted time-to-go from the lane's far end `g` (both avoiding lanes the convoy believes closed), and the lane reports the convoy has heard. Options with infinite `e + g` are not viable.

### 6.1 ACO (classical)

Lane reports act as pheromone that fades with age:

```
τ(lane) = τ₀ + e · Σ_reports exp(−(now − finished_at) / memory) / transit
P(lane) ∝ τ^α · (1 / (e + g))^β
```

- The pheromone term is the lane's estimate times the faded sum of reported speeds (1/transit). A lane reported faster than charted gains pheromone.
- The choice is sampled from `P` with the engine's own random stream.
- The faded sum is maintained incrementally per lane (`ReportLog.faded`): a call decays the stored total and adds only the reports that arrived since. That makes a decision O(new reports) rather than O(all reports).

| parameter | default | bounds | meaning |
| --- | ---: | --- | --- |
| `alpha` | 1.0 | 0–5 | weight of pheromone |
| `beta` | 2.0 | 0–5 | weight of the travel-time heuristic |
| `tau0` | 0.2 | 0.01–5 | base pheromone on unreported lanes |
| `memory` | 400 | 10–5000 | report fade time |

The word "pheromone" exists only inside this engine. The rest of the system speaks of lane reports.

### 6.2 Scored routing (LLM-authored policy)

A deterministic weighted score with an explicit avoid-list:

```
delay  = mean(recent transits) − e      # reports within report_horizon; 0 if none
score  = w_estimate·e + w_cost_to_go·g + w_report·delay + (avoid_penalty if lane ∈ avoid)
choice = argmin score
```

| parameter | default | bounds | meaning |
| --- | ---: | --- | --- |
| `w_estimate` | 1.0 | 0–5 | weight of this lane's estimated time |
| `w_cost_to_go` | 1.0 | 0–5 | weight of time-to-go from the lane's end |
| `w_report` | 1.0 | 0–5 | weight of recent reported delay |
| `report_horizon` | 400 | 10–5000 | maximum report age that counts |
| `avoid` | () | lane ids | lanes to avoid unless there is no other way |
| `avoid_penalty` | 10 000 | 0–100 000 | score added to avoided lanes |

With default parameters and no avoid-list, scored routing is a greedy shortest-time router corrected by reported delays. The LLM's lever is the weights and, mainly, the avoid-list.

---

## 7. Formation engines

Each ship's engine receives a `FormationObservation` with:
- its velocity and its radio neighbours (relative position, velocity, same-convoy flag);
- the vector to the convoy's target node and the offset to the lane's centreline;
- the lane half-width, the speed limit (reduced by lane congestion);
- its slot index, the convoy size, and whether the convoy is holding at a node.

The engine returns a desired ground velocity, clamped to the speed limit. Both engines share two terms:
- **goal velocity:** full speed towards the target, slowing within 30 units when holding at a node;
- **separation:** a push away from any neighbour closer than a radius.

### 7.1 Boids (classical)

Convoy-mates within `neighbour_radius` contribute:
- **alignment:** mean velocity minus own;
- **cohesion:** towards the mates' centre, scaled by distance over the radius.

A **lane-keeping** term pushes back towards the centreline once the ship is beyond half the half-width. The steer is:

```
goal·w_goal + max_speed·(sep·w_sep + coh·w_coh + lane·w_lane) + ali·w_ali
```

| parameter | default | bounds |
| --- | ---: | --- |
| `neighbour_radius` | 80 | 5–300 |
| `separation` | 12 | 1–60 |
| `w_sep`, `w_ali`, `w_coh`, `w_goal`, `w_lane` | 2.0, 0.5, 0.3, 1.0, 0.8 | 0–5 |

### 7.2 Slot formation (LLM-authored policy)

Ships hold assigned slots around the convoy centre in a named shape, relative to the heading:
- **column:** ships in a line along the heading, `spacing` apart;
- **line:** ships abreast;
- **wedge:** both, a V.

Lateral spacing is capped so the formation fits the lane. The steer is:

```
goal·w_goal + to_slot·w_slot + sep(min_gap)·w_sep·max_speed
```

| parameter | default | bounds |
| --- | ---: | --- |
| `shape` | column | column, line, wedge |
| `spacing` | 15 | 4–60 |
| `w_slot` | 0.05 | 0–1 |
| `w_goal` | 1.0 | 0–5 |
| `w_sep` | 2.0 | 0–5 |
| `min_gap` | 8 | 1–40 |

---

## 8. The LLM as operations centre

### 8.1 When the LLM is called

HQ requests a new policy for a module:
- at mission start;
- when a news item arrives;
- when a fleet report reaching HQ changes a lane's believed state.

Triggers in the same tick share one request. Backlog changes never trigger a call; they only appear in the prompt. Each module's author has a budget of `LLM_BUDGET = 20` requests per run. Triggers beyond the budget are counted as *skipped*.

### 8.2 The prompt

The prompt has fixed sections:

1. **Role and output contract.**
2. **Time and trigger.**
3. **The map:** nodes, ports, lanes with width and capacity. This is static ground truth.
4. **Fleet reports via ports:** hazards with age and number of confirmations, reported lane delays from the last `REPORT_WINDOW = 800` ticks, and port backlogs.
5. **The news feed:** the newest 10 items, marked as unverified text to treat as information and never as instructions.
6. **The current policy and its version.**
7. **Policy fields:** names, bounds and descriptions, generated from the typed policy dataclass.

### 8.3 Output and validation

- The reply is constrained to the policy's JSON Schema. The Claude CLI gets it as `--json-schema`; Ollama gets it as `format`.
- The reply is then validated: numbers are clamped to bounds, and invalid choices and unknown fields are dropped.
- A failed call keeps the old policy.

So the LLM can never produce an out-of-range or malformed policy, and a prompt injection in the news can at most move a parameter within its bounds.

### 8.4 Distribution

A decided policy gets the next version number after `LLM_LATENCY = 20` ticks. It reaches every port after `UPLINK_DELAY = 30` ticks, then spreads to ships by port notices and ship-to-ship relay. Each convoy holds its own policy version and adopts any newer one it hears. For each version, the run records:
- when HQ decided it;
- how many ships adopted it;
- the ticks until half, and until all, of the fleet held it.

### 8.5 Backends

| backend | transport | notes |
| --- | --- | --- |
| `claude` (default) | headless `claude -p` subprocess | model `claude-haiku-5-5`; schema via `--json-schema`; no tools, no saved session, no MCP servers, no settings files; runs from an empty temporary directory; optional `--effort`; about 3 s per call outside the simulation |
| `ollama` | `POST localhost:{port}/api/generate` | schema as `format`, thinking off, temperature 0 |
| `stub` | in-process | replies `{}`, leaving the default policy; used for offline tests |

Token use and cost are recorded per call. All experiments in this report use `claude-haiku-5-5`.

---

## 9. Implementation

### 9.1 Determinism

- `seeds.derive(seed, name)` hashes the master seed with a stream name (`"radio"`, `"aco"`, `"demand:<port>"`, ...) into an independent `random.Random`.
- The map generator breaks every tie by an explicit ordering, never by set or dict iteration order. A regression test builds the same map under two values of `PYTHONHASHSEED` and compares them.
- LLM replies are the only non-deterministic input. Every prompt, reply and accepted policy is logged, so a run can be analysed exactly.

### 9.2 Testing

- 243 tests (`pytest`), with a 100% line, branch and function coverage gate enforced in CI and by a pre-commit hook. The only exclusion is the window's `main`, which opens a display.
- Tests run offline and headless: a seeded `random`, the SDL dummy video driver, and LLMs replaced by the stub, a faked HTTP call or a faked `claude` process.
- Every optimised algorithm has a plain reference implementation in the tests, checked on random inputs. This covers farthest-point sampling, the lane builder, closure selection, bridge finding, current lookup and Dijkstra.
- Every optimisation was also checked end to end: run summaries before and after are identical at 10, 20, 40 and 100 ports.

### 9.3 Performance

Scaling to 1000 ports (2148 nodes, 7662 lanes, 3000 ships) needed these techniques, all exact:

| component | technique |
| --- | --- |
| lane costs | drift-integrated transit estimates computed once per set of active storms and shared by every search |
| shortest times | one search per source instead of one per destination; results cached per (direction, node, believed closures) until lane costs change; SciPy's compiled Dijkstra on a CSR matrix cached per cost table |
| why SciPy is exact | Dijkstra's distance is the minimum over paths of a left-to-right float sum along the path; both implementations compute that same sum, so results agree to the bit |
| drift | currents indexed by a grid whose cell is the largest radius, so a lookup adds only the currents that can reach the point, in their original order; active storms worked out once per tick |
| ports | convoys grouped by node once per tick, instead of each port scanning every convoy |
| map generation | farthest-point sampling with a lazy max-heap and a bucket grid; one global sorted pair list with lazy line-of-sight tests; Kruskal with path compression; iterative Tarjan bridges |
| ACO | incremental faded report sums |
| neighbours | spatial hash grid for radio and near misses |

Measured cost, from the r9 sweep (6000 ticks, one process per run on an 80-core server):

| ports | ships | classical ms/tick | llm_routing ms/tick | map build (s) | slowest run (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 15 | 2.2 | 11.0 | 0.02 | 82 |
| 10 | 30 | 4.4 | 23.0 | 0.05 | 164 |
| 50 | 150 | 17.4 | 47.1 | 0.6 | 324 |
| 100 | 300 | 29.2 | 58.6 | 1.8 | 427 |
| 250 | 750 | 60.6 | 82.1 | 9 | 520 |

At 1000 ports a tick costs about 65 ms and the map builds in about 64 s.

Without the LLM, cost grows about as ships^0.85. Ship movement (formation and physics) is the largest phase from 50 ports on, followed by radio and ACO. In llm_routing runs, 40–85% of wall time is spent waiting on Claude (7.5–10 s per call inside a busy sweep).

### 9.4 Tooling

- **Headless runner.** `apeks-run` expands list-valued `seed`, `ports` and `adjacency` keys into one run per combination. It runs them in parallel, one process per CPU by default (`-j N`). Each run writes a summary JSON and an optional JSONL trace (`--snapshot-every N` thins snapshots).
- **Interactive dashboard.** `apeks` opens a three-column pygame window:
  - **Controls:** every config field editable, with a draft-and-apply model.
  - **Map:** pan, zoom and minimap; clicking a ship pauses the simulation and opens a dialog with its cargo and next destinations.
  - **Figures:** the headline figures, charts, a port table and the event log.

---

## 10. Experimental method

### 10.1 Arms

| arm | routing | formation |
| --- | --- | --- |
| `classical` | ACO | Boids |
| `llm_routing` | LLM (scored) | Boids |
| `llm_formation` | ACO | LLM (slot) |
| `llm` (combined) | LLM (scored) | LLM (slot) |

This 2 × 2 design separates the effect of each module and tests their interaction:

```
interaction = llm − llm_routing − llm_formation + classical
```

### 10.2 Sweeps

| report | map | arms | seeds | runs | conditions |
| ---: | --- | --- | --- | ---: | --- |
| 1 | classic | 4 | 1 | 4 | privileged information (see §14) |
| 2 | classic | 4 | 0–10 | 44 | privileged information |
| 3 | classic | 4 | 0–10 | 44 | realistic information model; convoys shuttle between fixed ports |
| 4 | classic | 4 | 0–10 | 44 | cargo, purposeful ships, port regrouping |
| 5 | classic | 4 | 0–10 | 44 | loading on the way, stopovers |
| 6 | classic | 4 | 0–10 | 44 | demand 1.5× |
| 7 | classic | 4 | 0–49 | 200 | as report 6, 50 seeds |
| 8 | generated, 5–40 ports × adjacency 2–4 | classical | 1 | 12 | cost scaling |
| 9 | generated, 5/10/50/100/250 ports | classical, llm_routing | 0–10 | 110 | scaling |

All runs last 6000 ticks.

### 10.3 Metrics

| group | metrics |
| --- | --- |
| cargo | deliveries, packages generated, mean latency (creation → delivery), backlog (end and mean), berth arrivals, load factor (packages aboard / capacity per sailing ship-tick), empty sailing distance |
| efficiency | energy, energy per delivery |
| flow | port wait ticks, lane congestion ticks (convoy-ticks beyond capacity) |
| safety | near misses |
| information | blocked-lane discoveries; ship- and HQ-ticks believing a false closure or missing a true one; radio sent, delivered, lost, overflow, relayed, by type |
| LLM | calls, failures, skipped triggers, input and output tokens, cost; policy spread (ticks to half and all of the fleet) |
| cost | build time, ms per tick, time per phase |

### 10.4 Statistics

Every seed gives every arm the same map, demand and hazards, so all comparisons are **paired by seed**. For each metric and comparison:

- **Wilcoxon signed-rank test**, exact, two-sided, on paired differences. This is the primary test.
- **Paired t-test** as a cross-check.
- **Bootstrap 95% CI** of the mean paired difference (10 000 resamples, seeded).
- **d_z** = mean difference / sd of differences: the paired effect size.
- **"Better in k/n seeds"**: the direction count, with the direction set per metric (higher is better for deliveries, load factor and arrivals; lower for the rest).
- **Holm–Bonferroni correction** across the family of comparisons in a report: 30 for the 4-arm reports (10 metrics × 3 arms against classical), 50 for report 9 (10 metrics × 5 sizes).

**A note on power.** With n = 11 pairs, the smallest possible exact two-sided Wilcoxon p is 2/2¹¹ ≈ 0.00098, reached when every seed agrees. Under Holm over 50 tests, a unanimous result sits at exactly 0.049. Unanimity is the strongest evidence 11 pairs can give, so for report 9 the t-test p values and the CIs carry the strength of evidence. Report 7's 50 seeds remove this floor.

---

## 11. Results on the classic map

The five-port classic map has 15 ships, 6000 ticks, a storm at t = 1500–2500 and the strait closed at t = 3000–4000. The definitive numbers come from report 7 (50 seeds, demand 1.5×, about 456 packages generated per run). Reports 3–6 (11 seeds each) agree in direction.

### 11.1 Headline (report 7, 50 seeds)

| comparison | result |
| --- | --- |
| LLM routing vs classical, deliveries | 388–392 vs 178 of 456 (2.2×), better in 50/50 seeds, d_z 4.6–5.9 |
| LLM routing vs classical, latency | 38% of classical's |
| LLM routing vs classical, energy per delivery | 44% of classical's |
| LLM routing vs classical, congestion | lower in 47–50 of 50 seeds, d_z ≈ −2 |
| LLM routing vs classical, empty distance | +2.3–2.8k, significant after Holm |
| LLM routing vs classical, total energy | 3–4k lower (36/50 seeds), weak (d_z ≈ −0.4) |
| LLM formation vs classical, cargo | no difference in any cargo metric |
| LLM formation vs classical, near misses | +81 per run (classical mean 148), worse in 45/50 seeds, p = 3×10⁻⁸ |
| combined vs LLM routing, deliveries | +4.8, p = 0.12, no difference |
| combined vs LLM routing, near misses | +28.4, p = 0.0007 |
| interaction | none for cargo (additive); sub-additive for near misses (−52.5, p = 4×10⁻⁵) |

### 11.2 How the conclusions held across reports 3–7

| conclusion | r3 | r4 | r5 | r6 | r7 (50 seeds) |
| --- | --- | --- | --- | --- | --- |
| LLM routing beats classical on deliveries | +141%, 11/11 | +90%, 11/11 | +85%, 11/11 | 2.3×, 11/11 | 2.2×, 50/50 |
| LLM routing roughly halves energy per delivery | −60% | about −50% | about −50% | −60% | −56%, 50/50 |
| LLM routing cuts congestion | n.s. | significant | — | 11/11 | 47–50/50 |
| LLM routing sails more empty distance | — | 5× | — | n.s. after Holm | Holm ✓ |
| LLM formation raises near misses | ×2.9, 11/11 | +65%, n.s. after Holm | +95, Holm ✓ | +34, p = 0.15 | +81, p = 3×10⁻⁸ |
| LLM formation matches classical on cargo | yes | yes | latency −159, Holm ✓ | yes | yes, no cargo metric differs |
| combined adds near misses over LLM routing | +53% vs classical | +54, p = 0.002 | more | +31.6, p = 0.058 | +28.4, p = 0.0007 |

Two apparent effects at 11 seeds did not replicate at 50:
- LLM formation's latency gain in report 5;
- the combined arm's delivery edge and deficit in reports 5 and 6.

They show why 11-seed results near the significance threshold need confirmation.

### 11.3 Mechanisms

**Why LLM routing wins.** At mission start, the LLM router puts the single-file strait `W2>W3`/`W3>W2` on its avoid-list in most runs, before anything has happened there. ACO keeps converging on the strait and its feeder corridor, because each traversal deposits pheromone.

A controlled prompt experiment showed the cue behind this choice. It sent the mission-start prompt 5 times per variant to `claude-haiku-5-5`, with no simulation:

| variant | replies avoiding the strait |
| --- | ---: |
| original prompt | 3/5 |
| schema's example lane id changed from `W2>W3` to `W1>W2` | 4/5 |
| strait drawn as width 60, capacity 2 | 0/5 |
| both changes | 0/5 |

The model avoids the strait because the map shows it as the only single-file lane: a bottleneck-aversion reading of the network data it is given. This is not leakage of the scripted blockade. Runs whose first policy used the strait did no worse (report 2, p = 0.21 and 0.30), so foresight is not where the routing gain comes from. The gain comes from steady, deterministic, congestion-aware routing. ACO's stochastic choice and reinforcement concentrate convoys on a few corridors, which turns cargo into queues.

**Why classical's load factor is high.** ACO's slow legs pile up packages at ports, so departing ships are usually full (load factor 0.73–0.74). This is a symptom of slowness, not efficiency. LLM routing clears ports, so ships more often leave after `MAX_WAIT` partly full or sail empty towards a reported backlog. That raises empty distance, but energy per delivery still halves.

**Why LLM formation is unsafe.** Its policies change spacing, gap and shape during events. Each convoy holds its own policy version and adoption spreads over about 40 ticks, so neighbouring convoys can run different formations at once. Slot steering also has a weak separation response compared with Boids' always-on cohesion and alignment. The result is more close approaches (+81 per run) without any cargo benefit.

**Information quality.** Beliefs are often wrong. Convoys spend thousands of ship-ticks per run believing stale closures or missing real ones, and HQ, fed by every port, is wrong far less. The LLM-routing arms log more false and missed ship-ticks than classical. A plausible cause, not verified in traces, is that faster ships spend more time at sea, away from port notices.

---

## 12. Results on generated maps: scaling

Report 9 compares `classical` and `llm_routing` on generated maps of 5, 10, 50, 100 and 250 ports:
- adjacency 3, seeds 0–10, 6000 ticks;
- demand 1× per port, so total demand grows linearly with ports;
- an LLM budget of 20 calls per run.

### 12.1 Outcomes (means over 11 seeds)

| ports | generated | classical delivered | llm_routing delivered | ratio | classical share | llm_routing share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 452 | 97 ± 96 | 331 ± 72 | 3.4× | 21% | 73% |
| 10 | 906 | 131 ± 66 | 534 ± 126 | 4.1× | 14% | 59% |
| 50 | 4 473 | 88 ± 32 | 931 ± 342 | 10.6× | 2% | 21% |
| 100 | 8 968 | 81 ± 29 | 1 086 ± 464 | 13.5× | 1% | 12% |
| 250 | 22 565 | 57 ± 10 | 863 ± 226 | 15× | 0.3% | 4% |

| ports | arm | latency | load factor | mean backlog | arrivals | energy/delivery | congestion | near misses |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | classical | 1 600 | 0.66 | 144 | 41 | 2 702 | 20 689 | 95 |
| 5 | llm_routing | 1 251 | 0.85 | 41 | 137 | 399 | 9 799 | 60 |
| 10 | classical | 2 265 | 0.65 | 322 | 66 | 2 374 | 29 299 | 267 |
| 10 | llm_routing | 1 348 | 0.88 | 123 | 292 | 474 | 29 771 | 115 |
| 50 | classical | 2 235 | 0.42 | 1 951 | 72 | 16 314 | 149 932 | 2 396 |
| 50 | llm_routing | 2 093 | 0.69 | 1 496 | 686 | 1 186 | 317 617 | 642 |
| 100 | classical | 2 373 | 0.37 | 4 005 | 72 | 35 908 | 299 612 | 6 833 |
| 100 | llm_routing | 2 233 | 0.58 | 3 402 | 885 | 1 804 | 803 273 | 1 878 |
| 250 | classical | 2 476 | 0.34 | 10 289 | 55 | 120 047 | 721 999 | 22 023 |
| 250 | llm_routing | 2 570 | 0.45 | 9 682 | 785 | 4 476 | 2 489 854 | 5 782 |

### 12.2 Paired tests (llm_routing − classical, Holm over 50)

| metric | 5 | 10 | 50 | 100 | 250 |
| --- | --- | --- | --- | --- | --- |
| deliveries | +234 *, d_z 4.2 | +403 *, 4.6 | +843 *, 2.6 | +1 005 *, 2.3 | +805 *, 3.6 |
| mean latency | −349, n.s. | −917 *, −3.6 | −143, n.s. | −139, n.s. | +93, n.s. |
| load factor | +0.18, n.s. | +0.22 *, 2.5 | +0.27 *, 3.5 | +0.21 *, 3.6 | +0.11 *, 4.6 |
| mean backlog | −104 *, −3.7 | −199 *, −5.9 | −455 *, −3.0 | −603 *, −2.8 | −606 *, −4.2 |
| arrivals | +97 *, 5.9 | +227 *, 3.3 | +614 *, 2.4 | +813 *, 2.2 | +729 *, 3.5 |
| energy | +24k, n.s. | −11k, n.s. | −283k *, −1.5 | −856k *, −2.7 | −3.0M *, −6.9 |
| congestion | −10.9k, n.s. | +0.5k, n.s. | +168k * (worse), 1.3 | +504k * (worse), 2.7 | +1.77M * (worse), 7.4 |
| near misses | −34, n.s. | −153 *, −1.9 | −1 754 *, −2.3 | −4 954 *, −3.9 | −16 240 *, −5.8 |
| energy/delivery | −2 303 *, −1.0 | −1 900 *, −1.9 | −15 129 *, −3.4 | −34 104 *, −3.5 | −115 571 *, −5.9 |

`*` marks Holm p < 0.05, which at n = 11 means all 11 seeds agree (§10.4). Paired t-test p values for the starred rows range from 0.008 to 5×10⁻¹⁰.

### 12.3 LLM use and policy spread

| ports | LLM calls | skipped triggers | input tokens / run | cost / run (USD) | policies reaching every ship | radio overflow |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 7.5 | 0 | 26.6k | 0.008 | 76 of 83 (median 40 ticks) | 0 |
| 10 | 14.2 | 0 | 74.0k | 0.019 | 144 of 156 (median 39 ticks) | 0 |
| 50 | 20 | 42.5 | 334k | 0.080 | 0 of 220 | 8.0M |
| 100 | 20 | 99.8 | 611k | 0.121 | 0 of 220 | 29.0M |
| 250 | 20 | 265 | 1.41M | 0.293 | 0 of 220 | 91.2M |

No call failed. The prompt grows with the map, from about 3.5k tokens per call at 5 ports to 70k at 250. Whole-sweep Claude cost was about US$5.7.

### 12.4 Scaling findings

1. **LLM routing's advantage grows with the map.** It goes from 3.4× deliveries at 5 ports to 15× at 250, with every seed agreeing at every size.
2. **Classical routing does not scale.** ACO delivers about the same 57–131 packages whether the map has 5 or 250 ports, so its share falls from 21% to 0.3%, and its load factor falls to 0.34. A likely cause, not isolated here, is that the reinforcement signal spreads thin over hundreds of lanes: most lanes never receive enough reports to stand out from the base level `τ₀`.
3. **LLM routing peaks at 100 ports.** Deliveries go 931 → 1 086 → 863 from 50 to 250 ports, and share falls 21% → 12% → 4%. Two limits bind from 50 ports on:
   - the 20-call budget is spent early, and 42–265 later triggers are skipped;
   - radio queues overflow, so no policy version ever reaches every ship.

   The fleet therefore runs a mixture of policy versions.
4. **LLM routing buys throughput with congestion.** From 50 ports on, it concentrates traffic on the lanes its scores favour, giving 2.1–3.4× classical's congestion ticks. Part of this is volume: it completes 10–14× as many legs.
5. **Near misses fall under LLM routing at every size from 10 ports.** At 250 ports they drop to 26% of classical's. Classical's ships linger in crowded areas longer.
6. **Latency stops separating the arms beyond 10 ports.** Within 6000 ticks only short trips finish on large maps, so the delivered packages' latency is censored similarly in both arms.

---

## 13. Discussion

**The LLM is useful as a policy author, not a controller.** Across every condition tested, the LLM did best where its output is a few high-level choices with large consequences: which lanes to avoid, and how to weigh estimates against reports. The deterministic engine then applied those choices consistently thousands of times. This splits the work between the LLM's strengths (reading a whole network and unstructured news) and the simulator's real-time needs. The LLM makes 7–20 calls in 6000 ticks; the routing engine makes thousands of decisions.

**Much of the routing gain is structural.** The scored engine with an avoid-list is deterministic and greedy over charted times plus reported delays. ACO is stochastic and self-reinforcing. Part of LLM routing's advantage may therefore come from the engine rather than the LLM. Even an LLM that returned default parameters would run a deterministic shortest-time router. The prompt experiment (§11.3) shows the LLM does make meaningful structural choices, but separating engine from author needs a scored-routing arm with a fixed, non-LLM policy (§15).

**Formation is a bad fit for an LLM author.** Formation is local, continuous and safety-critical. The LLM's slot policies changed spacing and shape around events without regard to neighbouring convoys that held different versions. Boids' always-on local rules handled close-quarters safety better. Cargo outcomes were unaffected because formation does not decide where ships go.

**Scale exposes the communication layer.** At 50+ ports the bottleneck is not the LLM's reasoning but policy distribution. The fixed budget throttles how often policy can change, and radio overflow stops versions from spreading. A realistic deployment would need a policy-distribution channel sized to the fleet (port relays, priority for policy messages) and a call budget that scales with the network.

**Cost is not a barrier.** A 6000-tick run at 250 ports costs US$0.29 in Haiku tokens. The real cost is wall-time latency, which the design hides from control by keeping the LLM out of the loop.

---

## 14. Threats to validity

**Internal validity**
- **Engine confound.** LLM routing changes both the author (LLM vs none) and the engine (scored vs ACO). See §13.
- **Information model in reports 1–2.** These used privileged information: ports broadcast true closures, HQ read every report instantly, and policies reached every convoy at once. Their numbers are not comparable with reports 3–9, which use the realistic model in §5. Conclusions in this document rest on reports 3–9.
- **LLM non-determinism.** Claude replies vary between runs, so a run cannot be reproduced bit-for-bit. Paired seeds control the world, not the model. The full prompt and reply logs allow audit.
- **Wall-clock-dependent LLM phase.** Time spent waiting on the CLI varies with service load. It does not affect simulated outcomes (policies take effect after a fixed simulated `LLM_LATENCY`), only measured run time.

**Statistical validity**
- **11 seeds** in reports 2–6 and 9. The exact-Wilcoxon floor makes Holm-corrected significance coarse (§10.4). Report 7's 50 seeds confirm the classic-map conclusions; two 11-seed effects did not replicate.
- **Multiple comparisons** are controlled within each report, not across reports.

**External validity**
- **One model** (`claude-haiku-5-5`), one budget (20 calls), one prompt design.
- **Synthetic world.** Maps, physics, radio and news are simplified models. Absolute numbers (deliveries, energy) are in simulator units and do not transfer, but relative comparisons under identical conditions do.
- **Transient horizon.** 6000 ticks is short for maps of 50+ ports. Backlog grows all run in both arms, so those are transient results, not steady state.
- **Fixed logistics heuristic.** Destinations, loading and grouping are the same in every arm and are not optimised. A different heuristic could change the size of routing effects.
- **Formation is held at Boids** in the scaling study, so report 9 says nothing about formation at scale.

---

## 15. Future work

1. **Separate engine from author.** Add an arm with scored routing under a fixed default policy, and one with ACO parameters tuned by the LLM. This splits LLM routing's gain into engine and reasoning.
2. **Scale the budget.** Make calls proportional to network size (for example per 10 ports) and rerun 100 and 250 ports to test whether the decline at 250 is budget-limited.
3. **Policy distribution at scale.** Give policy messages priority on the radio, or relay them via port stations, and measure reach and lag.
4. **Longer horizons.** Run 20 000+ ticks on large maps to reach, or rule out, a steady backlog.
5. **Formation safety.** Constrain slot policies (minimum gap tied to convoy speed), or synchronise version changes within a convoy, and test whether LLM formation's near-miss cost disappears.
6. **Model comparison.** Repeat key sweeps with larger models and a local Ollama model, and with `--effort` levels.
7. **Prompt-injection robustness.** Seed adversarial news items and measure how far bounded policies can be pushed.

---

## 16. Reproducing the results

### Setup

```sh
uv sync
git config core.hooksPath .githooks
uv run pytest            # 100% coverage gate
```

The `claude` backend needs a logged-in Claude Code CLI on `PATH`. `--llm stub` runs every arm offline.

### Interactive

```sh
uv run apeks                              # classical arm
uv run apeks experiments/llm.toml         # any arm
```

### Sweeps

Each arm has `experiments/<arm>_sweep.toml`. A list-valued `seed`, `ports` or `adjacency` expands into runs named `<arm>[_p<ports>][_k<adjacency>]_s<seed>`. A report sweep, from the repository root:

```sh
mkdir -p logs/rN
TIMEFORMAT='WALL %R s'
{ time uv run apeks-run experiments/*_sweep.toml --snapshot-every 10 --logs logs/rN \
    > logs/rN/summary.jsonl 2> logs/rN/stderr.log; } 2>> logs/rN/stderr.log
```

Report 9 sets `seed = [0, …, 10]` and `ports = [5, 10, 50, 100, 250]` in `classical_sweep.toml` and `llm_routing_sweep.toml`, and runs those two files only.

Outputs, per run:
- `out_<run>_<stamp>.json`: config, summary, event log, LLM log;
- `trace_<run>_<stamp>.jsonl`: the trace;
- one summary line per run in `summary.jsonl`.

Each report in `reports/` states its exact command, commit and wall time.

### Hardware

Sweeps are CPU-bound and run one process per run. On an 80-core server:
- report 7 (200 runs) took 22 minutes;
- report 9 (110 runs, up to 250 ports) took 10 minutes.

---

## 17. Appendix: constants, metrics and trace format

### 17.1 Constants (`config.py`)

| group | constant | value | meaning |
| --- | --- | ---: | --- |
| time | `DT` | 1 | tick length |
| vessels | `MAX_SPEED` | 2.0 | water speed limit |
| | `MAX_ACCEL` | 0.2 | per-tick velocity change |
| | `CRUISE_FRACTION` | 0.9 | share of max speed used for estimates |
| | `ARRIVE_RADIUS` | 30 | convoy centre within this of a node arrives |
| | `NEAR_MISS` | 4 | near-miss distance |
| | `HOLD_WIDTH` | 100 | corridor width while holding at a node |
| ports | `BERTHS` | 4 | berths per port, in ships |
| | `SHIPS_PER_PORT` | 3 | starting ships per port |
| | `SERVICE_TICKS` | 40 | mooring and casting-off time |
| cargo | `CAPACITY` | 4 | packages per ship |
| | `MAX_CONVOY` | 4 | ships per convoy |
| | `HANDLING_TIME` | 5 | ticks per package loaded or unloaded |
| | `REGROUP_EVERY` | 10 | ticks between port regroupings |
| | `MAX_WAIT` | 100 | docked ticks before a partly full convoy sails |
| | `DEMAND_RATE` | 0.015 | packages per tick per port at demand 1 |
| radio | `RADIO_RANGE` | 260 | broadcast range |
| | `RADIO_LOSS` | 0.05 | per-receiver loss |
| | `RADIO_BANDWIDTH` | 96 | transmissions per tick |
| | `RADIO_NOISE` | 0.03 | position error per unit distance |
| | `NOTICE_EVERY` | 20 | port notice interval |
| information | `REPORT_NOISE` | 0.1 | lane-report error sd, share of transit |
| | `RELAY_HOPS` | 3 | ship-to-ship relay hops |
| | `RELAY_MAX_AGE` | 300 | hazards older than this are not relayed |
| | `NOTICE_MAX_AGE` | 600 | reopenings repeated for this long |
| | `UPLINK_DELAY` | 30 | port ↔ HQ, each way |
| | `REPORT_WINDOW` | 800 | report age shown in HQ's briefing |
| | `NEWS_KEEP` | 10 | news items shown to HQ |
| LLM | `LLM_LATENCY` | 20 | ticks before a decided policy takes effect |
| | `LLM_BUDGET` | 20 | requests per author per run |

### 17.2 Run summary fields

`ticks`, `deliveries`, `mean_latency`, `generated`, `backlog`, `mean_backlog`, `arrivals`, `load_factor`, `empty_distance`, `energy`, `energy_per_delivery`, `wait_ticks`, `congestion_ticks`, `near_misses`, `blocked_discoveries`, `beliefs` (`belief_false`, `belief_missed`, `hq_belief_false`, `hq_belief_missed`), `radio` (`sent`, `delivered`, `lost`, `overflow`, `relayed`, `by_type`), `llm` (per module: `calls`, `failures`, `skipped`, `input_tokens`, `output_tokens`, `cost_usd`), `policies` (per version: decided at, adopters, ticks to half and to all), `map` (`kind`, `ports`, `nodes`, `lanes`, `adjacency`), `timing` (`build_s`, `wall_s`, `per_tick_ms`, `phases`), `delivery_times`.

### 17.3 Trace records

One JSON object per line; every record has `t` and `kind`.

| kind | contents |
| --- | --- |
| `meta` | config, nodes, ports, lanes, currents, storms, scheduled events, news items, packages, starting convoys |
| `snapshot` | every vessel's position, velocities, drift, energy, cargo; every convoy's state, lane, destination, centre, beliefs, policy versions; port queues and backlogs; true and HQ closures; metrics; radio counters; HQ policies |
| `route_choice` | convoy, node, destination, each option's estimate, cost-to-go and report count, the choice, the policy version |
| `lane_start`, `lane_end` | convoy, lane, transit time |
| `formed`, `departed`, `queued`, `berthed`, `stopover`, `delivered` | convoy life cycle and cargo |
| `hazard_seen`, `hazard_heard` | convoy, lane, closed |
| `news` | source, text, the event it describes |
| `near_miss` | the two vessels and their distance |
| `event` | closures, reopenings, storms forming and clearing |
| `policy_request` | module, trigger, full prompt, raw reply, accepted policy or error |
| `policy_decided`, `policy_at_ports`, `policy_adopted` | policy distribution |

Load a trace with `apeks.trace.read_trace(path, kind=None)` or `pandas.read_json(path, lines=True)`.

### 17.4 Per-report index

| report | file | subject |
| ---: | --- | --- |
| 1 | `reports/1.md` | four arms, one seed; the strait-avoidance prompt experiment |
| 2 | `reports/2.md` | 11-seed sweep, privileged information |
| 3 | `reports/3.md` | realistic information model |
| 4 | `reports/4.md` | cargo, purposeful ships, port regrouping |
| 5 | `reports/5.md` | loading on the way, stopovers |
| 6 | `reports/6.md` | demand 1.5× |
| 7 | `reports/7.md` | 50 seeds; consolidation of reports 3–6 |
| 8 | `reports/8.md` | cost per tick on generated maps, classical only |
| 9 | `reports/9.md` | LLM routing vs classical at 5–250 ports |
