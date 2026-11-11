# Plan: cargo, purposeful ships and port regrouping

## Goal

Ships carry packages, and packages decide where ships go. In report 3, each convoy shuttles between two fixed ports and a "delivery" is any berth arrival. Here a fixed heuristic, the same in every arm, picks each ship's next port from its cargo and the demand it knows of. Ships regroup into convoys only at ports. Routing (ACO or LLM) still picks lanes, and formation (Boids or LLM) still keeps a convoy's shape between ports.

## Decisions

| # | decision |
| --- | --- |
| 1 | Ports generate packages by a Poisson process, one stream per port: `derive(seed, "demand:<port>")`. Every arm faces the same demand. |
| 2 | A package has an origin, a destination (uniform over the other ports) and a creation time. No deadlines or priorities. |
| 3 | Capacity is per ship: an integer `CAPACITY`, the same for every ship. |
| 4 | "Deliveries" counts packages delivered. Also measured: package latency (creation to delivery), load factor, empty sailing distance, backlog, and berth arrivals. |
| 5 | The next port is chosen by a fixed heuristic in every arm. The LLM does not choose destinations. |
| 6 | Each ship loads first the waiting packages for the destination its cargo already favours, then the most common waiting destination. Its next port is the most common destination aboard, with ties going to the nearest by charted time. An empty ship heads for the nearest port it knows has waiting packages. |
| 7 | Ships learn other ports' backlogs only from port-station notices; HQ learns them by uplink. The r3 information model holds. |
| 8 | The regrouping rule is fixed and not part of any policy. |
| 9 | All ships start empty and docked. The scenario gives a ship count per port. |
| 10 | A convoy lasts one leg. It forms at departure and dissolves into the port's dock after unloading. Packages belong to ships. |
| 11 | Berths count ships. A convoy berths when its ships fit in the free berths, or when the port has no berthed ships at all. Berthing takes `SERVICE_TICKS` plus `HANDLING_TIME` per package unloaded. |
| 12 | Every `REGROUP_EVERY` ticks each port regroups its docked ships. Ships with the same next port form convoys of at most `MAX_CONVOY`. A convoy departs when every member is full, or when one member has been docked for `MAX_WAIT` ticks. Before sailing, it loads for `HANDLING_TIME` × the most packages any member loaded. Ships with nowhere to go stay docked. |
| 13 | A ship may sail alone, as a convoy of one. |
| 14 | Every port has the same base rate, `DEMAND_RATE`, scaled by the config's `demand`, and its queue has no limit. |
| 15 | Loading is per ship, in ship-id order. Grouping comes after loading, so a convoy never splits cargo. |
| 16 | The arms stay the same. HQ's prompt adds the port backlogs it knows of. Routing picks lanes and formation sets shape, spacing and gains. |
| 17 | Knowledge (hazards, lane reports, backlogs, policies) is held by a convoy and carried by its ships. When convoys merge at a port, the new convoy takes the union of their hazards, reports and backlogs, plus the newest policy version any member holds. A ship that moves to a newer policy this way counts as having adopted it. |
| 18 | A convoy that meets a closure mid-leg reroutes to the same port. It never splits or changes destination away from a port. |
| 19 | Backlog changes never trigger an LLM call. They only appear in the prompt. |
| 20 | The fleet keeps r3's 16 ships: 4 at north, and 3 at each other port. |

## Model details

- **Package lifecycle:** generated → waiting at its origin → aboard a ship → delivered when that ship berths at its destination.
- **Docked ships:** ships at a port that are not in a berth or loading belong to one docked convoy for that port. That way every ship is always in exactly one convoy, and the radio, beliefs and formation code stay unchanged. Docked ships hold at the port node and do not take up berths.
- **Backlog message:** port stations broadcast their waiting-package count, with the time observed, alongside their notices. Ships keep the newest count per port. These messages are not relayed. Stations forward their own count to HQ.
- **Policy adoption** is counted per ship, since convoys don't last.
- **Belief metrics** are counted per ship-tick, for the same reason.

## Constants

| name | value | meaning |
| --- | ---: | --- |
| `CAPACITY` | 4 | packages per ship |
| `MAX_CONVOY` | 8 | ships per convoy |
| `HANDLING_TIME` | 5 | ticks per package loaded or unloaded |
| `REGROUP_EVERY` | 10 | ticks between port regroupings |
| `MAX_WAIT` | 100 | most ticks a docked ship waits before a partly full convoy sails |
| `DEMAND_RATE` | tuned | packages per tick per port at `demand = 1`; set so classical runs at about 70–80% of capacity |

## Stages

Each stage is test-first and keeps the suite at 100%.

1. **Demand.** `cargo.py` with `Package` and `generate_demand(ports, rate, seed, horizon)`; `demand` key in the experiment config.
2. **Heuristics.** Pure functions in `cargo.py`: `load(ship, waiting)`, `next_port(cargo, here, times)` and `group(ships)`.
3. **World.** Ship cargo, one-leg convoys, berths counted in ships, unloading and delivery, regrouping and loading, knowledge merge, the backlog message, and metrics.
4. **Scenario and experiment.** Fleet placement, berths, demand. Tune `DEMAND_RATE` with a stub run of the classical arm.
5. **Docs.** README and CLAUDE.md.
6. **Report 4.** Four arms × 11 seeds, with the same statistics as report 3 plus the new metrics.
