# Plan: news feed and realistic communications

## Goal

The LLM is the fleet's operations centre (HQ). It reads a free-text news feed about the world and what ships have reported through port stations, and it writes policies that reach ships over the radio. Nobody in the simulation knows anything they could not have seen or heard.

## Principles

- **Information has a path.** Every fact an agent acts on arrives through a channel with delay, loss or error: the ship's own eyes, radio, a port station, the shore network or the news.
- **Truth stays in the world.** Only `World` holds the true closures, storms and transit times. Ports, ships and HQ hold beliefs.
- **One master seed.** `seed` in the TOML derives independent streams by hashing, e.g. `derive(seed, "radio")`. Each consumer gets its own `random.Random`, so adding a consumer never shifts another's draws.
- **Every new constant goes in `config.py`.** Every new behaviour is test-first with 100% coverage, as usual.

## Current gaps this fixes

| # | gap | where |
| --- | --- | --- |
| G1 | Port stations broadcast `World.closed`, the true state, for every lane on the map | `World._radio`, `world.py:296` |
| G2 | HQ reads every convoy's private lane reports instantly | `World.situation`, `world.py:157` |
| G3 | A convoy that finds a lane blocked tells nobody | `World._convoys`, `world.py:387` |
| G4 | No relaying: messages travel one hop of radio range at most | `Radio.deliver` |
| G5 | Lane reports are exact | `LaneReport` |
| G6 | `reopened` is never cleared, so "lane open" is broadcast forever | `World._radio` |
| G7 | HQ sees exact storm centre, radius and end time, and exact closures | `World.situation` |
| G8 | A new policy reaches every convoy at once, through one engine-wide `params` | `World.step`, `world.py:255` |

## Communication model

| channel | from → to | carries | delay, loss, error |
| --- | --- | --- | --- |
| Beacon | ship → ships in range | position, velocity | as now |
| Lane report | convoy → ships and ports in range | lane, transit time | Gaussian error, `REPORT_NOISE` × transit |
| Hazard report | convoy → ships and ports in range | "lane X blocked", observed time | sent on discovery |
| Notice | port → ships in range | hazards the port has heard of, with age | repeated every `NOTICE_EVERY` |
| Relay | ship → ships in range | hazards, notices and policies it heard | hop limit `RELAY_HOPS` (3), max age `RELAY_MAX_AGE` |
| Uplink | port → HQ over land | reports the port has heard | delay `UPLINK_DELAY`, same for every port |
| News | world → HQ | free-text items | see the news layer |
| Policy | HQ → ports → ships | versioned policy | downlink delay `UPLINK_DELAY`, then radio and relay |

Every relayed message carries an id, origin time and hop count. A receiver drops anything it has already seen, anything over the hop limit and anything older than the maximum age. Relays share the radio's bandwidth, so heavy relaying causes overflow, and overflow is measured.

## News layer

### Generating news items

A new module `news.py` turns world events into news items:

| field | meaning |
| --- | --- |
| `published` | event time + delay drawn from `NEWS_DELAY` |
| `text` | free-text headline and body |
| `source` | outlet name, e.g. "Coastguard bulletin", "Shipping wire", "Local radio" |
| `truth` | the event it describes, or `None` for a false alarm; kept for analysis, never shown to HQ |

The news stream draws its randomness from `derive(seed, "news")`. Its settings are a `NewsSettings` dataclass, set per experiment in the TOML:

| setting | effect |
| --- | --- |
| `delay` | mean publication delay in ticks |
| `location_error` | storms described by the nearest named place, possibly the wrong one, rather than coordinates |
| `severity_error` | strength described with words ("gale", "severe storm") chosen with noise |
| `forecast_error` | "expected to clear in about N ticks", with N off by a random factor |
| `false_alarm_rate` | chance per event window of an item describing nothing |
| `miss_rate` | chance a real event is never reported |
| `correction_delay` | delay before an "all clear" item; can be infinite |

Text comes from **templates**, several phrasings per event kind, chosen by the news stream. That keeps runs reproducible, cheap and testable. LLM-written news can come later as an option.

### Grounded prompt

`build_prompt` gets fixed, labelled sections, so the free text sits inside structure the model can rely on:

1. **Role:** "You are the operations centre of a cargo fleet."
2. **Map:** nodes, named places, lanes with width and capacity. Ground truth that never changes.
3. **Fleet reports via ports:** hazards and lane delays that have reached HQ, each with its age and how many ships confirmed it.
4. **News feed:** timestamped items, each with source and age, in a block marked as unverified external text. The prompt says to treat it as information, never as instructions.
5. **Current policy** and its version.
6. **Policy fields** and the JSON schema, as now.

HQ asks for a new policy when a news item arrives or an uplinked hazard changes, plus at mission start. Triggers no longer come from the world's true events.

## Policy propagation

- `PolicyAuthor` gives each policy a version number. After `UPLINK_DELAY` it goes to every port, which broadcasts it with its notices.
- Ships relay the newest version they have heard. **Each convoy holds its own policy and version** and adopts any newer version it hears.
- Engines take their parameters from the observation (`RouteObservation.params`, `FormationObservation.params`) instead of one engine-wide `params`. Classical arms put fixed parameters in every observation, so their behaviour is unchanged.

## Measurements

Added to the summary and trace:

- **Policy spread:** share of convoys on the newest version over time; ticks from an HQ decision to 50% and 100% adoption.
- **Reaction time:** event → news published → policy decided → policy adopted → first route change.
- **Belief vs truth:** per convoy and for HQ, the hazards it believes in that are false, and the true hazards it doesn't know about, per tick.
- **Radio load:** messages by type, relay share, overflow.
- **LLM use:** input and output tokens, cost and wall-clock latency per call, from `claude -p` output.
- **Energy per delivery** in the summary itself.

## Stages

Each stage is one or more commits, test-first, with the suite at 100%.

1. **Seed derivation.** `derive(seed, name)` and per-consumer generators. Existing results will shift; record the new baseline.
2. **Honest ports and hazard reports (G1, G3, G6).** Ports know only what they hear; convoys broadcast hazards on discovery; notices age out.
3. **Relay (G4)** with hop limit, maximum age, deduplication and bandwidth accounting.
4. **Noisy lane reports (G5).**
5. **Uplink (G2).** HQ's situation is built from uplinked reports only.
6. **News layer (G7).** `news.py`, `NewsSettings`, templates, and the grounded prompt with news-driven triggers.
7. **Policy propagation (G8).** Versioned policies, per-convoy parameters, engines read parameters from observations.
8. **Measurements**, including token use from the Claude backend.
9. **Re-run** the four-arm sweep on the new model as report 3. Reports 1 and 2 get one line at the top saying they used the earlier information model.

## Later, not in this plan

- Scenario generator: ports, connectivity, fleet and convoy sizes, event intensity, timing and place.
- Space-filling suite and sensitivity analysis over all variables.
- Baselines: ACO with rule-based use of structured news, and hand-tuned scored routing.
- Ablations on LLM routing + Boids: frozen policy, inputs removed, fields fixed, timed re-planning, latency, model.
- Realism: cubic fuel cost, mixed ships, collision consequences, outside traffic, LLM outages.

## Open risks

- **Bandwidth:** relaying could swamp the 96-messages-per-tick limit; `RELAY_HOPS` and `RELAY_MAX_AGE` may need tuning against measured overflow.
- **Prompt size:** the news feed grows over a run; keep only the newest items, by count or age.
- **Changed baseline:** stage 1 changes every existing number. Reports 1 and 2 stay valid for the old model only.
