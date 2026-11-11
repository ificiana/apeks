"""The fleet simulation: vessels, convoys, ports, radio, events and metrics.

The world owns physical truth (positions, closures, storms). Everyone else acts
on beliefs that arrived by a path:

- a convoy sees the lanes leading out of the node it stands at, and hears lane
  reports, hazards and policies by radio, relaying new hazards and policies;
- a port station hears the radio, repeats the hazards and policies it holds,
  and forwards what it hears to HQ over a land link with a delay;
- HQ (the policy authors) reads those forwarded reports and the news feed, and
  sends versioned policies back down through the port stations.

Ships carry packages. A convoy lasts one leg: at a port its ships unload, wait
docked, load by the fixed heuristics in ``cargo`` and regroup by next port.

Decisions come from the plugged-in engines, which read the policy each convoy
holds from their observation.
"""

import copy
import math
import random
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

from apeks import config
from apeks.cargo import Package, detours, group, load, next_port, target_port
from apeks.formation import FormationObservation, Neighbour
from apeks.geometry import (
    Vec,
    add,
    clamp_length,
    closest_on_segment,
    dist,
    length,
    scale,
    sub,
)
from apeks.hazards import HazardBook, observe_channel, reverse
from apeks.hq import Hq
from apeks.llm import PolicyAuthor
from apeks.network import SeaNetwork
from apeks.news import NewsItem
from apeks.ocean import Ocean
from apeks.radio import (
    Backlog,
    Hazard,
    LaneReport,
    Message,
    PolicyMsg,
    PositionReport,
    Radio,
)
from apeks.routing import Report, RouteObservation, RouteOption
from apeks.trace import Tracer

if TYPE_CHECKING:
    from apeks.mapgen import Geography


class RoutingEngine(Protocol):
    params: Any

    def choose(self, obs: RouteObservation) -> str | None: ...


class FormationEngine(Protocol):
    params: Any

    def steer(self, obs: FormationObservation) -> Vec: ...


@dataclass
class Vessel:
    id: str
    convoy: str
    pos: Vec
    vel: Vec = (0.0, 0.0)
    water_vel: Vec = (0.0, 0.0)
    max_speed: float = config.MAX_SPEED
    energy: float = 0.0
    capacity: int = config.CAPACITY
    cargo: list[Package] = field(default_factory=list)
    docked_since: float = 0.0
    loaded_here: int = 0

    @property
    def full(self) -> bool:
        return len(self.cargo) >= self.capacity

    def propel(self, desired: Vec, drift: Vec, dt: float) -> None:
        """Ask for ground velocity ``desired``; thrust and acceleration are limited."""
        want = clamp_length(sub(desired, drift), self.max_speed)
        change = clamp_length(sub(want, self.water_vel), config.MAX_ACCEL * dt)
        self.water_vel = add(self.water_vel, change)
        self.vel = add(self.water_vel, drift)
        self.pos = add(self.pos, scale(self.vel, dt))
        self.energy += length(self.water_vel) * dt


@dataclass
class Convoy:
    """Ships sailing one leg together, or waiting together at a port.

    It holds what its ships know; they carry that into the next convoy.
    """

    id: str
    vessels: list[str]
    node: str
    destination: str | None = None
    lane: str | None = None
    lane_started: float = 0.0
    state: str = "sailing"  # sailing | queued | berthed | docked | loading
    service_until: float = 0.0
    hazards: HazardBook = field(default_factory=HazardBook)
    reports: dict[str, list[Report]] = field(default_factory=dict)
    policies: dict[str, tuple[int, Any]] = field(default_factory=dict)
    backlogs: dict[str, tuple[int, float]] = field(default_factory=dict)
    outbox: list[Message] = field(default_factory=list)
    _seen: dict[tuple[str, str, float], Report] = field(default_factory=dict)

    @property
    def known_closed(self) -> set[str]:
        return self.hazards.closed()

    def absorb(self, other: Convoy) -> None:
        """Take in everything ``other`` knows; keep the newer of each fact."""
        for lane, b in other.hazards.entries.items():
            for source in sorted(b.sources):
                self.hazards.update(lane, b.closed, b.observed_at, source)
        for key, report in other._seen.items():
            self._add_report(key, report)
        for module, (version, params) in other.policies.items():
            if version > self.policies.get(module, (-1, None))[0]:
                self.policies[module] = (version, params)
        for port, (n, at) in other.backlogs.items():
            if at > self.backlogs.get(port, (0, -1.0))[1]:
                self.backlogs[port] = (n, at)

    def knowledge(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "hazards": self.hazards,
                "reports": self.reports,
                "policies": self.policies,
                "backlogs": self.backlogs,
                "_seen": self._seen,
            }
        )

    def _add_report(self, key: tuple[str, str, float], report: Report) -> None:
        if key not in self._seen:
            self._seen[key] = report
            self.reports.setdefault(key[1], []).append(report)

    def hear(
        self, msg: LaneReport | Hazard | PolicyMsg | Backlog, now: float
    ) -> str | None:
        """Take in a message; queue a relay for new hazards and policies.

        Returns ``"changed"``/``"refreshed"`` for hazard news, ``"adopted"``
        for a newer policy, ``None`` otherwise.
        """
        if isinstance(msg, PolicyMsg):
            if self.policies.get(msg.module, (-1, None))[0] >= msg.version:
                return None
            self.policies[msg.module] = (msg.version, msg.params)
            self._relay(msg)
            return "adopted"
        if isinstance(msg, Hazard):
            got = observe_channel(
                self.hazards, msg.lane, msg.closed, msg.observed_at, msg.origin
            )
            if got is not None and now - msg.observed_at <= config.RELAY_MAX_AGE:
                self._relay(msg)
            return got
        if isinstance(msg, Backlog):
            if msg.observed_at > self.backlogs.get(msg.port, (0, -1.0))[1]:
                self.backlogs[msg.port] = (msg.waiting, msg.observed_at)
            return None
        key = (msg.sender, msg.lane, msg.finished_at)
        self._add_report(key, Report(msg.transit, msg.finished_at))
        return None

    def _relay(self, msg: Hazard | PolicyMsg) -> None:
        if msg.hops < config.RELAY_HOPS:
            self.outbox.append(replace(msg, sender=self.vessels[0], hops=msg.hops + 1))


@dataclass
class Station:
    """A port's radio station: what it has heard, and the policies it sends."""

    id: str
    hazards: HazardBook = field(default_factory=HazardBook)
    policies: dict[str, tuple[int, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class Event:
    at: float
    kind: str  # close | open
    lane: str
    text: str


PHASES = ("events", "llm", "radio", "convoys", "ports", "move", "safety", "trace")
"""Parts of a tick that ``World.timings`` times separately."""


@dataclass
class World:
    network: SeaNetwork
    ocean: Ocean
    radio: Radio
    vessels: dict[str, Vessel]
    convoys: dict[str, Convoy]
    routing: RoutingEngine
    formation: FormationEngine
    events: list[Event] = field(default_factory=list)
    authors: dict[str, PolicyAuthor] = field(default_factory=dict)
    news: list[NewsItem] = field(default_factory=list)
    hq: Hq = field(default_factory=lambda: Hq(config.UPLINK_DELAY))
    packages: list[Package] = field(default_factory=list)
    waiting: dict[str, list[Package]] = field(default_factory=dict)
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    t: float = 0.0
    closed: set[str] = field(default_factory=set)
    queues: dict[str, list[str]] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    tracer: Tracer | None = None
    geography: Geography | None = None
    _costs: tuple[Any, dict[str, float]] | None = None
    timings: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(PHASES, 0.0)
    )
    metrics: dict[str, Any] = field(
        default_factory=lambda: {
            "deliveries": 0,
            "latencies": [],
            "delivery_times": [],
            "generated": 0,
            "arrivals": 0,
            "load_ticks": 0.0,
            "sail_ticks": 0.0,
            "empty_distance": 0.0,
            "backlog_ticks": 0.0,
            "wait_ticks": 0.0,
            "congestion_ticks": 0.0,
            "near_misses": 0,
            "blocked_discoveries": 0,
            "belief_false": 0,
            "belief_missed": 0,
            "hq_belief_false": 0,
            "hq_belief_missed": 0,
        }
    )
    stations: dict[str, Station] = field(default_factory=dict)
    policies: dict[str, tuple[int, Any]] = field(default_factory=dict)
    adoption: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    _downlink: list[tuple[float, str, int, Any]] = field(default_factory=list)
    _formed: int = 0
    _close_pairs: set[tuple[str, str]] = field(default_factory=set)
    _storms_active: frozenset[int] = frozenset()
    _neighbours: dict[str, tuple[Neighbour, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.news = sorted(self.news, key=lambda item: item.published)
        self.packages = sorted(self.packages, key=lambda p: (p.created, p.id))
        self.waiting = {p: [] for p in self.network.ports}
        self.stations = {p: Station(f"port:{p}") for p in self.network.ports}
        start = {"routing": self.routing.params, "formation": self.formation.params}
        self.policies = {m: (0, params) for m, params in start.items()}
        for convoy in self.convoys.values():
            convoy.policies = dict(self.policies)

    # ----- helpers -------------------------------------------------------

    def centre(self, convoy: Convoy) -> Vec:
        pts = [self.vessels[v].pos for v in convoy.vessels]
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def cruise(self) -> float:
        return config.MAX_SPEED * config.CRUISE_FRACTION

    def lane_costs(self) -> dict[str, float]:
        """Charted transit time of every lane, recomputed when the drift changes.

        The mean drift depends on time only through which storms are active.
        """
        key = (self.ocean, self.ocean.active_storms(self.t))
        if self._costs is None or self._costs[0] != key:
            costs = self.network.lane_times(self.ocean, self.t, self.cruise())
            self._costs = (key, costs)
        return self._costs[1]

    def situation(self, trigger: str) -> dict[str, Any]:
        """HQ's briefing: the chart, what ships reported via ports, the news."""
        delays = {}
        for lane_id, reps in self.hq.reports.items():
            recent = [
                r.transit
                for r in reps
                if self.t - r.finished_at <= config.REPORT_WINDOW
            ]
            if recent:
                est = self.network.transit_estimate(
                    self.network.lanes[lane_id], self.ocean, self.t, self.cruise()
                )
                delays[lane_id] = round(sum(recent) / len(recent) - est, 1)
        net = self.network
        return {
            "time": self.t,
            "trigger": trigger,
            "map": {
                "nodes": {name: list(pos) for name, pos in net.nodes.items()},
                "ports": sorted(net.ports),
                "lanes": {
                    lane.id: {"width": lane.width, "capacity": lane.capacity}
                    for lane in net.lanes.values()
                },
            },
            "fleet_reports": {
                "hazards": [
                    {
                        "lane": lane,
                        "state": "closed" if b.closed else "open",
                        "age": round(self.t - b.observed_at),
                        "confirmations": len(b.sources),
                    }
                    for lane, b in sorted(self.hq.hazards.entries.items())
                ],
                "reported_delay_by_lane": delays,
            },
            "news": [
                {
                    "published": item.published,
                    "age": round(self.t - item.published),
                    "source": item.source,
                    "text": item.text,
                }
                for item in self.hq.news[-config.NEWS_KEEP :]
            ],
            "port_backlogs": {
                port: {"waiting": n, "age": round(self.t - at)}
                for port, (n, at) in sorted(self.hq.backlogs.items())
            },
            "deliveries": self.metrics["deliveries"],
        }

    def _emit(self, kind: str, **data: Any) -> None:
        if self.tracer is not None:
            self.tracer.emit(self.t, kind, **data)

    def static(self) -> dict[str, Any]:
        """The parts of the world that never change, for a trace header."""
        net = self.network
        return {
            "nodes": net.nodes,
            "ports": net.ports,
            "lanes": list(net.lanes.values()),
            "ocean": self.ocean,
            "events": self.events,
            "news": self.news,
            "packages": self.packages,
            "convoys": [
                {"id": c.id, "port": c.node, "vessels": c.vessels}
                for c in self.convoys.values()
            ],
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "vessels": [
                {
                    "id": v.id,
                    "pos": v.pos,
                    "vel": v.vel,
                    "water_vel": v.water_vel,
                    "drift": self.ocean.drift(v.pos, self.t),
                    "energy": v.energy,
                    "cargo": [p.destination for p in v.cargo],
                }
                for v in self.vessels.values()
            ],
            "convoys": [
                {
                    "id": c.id,
                    "state": c.state,
                    "node": c.node,
                    "lane": c.lane,
                    "destination": c.destination,
                    "centre": self.centre(c),
                    "known_closed": sorted(c.known_closed),
                    "reports": sum(len(r) for r in c.reports.values()),
                    "policy_versions": {m: v for m, (v, _) in c.policies.items()},
                }
                for c in self.convoys.values()
            ],
            "queues": {k: list(v) for k, v in self.queues.items()},
            "waiting": {k: len(v) for k, v in self.waiting.items()},
            "closed": sorted(self.closed),
            "hq_closed": sorted(self.hq.hazards.closed()),
            "metrics": {
                k: v for k, v in self.metrics.items() if not isinstance(v, list)
            },
            "radio": dict(self.radio.stats),
            "policies": {
                m: {"version": v, "params": p} for m, (v, p) in self.policies.items()
            },
        }

    # ----- step ----------------------------------------------------------

    def step(self) -> None:
        for name, phase in (
            ("events", self._events_phase),
            ("llm", self._llm_phase),
            ("radio", self._radio),
            ("convoys", self._convoys),
            ("ports", self._ports),
            ("move", self._move),
            ("safety", self._safety_phase),
            ("trace", self._trace_phase),
        ):
            start = time.perf_counter()
            phase()
            self.timings[name] += time.perf_counter() - start
        self.t += config.DT

    def _events_phase(self) -> None:
        self._apply_events()
        self._release()

    def _llm_phase(self) -> None:
        triggers = ["mission start"] if self.t == 0 else []
        triggers += self._read_news()
        triggers += self.hq.receive(self.t)
        self._authors(triggers)
        self._send_down()

    def _safety_phase(self) -> None:
        self._near_misses()
        self._beliefs()

    def _trace_phase(self) -> None:
        if self.tracer is not None and self.tracer.wants_snapshot(self.t):
            self._emit("snapshot", **self.snapshot())

    def _apply_events(self) -> None:
        """Change the true state of the sea. Nobody is told."""
        out = []
        for e in [e for e in self.events if e.at <= self.t]:
            self.events.remove(e)
            pair = {e.lane, reverse(e.lane)}
            if e.kind == "close":
                self.closed |= pair
            else:
                self.closed -= pair
            out.append(e.text)
        active = frozenset(
            i for i, s in enumerate(self.ocean.storms) if s.active(self.t)
        )
        for i in sorted(active - self._storms_active):
            out.append(f"storm forming at {self.ocean.storms[i].centre}")
        for i in sorted(self._storms_active - active):
            out.append(f"storm at {self.ocean.storms[i].centre} cleared")
        self._storms_active = active
        self.log += [f"t={self.t:.0f} {text}" for text in out]
        for text in out:
            self._emit("event", text=text)

    def _release(self) -> None:
        """Packages appear at their origin port."""
        while self.packages and self.packages[0].created <= self.t:
            p = self.packages.pop(0)
            self.waiting[p.origin].append(p)
            self.metrics["generated"] += 1

    def _read_news(self) -> list[str]:
        out = []
        while self.news and self.news[0].published <= self.t:
            item = self.news.pop(0)
            self.hq.news.append(item)
            self._emit("news", source=item.source, text=item.text, truth=item.truth)
            out.append(f"news: {item.text}")
        return out

    def _authors(self, triggers: list[str]) -> None:
        for module, author in self.authors.items():
            if triggers:
                before = len(author.log)
                situation = self.situation("; ".join(triggers))
                situation["policy_version"] = self.policies[module][0]
                author.request(situation, self.policies[module][1], self.t)
                if len(author.log) > before:
                    entry = {k: v for k, v in author.log[-1].items() if k != "t"}
                    self._emit("policy_request", module=module, **entry)
            policy = author.due(self.t)
            if policy is not None:
                version = self.policies[module][0] + 1
                self.policies[module] = (version, policy)
                self._downlink.append((self.t + self.hq.delay, module, version, policy))
                self.adoption[(module, version)] = {"decided": self.t, "adopted": {}}
                self.log.append(f"t={self.t:.0f} new {module} policy v{version}")
                self._emit(
                    "policy_decided", module=module, version=version, params=policy
                )

    def _send_down(self) -> None:
        """Policies reach the port stations after the land-link delay."""
        due = [d for d in self._downlink if d[0] <= self.t]
        self._downlink = [d for d in self._downlink if d[0] > self.t]
        for _, module, version, params in due:
            for station in self.stations.values():
                station.policies[module] = (version, params)
            self._emit("policy_at_ports", module=module, version=version)

    def _radio(self) -> None:
        for v in self.vessels.values():
            self.radio.broadcast(PositionReport(v.id, v.convoy, v.vel), v.pos)
        for convoy in self.convoys.values():
            for msg in convoy.outbox:
                self.radio.broadcast(msg, self.centre(convoy))
            convoy.outbox = []
        if self.t % config.NOTICE_EVERY == 0:
            for port, station in self.stations.items():
                pos = self.network.nodes[port]
                for lane, b in station.hazards.fresh(self.t, config.NOTICE_MAX_AGE):
                    origin = min(b.sources)
                    msg = Hazard(station.id, lane, b.closed, b.observed_at, origin)
                    self.radio.broadcast(msg, pos)
                for module, (version, params) in sorted(station.policies.items()):
                    self.radio.broadcast(
                        PolicyMsg(station.id, module, version, params), pos
                    )
                count = Backlog(station.id, port, len(self.waiting[port]), self.t)
                self.radio.broadcast(count, pos)
                self.hq.send(self.t, count)
        receivers = {v.id: v.pos for v in self.vessels.values()}
        by_id = {s.id: (port, s) for port, s in self.stations.items()}
        for sid, (port, _) in by_id.items():
            receivers[sid] = self.network.nodes[port]
        heard = self.radio.deliver(receivers)
        self._neighbours = {}
        for rid, messages in heard.items():
            if rid in by_id:
                self._station_hears(by_id[rid][1], messages)
                continue
            mine = self.vessels[rid]
            convoy = self.convoys[mine.convoy]
            nbrs = []
            for rx in messages:
                msg = rx.message
                if isinstance(msg, PositionReport):
                    nbrs.append(
                        Neighbour(rx.rel_pos, msg.velocity, msg.convoy == mine.convoy)
                    )
                    continue
                got = convoy.hear(msg, self.t)
                if got == "changed":
                    assert isinstance(msg, Hazard)
                    self._emit(
                        "hazard_heard",
                        convoy=convoy.id,
                        lane=msg.lane,
                        closed=msg.closed,
                        hops=msg.hops,
                    )
                elif got == "adopted":
                    assert isinstance(msg, PolicyMsg)
                    self._adopted(convoy, msg)
            self._neighbours[rid] = tuple(nbrs)

    def _station_hears(self, station: Station, messages: list[Any]) -> None:
        for rx in messages:
            msg = rx.message
            if isinstance(msg, Hazard):
                got = observe_channel(
                    station.hazards, msg.lane, msg.closed, msg.observed_at, msg.origin
                )
                if got is not None:
                    self.hq.send(self.t, msg)
            elif isinstance(msg, LaneReport):
                self.hq.send(self.t, msg)

    def _adopted(self, convoy: Convoy, msg: PolicyMsg) -> None:
        self._adopt(convoy.vessels, msg.module, msg.version)
        self._emit(
            "policy_adopted",
            convoy=convoy.id,
            module=msg.module,
            version=msg.version,
            hops=msg.hops,
        )

    def _adopt(self, vessels: list[str], module: str, version: int) -> None:
        record = self.adoption.get((module, version))
        if record is not None:
            for vid in vessels:
                record["adopted"].setdefault(vid, self.t)

    def _look(self, convoy: Convoy) -> None:
        """A convoy at a node sees whether the lanes leaving it are blocked."""
        for lane in self.network.out_lanes(convoy.node):
            closed = lane.id in self.closed
            if closed == (lane.id in convoy.known_closed):
                continue
            observe_channel(convoy.hazards, lane.id, closed, self.t, convoy.id)
            msg = Hazard(convoy.vessels[0], lane.id, closed, self.t, convoy.id)
            self.radio.broadcast(msg, self.centre(convoy))
            if closed:
                self.metrics["blocked_discoveries"] += 1
            self._emit("hazard_seen", convoy=convoy.id, lane=lane.id, closed=closed)

    def _route_options(self, convoy: Convoy) -> RouteObservation:
        net = self.network
        known = convoy.known_closed
        costs = self.lane_costs()
        to_go = net.times_to(
            str(convoy.destination), self.ocean, self.t, self.cruise(), known, costs
        )
        options = tuple(
            RouteOption(
                lane=lane.id,
                to_node=lane.b,
                estimate=costs[lane.id],
                cost_to_go=to_go.get(lane.b, math.inf),
                reports=tuple(convoy.reports.get(lane.id, ())),
            )
            for lane in net.out_lanes(convoy.node, known)
        )
        return RouteObservation(
            self.t,
            convoy.node,
            str(convoy.destination),
            options,
            convoy.policies["routing"][1],
        )

    def _convoys(self) -> None:
        for convoy in self.convoys.values():
            if convoy.state != "sailing":
                continue
            if convoy.lane is not None:
                lane = self.network.lanes[convoy.lane]
                centre = self.centre(convoy)
                if dist(centre, self.network.nodes[lane.b]) > config.ARRIVE_RADIUS:
                    continue
                transit = self.t - convoy.lane_started
                lead = convoy.vessels[0]
                error = max(0.1, 1.0 + self.rng.gauss(0.0, config.REPORT_NOISE))
                told = LaneReport(lead, lane.id, transit * error, self.t)
                self.radio.broadcast(told, centre)
                convoy.hear(LaneReport(lead, lane.id, transit, self.t), self.t)
                self._emit("lane_end", convoy=convoy.id, lane=lane.id, transit=transit)
                convoy.node, convoy.lane = lane.b, None
                if convoy.node != convoy.destination and self._delivers_at(
                    convoy, convoy.node
                ):
                    convoy.destination = convoy.node
                    self._emit("stopover", convoy=convoy.id, port=convoy.node)
            if convoy.node == convoy.destination:
                convoy.state = "queued"
                self.queues.setdefault(convoy.node, []).append(convoy.id)
                self._emit("queued", convoy=convoy.id, port=convoy.node)
                continue
            self._look(convoy)
            obs = self._route_options(convoy)
            choice = self.routing.choose(obs)
            self._emit(
                "route_choice",
                convoy=convoy.id,
                node=obs.node,
                destination=obs.destination,
                options=[
                    {
                        "lane": o.lane,
                        "estimate": o.estimate,
                        "cost_to_go": o.cost_to_go,
                        "reports": len(o.reports),
                    }
                    for o in obs.options
                ],
                choice=choice,
                policy_version=convoy.policies["routing"][0],
            )
            if choice is None:
                continue
            convoy.lane, convoy.lane_started = choice, self.t
            self._emit("lane_start", convoy=convoy.id, lane=choice)

    def _delivers_at(self, convoy: Convoy, node: str) -> bool:
        """Whether ``node`` is a port some member has packages for."""
        return node in self.network.ports and any(
            p.destination == node for v in convoy.vessels for p in self.vessels[v].cargo
        )

    def _ports(self) -> None:
        for port, berths in self.network.ports.items():
            here = [c for c in self.convoys.values() if c.node == port]
            for convoy in here:
                if convoy.state == "berthed" and self.t >= convoy.service_until:
                    self._unload(convoy, port)
                elif convoy.state == "loading" and self.t >= convoy.service_until:
                    convoy.state = "sailing"
                    self._emit("departed", convoy=convoy.id, port=port)
            queue = self.queues.setdefault(port, [])
            used = sum(len(c.vessels) for c in here if c.state == "berthed")
            while queue:
                convoy = self.convoys[queue[0]]
                if used and used + len(convoy.vessels) > berths:
                    break
                queue.pop(0)
                used += len(convoy.vessels)
                most = max(
                    sum(p.destination == port for p in self.vessels[v].cargo)
                    for v in convoy.vessels
                )
                convoy.state = "berthed"
                convoy.service_until = (
                    self.t + config.SERVICE_TICKS + config.HANDLING_TIME * most
                )
                self.metrics["arrivals"] += 1
                self._emit("berthed", convoy=convoy.id, port=port)
            self.metrics["wait_ticks"] += (
                sum(len(self.convoys[c].vessels) for c in queue) * config.DT
            )
            if self.t % config.REGROUP_EVERY == 0:
                self._regroup(port)

    def _unload(self, convoy: Convoy, port: str) -> None:
        """Hand over the packages bound here; the ships join the dock."""
        latencies = []
        for vid in convoy.vessels:
            v = self.vessels[vid]
            mine = [p for p in v.cargo if p.destination == port]
            v.cargo = [p for p in v.cargo if p.destination != port]
            latencies += [self.t - p.created for p in mine]
            v.docked_since, v.loaded_here = self.t, 0
        self.metrics["deliveries"] += len(latencies)
        self.metrics["latencies"] += latencies
        self.metrics["delivery_times"] += [self.t] * len(latencies)
        self._emit("delivered", convoy=convoy.id, port=port, latencies=latencies)
        convoy.state, convoy.destination = "docked", None

    def _charted_times(self, port: str, known: set[str]) -> dict[str, float]:
        """Charted sailing time from ``port`` to every port."""
        net, speed, costs = self.network, self.cruise(), self.lane_costs()
        return {
            p: net.times_to(p, self.ocean, self.t, speed, known, costs).get(
                port, math.inf
            )
            for p in net.ports
        }

    def _regroup(self, port: str) -> None:
        """Docked ships load, pick a next port, and sail in convoys of the same."""
        docked = [
            c for c in self.convoys.values() if c.node == port and c.state == "docked"
        ]
        if not docked:
            return
        pool = Convoy("pool", [], port)
        for c in docked:
            pool.absorb(c)
        ships = sorted(v for c in docked for v in c.vessels)
        times = self._charted_times(port, pool.known_closed)
        backlogs = {p: n for p, (n, _) in pool.backlogs.items()}
        for vid in ships:
            v = self.vessels[vid]
            target = target_port(v.cargo, self.waiting[port], times)
            if target is None:
                continue
            to_target = self.network.times_to(
                target,
                self.ocean,
                self.t,
                self.cruise(),
                pool.known_closed,
                self.lane_costs(),
            )
            v.loaded_here += len(
                load(
                    v.cargo,
                    v.capacity,
                    self.waiting[port],
                    target,
                    detours(times, to_target, target),
                    config.DETOUR_FRACTION * times[target],
                )
            )
        plans = [
            (vid, next_port(self.vessels[vid].cargo, port, times, backlogs))
            for vid in ships
        ]
        leaving = [
            (dest, members)
            for dest, members in group(plans, config.MAX_CONVOY)
            if all(self.vessels[m].full for m in members)
            or any(
                self.t - self.vessels[m].docked_since >= config.MAX_WAIT
                for m in members
            )
        ]
        if not leaving and len(docked) == 1:
            return
        for module, (version, _) in pool.policies.items():
            behind = [
                v
                for c in docked
                if c.policies.get(module, (-1, None))[0] < version
                for v in c.vessels
            ]
            self._adopt(behind, module, version)
        for c in docked:
            del self.convoys[c.id]
        gone = {m for _, members in leaving for m in members}
        for dest, members in leaving:
            convoy = self._form(pool, port, members)
            convoy.state, convoy.destination = "loading", dest
            most = max(self.vessels[m].loaded_here for m in members)
            convoy.service_until = self.t + config.HANDLING_TIME * most
            self._emit(
                "formed",
                convoy=convoy.id,
                port=port,
                destination=dest,
                vessels=members,
                load=[len(self.vessels[m].cargo) for m in members],
            )
        rest = [v for v in ships if v not in gone]
        if rest:
            self._form(pool, port, rest).state = "docked"

    def _form(self, pool: Convoy, port: str, members: list[str]) -> Convoy:
        convoy = Convoy(f"{port}-{self._formed}", list(members), port)
        self._formed += 1
        for key, value in pool.knowledge().items():
            setattr(convoy, key, value)
        for m in members:
            self.vessels[m].convoy = convoy.id
        self.convoys[convoy.id] = convoy
        return convoy

    def _move(self) -> None:
        net = self.network
        occupancy: dict[str, int] = {}
        for c in self.convoys.values():
            if c.lane is not None:
                occupancy[c.lane] = occupancy.get(c.lane, 0) + 1
        for lane_id, n in occupancy.items():
            excess = n - net.lanes[lane_id].capacity
            self.metrics["congestion_ticks"] += max(0, excess) * config.DT
        for convoy in self.convoys.values():
            if convoy.lane is not None:
                lane = net.lanes[convoy.lane]
                a, b = net.nodes[lane.a], net.nodes[lane.b]
                target, half, holding = b, lane.width / 2, False
                factor = min(1.0, lane.capacity / occupancy[lane.id])
            else:
                a = b = target = net.nodes[convoy.node]
                half, holding, factor = config.HOLD_WIDTH / 2, True, 1.0
            for index, vid in enumerate(convoy.vessels):
                v = self.vessels[vid]
                obs = FormationObservation(
                    velocity=v.vel,
                    neighbours=self._neighbours.get(vid, ()),
                    goal=sub(target, v.pos),
                    lane_offset=sub(closest_on_segment(v.pos, a, b), v.pos),
                    half_width=half,
                    max_speed=v.max_speed * factor,
                    slot_index=index,
                    convoy_size=len(convoy.vessels),
                    holding=holding,
                    params=convoy.policies["formation"][1],
                )
                desired = self.formation.steer(obs)
                v.propel(desired, self.ocean.drift(v.pos, self.t), config.DT)
                if convoy.state == "sailing":
                    self.metrics["sail_ticks"] += config.DT
                    self.metrics["load_ticks"] += len(v.cargo) / v.capacity
                    if not v.cargo:
                        self.metrics["empty_distance"] += length(v.vel) * config.DT

    def _near_misses(self) -> None:
        ids = sorted(self.vessels)
        close = set()
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if dist(self.vessels[a].pos, self.vessels[b].pos) < config.NEAR_MISS:
                    close.add((a, b))
        self.metrics["near_misses"] += len(close - self._close_pairs)
        for a, b in sorted(close - self._close_pairs):
            gap = dist(self.vessels[a].pos, self.vessels[b].pos)
            self._emit("near_miss", a=a, b=b, distance=gap)
        self._close_pairs = close

    def _beliefs(self) -> None:
        """Count believed closures that are false, and true ones not known."""
        m = self.metrics
        for convoy in self.convoys.values():
            known, n = convoy.known_closed, len(convoy.vessels)
            m["belief_false"] += n * len(known - self.closed)
            m["belief_missed"] += n * len(self.closed - known)
        m["backlog_ticks"] += sum(len(w) for w in self.waiting.values()) * config.DT
        hq = self.hq.hazards.closed()
        m["hq_belief_false"] += len(hq - self.closed)
        m["hq_belief_missed"] += len(self.closed - hq)

    # ----- reporting -----------------------------------------------------

    def summary(self) -> dict[str, Any]:
        m = self.metrics
        lat = m["latencies"]
        energy = round(sum(v.energy for v in self.vessels.values()), 1)
        sail = m["sail_ticks"]
        out: dict[str, Any] = {
            "ticks": self.t,
            "deliveries": m["deliveries"],
            "mean_latency": round(sum(lat) / len(lat), 1) if lat else None,
            "generated": m["generated"],
            "backlog": sum(len(w) for w in self.waiting.values()),
            "mean_backlog": round(m["backlog_ticks"] / self.t, 1) if self.t else None,
            "arrivals": m["arrivals"],
            "load_factor": round(m["load_ticks"] / sail, 3) if sail else None,
            "empty_distance": round(m["empty_distance"], 1),
            "energy": energy,
            "energy_per_delivery": (
                round(energy / m["deliveries"], 1) if m["deliveries"] else None
            ),
            "wait_ticks": m["wait_ticks"],
            "congestion_ticks": m["congestion_ticks"],
            "near_misses": m["near_misses"],
            "blocked_discoveries": m["blocked_discoveries"],
            "beliefs": {
                k: m[k]
                for k in (
                    "belief_false",
                    "belief_missed",
                    "hq_belief_false",
                    "hq_belief_missed",
                )
            },
            "radio": {**self.radio.stats, "by_type": dict(self.radio.by_type)},
            "delivery_times": list(m["delivery_times"]),
        }
        out["llm"] = {
            module: {
                "calls": a.calls,
                "failures": a.failures,
                "skipped": a.skipped,
                **a.usage,
            }
            for module, a in self.authors.items()
        }
        out["policies"] = [
            {"module": module, "version": version, **self._spread(record)}
            for (module, version), record in self.adoption.items()
        ]
        return out

    def _spread(self, record: dict[str, Any]) -> dict[str, Any]:
        """When a decided policy reached half, and all, of the ships."""
        times = sorted(record["adopted"].values())
        n = len(self.vessels)
        half = (n + 1) // 2

        def after(k: int) -> float | None:
            return times[k - 1] - record["decided"] if len(times) >= k else None

        return {
            "decided": record["decided"],
            "adopted_by": len(times),
            "to_half": after(half),
            "to_all": after(n),
        }


def make_fleet(
    network: SeaNetwork,
    specs: list[tuple[str, str, int, str | None]],
    spacing: float = 10.0,
) -> tuple[dict[str, Vessel], dict[str, Convoy]]:
    """Convoys ``(id, port, size, destination)`` at ``port``, in a row.

    With no destination the convoy starts docked, waiting for cargo.
    """
    vessels: dict[str, Vessel] = {}
    convoys: dict[str, Convoy] = {}
    for cid, port, size, dest in specs:
        x, y = network.nodes[port]
        ids = [f"{cid}.{i}" for i in range(size)]
        for i, vid in enumerate(ids):
            vessels[vid] = Vessel(vid, cid, (x + (i - (size - 1) / 2) * spacing, y))
        state = "docked" if dest is None else "sailing"
        convoys[cid] = Convoy(cid, ids, port, dest, state=state)
    return vessels, convoys
