"""The fleet simulation: vessels, convoys, ports, radio, events and metrics.

The world owns physical truth (positions, closures, storms). Everyone else acts
on beliefs that arrived by a path:

- a convoy sees the lanes leading out of the node it stands at, and hears lane
  reports, hazards and policies by radio, relaying new hazards and policies;
- a port station hears the radio, repeats the hazards and policies it holds,
  and forwards what it hears to HQ over a land link with a delay;
- HQ (the policy authors) reads those forwarded reports and the news feed, and
  sends versioned policies back down through the port stations.

Decisions come from the plugged-in engines, which read the policy each convoy
holds from their observation.
"""

import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from apeks import config
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
from apeks.radio import Hazard, LaneReport, Message, PolicyMsg, PositionReport, Radio
from apeks.routing import Report, RouteObservation, RouteOption
from apeks.trace import Tracer


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
    id: str
    vessels: list[str]
    orders: tuple[str, ...]
    node: str
    order_idx: int = 1
    lane: str | None = None
    lane_started: float = 0.0
    state: str = "sailing"  # sailing | queued | berthed
    service_until: float = 0.0
    trip_started: float = 0.0
    hazards: HazardBook = field(default_factory=HazardBook)
    reports: dict[str, list[Report]] = field(default_factory=dict)
    policies: dict[str, tuple[int, Any]] = field(default_factory=dict)
    outbox: list[Message] = field(default_factory=list)
    _seen: set[tuple[str, str, float]] = field(default_factory=set)

    @property
    def destination(self) -> str:
        return self.orders[self.order_idx]

    @property
    def known_closed(self) -> set[str]:
        return self.hazards.closed()

    def hear(self, msg: LaneReport | Hazard | PolicyMsg, now: float) -> str | None:
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
        key = (msg.sender, msg.lane, msg.finished_at)
        if key not in self._seen:
            self._seen.add(key)
            self.reports.setdefault(msg.lane, []).append(
                Report(msg.transit, msg.finished_at)
            )
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
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    t: float = 0.0
    closed: set[str] = field(default_factory=set)
    queues: dict[str, list[str]] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    tracer: Tracer | None = None
    metrics: dict[str, Any] = field(
        default_factory=lambda: {
            "deliveries": 0,
            "trip_times": [],
            "delivery_times": [],
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
    _close_pairs: set[tuple[str, str]] = field(default_factory=set)
    _storms_active: frozenset[int] = frozenset()
    _neighbours: dict[str, tuple[Neighbour, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.news = sorted(self.news, key=lambda item: item.published)
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
            "convoys": [
                {"id": c.id, "orders": c.orders, "vessels": c.vessels}
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
        self._apply_events()
        triggers = ["mission start"] if self.t == 0 else []
        triggers += self._read_news()
        triggers += self.hq.receive(self.t)
        self._authors(triggers)
        self._send_down()
        self._radio()
        self._convoys()
        self._ports()
        self._move()
        self._near_misses()
        self._beliefs()
        if self.tracer is not None and self.tracer.wants_snapshot(self.t):
            self._emit("snapshot", **self.snapshot())
        self.t += config.DT

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
        self.adoption[(msg.module, msg.version)]["adopted"][convoy.id] = self.t
        self._emit(
            "policy_adopted",
            convoy=convoy.id,
            module=msg.module,
            version=msg.version,
            hops=msg.hops,
        )

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
        to_go = net.times_to(
            convoy.destination, self.ocean, self.t, self.cruise(), known
        )
        options = tuple(
            RouteOption(
                lane=lane.id,
                to_node=lane.b,
                estimate=net.transit_estimate(lane, self.ocean, self.t, self.cruise()),
                cost_to_go=to_go.get(lane.b, math.inf),
                reports=tuple(convoy.reports.get(lane.id, ())),
            )
            for lane in net.out_lanes(convoy.node, known)
        )
        return RouteObservation(
            self.t,
            convoy.node,
            convoy.destination,
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

    def _ports(self) -> None:
        for port, berths in self.network.ports.items():
            queue = self.queues.setdefault(port, [])
            berthed = [
                c
                for c in self.convoys.values()
                if c.state == "berthed" and c.node == port
            ]
            for convoy in berthed:
                if self.t < convoy.service_until:
                    continue
                self.metrics["deliveries"] += 1
                self.metrics["trip_times"].append(self.t - convoy.trip_started)
                self.metrics["delivery_times"].append(self.t)
                self._emit(
                    "delivered",
                    convoy=convoy.id,
                    port=port,
                    trip=self.t - convoy.trip_started,
                )
                convoy.order_idx = (convoy.order_idx + 1) % len(convoy.orders)
                convoy.trip_started = self.t
                convoy.state = "sailing"
            free = berths - sum(1 for c in berthed if c.state == "berthed")
            while queue and free > 0:
                convoy = self.convoys[queue.pop(0)]
                convoy.state = "berthed"
                convoy.service_until = self.t + config.SERVICE_TICKS
                free -= 1
                self._emit("berthed", convoy=convoy.id, port=port)
            self.metrics["wait_ticks"] += len(queue) * config.DT

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
            known = convoy.known_closed
            m["belief_false"] += len(known - self.closed)
            m["belief_missed"] += len(self.closed - known)
        hq = self.hq.hazards.closed()
        m["hq_belief_false"] += len(hq - self.closed)
        m["hq_belief_missed"] += len(self.closed - hq)

    # ----- reporting -----------------------------------------------------

    def summary(self) -> dict[str, Any]:
        m = self.metrics
        trips = m["trip_times"]
        energy = round(sum(v.energy for v in self.vessels.values()), 1)
        out: dict[str, Any] = {
            "ticks": self.t,
            "deliveries": m["deliveries"],
            "mean_trip_time": round(sum(trips) / len(trips), 1) if trips else None,
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
        """When a decided policy reached half, and all, of the convoys."""
        times = sorted(record["adopted"].values())
        n = len(self.convoys)
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
    specs: list[tuple[str, tuple[str, ...], int]],
    spacing: float = 10.0,
) -> tuple[dict[str, Vessel], dict[str, Convoy]]:
    """Convoys ``(id, port orders, size)`` starting at their first port, in a row."""
    vessels: dict[str, Vessel] = {}
    convoys: dict[str, Convoy] = {}
    for cid, orders, size in specs:
        x, y = network.nodes[orders[0]]
        ids = [f"{cid}.{i}" for i in range(size)]
        for i, vid in enumerate(ids):
            vessels[vid] = Vessel(vid, cid, (x + (i - (size - 1) / 2) * spacing, y))
        convoys[cid] = Convoy(cid, ids, orders, node=orders[0])
    return vessels, convoys
