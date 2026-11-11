"""The fleet simulation: vessels, convoys, ports, radio, events and metrics.

The world owns physical truth (positions, closures, storms). Convoys act only
on what they know: lane reports and notices that reached them by radio, and
blockages they run into. Decisions come from the plugged-in engines.
"""

import math
from dataclasses import dataclass, field
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
from apeks.llm import PolicyAuthor
from apeks.network import SeaNetwork
from apeks.ocean import Ocean
from apeks.radio import LaneReport, Notice, PositionReport, Radio
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
    known_closed: set[str] = field(default_factory=set)
    reports: dict[str, list[Report]] = field(default_factory=dict)
    _seen: set[tuple[str, str, float]] = field(default_factory=set)

    @property
    def destination(self) -> str:
        return self.orders[self.order_idx]

    def hear(self, msg: LaneReport | Notice) -> None:
        if isinstance(msg, Notice):
            if msg.closed:
                self.known_closed.add(msg.lane)
            else:
                self.known_closed.discard(msg.lane)
            return
        key = (msg.sender, msg.lane, msg.finished_at)
        if key not in self._seen:
            self._seen.add(key)
            self.reports.setdefault(msg.lane, []).append(
                Report(msg.transit, msg.finished_at)
            )


@dataclass(frozen=True)
class Event:
    at: float
    kind: str  # close | open
    lane: str
    text: str


def reverse(lane_id: str) -> str:
    a, b = lane_id.split(">")
    return f"{b}>{a}"


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
    t: float = 0.0
    closed: set[str] = field(default_factory=set)
    reopened: set[str] = field(default_factory=set)
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
        }
    )
    _close_pairs: set[tuple[str, str]] = field(default_factory=set)
    _storms_active: frozenset[int] = frozenset()
    _neighbours: dict[str, tuple[Neighbour, ...]] = field(default_factory=dict)

    # ----- helpers -------------------------------------------------------

    def centre(self, convoy: Convoy) -> Vec:
        pts = [self.vessels[v].pos for v in convoy.vessels]
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def cruise(self) -> float:
        return config.MAX_SPEED * config.CRUISE_FRACTION

    def situation(self, trigger: str) -> dict[str, Any]:
        """What a shore operations centre knows, for policy authors."""
        delays = {}
        for c in self.convoys.values():
            for lane_id, reps in c.reports.items():
                recent = [r.transit for r in reps if self.t - r.finished_at <= 800]
                if recent:
                    est = self.network.transit_estimate(
                        self.network.lanes[lane_id], self.ocean, self.t, self.cruise()
                    )
                    delays[lane_id] = round(sum(recent) / len(recent) - est, 1)
        storms = [
            {"centre": s.centre, "radius": s.radius, "until": s.end}
            for s in self.ocean.active_storms(self.t)
        ]
        lanes = {
            lane.id: {"width": lane.width, "capacity": lane.capacity}
            for lane in self.network.lanes.values()
        }
        return {
            "time": self.t,
            "trigger": trigger,
            "closed_lanes": sorted(self.closed),
            "storms": storms,
            "reported_delay_by_lane": delays,
            "lanes": lanes,
            "deliveries": self.metrics["deliveries"],
            "near_misses": self.metrics["near_misses"],
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
                    "known_closed": c.known_closed,
                    "reports": sum(len(r) for r in c.reports.values()),
                }
                for c in self.convoys.values()
            ],
            "queues": {k: list(v) for k, v in self.queues.items()},
            "closed": self.closed,
            "metrics": {
                k: v for k, v in self.metrics.items() if not isinstance(v, list)
            },
            "radio": dict(self.radio.stats),
            "routing_params": self.routing.params,
            "formation_params": self.formation.params,
        }

    def _engine(self, module: str) -> Any:
        return self.routing if module == "routing" else self.formation

    # ----- step ----------------------------------------------------------

    def step(self) -> None:
        triggers = ["mission start"] if self.t == 0 else []
        triggers += self._apply_events()
        for module, author in self.authors.items():
            engine = self._engine(module)
            for trigger in triggers:
                before = len(author.log)
                author.request(self.situation(trigger), engine.params, self.t)
                if len(author.log) > before:
                    entry = {k: v for k, v in author.log[-1].items() if k != "t"}
                    self._emit("policy_request", module=module, **entry)
            policy = author.due(self.t)
            if policy is not None:
                engine.params = policy
                self.log.append(f"t={self.t:.0f} new {module} policy")
                self._emit("policy_applied", module=module, params=policy)
        self._radio()
        self._convoys()
        self._ports()
        self._move()
        self._near_misses()
        if self.tracer is not None and self.tracer.wants_snapshot(self.t):
            self._emit("snapshot", **self.snapshot())
        self.t += config.DT

    def _apply_events(self) -> list[str]:
        out = []
        for e in [e for e in self.events if e.at <= self.t]:
            self.events.remove(e)
            pair = {e.lane, reverse(e.lane)}
            if e.kind == "close":
                self.closed |= pair
                self.reopened -= pair
            else:
                self.closed -= pair
                self.reopened |= pair
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
        return out

    def _radio(self) -> None:
        for v in self.vessels.values():
            self.radio.broadcast(PositionReport(v.id, v.convoy, v.vel), v.pos)
        if self.t % config.NOTICE_EVERY == 0:
            for port in self.network.ports:
                notices = [(lane, True) for lane in sorted(self.closed)]
                notices += [(lane, False) for lane in sorted(self.reopened)]
                for lane_id, closed in notices:
                    text = "lane closed" if closed else "lane open"
                    self.radio.broadcast(
                        Notice(port, lane_id, closed, text), self.network.nodes[port]
                    )
        heard = self.radio.deliver({v.id: v.pos for v in self.vessels.values()})
        self._neighbours = {}
        for vid, messages in heard.items():
            mine = self.vessels[vid]
            convoy = self.convoys[mine.convoy]
            nbrs = []
            for rx in messages:
                msg = rx.message
                if isinstance(msg, PositionReport):
                    nbrs.append(
                        Neighbour(rx.rel_pos, msg.velocity, msg.convoy == mine.convoy)
                    )
                else:
                    was = msg.lane in convoy.known_closed
                    convoy.hear(msg)
                    if isinstance(msg, Notice) and msg.closed != was:
                        self._emit(
                            "notice_heard",
                            convoy=convoy.id,
                            lane=msg.lane,
                            closed=msg.closed,
                        )
            self._neighbours[vid] = tuple(nbrs)

    def _route_options(self, convoy: Convoy) -> RouteObservation:
        net = self.network
        to_go = net.times_to(
            convoy.destination, self.ocean, self.t, self.cruise(), convoy.known_closed
        )
        options = tuple(
            RouteOption(
                lane=lane.id,
                to_node=lane.b,
                estimate=net.transit_estimate(lane, self.ocean, self.t, self.cruise()),
                cost_to_go=to_go.get(lane.b, math.inf),
                reports=tuple(convoy.reports.get(lane.id, ())),
            )
            for lane in net.out_lanes(convoy.node, convoy.known_closed)
        )
        return RouteObservation(self.t, convoy.node, convoy.destination, options)

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
                report = LaneReport(convoy.vessels[0], lane.id, transit, self.t)
                self.radio.broadcast(report, centre)
                convoy.hear(report)
                self._emit("lane_end", convoy=convoy.id, lane=lane.id, transit=transit)
                convoy.node, convoy.lane = lane.b, None
            if convoy.node == convoy.destination:
                convoy.state = "queued"
                self.queues.setdefault(convoy.node, []).append(convoy.id)
                self._emit("queued", convoy=convoy.id, port=convoy.node)
                continue
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
            )
            if choice is None:
                continue
            if choice in self.closed:
                convoy.known_closed.add(choice)
                self.metrics["blocked_discoveries"] += 1
                self._emit("blocked", convoy=convoy.id, lane=choice)
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

    # ----- reporting -----------------------------------------------------

    def summary(self) -> dict[str, Any]:
        m = self.metrics
        trips = m["trip_times"]
        out: dict[str, Any] = {
            "ticks": self.t,
            "deliveries": m["deliveries"],
            "mean_trip_time": round(sum(trips) / len(trips), 1) if trips else None,
            "energy": round(sum(v.energy for v in self.vessels.values()), 1),
            "wait_ticks": m["wait_ticks"],
            "congestion_ticks": m["congestion_ticks"],
            "near_misses": m["near_misses"],
            "blocked_discoveries": m["blocked_discoveries"],
            "radio": dict(self.radio.stats),
            "delivery_times": list(m["delivery_times"]),
        }
        out["llm"] = {
            module: {"calls": a.calls, "failures": a.failures, "skipped": a.skipped}
            for module, a in self.authors.items()
        }
        return out


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
