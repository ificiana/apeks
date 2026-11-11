"""Broadcast radio between vessels and port stations.

Every transmitter is also a beacon: a receiver estimates where the sender is,
relative to itself, from the signal (with error growing with distance). The
messages therefore carry no positions of their own.
"""

import random
from dataclasses import dataclass, field
from typing import Any

from apeks.geometry import Vec, dist, sub


@dataclass(frozen=True)
class PositionReport:
    """A vessel's periodic beacon: who it is, its convoy and its velocity."""

    sender: str
    convoy: str
    velocity: Vec


@dataclass(frozen=True)
class LaneReport:
    """A convoy reporting how long a lane actually took to cross."""

    sender: str
    lane: str
    transit: float
    finished_at: float


@dataclass(frozen=True)
class Hazard:
    """``origin`` saw ``lane`` closed (or open) at ``observed_at``.

    Convoys send one when their own eyes disagree with what they believed;
    port stations repeat the hazards they have heard; ships relay new ones.
    ``hops`` counts relays since the origin or the last port station.
    """

    sender: str
    lane: str
    closed: bool
    observed_at: float
    origin: str
    hops: int = 0


@dataclass(frozen=True)
class PolicyMsg:
    """A versioned policy from HQ, sent by port stations and relayed by ships."""

    sender: str
    module: str
    version: int
    params: Any
    hops: int = 0


@dataclass(frozen=True)
class Backlog:
    """A port station's count of packages waiting at ``port``. Never relayed."""

    sender: str
    port: str
    waiting: int
    observed_at: float


type Message = PositionReport | LaneReport | Hazard | PolicyMsg | Backlog


@dataclass(frozen=True)
class Received:
    message: Message
    rel_pos: Vec


@dataclass
class Radio:
    rng: random.Random
    range: float
    loss: float
    bandwidth: int
    noise: float
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "sent": 0,
            "delivered": 0,
            "lost": 0,
            "overflow": 0,
            "relayed": 0,
        }
    )
    by_type: dict[str, int] = field(default_factory=dict)
    _queue: list[tuple[Message, Vec]] = field(default_factory=list)

    def broadcast(self, message: Message, origin: Vec) -> None:
        self.stats["sent"] += 1
        kind = type(message).__name__
        self.by_type[kind] = self.by_type.get(kind, 0) + 1
        if getattr(message, "hops", 0) > 0:
            self.stats["relayed"] += 1
        self._queue.append((message, origin))

    def deliver(self, receivers: dict[str, Vec]) -> dict[str, list[Received]]:
        """Deliver this tick's transmissions; at most ``bandwidth`` get through."""
        queue, self._queue = self._queue, []
        self.stats["overflow"] += max(0, len(queue) - self.bandwidth)
        out: dict[str, list[Received]] = {}
        for message, origin in queue[: self.bandwidth]:
            for rid, pos in receivers.items():
                d = dist(pos, origin)
                if rid == message.sender or d > self.range:
                    continue
                if self.rng.random() < self.loss:
                    self.stats["lost"] += 1
                    continue
                rel = sub(origin, pos)
                sigma = self.noise * d
                est = (
                    rel[0] + self.rng.gauss(0, sigma),
                    rel[1] + self.rng.gauss(0, sigma),
                )
                out.setdefault(rid, []).append(Received(message, est))
                self.stats["delivered"] += 1
        return out
