"""The sea-lane network: ports and waypoints joined by directed lanes."""

import heapq
import math
from dataclasses import dataclass, field

from apeks.geometry import Vec, add, dist, dot, normalize, perp, scale, sub
from apeks.ocean import Ocean


@dataclass(frozen=True)
class Lane:
    """A directed lane from node ``a`` to node ``b``.

    ``width`` is the navigable width of the channel; ``capacity`` is how many
    convoys may use it at full speed at the same time.
    """

    id: str
    a: str
    b: str
    width: float
    capacity: int


@dataclass
class SeaNetwork:
    nodes: dict[str, Vec] = field(default_factory=dict)
    ports: dict[str, int] = field(default_factory=dict)
    lanes: dict[str, Lane] = field(default_factory=dict)

    def add_node(self, name: str, pos: Vec) -> None:
        self.nodes[name] = pos

    def add_port(self, name: str, pos: Vec, berths: int) -> None:
        self.add_node(name, pos)
        self.ports[name] = berths

    def add_channel(self, a: str, b: str, width: float, capacity: int) -> None:
        for x, y in ((a, b), (b, a)):
            lane_id = f"{x}>{y}"
            self.lanes[lane_id] = Lane(lane_id, x, y, width, capacity)

    def length(self, lane: Lane) -> float:
        return dist(self.nodes[lane.a], self.nodes[lane.b])

    def out_lanes(self, node: str, exclude: set[str] | None = None) -> list[Lane]:
        skip = exclude or set()
        return [
            lane
            for lane in self.lanes.values()
            if lane.a == node and lane.id not in skip
        ]

    def transit_estimate(
        self, lane: Lane, ocean: Ocean, t: float, speed: float, samples: int = 8
    ) -> float:
        """Time to cross ``lane`` at water speed ``speed`` under the mean drift.

        The vessel crabs against the cross-track drift and gains or loses the
        along-track drift. Returns ``inf`` when the drift cannot be overcome.
        """
        a, b = self.nodes[lane.a], self.nodes[lane.b]
        u = normalize(sub(b, a))
        step = self.length(lane) / samples
        total = 0.0
        for i in range(samples):
            p = add(a, scale(u, step * (i + 0.5)))
            d = ocean.mean_drift(p, t)
            cross = abs(dot(d, perp(u)))
            if cross >= speed:
                return math.inf
            ground = math.sqrt(speed * speed - cross * cross) + dot(d, u)
            if ground <= 0:
                return math.inf
            total += step / ground
        return total

    def lane_times(self, ocean: Ocean, t: float, speed: float) -> dict[str, float]:
        """Transit estimate of every lane, to share across many ``times_to``."""
        return {
            lid: self.transit_estimate(lane, ocean, t, speed)
            for lid, lane in self.lanes.items()
        }

    def times_to(
        self,
        dest: str,
        ocean: Ocean,
        t: float,
        speed: float,
        exclude: set[str] | None = None,
        costs: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Estimated time from every node that can reach ``dest``."""
        skip = exclude or set()
        cost_of = costs if costs is not None else self.lane_times(ocean, t, speed)
        incoming: dict[str, list[Lane]] = {}
        for lane in self.lanes.values():
            if lane.id not in skip:
                incoming.setdefault(lane.b, []).append(lane)
        best = {dest: 0.0}
        heap = [(0.0, dest)]
        while heap:
            cost, node = heapq.heappop(heap)
            if cost > best[node]:
                continue
            for lane in incoming.get(node, []):
                total = cost + cost_of[lane.id]
                if total < best.get(lane.a, math.inf):
                    best[lane.a] = total
                    heapq.heappush(heap, (total, lane.a))
        return best
