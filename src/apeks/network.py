"""The sea-lane network: ports and waypoints joined by directed lanes."""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

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
    _index: tuple[dict[str, list[Lane]], dict[str, list[Lane]]] | None = field(
        default=None, repr=False, compare=False
    )
    _matrix: Any = field(default=None, repr=False, compare=False)

    def add_node(self, name: str, pos: Vec) -> None:
        self.nodes[name] = pos

    def add_port(self, name: str, pos: Vec, berths: int) -> None:
        self.add_node(name, pos)
        self.ports[name] = berths

    def add_channel(self, a: str, b: str, width: float, capacity: int) -> None:
        for x, y in ((a, b), (b, a)):
            lane_id = f"{x}>{y}"
            self.lanes[lane_id] = Lane(lane_id, x, y, width, capacity)
        self._index = None

    def _lanes_by_end(self) -> tuple[dict[str, list[Lane]], dict[str, list[Lane]]]:
        """Lanes into and out of each node, built once per set of channels."""
        if self._index is None:
            into: dict[str, list[Lane]] = {}
            out: dict[str, list[Lane]] = {}
            for lane in self.lanes.values():
                into.setdefault(lane.b, []).append(lane)
                out.setdefault(lane.a, []).append(lane)
            self._index = (into, out)
        return self._index

    def _lane_arrays(self, costs: dict[str, float]) -> Any:
        """Node order, lane ids, ends and costs as arrays, kept per ``costs``."""
        if self._matrix is None or self._matrix[0] is not costs:
            names = list(self.nodes)
            at = {n: i for i, n in enumerate(names)}
            ids = list(self.lanes)
            a = np.array([at[self.lanes[i].a] for i in ids], dtype=np.int32)
            b = np.array([at[self.lanes[i].b] for i in ids], dtype=np.int32)
            w = np.array([costs[i] for i in ids], dtype=float)
            self._matrix = (
                costs,
                names,
                at,
                {i: k for k, i in enumerate(ids)},
                a,
                b,
                w,
            )
        return self._matrix

    def length(self, lane: Lane) -> float:
        return dist(self.nodes[lane.a], self.nodes[lane.b])

    def out_lanes(self, node: str, exclude: set[str] | None = None) -> list[Lane]:
        skip = exclude or set()
        _, out = self._lanes_by_end()
        return [lane for lane in out.get(node, []) if lane.id not in skip]

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
        return self._dijkstra(dest, "a", ocean, t, speed, exclude, costs)

    def times_from(
        self,
        src: str,
        ocean: Ocean,
        t: float,
        speed: float,
        exclude: set[str] | None = None,
        costs: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Estimated time from ``src`` to every node it can reach."""
        return self._dijkstra(src, "b", ocean, t, speed, exclude, costs)

    def _dijkstra(
        self,
        start: str,
        far_end: str,
        ocean: Ocean,
        t: float,
        speed: float,
        exclude: set[str] | None,
        costs: dict[str, float] | None,
    ) -> dict[str, float]:
        """Shortest times from ``start``, walking lanes towards ``far_end``.

        SciPy's Dijkstra adds lane costs along each path in the same order as
        a plain search, so the times are the same to the last bit.
        """
        cost_of = costs if costs is not None else self.lane_times(ocean, t, speed)
        _, names, at, index, a, b, w = self._lane_arrays(cost_of)
        keep = np.isfinite(w)
        for lane_id in exclude or ():
            if lane_id in index:
                keep[index[lane_id]] = False
        src, dst = (a, b) if far_end == "b" else (b, a)
        n = len(names)
        graph = csr_matrix((w[keep], (src[keep], dst[keep])), shape=(n, n))
        found = dijkstra(graph, directed=True, indices=at[start])
        reached = np.flatnonzero(np.isfinite(found))
        return {names[i]: float(found[i]) for i in reached}
