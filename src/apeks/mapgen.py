"""A seeded synthetic sea: a noise landmass, ports on its shores, lanes at sea.

``generate(seed, ports, adjacency, ticks)`` builds the land on a grid of
``CELL``-sized cells, keeps the largest connected body of water as the ocean,
puts the ports on its coast and as many waypoints in open water, and joins
every node to its ``adjacency`` nearest neighbours in sight across water. A
spanning step then joins any parts left apart, so every node reaches every
other. The sea area grows with the number of ports, so port density stays the
same. Storms, currents and lane closures are drawn from the seed and sit in
the ocean; a closure never cuts the map in two.
"""

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

from apeks import config
from apeks.network import SeaNetwork
from apeks.ocean import Current, Ocean, Storm
from apeks.seeds import derive
from apeks.world import Event

Vec = tuple[float, float]
Grid = tuple[tuple[bool, ...], ...]

CELL = 20.0
BASE_WIDTH, BASE_HEIGHT, BASE_PORTS = 1000.0, 700.0, 5
LAND_SHARE = 0.4  # share of cells that start as land, before lakes are filled
NOISE_SCALES = ((12, 1.0), (6, 0.5), (3, 0.25))  # lattice spacing in cells, weight
LANE_WIDTH, LANE_CAPACITY = 60.0, 2
CLEARANCE = 2  # cells of open water around a waypoint
EVENT_SPAN = 1000.0  # ticks a storm lasts or a lane stays closed
EVENT_MARGIN = 500.0  # no event starts before this or ends within it of the end
PORTS_PER_EVENT = 5  # one storm, one closure and two currents per this many ports
MAX_ATTEMPTS = 20


@dataclass(frozen=True)
class Geography:
    """The land grid. ``land[row][col]`` is True for land; off the map is land."""

    width: float
    height: float
    cell: float
    land: Grid

    @classmethod
    def open_sea(cls, width: float, height: float, cell: float = 50.0) -> Geography:
        cols, rows = math.ceil(width / cell), math.ceil(height / cell)
        return cls(width, height, cell, ((False,) * cols,) * rows)

    @property
    def cols(self) -> int:
        return len(self.land[0])

    @property
    def rows(self) -> int:
        return len(self.land)

    def _land_at(self, col: int, row: int) -> bool:
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return self.land[row][col]
        return True

    def is_land(self, pos: Vec) -> bool:
        return self._land_at(int(pos[0] // self.cell), int(pos[1] // self.cell))

    def near_land(self, pos: Vec) -> bool:
        col, row = int(pos[0] // self.cell), int(pos[1] // self.cell)
        return any(
            self._land_at(col + dc, row + dr) for dc in (-1, 0, 1) for dr in (-1, 0, 1)
        )


@dataclass(frozen=True)
class Sea:
    network: SeaNetwork
    ocean: Ocean
    events: list[Event]
    fleet: dict[str, int]
    geography: Geography


def segment_clear(geo: Geography, a: Vec, b: Vec) -> bool:
    """True when the straight line from ``a`` to ``b`` stays on water."""
    steps = math.ceil(math.dist(a, b) / (geo.cell / 2)) + 1
    return not any(
        geo.is_land(
            (a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps)
        )
        for i in range(steps + 1)
    )


def is_connected(nodes: Iterable[str], edges: set[tuple[str, str]]) -> bool:
    """Whether the undirected graph reaches every node from the first."""
    names = list(nodes)
    if not names:
        return True
    neighbours: dict[str, set[str]] = {n: set() for n in names}
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    seen, stack = {names[0]}, [names[0]]
    while stack:
        for n in neighbours[stack.pop()] - seen:
            seen.add(n)
            stack.append(n)
    return len(seen) == len(names)


def generate(seed: int, ports: int, adjacency: int, ticks: int) -> Sea:
    if ports < 2 or adjacency < 1:
        raise ValueError("a sea needs at least 2 ports and 1 lane per node")
    for attempt in range(MAX_ATTEMPTS):
        sea = _attempt(seed, ports, adjacency, ticks, attempt)
        if sea is not None:
            return sea
    raise RuntimeError(f"no connected sea for seed {seed} with {ports} ports")


def _attempt(
    seed: int, ports: int, adjacency: int, ticks: int, attempt: int
) -> Sea | None:
    rng = derive(seed, f"map:{attempt}")
    grow = math.sqrt(ports / BASE_PORTS)
    width, height = BASE_WIDTH * grow, BASE_HEIGHT * grow
    cols, rows = math.ceil(width / CELL), math.ceil(height / CELL)
    land = _ocean_only(_landmass(rng, cols, rows))
    geo = Geography(width, height, CELL, land)
    coast = [
        (c, r)
        for r in range(rows)
        for c in range(cols)
        if not land[r][c]
        and any(
            0 <= c + dc < cols and 0 <= r + dr < rows and land[r + dr][c + dc]
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
    ]
    open_water = [
        (c, r)
        for r in range(rows)
        for c in range(cols)
        if not any(
            geo._land_at(c + dc, r + dr)
            for dc in range(-CLEARANCE, CLEARANCE + 1)
            for dr in range(-CLEARANCE, CLEARANCE + 1)
        )
    ]
    if len(coast) < ports or len(open_water) < 2:
        return None
    port_cells = _spread(rng, coast, ports, [])
    way_cells = _spread(rng, open_water, min(ports, len(open_water)), port_cells)
    nodes = {f"P{i + 1}": _centre(cell) for i, cell in enumerate(port_cells)}
    nodes |= {f"W{i + 1}": _centre(cell) for i, cell in enumerate(way_cells)}
    edges = _lanes(geo, nodes, adjacency)
    net = SeaNetwork()
    for name, pos in nodes.items():
        if name.startswith("P"):
            net.add_port(name, pos, berths=config.BERTHS)
        else:
            net.add_node(name, pos)
    for a, b in sorted(edges):
        net.add_channel(a, b, LANE_WIDTH, LANE_CAPACITY)
    count = max(1, ports // PORTS_PER_EVENT)
    deep = [_centre(cell) for cell in open_water]
    ocean = Ocean(
        currents=tuple(
            Current(rng.choice(deep), 260.0, _heading(rng, 0.4))
            for _ in range(2 * count)
        ),
        storms=tuple(
            Storm(rng.choice(deep), 150.0, 0.8, start, start + EVENT_SPAN)
            for start in (_start(rng, ticks) for _ in range(count))
        ),
        wave_amplitude=0.1,
        wave_period=80.0,
        wave_dir=(0.0, 1.0),
    )
    events = _closures(rng, nodes, edges, count, ticks)
    fleet = dict.fromkeys(net.ports, config.SHIPS_PER_PORT)
    return Sea(net, ocean, events, fleet, geo)


def _landmass(rng: random.Random, cols: int, rows: int) -> list[list[bool]]:
    height = [[0.0] * cols for _ in range(rows)]
    for spacing, weight in NOISE_SCALES:
        lattice = [
            [rng.random() for _ in range(cols // spacing + 2)]
            for _ in range(rows // spacing + 2)
        ]
        for r in range(rows):
            gy, fy = divmod(r / spacing, 1.0)
            sy = fy * fy * (3 - 2 * fy)
            for c in range(cols):
                gx, fx = divmod(c / spacing, 1.0)
                sx = fx * fx * (3 - 2 * fx)
                i, j = int(gy), int(gx)
                top = lattice[i][j] + (lattice[i][j + 1] - lattice[i][j]) * sx
                low = (
                    lattice[i + 1][j] + (lattice[i + 1][j + 1] - lattice[i + 1][j]) * sx
                )
                height[r][c] += weight * (top + (low - top) * sy)
    values = sorted(v for row in height for v in row)
    cut = values[min(len(values) - 1, int((1 - LAND_SHARE) * len(values)))]
    return [[v >= cut for v in row] for row in height]


def _ocean_only(land: list[list[bool]]) -> Grid:
    """Fill every lake, keeping only the largest body of water."""
    rows, cols = len(land), len(land[0])
    label = [[-1] * cols for _ in range(rows)]
    sizes: list[int] = []
    for r0 in range(rows):
        for c0 in range(cols):
            if land[r0][c0] or label[r0][c0] >= 0:
                continue
            label[r0][c0] = len(sizes)
            stack, size = [(c0, r0)], 0
            while stack:
                c, r = stack.pop()
                size += 1
                for nc, nr in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
                    if 0 <= nc < cols and 0 <= nr < rows:
                        if not land[nr][nc] and label[nr][nc] < 0:
                            label[nr][nc] = len(sizes)
                            stack.append((nc, nr))
            sizes.append(size)
    biggest = sizes.index(max(sizes))
    return tuple(
        tuple(label[r][c] != biggest for c in range(cols)) for r in range(rows)
    )


def _centre(cell: tuple[int, int]) -> Vec:
    return ((cell[0] + 0.5) * CELL, (cell[1] + 0.5) * CELL)


def _spread(
    rng: random.Random,
    cells: list[tuple[int, int]],
    n: int,
    taken: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Pick ``n`` cells, each as far as possible from those already picked."""
    picked: list[tuple[int, int]] = []
    if not taken:
        picked.append(rng.choice(cells))
    gap = {
        c: min((math.dist(c, t) for t in taken + picked), default=0.0) for c in cells
    }
    while len(picked) < n:
        best = max(cells, key=lambda c: gap[c])
        picked.append(best)
        for c in cells:
            gap[c] = min(gap[c], math.dist(c, best))
    return picked


def _lanes(
    geo: Geography, nodes: dict[str, Vec], adjacency: int
) -> set[tuple[str, str]]:
    names = sorted(nodes)
    sight = {
        (a, b): math.dist(nodes[a], nodes[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if segment_clear(geo, nodes[a], nodes[b])
    }
    edges: set[tuple[str, str]] = set()
    for a in names:
        near = sorted((d, pair) for pair, d in sight.items() if a in pair)[:adjacency]
        edges |= {pair for _, pair in near}
    group = {n: n for n in names}

    def root(n: str) -> str:
        while group[n] != n:
            n = group[n]
        return n

    for a, b in edges:
        group[root(a)] = root(b)
    for _, (a, b) in sorted((d, pair) for pair, d in sight.items()):
        if root(a) != root(b):
            group[root(a)] = root(b)
            edges.add((a, b))
    while not is_connected(nodes, edges):
        _bridge(geo, nodes, edges)
    return edges


def _bridge(geo: Geography, nodes: dict[str, Vec], edges: set[tuple[str, str]]) -> None:
    """Join the part holding the first node to the nearest node outside it.

    The lane follows the shortest water path, with a waypoint wherever the
    straight line of sight runs out.
    """
    names = sorted(nodes)
    part = {names[0]}
    grew = True
    while grew:
        grew = False
        for a, b in edges:
            if (a in part) != (b in part):
                part |= {a, b}
                grew = True
    a, b = min(
        ((x, y) for x in part for y in names if y not in part),
        key=lambda pair: math.dist(nodes[pair[0]], nodes[pair[1]]),
    )
    path = _water_path(geo, nodes[a], nodes[b])
    here, i = a, 0
    while not segment_clear(geo, nodes[here], nodes[b]):
        j = max(
            k
            for k in range(i + 1, len(path))
            if segment_clear(geo, nodes[here], path[k])
        )
        name = f"W{sum(n.startswith('W') for n in nodes) + 1}"
        nodes[name] = path[j]
        edges.add((here, name))
        here, i = name, j
    edges.add((here, b))


def _water_path(geo: Geography, start: Vec, end: Vec) -> list[Vec]:
    """Cell centres of a shortest 4-connected water path, start to end.

    Steps never cut a land corner, so each step is in sight of the last.
    """

    def cell(pos: Vec) -> tuple[int, int]:
        return int(pos[0] // geo.cell), int(pos[1] // geo.cell)

    goal, first = cell(end), cell(start)
    came: dict[tuple[int, int], tuple[int, int] | None] = {first: None}
    queue = [first]
    for c, r in queue:  # the ocean is one body of water, so this reaches goal
        for nxt in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
            if nxt not in came and not geo._land_at(*nxt):
                came[nxt] = (c, r)
                queue.append(nxt)
    path: list[Vec] = []
    step: tuple[int, int] | None = goal
    while step is not None:
        path.append(((step[0] + 0.5) * geo.cell, (step[1] + 0.5) * geo.cell))
        step = came[step]
    return path[::-1]


def _heading(rng: random.Random, speed: float) -> Vec:
    angle = rng.uniform(0.0, 2 * math.pi)
    return (speed * math.cos(angle), speed * math.sin(angle))


def _start(rng: random.Random, ticks: int) -> float:
    return rng.uniform(
        EVENT_MARGIN, max(EVENT_MARGIN, ticks - EVENT_MARGIN - EVENT_SPAN)
    )


def _closures(
    rng: random.Random,
    nodes: dict[str, Vec],
    edges: set[tuple[str, str]],
    count: int,
    ticks: int,
) -> list[Event]:
    """Close ``count`` lanes, each for ``EVENT_SPAN`` ticks, never cutting the map."""
    events: list[Event] = []
    closed: set[tuple[str, str]] = set()
    for _ in range(count):
        safe = [
            e
            for e in sorted(edges - closed)
            if is_connected(nodes, edges - closed - {e})
        ]
        if not safe:
            break
        a, b = rng.choice(safe)
        closed.add((a, b))
        at = _start(rng, ticks)
        lane = f"{a}>{b}"
        events.append(Event(at, "close", lane, f"lane {lane} blocked"))
        events.append(Event(at + EVENT_SPAN, "open", lane, f"lane {lane} reopened"))
    return sorted(events, key=lambda e: e.at)
