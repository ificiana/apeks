import random
from collections import deque
from dataclasses import dataclass

from apeks.config import (
    AGENT_COLOR,
    CHARGE_RATE,
    DIRECTIONS,
    GRID_SIZE,
    LOW_CHARGE,
    MARK_DECAY,
    MARK_VALUE,
    MAX_CHARGE,
    MOVE_COST,
    PRESSURE_LIMIT,
    PUSH_FLOOR_SEEK,
    PUSH_FLOOR_WALK,
    PUSH_REACH,
    STATION_COUNT,
    STEP_INTERVAL,
    URGENT_SLACK,
    WALK_NOISE,
)

Tile = tuple[int, int]


def cardinal(dx: int, dy: int) -> tuple[int, int]:
    """Unit step along the axis with the longer remaining distance."""
    if abs(dx) >= abs(dy):
        return (1 if dx > 0 else -1), 0
    return 0, (1 if dy > 0 else -1)


class Trail:
    """Per-tile value agents write (MARK_VALUE) and that decays over time.

    Stores the last write time per tile, so decay costs nothing per frame:
    value = MARK_VALUE * MARK_DECAY ** seconds_since_write (0 if never).
    """

    def __init__(self) -> None:
        self.stamp: list[list[float | None]] = [
            [None] * GRID_SIZE for _ in range(GRID_SIZE)
        ]

    def mark(self, col: int, row: int, now: float) -> None:
        self.stamp[row][col] = now

    def value(self, col: int, row: int, now: float) -> float:
        t = self.stamp[row][col]
        return 0.0 if t is None else MARK_VALUE * MARK_DECAY ** (now - t)

    def cone_mean(self, col: int, row: int, cone: list[Tile], now: float) -> float:
        """Mean value over the in-grid tiles of a sensing cone."""
        total = n = 0
        for dx, dy in cone:
            c, r = col + dx, row + dy
            if 0 <= c < GRID_SIZE and 0 <= r < GRID_SIZE:
                total += self.value(c, r, now)
                n += 1
        return total / n if n else 0.0


def _cone(dc: int, dr: int) -> list[Tile]:
    """Tiles within PUSH_REACH ahead of an agent facing (dc, dr)."""
    cells = []
    for k in range(1, PUSH_REACH + 1):
        for j in range(-(k - 1), k):
            if k + abs(j) <= PUSH_REACH:
                cells.append((dc * k + dr * j, dr * k + dc * j))
    return cells


CONES = {d: _cone(*d) for d in DIRECTIONS}


class Agent:
    def __init__(self, ident: int, col: int, row: int) -> None:
        self.id = ident
        self.col = col
        self.row = row
        self.charge: float = 6000  # U
        self.state = "walk"  # walk | seek | charge
        self.age = 0  # steps spent seeking (queue position: older goes first)
        self.released = False  # just finished charging, still on the station
        self.moved = False  # already moved during this step

    @property
    def color(self) -> tuple[int, int, int]:
        """AGENT_COLOR darkened in proportion to the remaining charge."""
        f = self.charge / MAX_CHARGE
        return tuple(round(c * f) for c in AGENT_COLOR)

    def charge_up(self, dt: float) -> None:
        """Gain charge while on a station; resume walking when full."""
        self.charge = min(MAX_CHARGE, self.charge + CHARGE_RATE * dt)
        if self.charge >= MAX_CHARGE:
            self.state = "walk"
            self.released = True

    def slack(self, nav: NavMap) -> float:
        """Spare moves: charge left minus the cost of walking to a station."""
        return self.charge / MOVE_COST - nav.dist[self.row][self.col]

    def priority(self, nav: NavMap) -> tuple:
        """Total order used to decide who yields; higher goes first.

        Released station occupants, then seekers about to run dry, then
        seekers by queue age, then walkers (least slack first).
        """
        if self.released:
            return 4, 0, self.id
        if self.state == "seek":
            slack = self.slack(nav)
            if slack < URGENT_SLACK:
                return 3, -slack, self.id
            return 2, self.age, -self.id
        return 1, -self.slack(nav), self.id

    def pushable(self, nav: NavMap, claim: tuple) -> bool:
        """Can give way to a claimant of priority `claim`: not charging, lower
        priority, and enough slack left to afford the detour."""
        floor = PUSH_FLOOR_SEEK if self.state == "seek" else PUSH_FLOOR_WALK
        return (
            self.state != "charge"
            and self.priority(nav) < claim
            and self.slack(nav) >= floor
        )

    def step(self, world: World, trail: Trail, now: float) -> None:
        """One tile move per step; a completed move costs MOVE_COST charge.

        Walking steps to the free neighbour whose sensing cone (tiles within
        PUSH_REACH ahead) has the lowest mean trail value, and stays out of
        queue zones.
        Below LOW_CHARGE the agent seeks: it follows the arrow on its tile to
        the nearest station. If a lower-priority agent is in the way, that
        agent is pushed aside (see World.push); otherwise the seeker waits.
        """
        if self.state == "charge" or self.moved or self.charge < MOVE_COST:
            return
        nav = world.nav
        here = (self.col, self.row)
        if self.state == "walk" and self.charge < LOW_CHARGE:
            self.state = "seek"
            self.age = 0
            world.press(here, 1)

        if self.state == "seek":
            if nav.offset[self.row][self.col] == (0, 0):
                self.state = "charge"
                world.press(here, -1)
                return
            dc, dr = nav.arrow[self.row][self.col]
            target = (self.col + dc, self.row + dr)
            if world.free(target):
                world.move(self, target)
                return
            blocker = world.occ[target]
            claim = self.priority(nav)
            if blocker.pushable(nav, claim) and world.push(blocker, claim, here):
                world.move(self, target)
            return

        options = [
            (self.col + dc, self.row + dr)
            for dc, dr in DIRECTIONS
            if world.free((self.col + dc, self.row + dr))
        ]
        if world.pressure[self.row][self.col] >= PRESSURE_LIMIT:  # retreat
            away = [
                t
                for t in options
                if nav.dist[t[1]][t[0]] > nav.dist[self.row][self.col]
            ]
            options = away or options
        else:  # stay out of queue zones
            options = [
                t for t in options if world.pressure[t[1]][t[0]] < PRESSURE_LIMIT
            ]
        if options:
            # Head where the trail is coldest within sensing range.
            def heat(t: Tile) -> float:
                cone = CONES[(t[0] - self.col, t[1] - self.row)]
                mean = trail.cone_mean(self.col, self.row, cone, now)
                return mean + random.random() * WALK_NOISE

            world.move(self, min(options, key=heat))
        elif self.released:  # boxed in on a station: push someone aside
            claim = self.priority(nav)
            for dc, dr in random.sample(DIRECTIONS, 4):
                t = (self.col + dc, self.row + dr)
                b = world.occ.get(t)
                if b and b.pushable(nav, claim) and world.push(b, claim, here):
                    world.move(self, t)
                    break


class World:
    """Occupancy plus the shared tile state agents coordinate through.

    `pressure` is a per-tile count of seekers within PUSH_REACH tiles, i.e.
    what an agent could read off the tiles around it. Nothing here lets an
    agent query another agent directly.
    """

    def __init__(self, nav: NavMap, agents: list[Agent]) -> None:
        self.nav = nav
        self.occ: dict[Tile, Agent] = {(a.col, a.row): a for a in agents}
        self.pressure = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

    def free(self, tile: Tile) -> bool:
        return (
            0 <= tile[0] < GRID_SIZE
            and 0 <= tile[1] < GRID_SIZE
            and tile not in self.occ
        )

    def is_station(self, tile: Tile) -> bool:
        return self.nav.dist[tile[1]][tile[0]] == 0

    def press(self, tile: Tile, delta: int) -> None:
        r = PUSH_REACH
        for dx in range(-r, r + 1):
            span = r - abs(dx)
            for dy in range(-span, span + 1):
                c, row = tile[0] + dx, tile[1] + dy
                if 0 <= c < GRID_SIZE and 0 <= row < GRID_SIZE:
                    self.pressure[row][c] += delta

    def move(self, agent: Agent, tile: Tile) -> None:
        old = (agent.col, agent.row)
        del self.occ[old]
        self.occ[tile] = agent
        agent.col, agent.row = tile
        agent.charge -= MOVE_COST
        agent.moved = True
        if agent.released and not self.is_station(tile):
            agent.released = False
        if agent.state == "seek":
            self.press(old, -1)
            if self.is_station(tile):
                agent.state = "charge"
            else:
                self.press(tile, 1)

    def push(self, blocker: Agent, claim: tuple, claimant: Tile) -> bool:
        """Shift `blocker` one tile out of the way, within PUSH_REACH tiles.

        Searches through pushable agents for the nearest free (non-station)
        tile; every agent on the path then steps forward one tile, like a
        train, so the blocker's tile is freed. Nobody is pushed unless the
        whole path works.
        """
        start = (blocker.col, blocker.row)
        queue = deque([[start]])
        seen = {start, claimant}
        while queue:
            path = queue.popleft()
            if len(path) > PUSH_REACH:
                continue
            u = path[-1]
            for dc, dr in random.sample(DIRECTIONS, 4):
                v = (u[0] + dc, u[1] + dr)
                if v in seen or not (0 <= v[0] < GRID_SIZE and 0 <= v[1] < GRID_SIZE):
                    continue
                seen.add(v)
                if v not in self.occ:
                    if self.is_station(v):
                        continue
                    chain = [self.occ[p] for p in path]
                    if all(y.pushable(self.nav, claim) for y in chain):
                        for y, t in reversed(
                            list(zip(chain, [*path[1:], v], strict=True))
                        ):
                            self.move(y, t)
                        return True
                    continue
                if self.occ[v].pushable(self.nav, claim):
                    queue.append([*path, v])
        return False

    def tick(self, agents: list[Agent], trail: Trail, now: float) -> None:
        """One step for everyone, highest priority first."""
        for agent in agents:
            agent.moved = False
        for agent in sorted(agents, key=lambda a: a.priority(self.nav), reverse=True):
            agent.step(self, trail, now)
        for agent in agents:
            if agent.state == "seek":
                agent.age += 1


@dataclass
class NavMap:
    offset: list[list[Tile]]  # (dx, dy) to the target station
    arrow: list[list[Tile]]  # cardinal unit step an agent should take
    dist: list[list[int]]  # moves to the nearest station


def make_stations() -> list[Tile]:
    """STATION_COUNT charging points, equally spaced along the walls.

    Offset by half a spacing so none lands on a corner.
    """
    last = GRID_SIZE - 1
    ring = (
        [(c, 0) for c in range(GRID_SIZE)]
        + [(last, r) for r in range(1, GRID_SIZE)]
        + [(c, last) for c in range(last - 1, -1, -1)]
        + [(0, r) for r in range(last - 1, 0, -1)]
    )
    stations = [
        ring[round((i + 0.5) * len(ring) / STATION_COUNT) % len(ring)]
        for i in range(STATION_COUNT)
    ]
    corners = {(0, 0), (last, 0), (0, last), (last, last)}
    assert not corners & set(stations)
    return stations


def build_nav(stations: list[Tile]) -> NavMap:
    """Shortest-path arrows to the nearest station (multi-source BFS)."""
    dist: dict[Tile, int] = {s: 0 for s in stations}
    src: dict[Tile, Tile] = {s: s for s in stations}
    queue = deque(stations)
    while queue:
        u = queue.popleft()
        for dc, dr in DIRECTIONS:
            v = (u[0] + dc, u[1] + dr)
            if not (0 <= v[0] < GRID_SIZE and 0 <= v[1] < GRID_SIZE) or v in dist:
                continue
            dist[v] = dist[u] + 1
            src[v] = src[u]
            queue.append(v)

    offset = [[(0, 0)] * GRID_SIZE for _ in range(GRID_SIZE)]
    arrow = [[(0, 0)] * GRID_SIZE for _ in range(GRID_SIZE)]
    grid_dist = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    for (col, row), d in dist.items():
        sx, sy = src[(col, row)]
        offset[row][col] = (sx - col, sy - row)
        grid_dist[row][col] = d
        if d == 0:
            continue
        steps = [
            (dc, dr)
            for dc, dr in DIRECTIONS
            if dist.get((col + dc, row + dr)) == d - 1
            and src[(col + dc, row + dr)] == (sx, sy)
        ]
        want = cardinal(sx - col, sy - row)
        arrow[row][col] = want if want in steps else steps[0]
    return NavMap(offset, arrow, grid_dist)


def advance(
    world: World,
    agents: list[Agent],
    trail: Trail,
    now: float,
    dt: float,
    step_timer: float,
) -> float:
    """Run one frame: whole steps owed so far, trail marks, charging.

    Returns the leftover step timer.
    """
    step_timer += dt
    while step_timer >= STEP_INTERVAL:
        step_timer -= STEP_INTERVAL
        world.tick(agents, trail, now)
    for agent in agents:
        trail.mark(agent.col, agent.row, now)
    for agent in agents:
        if agent.state == "charge":
            agent.charge_up(dt)
    return step_timer
