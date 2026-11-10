from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import dataclass

from apeks.config import (
    AGENT_COLOR,
    BARRIER_LIFETIME_MAX,
    BARRIER_LIFETIME_MIN,
    BARRIER_SPAWN_INTERVAL_MAX,
    BARRIER_SPAWN_INTERVAL_MIN,
    CHARGE_RATE,
    DIRECTIONS,
    FOOD_AMOUNT_MAX,
    FOOD_AMOUNT_MIN,
    GRID_SIZE,
    HAZARD_LIFETIME_MAX,
    HAZARD_LIFETIME_MIN,
    HAZARD_SHAPES,
    HAZARD_SPAWN_INTERVAL_MAX,
    HAZARD_SPAWN_INTERVAL_MIN,
    LOW_CHARGE,
    MARK_DECAY,
    MARK_VALUE,
    MAX_CHARGE,
    MAX_MOVING_BARRIERS,
    MOVE_COST,
    NEST_TILES,
    OBSTACLE_DRIFT_INTERVAL,
    PRESSURE_LIMIT,
    PUSH_FLOOR_SEEK,
    PUSH_FLOOR_WALK,
    PUSH_REACH,
    STATION_COUNT,
    STEP_INTERVAL,
    TOKEN_TTL,
    URGENT_SLACK,
    WALK_NOISE,
)

Tile = tuple[int, int]


def cardinal(dx: int, dy: int) -> tuple[int, int]:
    """Unit step along the axis with the longer remaining distance."""
    if abs(dx) >= abs(dy):
        return (1 if dx > 0 else -1), 0
    return 0, (1 if dy > 0 else -1)


@dataclass
class MovingBarrier:
    coords: list[Tile]
    axis: str  # 'x' or 'y'
    direction: int  # +1 or -1
    min_pos: int
    max_pos: int
    timer: float = 0.0
    interval: float = OBSTACLE_DRIFT_INTERVAL
    expiry: float = float("inf")


class Trail:
    """Per-tile value agents write (MARK_VALUE) and that decays over time.

    Stores the last write time per tile, so decay costs nothing per frame:
    value = MARK_VALUE * MARK_DECAY ** seconds_since_write (0 if never).
    When accumulate=True (used for food trails), multiple marks on the same
    tile reinforce the trail by adding MARK_VALUE to the decayed level.
    """

    def __init__(self, accumulate: bool = False) -> None:
        self.accumulate = accumulate
        self.stamp: list[list[float | None]] = [
            [None] * GRID_SIZE for _ in range(GRID_SIZE)
        ]
        self.levels: list[list[float]] = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.max_level: float = 0.0

    def mark(self, col: int, row: int, now: float) -> None:
        if self.accumulate:
            t = self.stamp[row][col]
            if t is None:
                self.levels[row][col] = float(MARK_VALUE)
            else:
                current = self.levels[row][col] * (MARK_DECAY ** (now - t))
                self.levels[row][col] = (
                    current if current >= 0.5 else 0.0
                ) + MARK_VALUE
            if self.levels[row][col] > self.max_level:
                self.max_level = self.levels[row][col]
        self.stamp[row][col] = now

    def value(self, col: int, row: int, now: float) -> float:
        t = self.stamp[row][col]
        if t is None:
            return 0.0
        base = self.levels[row][col] if self.accumulate else MARK_VALUE
        return base * (MARK_DECAY ** (now - t))

    def cone_mean(self, col: int, row: int, cone: list[Tile], now: float) -> float:
        """Mean value over the in-grid tiles of a sensing cone."""
        total = n = 0
        for dx, dy in cone:
            c, r = col + dx, row + dy
            if 0 <= c < GRID_SIZE and 0 <= r < GRID_SIZE:
                total += self.value(c, r, now)
                n += 1
        return total / n if n else 0.0

    def clear_tile(self, col: int, row: int) -> None:
        """Clear pheromone level and timestamp on a tile."""
        if 0 <= col < GRID_SIZE and 0 <= row < GRID_SIZE:
            self.stamp[row][col] = None
            self.levels[row][col] = 0.0

    def clear_area(self, col: int, row: int, radius: int = 4) -> None:
        """Clear pheromone trail within a radius around a depleted location."""
        for dx in range(-radius, radius + 1):
            for dy in range(-(radius - abs(dx)), radius - abs(dx) + 1):
                self.clear_tile(col + dx, row + dy)

    def clear_trail_branch(
        self,
        start_col: int,
        start_row: int,
        nest_nav: NavMap,
        food: dict[Tile, int],
        max_steps: int = 200,
    ) -> None:
        """Trace and clear an obsolete pheromone trail branch from a depleted food
        source back towards the nest.

        Follows connected active trail tiles towards the nest, stopping immediately
        when reaching any tile that lies on the active trail path of any remaining
        active food cluster, or near the nest (dist <= 6). This guarantees that
        shared trunks and forked branches to other food sources
        are completely preserved.
        """
        # 1. Identify all protected tiles that belong to the active trails of
        # remaining food clusters
        protected_tiles: set[Tile] = set()
        for f_tile, amt in food.items():
            if amt <= 0:
                continue
            # If the active food is this exact depleted source, skip it
            if abs(f_tile[0] - start_col) + abs(f_tile[1] - start_row) <= 6:
                continue
            # Trace the gradient path from active food cluster to nest
            curr_f = f_tile
            while nest_nav.dist[curr_f[1]][curr_f[0]] > 6:
                protected_tiles.add(curr_f)
                for dc, dr in DIRECTIONS:
                    protected_tiles.add((curr_f[0] + dc, curr_f[1] + dr))
                dc, dr = nest_nav.arrow[curr_f[1]][curr_f[0]]
                curr_f = (curr_f[0] + dc, curr_f[1] + dr)
            protected_tiles.add(curr_f)

        # 2. Trace and clear the unique branch of the depleted food source
        curr = (start_col, start_row)
        visited = {curr}
        queue = [curr]
        for dx in range(-4, 5):
            for dy in range(-(4 - abs(dx)), 4 - abs(dx) + 1):
                tc, tr = start_col + dx, start_row + dy
                if (
                    0 <= tc < GRID_SIZE
                    and 0 <= tr < GRID_SIZE
                    and (tc, tr) not in protected_tiles
                ):
                    if self.levels[tr][tc] >= 0.5 and (tc, tr) not in visited:
                        visited.add((tc, tr))
                        queue.append((tc, tr))
        steps = 0

        while queue and steps < max_steps:
            c, r = queue.pop(0)
            steps += 1

            # If this tile is on an active food source's path, DO NOT clear it!
            # Stop here!
            if (c, r) in protected_tiles:
                continue

            self.clear_tile(c, r)

            if nest_nav.dist[r][c] <= 6:
                continue

            curr_dist = nest_nav.dist[r][c]
            for dc, dr in DIRECTIONS:
                nc, nr = c + dc, r + dr
                if (
                    0 <= nc < GRID_SIZE
                    and 0 <= nr < GRID_SIZE
                    and (nc, nr) not in visited
                ):
                    if (
                        self.levels[nr][nc] >= 0.5
                        and nest_nav.dist[nr][nc] <= curr_dist
                    ):
                        visited.add((nc, nr))
                        queue.append((nc, nr))


def _cone(dc: int, dr: int) -> list[Tile]:
    """Tiles within PUSH_REACH ahead of an agent facing (dc, dr)."""
    cells = []
    for k in range(1, PUSH_REACH + 1):
        for j in range(-(k - 1), k):
            if k + abs(j) <= PUSH_REACH:
                cells.append((dc * k + dr * j, dr * k + dc * j))
    return cells


CONES = {d: _cone(*d) for d in DIRECTIONS}


def has_active_branch_outward(world: World, tile: Tile, now: float) -> bool:
    """Check if 'tile' has an outward food trail branch leading to an active
    food source."""
    active_foods = {f for f, amt in world.food.items() if amt > 0}
    if not active_foods:
        return False
    if any(abs(f[0] - tile[0]) + abs(f[1] - tile[1]) <= 1 for f in active_foods):
        return True
    visited = {tile}
    queue = deque([tile])
    while queue:
        curr = queue.popleft()
        curr_dist = world.nest_nav.dist[curr[1]][curr[0]]
        for dc, dr in DIRECTIONS:
            nxt = (curr[0] + dc, curr[1] + dr)
            if not (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE):
                continue
            if nxt in visited:
                continue
            if any(abs(f[0] - nxt[0]) + abs(f[1] - nxt[1]) <= 1 for f in active_foods):
                return True
            if world.food_trail.value(nxt[0], nxt[1], now) >= 0.5:
                nxt_dist = world.nest_nav.dist[nxt[1]][nxt[0]]
                if nxt_dist >= curr_dist:
                    visited.add(nxt)
                    queue.append(nxt)
    return False


class Agent:
    def __init__(
        self, ident: int, col: int, row: int, charge: float = MAX_CHARGE
    ) -> None:
        self.id = ident
        self.col = col
        self.row = row
        self.charge: float = charge  # U
        self.state = "walk"  # walk | seek | charge | return_nest
        self.age = 0  # steps spent seeking (queue position: older goes first)
        self.released = False  # just finished charging, still on the station
        self.moved = False  # already moved during this step
        self.has_food = False
        self.food_trail_valid = True
        self.last_food_tile: Tile | None = None
        self.prev_tile: Tile | None = None
        self.last_food_mark_pos: Tile | None = None
        self.role: str = "scout"
        self.waypoint: Tile | None = None
        self.last_llm_time: float = 0.0
        self.llm_thought: str = ""
        self.erasing_trail: bool = False
        self.avoided_collision_from_trail: bool = False
        self.llm_mode: str = "classical"

    @property
    def color(self) -> tuple[int, int, int]:
        """AGENT_COLOR darkened in proportion to the remaining charge."""
        f = self.charge / MAX_CHARGE
        return tuple(round(c * f) for c in AGENT_COLOR)

    @property
    def is_dead(self) -> bool:
        """True if the agent has run out of charge and is not charging on a station."""
        return self.charge < MOVE_COST and self.state != "charge"

    def charge_up(self, dt: float) -> None:
        """Gain charge while on a station; resume walking when full."""
        self.charge = min(MAX_CHARGE, self.charge + CHARGE_RATE * dt)
        if self.charge >= MAX_CHARGE:
            self.state = "return_nest" if self.has_food else "walk"
            self.released = True

    def slack(self, nav: NavMap) -> float:
        """Spare moves: charge left minus the cost of walking to a station."""
        return self.charge / MOVE_COST - nav.dist[self.row][self.col]

    def priority(self, nav: NavMap) -> tuple:
        """Total order used to decide who yields; higher goes first.

        Released station occupants, then seekers about to run dry, then
        seekers by queue age, then foragers returning to nest, then walkers (least
        slack first).
        """
        if self.released:
            return 5, 0, self.id
        if self.state == "seek":
            slack = self.slack(nav)
            if slack < URGENT_SLACK:
                return 4, -slack, self.id
            return 3, self.age, -self.id
        if self.state == "return_nest":
            return 2, -self.slack(nav), self.id
        return 1, -self.slack(nav), self.id

    def pushable(self, nav: NavMap, claim: tuple) -> bool:
        """Can give way to a claimant of priority `claim`: not charging, lower
        priority, and enough slack left to afford the detour."""
        floor = PUSH_FLOOR_SEEK if self.state == "seek" else PUSH_FLOOR_WALK
        if (
            getattr(self, "llm_mode", "classical")
            in ("global_contextual", "pure_llm", "coarse_fine")
            and claim[0] >= 5
            and self.charge >= MOVE_COST
        ):
            return not self.is_dead and self.state != "charge"
        return (
            not self.is_dead
            and self.state != "charge"
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

        if self.has_food and world.is_nest(here):
            self.has_food = False
            self.food_trail_valid = True
            world.food_collected += 1
            self.last_food_mark_pos = None
            self.waypoint = None
            self.role = "forager"
            if self.state == "return_nest":
                self.state = "walk"
                return

        if self.state in ("walk", "return_nest") and self.charge < LOW_CHARGE:
            self.state = "seek"
            self.food_trail_valid = False
            self.age = 0
            if self.waypoint is not None and not world.is_station(self.waypoint):
                self.waypoint = None
            world.press(here, 1)

        if self.state == "seek":
            if nav.offset[self.row][self.col] == (0, 0):
                self.state = "charge"
                world.press(here, -1)
                self.waypoint = None
                return
            if self.waypoint is not None:
                if here == self.waypoint:
                    self.waypoint = None
                elif self.waypoint in world.obstacles:
                    self.waypoint = None
                else:
                    target_dist = abs(self.col - self.waypoint[0]) + abs(
                        self.row - self.waypoint[1]
                    )
                    if target_dist == 1:
                        if not world.free(self.waypoint):
                            occ = world.occ.get(self.waypoint)
                            if (
                                occ
                                and getattr(occ, "released", False)
                                and getattr(self, "llm_mode", "classical")
                                in ("global_contextual", "pure_llm", "coarse_fine")
                            ):
                                side_options = [
                                    (self.col + ddc, self.row + ddr)
                                    for ddc, ddr in DIRECTIONS
                                    if world.free((self.col + ddc, self.row + ddr))
                                ]
                                if side_options:
                                    world.move(self, side_options[0])
                                    return
                            return
                        world.move(self, self.waypoint)
                        return
                    options = [
                        (self.col + ddc, self.row + ddr)
                        for ddc, ddr in DIRECTIONS
                        if world.free((self.col + ddc, self.row + ddr))
                    ]
                    if options:
                        closer = [
                            t
                            for t in options
                            if abs(t[0] - self.waypoint[0])
                            + abs(t[1] - self.waypoint[1])
                            < target_dist
                        ]
                        if closer:
                            if (
                                world.is_station(self.waypoint)
                                and target_dist == 2
                                and not world.free(self.waypoint)
                                and getattr(
                                    world.occ.get(self.waypoint), "released", False
                                )
                                and getattr(self, "llm_mode", "classical")
                                in ("global_contextual", "pure_llm", "coarse_fine")
                            ):
                                return
                            best = min(
                                closer,
                                key=lambda t: (
                                    abs(t[0] - self.waypoint[0])
                                    + abs(t[1] - self.waypoint[1])
                                ),
                            )
                            world.move(self, best)
                            return
                        if world.is_station(self.waypoint):
                            cd = cardinal(
                                self.waypoint[0] - self.col, self.waypoint[1] - self.row
                            )
                            fwd = (self.col + cd[0], self.row + cd[1])
                            b = world.occ.get(fwd)
                            is_obstacle_ahead = fwd in world.obstacles or (
                                not world.free(fwd) and b is None
                            )
                            if is_obstacle_ahead:
                                cand = [
                                    t for t in options if t != self.prev_tile
                                ] or options
                                best = min(
                                    cand,
                                    key=lambda t: (
                                        abs(t[0] - self.waypoint[0])
                                        + abs(t[1] - self.waypoint[1])
                                    ),
                                )
                                world.move(self, best)
                                return
                            is_llm = getattr(self, "llm_mode", "classical") in (
                                "global_contextual",
                                "pure_llm",
                                "coarse_fine",
                            )
                            if b is not None and (
                                b.state in ("seek", "charge")
                                or b.waypoint == self.waypoint
                            ):
                                if is_llm and getattr(
                                    self, "llm_mode", "classical"
                                ) in ("global_contextual", "pure_llm"):
                                    st_occ = world.occ.get(self.waypoint)
                                    queue_load = (1 if st_occ else 0) + sum(
                                        1
                                        for other in world.occ.values()
                                        if other.state == "seek"
                                        and other.waypoint == self.waypoint
                                    )
                                    if queue_load >= 2:
                                        alt_candidates = [
                                            st
                                            for st in getattr(world, "stations", [])
                                            if st != self.waypoint
                                        ]
                                        best_alt = None
                                        best_alt_dist = 999999
                                        for alt in alt_candidates:
                                            alt_dist = abs(self.col - alt[0]) + abs(
                                                self.row - alt[1]
                                            )
                                            if (
                                                self.charge
                                                >= (alt_dist + 8) * MOVE_COST
                                                and alt_dist < best_alt_dist
                                            ):
                                                alt_occ = world.occ.get(alt)
                                                alt_q = (1 if alt_occ else 0) + sum(
                                                    1
                                                    for o in world.occ.values()
                                                    if o.state == "seek"
                                                    and o.waypoint == alt
                                                )
                                                if alt_q < queue_load:
                                                    best_alt = alt
                                                    best_alt_dist = alt_dist
                                        if best_alt is not None:
                                            self.waypoint = best_alt
                                            cand = [
                                                t
                                                for t in options
                                                if t != self.prev_tile
                                            ] or options
                                            best = min(
                                                cand,
                                                key=lambda t: (
                                                    abs(t[0] - best_alt[0])
                                                    + abs(t[1] - best_alt[1])
                                                ),
                                            )
                                            world.move(self, best)
                                            return
                                return
                            elif b is not None and (b.state == "return_nest" or is_llm):
                                cand = [
                                    t for t in options if t != self.prev_tile
                                ] or options
                                best = min(
                                    cand,
                                    key=lambda t: (
                                        abs(t[0] - self.waypoint[0])
                                        + abs(t[1] - self.waypoint[1])
                                    ),
                                )
                                world.move(self, best)
                                return
                            return
                        cand = [t for t in options if t != self.prev_tile] or options
                        best = min(
                            cand,
                            key=lambda t: (
                                abs(t[0] - self.waypoint[0])
                                + abs(t[1] - self.waypoint[1])
                            ),
                        )
                        world.move(self, best)
                        return
            dc, dr = nav.arrow[self.row][self.col]
            target = (self.col + dc, self.row + dr)
            if world.free(target):
                world.move(self, target)
                return
            blocker = world.occ.get(target)
            if blocker:
                claim = self.priority(nav)
                if blocker.pushable(nav, claim) and world.push(blocker, claim, here):
                    world.move(self, target)
                    return
                if (
                    blocker.is_dead
                    or (
                        blocker.state == "return_nest"
                        and self.slack(nav) >= PUSH_FLOOR_SEEK
                    )
                    or (
                        getattr(self, "llm_mode", "classical")
                        in ("global_contextual", "pure_llm", "coarse_fine")
                        and blocker.state != "charge"
                    )
                ):
                    detours = [
                        (self.col + ddc, self.row + ddr)
                        for ddc, ddr in DIRECTIONS
                        if world.free((self.col + ddc, self.row + ddr))
                    ]
                    if detours:
                        cand = [t for t in detours if t != self.prev_tile] or detours
                        world.move(self, min(cand, key=lambda t: nav.dist[t[1]][t[0]]))
                        return
                return
            else:
                detours = [
                    (self.col + ddc, self.row + ddr)
                    for ddc, ddr in DIRECTIONS
                    if world.free((self.col + ddc, self.row + ddr))
                ]
                if detours:
                    cand = [t for t in detours if t != self.prev_tile] or detours
                    world.move(self, min(cand, key=lambda t: nav.dist[t[1]][t[0]]))
                    return
            return

        if self.state == "return_nest":
            dc, dr = world.nest_nav.arrow[self.row][self.col]
            target = (self.col + dc, self.row + dr)
            if world.free(target):
                world.move(self, target)
                return
            blocker = world.occ.get(target)
            if blocker:
                claim = self.priority(nav)
                if blocker.pushable(nav, claim) and world.push(blocker, claim, here):
                    world.move(self, target)
                    return
                elif self.released:
                    for rdc, rdr in random.sample(DIRECTIONS, 4):
                        t = (self.col + rdc, self.row + rdr)
                        b = world.occ.get(t)
                        if b and b.pushable(nav, claim) and world.push(b, claim, here):
                            world.move(self, t)
                            return
                if (
                    blocker.is_dead
                    or blocker.state != "charge"
                    or getattr(self, "llm_mode", "classical")
                    in ("global_contextual", "pure_llm", "coarse_fine")
                ):
                    detours = [
                        (self.col + ddc, self.row + ddr)
                        for ddc, ddr in DIRECTIONS
                        if world.free((self.col + ddc, self.row + ddr))
                    ]
                    if detours:
                        cand = [t for t in detours if t != self.prev_tile] or detours
                        world.move(
                            self,
                            min(cand, key=lambda t: world.nest_nav.dist[t[1]][t[0]]),
                        )
                        return
                return
            else:
                detours = [
                    (self.col + ddc, self.row + ddr)
                    for ddc, ddr in DIRECTIONS
                    if world.free((self.col + ddc, self.row + ddr))
                ]
                if detours:
                    cand = [t for t in detours if t != self.prev_tile] or detours
                    world.move(
                        self,
                        min(cand, key=lambda t: world.nest_nav.dist[t[1]][t[0]]),
                    )
                    return
            return

        if world.food.get(here, 0) > 0:
            world.food[here] -= 1
            cluster_empty = False
            if world.food[here] == 0:
                del world.food[here]
                cluster_empty = not any(
                    amt > 0
                    for t, amt in world.food.items()
                    if abs(t[0] - here[0]) + abs(t[1] - here[1]) <= 6
                )
            self.has_food = True
            self.food_trail_valid = not cluster_empty
            self.last_food_tile = here if not cluster_empty else None
            self.state = "return_nest"
            self.role = "forager"
            self.last_food_mark_pos = None
            self.waypoint = None
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
        elif self.released or nav.dist[self.row][self.col] == 0:
            away = [
                t
                for t in options
                if nav.dist[t[1]][t[0]] > nav.dist[self.row][self.col]
            ]
            options = away or options
        else:  # stay out of queue zones
            low_pressure = [
                t for t in options if world.pressure[t[1]][t[0]] < PRESSURE_LIMIT
            ]
            if getattr(self, "llm_mode", "classical") in (
                "global_contextual",
                "pure_llm",
                "coarse_fine",
            ):
                options = low_pressure or options
            else:
                options = low_pressure
        if options:
            candidates = [t for t in options if t != self.prev_tile] or options
            food_options = [t for t in candidates if world.food.get(t, 0) > 0]
            if food_options:
                world.move(self, food_options[0])
                return

            if (
                getattr(self, "llm_mode", "classical")
                in ("global_contextual", "pure_llm", "coarse_fine")
                and not self.has_food
            ):
                nearby_active_foods = [
                    f
                    for f, amt in world.food.items()
                    if amt > 0 and abs(f[0] - self.col) + abs(f[1] - self.row) <= 6
                ]
                if nearby_active_foods:
                    nearby_active_foods.sort(
                        key=lambda f: abs(f[0] - self.col) + abs(f[1] - self.row)
                    )
                    target_food = nearby_active_foods[
                        self.id % len(nearby_active_foods)
                    ]
                    self.role = "forager"
                    self.waypoint = target_food
                    self.erasing_trail = False
                    best_step = min(
                        candidates,
                        key=lambda t: (
                            abs(t[0] - target_food[0]) + abs(t[1] - target_food[1])
                        ),
                    )
                    world.move(self, best_step)
                    return

            if getattr(self, "erasing_trail", False) and not self.has_food:
                on_active_node = has_active_branch_outward(
                    world, (self.col, self.row), now
                )
                if not on_active_node:
                    world.food_trail.clear_tile(self.col, self.row)
                here_dist = world.nest_nav.dist[self.row][self.col]
                active_inward = [
                    t
                    for t in candidates
                    if world.food_trail.value(t[0], t[1], now) >= 0.5
                    and world.nest_nav.dist[t[1]][t[0]] < here_dist
                    and not has_active_branch_outward(world, t, now)
                ]
                has_active_food = any(
                    amt > 0
                    for f, amt in world.food.items()
                    if abs(f[0] - self.col) + abs(f[1] - self.row) <= 6
                )
                if (
                    active_inward
                    and here_dist > 6
                    and not has_active_food
                    and not on_active_node
                ):
                    next_tile = min(
                        active_inward, key=lambda t: world.nest_nav.dist[t[1]][t[0]]
                    )
                    world.move(self, next_tile)
                    world.food_trail.clear_tile(self.col, self.row)
                    return
                self.erasing_trail = False
                is_llm = getattr(self, "llm_mode", "classical") in (
                    "global_contextual",
                    "pure_llm",
                    "coarse_fine",
                )
                self.role = "forager" if is_llm else "scout"
                self.waypoint = None
                if is_llm:
                    active_branch_tiles = [
                        t
                        for t in candidates
                        if world.food_trail.value(t[0], t[1], now) >= 0.5
                        and has_active_branch_outward(world, t, now)
                    ]
                    if active_branch_tiles:
                        best_branch = max(
                            active_branch_tiles,
                            key=lambda t: world.nest_nav.dist[t[1]][t[0]],
                        )
                        world.move(self, best_branch)
                        return

            if self.waypoint is not None and here == self.waypoint:
                self.waypoint = None

            if self.last_food_tile is not None:
                if (
                    abs(self.last_food_tile[0] - self.col)
                    + abs(self.last_food_tile[1] - self.row)
                    <= 6
                    and world.food.get(self.last_food_tile, 0) == 0
                ):
                    self.last_food_tile = None

            here_dist = world.nest_nav.dist[self.row][self.col]

            def food_scent(t: Tile) -> float:
                cone = CONES[(t[0] - self.col, t[1] - self.row)]
                mean = world.food_trail.cone_mean(self.col, self.row, cone, now)
                tile_val = world.food_trail.value(t[0], t[1], now)
                return tile_val * 2.0 + mean

            is_llm = getattr(self, "llm_mode", "classical") in (
                "global_contextual",
                "pure_llm",
                "coarse_fine",
            )
            is_on_trail = world.food_trail.value(self.col, self.row, now) >= 0.5
            if is_on_trail:
                self.avoided_collision_from_trail = False

            if (
                is_llm
                and not is_on_trail
                and not getattr(self, "avoided_collision_from_trail", False)
            ):
                active_trail_tiles = []
            else:
                active_trail_tiles = [
                    t
                    for t in candidates
                    if world.food_trail.value(t[0], t[1], now) >= 0.5
                ]

            tokens_now = world.active_tokens(now)
            depleted_in_range = any(
                "[DEPLETED]" in text
                and abs(pos[0] - self.col) + abs(pos[1] - self.row) <= 6
                for pos, text in tokens_now.items()
            )
            nearby_has_food = any(
                amt > 0
                for t, amt in world.food.items()
                if abs(t[0] - self.col) + abs(t[1] - self.row) <= 6
            )

            in_depleted_zone = depleted_in_range and not nearby_has_food
            # Determine if this agent should follow the minimum-path food pheromone
            # trail
            is_foraging = (
                not in_depleted_zone
                and (
                    self.waypoint is None
                    or getattr(self, "role", "scout") == "forager"
                    or self.waypoint in world.food
                    or is_on_trail
                )
            ) or (
                getattr(self, "role", "scout") == "forager"
                and any(
                    world.nest_nav.dist[t[1]][t[0]] >= here_dist
                    for t in active_trail_tiles
                )
            )

            if is_foraging and (active_trail_tiles or is_on_trail):
                outward_trail = [
                    t
                    for t in active_trail_tiles
                    if world.nest_nav.dist[t[1]][t[0]] >= here_dist
                ]
                if is_llm:
                    active_outward_trail = [
                        t
                        for t in outward_trail
                        if has_active_branch_outward(world, t, now)
                    ]
                else:
                    active_outward_trail = outward_trail

                if not active_outward_trail and here_dist >= 15 and not nearby_has_food:
                    self.role = "scout"
                    self.waypoint = None
                    is_foraging = False
                    self.avoided_collision_from_trail = False
                    if is_llm or in_depleted_zone:
                        already_erasing = any(
                            other is not self
                            and getattr(other, "erasing_trail", False)
                            and abs(other.col - self.col) + abs(other.row - self.row)
                            <= 25
                            for other in world.occ.values()
                        )
                        if not already_erasing:
                            world.food_trail.clear_tile(self.col, self.row)
                            self.erasing_trail = True
                            active_inward = [
                                t
                                for t in candidates
                                if world.food_trail.value(t[0], t[1], now) >= 0.5
                                and world.nest_nav.dist[t[1]][t[0]] < here_dist
                                and not has_active_branch_outward(world, t, now)
                            ]
                            if active_inward and here_dist > 6:
                                next_tile = min(
                                    active_inward,
                                    key=lambda t: world.nest_nav.dist[t[1]][t[0]],
                                )
                                world.move(self, next_tile)
                                world.food_trail.clear_tile(self.col, self.row)
                                return
                        else:
                            self.erasing_trail = False
                            self.role = "forager"
                            self.waypoint = None
                elif active_outward_trail or (not is_llm and active_trail_tiles):
                    pool = active_outward_trail or active_trail_tiles
                    if len(pool) >= 2:
                        if is_llm:
                            world.move(self, pool[self.id % len(pool)])
                            return
                        # Trail split / fork: follow the less pheromone trail to
                        # balance traffic and prevent losing that food source
                        scents = [(t, food_scent(t)) for t in pool]
                        lowest_scent = min(s for _, s in scents)
                        best_options = [t for t, s in scents if s == lowest_scent]
                        world.move(self, random.choice(best_options))
                        return
                    else:
                        # Single continuous food trail: follow the trail tile
                        world.move(self, pool[0])
                        return
                elif is_llm:
                    self.role = "forager"
                    self.waypoint = None

            if is_foraging and not in_depleted_zone:
                # Approaching a trail from outside: steer toward strongest sensed
                # direction across candidates
                # Only steer towards active trails that are not in a depleted zone
                non_depleted_candidates = [
                    t
                    for t in candidates
                    if not (
                        any(
                            "[DEPLETED]" in text
                            and abs(pos[0] - t[0]) + abs(pos[1] - t[1]) <= 6
                            for pos, text in tokens_now.items()
                        )
                        and not any(
                            amt > 0
                            for f, amt in world.food.items()
                            if abs(f[0] - t[0]) + abs(f[1] - t[1]) <= 6
                        )
                    )
                ]
                if is_llm:
                    non_depleted_candidates = [
                        t
                        for t in non_depleted_candidates
                        if has_active_branch_outward(world, t, now)
                    ]
                if non_depleted_candidates:
                    scents = [(t, food_scent(t)) for t in non_depleted_candidates]
                    best_scent = max(s for _, s in scents)
                    if best_scent >= 0.5:
                        if is_llm:
                            active_candidates = [t for t, s in scents if s >= 0.5]
                            world.move(
                                self,
                                active_candidates[self.id % len(active_candidates)],
                            )
                            return
                        best_options = [t for t, s in scents if s == best_scent]
                        world.move(self, random.choice(best_options))
                        return

            if self.waypoint is not None:
                best = min(
                    candidates,
                    key=lambda t: (
                        abs(t[0] - self.waypoint[0]) + abs(t[1] - self.waypoint[1])
                    ),
                )
                world.move(self, best)
                return

            if (
                self.last_food_tile is not None
                and world.food.get(self.last_food_tile, 0) > 0
            ):
                cd = cardinal(
                    self.last_food_tile[0] - self.col,
                    self.last_food_tile[1] - self.row,
                )
                preferred = (self.col + cd[0], self.row + cd[1])
                if preferred in options:
                    world.move(self, preferred)
                    return

            # Perimeter reflection: if on a boundary wall, step inward into the world
            wall_dist_here = min(
                self.col, GRID_SIZE - 1 - self.col, self.row, GRID_SIZE - 1 - self.row
            )
            if wall_dist_here == 0:
                inward_moves = [
                    t
                    for t in candidates
                    if min(t[0], GRID_SIZE - 1 - t[0], t[1], GRID_SIZE - 1 - t[1]) > 0
                ]
                if inward_moves:
                    candidates = inward_moves

            # Head where the exploration trail is coldest (least pheromone) within
            # sensing range.
            def heat(t: Tile) -> float:
                dc, dr = t[0] - self.col, t[1] - self.row
                corridor = sum(
                    (
                        trail.value(self.col + k * dc, self.row + k * dr, now)
                        if 0 <= self.col + k * dc < GRID_SIZE
                        and 0 <= self.row + k * dr < GRID_SIZE
                        else 40.0
                    )
                    * (4 - k)
                    for k in (1, 2, 3)
                )
                cone = CONES[(dc, dr)]
                cone_sum = sum(
                    trail.value(self.col + dx, self.row + dy, now)
                    if 0 <= self.col + dx < GRID_SIZE and 0 <= self.row + dy < GRID_SIZE
                    else 40.0
                    for dx, dy in cone
                )
                mean = cone_sum / len(cone)
                h = corridor + mean + random.random() * WALK_NOISE
                if self.prev_tile is not None and (dc, dr) == (
                    self.col - self.prev_tile[0],
                    self.row - self.prev_tile[1],
                ):
                    h -= 15.0
                return h

            unvisited = [t for t in candidates if trail.value(t[0], t[1], now) < 0.5]
            pool = unvisited if unvisited else candidates
            world.move(self, min(pool, key=heat))
        elif self.released:  # boxed in on a station: push someone aside
            claim = self.priority(nav)
            for dc, dr in random.sample(DIRECTIONS, 4):
                t = (self.col + dc, self.row + dr)
                b = world.occ.get(t)
                if b and b.pushable(nav, claim) and world.push(b, claim, here):
                    world.move(self, t)
                    break


TOKEN_LINE_RE = re.compile(r"^\[([0-9.]+)(?:s)?\]\s*(.*)$")


def format_token_line(text: str, now: float) -> str:
    """Format a line with a timestamp header if not already formatted."""
    m = TOKEN_LINE_RE.match(text.strip())
    if m:
        return text.strip()
    return f"[{now:.1f}s] {text.strip()}"


def parse_token_line(line: str) -> tuple[float | None, str]:
    """Extract (timestamp, content) from a timestamped blackboard line."""
    m = TOKEN_LINE_RE.match(line.strip())
    if m:
        try:
            return float(m.group(1)), m.group(2)
        except ValueError:
            pass
    return None, line.strip()


def prune_token_lines(
    token_text: str,
    now: float,
    max_age: float = TOKEN_TTL,
    fallback_t: float | None = None,
) -> str:
    """Filter out lines older than max_age from a multi-line blackboard text."""
    fresh_lines: list[str] = []
    for line in token_text.splitlines():
        line = line.strip()
        if not line:
            continue
        ts, _ = parse_token_line(line)
        effective_t = ts if ts is not None else fallback_t
        if effective_t is not None and (now - effective_t) > max_age:
            continue
        fresh_lines.append(line)
    return "\n".join(fresh_lines)


def update_token_text(
    existing_text: str | None,
    new_line: str,
    now: float,
    max_age: float = TOKEN_TTL,
    fallback_t: float | None = None,
) -> str:
    """Merge a new line into existing blackboard text, pruning stale lines and
    overriding obsolete guides."""
    fresh_lines: list[str] = []
    target_coord: str | None = None
    if "[DEPLETED]" in new_line:
        coord_m = re.search(r"\(\s*\d+\s*,\s*\d+\s*\)", new_line)
        if coord_m:
            target_coord = coord_m.group(0).replace(" ", "")

    if existing_text:
        for line in existing_text.splitlines():
            line = line.strip()
            if not line:
                continue
            ts, content = parse_token_line(line)
            effective_t = ts if ts is not None else fallback_t
            if effective_t is not None and (now - effective_t) > max_age:
                continue
            # If new line marks target as depleted, override previous guides to that
            # target
            if target_coord:
                norm_content = content.replace(" ", "")
                if target_coord in norm_content and (
                    "[GUIDE]" in content or "[FOOD]" in content
                ):
                    continue
            fresh_lines.append(line)

    if new_line.strip():
        new_formatted = format_token_line(new_line, now)
        if new_formatted not in fresh_lines:
            fresh_lines.append(new_formatted)
    return "\n".join(fresh_lines)


class World:
    """Occupancy plus the shared tile state agents coordinate through.

    `pressure` is a per-tile count of seekers within PUSH_REACH tiles, i.e.
    what an agent could read off the tiles around it. Nothing here lets an
    agent query another agent directly.
    """

    def __init__(
        self,
        nav: NavMap,
        agents: list[Agent],
        nest_nav: NavMap | None = None,
        nest_set: set[Tile] | None = None,
        food: dict[Tile, int] | None = None,
        food_trail: Trail | None = None,
        static_obstacles: set[Tile] | None = None,
        moving_barriers: list[MovingBarrier] | None = None,
        stations: list[Tile] | None = None,
    ) -> None:
        self.nav = nav
        self.nest_nav = nest_nav if nest_nav is not None else nav
        self.nest_set = nest_set if nest_set is not None else set()
        self.food: dict[Tile, int] = food if food is not None else {}
        self.food_collected: int = 0
        self.food_trail: Trail = (
            food_trail if food_trail is not None else Trail(accumulate=True)
        )
        self.static_obstacles: set[Tile] = (
            static_obstacles if static_obstacles is not None else set()
        )
        self.moving_barriers: list[MovingBarrier] = (
            moving_barriers if moving_barriers is not None else []
        )
        self.stations: list[Tile] = (
            stations
            if stations is not None
            else [
                (c, r)
                for r in range(GRID_SIZE)
                for c in range(GRID_SIZE)
                if self.nav.dist[r][c] == 0
            ]
        )
        self.dynamic_hazards: dict[Tile, float] = {}
        self.tokens: dict[Tile, tuple[str, float]] = {}
        self.next_hazard_time: float = random.uniform(1.0, 3.0)
        self.next_barrier_time: float = random.uniform(
            BARRIER_SPAWN_INTERVAL_MIN, BARRIER_SPAWN_INTERVAL_MAX
        )
        self.occ: dict[Tile, Agent] = {(a.col, a.row): a for a in agents}
        self.pressure = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

    @property
    def barrier_coords(self) -> set[Tile]:
        return {c for b in self.moving_barriers for c in b.coords}

    @property
    def obstacles(self) -> set[Tile]:
        obs = set(self.static_obstacles)
        obs.update(self.dynamic_hazards.keys())
        obs.update(self.barrier_coords)
        return obs

    def free(self, tile: Tile) -> bool:
        return (
            0 <= tile[0] < GRID_SIZE
            and 0 <= tile[1] < GRID_SIZE
            and tile not in self.occ
            and tile not in self.obstacles
        )

    def write_token(self, tile: Tile, text: str, now: float) -> None:
        self.tokens[tile] = (text, now)

    def append_token(
        self, tile: Tile, line: str, now: float, max_age: float = TOKEN_TTL
    ) -> None:
        existing = self.tokens.get(tile)
        existing_text, t = existing if existing else (None, now)
        merged = update_token_text(existing_text, line, now, max_age, fallback_t=t)
        if merged.strip():
            self.tokens[tile] = (merged, now)
        elif tile in self.tokens:
            del self.tokens[tile]

    def read_token(self, tile: Tile, now: float, ttl: float = TOKEN_TTL) -> str | None:
        if tile in self.tokens:
            text, t = self.tokens[tile]
            pruned = prune_token_lines(text, now, ttl, fallback_t=t)
            if pruned.strip():
                self.tokens[tile] = (pruned, now)
                return pruned
            del self.tokens[tile]
        return None

    def active_tokens(self, now: float, ttl: float = TOKEN_TTL) -> dict[Tile, str]:
        active: dict[Tile, str] = {}
        to_delete: list[Tile] = []
        for tile, (text, t) in list(self.tokens.items()):
            pruned = prune_token_lines(text, now, ttl, fallback_t=t)
            if pruned.strip():
                self.tokens[tile] = (pruned, now)
                active[tile] = pruned
            else:
                to_delete.append(tile)
        for tile in to_delete:
            del self.tokens[tile]
        return active

    def spawn_hazard(
        self,
        now: float,
        candidate: Tile | None = None,
        shape: tuple[Tile, ...] | None = None,
        lifetime: float | None = None,
    ) -> list[Tile] | None:
        if candidate is not None:
            center = candidate
            selected_shape = shape if shape is not None else ((0, 0),)
        else:
            options = [
                (c, r)
                for r in range(10, GRID_SIZE - 10)
                for c in range(10, GRID_SIZE - 10)
                if abs(c - 50) + abs(r - 50) >= 8
                and (c, r) not in self.occ
                and (c, r) not in self.obstacles
                and (c, r) not in self.nest_set
                and (c, r) not in self.food
            ]
            if not options:
                return None
            center = random.choice(options)
            selected_shape = (
                shape if shape is not None else random.choice(HAZARD_SHAPES)
            )

        cluster: list[Tile] = []
        for dx, dy in selected_shape:
            t = (center[0] + dx, center[1] + dy)
            if (
                0 <= t[0] < GRID_SIZE
                and 0 <= t[1] < GRID_SIZE
                and t not in self.occ
                and t not in self.obstacles
                and t not in self.nest_set
                and t not in self.food
                and t not in self.static_obstacles
            ):
                cluster.append(t)

        if not cluster:
            return None

        dur = (
            lifetime
            if lifetime is not None
            else random.uniform(HAZARD_LIFETIME_MIN, HAZARD_LIFETIME_MAX)
        )
        exp = now + dur
        for t in cluster:
            self.dynamic_hazards[t] = exp
            self.write_token(t, "HAZARD: Temporary debris", now)
        return cluster

    def _step_barrier(self, b: MovingBarrier) -> None:
        dc, dr = (b.direction, 0) if b.axis == "x" else (0, b.direction)
        next_coords = [(c + dc, r + dr) for c, r in b.coords]
        other_coords = {
            coord
            for other in self.moving_barriers
            if other is not b
            for coord in other.coords
        }
        blocked = False
        for c, r in next_coords:
            pos = c if b.axis == "x" else r
            if not (b.min_pos <= pos <= b.max_pos):
                blocked = True
                break
            if (
                (c, r) in self.occ
                or (c, r) in self.nest_set
                or (c, r) in self.food
                or (c, r) in self.static_obstacles
                or (c, r) in self.dynamic_hazards
                or (c, r) in other_coords
            ):
                blocked = True
                break
        if blocked:
            b.direction = -b.direction
        else:
            b.coords = next_coords

    def spawn_moving_barrier(
        self,
        now: float,
        candidate: Tile | None = None,
        axis: str | None = None,
        length: int = 3,
        travel: int = 10,
        lifetime: float | None = None,
    ) -> MovingBarrier | None:
        sel_axis = axis if axis is not None else random.choice(["x", "y"])
        dur = (
            lifetime
            if lifetime is not None
            else random.uniform(BARRIER_LIFETIME_MIN, BARRIER_LIFETIME_MAX)
        )
        if candidate is not None:
            c0, r0 = candidate
        else:
            options = [
                (c, r)
                for r in range(12, GRID_SIZE - 12)
                for c in range(12, GRID_SIZE - 12)
                if abs(c - 50) + abs(r - 50) >= 10
                and (c, r) not in self.occ
                and (c, r) not in self.obstacles
                and (c, r) not in self.nest_set
                and (c, r) not in self.food
            ]
            if not options:
                return None
            c0, r0 = random.choice(options)

        coords: list[Tile] = []
        for i in range(length):
            tc, tr = (c0, r0 + i) if sel_axis == "x" else (c0 + i, r0)
            if not (0 <= tc < GRID_SIZE and 0 <= tr < GRID_SIZE):
                return None
            if (
                (tc, tr) in self.occ
                or (tc, tr) in self.obstacles
                or (tc, tr) in self.nest_set
                or (tc, tr) in self.food
                or (tc, tr) in self.static_obstacles
            ):
                return None
            coords.append((tc, tr))

        pos = c0 if sel_axis == "x" else r0
        min_pos = max(4, pos - travel)
        max_pos = min(GRID_SIZE - 5, pos + travel)
        barrier = MovingBarrier(
            coords=coords,
            axis=sel_axis,
            direction=random.choice([-1, 1]),
            min_pos=min_pos,
            max_pos=max_pos,
            interval=OBSTACLE_DRIFT_INTERVAL,
            expiry=now + dur,
        )
        self.moving_barriers.append(barrier)
        for t in coords:
            self.write_token(t, "ALERT: Moving barrier active", now)
        return barrier

    def tick_obstacles(self, now: float, dt: float) -> None:
        expired = [tile for tile, exp in self.dynamic_hazards.items() if now >= exp]
        for tile in expired:
            del self.dynamic_hazards[tile]

        if now >= self.next_hazard_time:
            if len(self.dynamic_hazards) < 20:
                self.spawn_hazard(now)
            self.next_hazard_time = now + random.uniform(
                HAZARD_SPAWN_INTERVAL_MIN, HAZARD_SPAWN_INTERVAL_MAX
            )

        self.moving_barriers = [b for b in self.moving_barriers if now < b.expiry]

        if now >= self.next_barrier_time:
            if len(self.moving_barriers) < MAX_MOVING_BARRIERS:
                self.spawn_moving_barrier(now)
            self.next_barrier_time = now + random.uniform(
                BARRIER_SPAWN_INTERVAL_MIN, BARRIER_SPAWN_INTERVAL_MAX
            )

        for b in self.moving_barriers:
            b.timer += dt
            if b.timer >= b.interval:
                b.timer -= b.interval
                self._step_barrier(b)

    def is_station(self, tile: Tile) -> bool:
        return self.nav.dist[tile[1]][tile[0]] == 0 or tile in getattr(
            self, "stations", []
        )

    def is_nest(self, tile: Tile) -> bool:
        return tile in self.nest_set

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
        agent.prev_tile = old
        agent.col, agent.row = tile
        agent.charge -= MOVE_COST
        agent.moved = True
        if (
            self.food_trail.stamp[tile[1]][tile[0]] is not None
            and self.food_trail.levels[tile[1]][tile[0]] >= 0.5
        ):
            agent.avoided_collision_from_trail = False
        elif (
            self.food_trail.stamp[old[1]][old[0]] is not None
            and self.food_trail.levels[old[1]][old[0]] >= 0.5
            and not getattr(agent, "erasing_trail", False)
            and not agent.has_food
        ):
            agent.avoided_collision_from_trail = True
        if agent.released and not self.is_station(tile):
            agent.released = False
        if agent.has_food and self.is_nest(tile):
            agent.has_food = False
            agent.food_trail_valid = True
            self.food_collected += 1
            agent.last_food_mark_pos = None
            agent.waypoint = None
            agent.role = "forager"
            if agent.state == "return_nest":
                agent.state = "walk"
        if agent.state == "seek":
            agent.food_trail_valid = False
            self.press(old, -1)
            if self.is_station(tile):
                agent.state = "charge"
                agent.waypoint = None
            else:
                self.press(tile, 1)
        elif (
            agent.state == "walk" and not agent.has_food and self.food.get(tile, 0) > 0
        ):
            self.food[tile] -= 1
            cluster_empty = False
            if self.food[tile] == 0:
                del self.food[tile]
                cluster_empty = not any(
                    amt > 0
                    for t, amt in self.food.items()
                    if abs(t[0] - tile[0]) + abs(t[1] - tile[1]) <= 6
                )
            agent.has_food = True
            agent.food_trail_valid = not cluster_empty
            agent.last_food_tile = tile if not cluster_empty else None
            agent.state = "return_nest"
            agent.role = "forager"
            agent.last_food_mark_pos = None

    def spawn_random_food(self, seed: int | None = None) -> None:
        count = 1 if seed is not None else random.randint(1, 2)
        new_food = make_food_clusters(
            cluster_count=count, seed=seed, obstacles=self.obstacles
        )
        for tile, amt in new_food.items():
            self.food[tile] = self.food.get(tile, 0) + amt

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
                    if self.is_station(v) or v in self.obstacles:
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


def make_nest() -> list[Tile]:
    """Tiles forming the central depot/nest."""
    return list(NEST_TILES)


def make_static_obstacles() -> set[Tile]:
    """Concave U-shaped barrier trap and corridor chokepoints."""
    obs: set[Tile] = set()
    # 1. U-shaped concave trap (rows 25..35, cols 25..35, open to North)
    for c in range(25, 36):
        obs.add((c, 35))
    for r in range(25, 36):
        obs.add((25, r))
        obs.add((35, r))
    # 2. Corridor chokepoint barrier at row 65 with a 4-tile gap
    for c in range(35, 48):
        obs.add((c, 65))
    for c in range(53, 66):
        obs.add((c, 65))
    return obs


def make_moving_barriers(now: float = 0.0) -> list[MovingBarrier]:
    """Dynamic sliding chokepoint barriers across multiple quadrants."""
    return [
        # Barrier 1: Horizontal sliding 3-tile patrol in opening camera view
        # (cols 6..18, rows 6..8)
        MovingBarrier(
            coords=[(10, 6), (10, 7), (10, 8)],
            axis="x",
            direction=1,
            min_pos=6,
            max_pos=18,
            expiry=now + 30.0,
        ),
        # Barrier 2: Vertical sliding 3-tile gate in the East
        MovingBarrier(
            coords=[(72, 48), (72, 49), (72, 50)],
            axis="y",
            direction=1,
            min_pos=40,
            max_pos=60,
            expiry=now + 40.0,
        ),
        # Barrier 3: Horizontal sliding 3-tile bar in the South
        MovingBarrier(
            coords=[(18, 75), (19, 75), (20, 75)],
            axis="x",
            direction=1,
            min_pos=12,
            max_pos=32,
            expiry=now + 35.0,
        ),
        # Barrier 4: 2x2 sliding block in the North-East
        MovingBarrier(
            coords=[(80, 20), (81, 20), (80, 21), (81, 21)],
            axis="y",
            direction=1,
            min_pos=15,
            max_pos=30,
            expiry=now + 45.0,
        ),
    ]


def make_food_clusters(
    cluster_count: int | None = None,
    food_per_cluster: int | None = None,
    seed: int | None = None,
    obstacles: set[Tile] | None = None,
) -> dict[Tile, int]:
    """Generate food clusters at random positions with random amounts."""
    obs = obstacles or set()
    rng = random.Random(seed) if seed is not None else random.Random()
    count = cluster_count if cluster_count is not None else rng.randint(4, 7)
    food: dict[Tile, int] = {}
    valid_coords = [
        (c, r)
        for r in range(8, GRID_SIZE - 8)
        for c in range(8, GRID_SIZE - 8)
        if abs(c - 50) + abs(r - 50) >= 15 and (c, r) not in obs
    ]
    centers = rng.sample(valid_coords, min(count, len(valid_coords)))

    for cx, cy in centers:
        total_items = (
            food_per_cluster
            if food_per_cluster is not None
            else rng.randint(FOOD_AMOUNT_MIN, FOOD_AMOUNT_MAX)
        )
        patch = [
            (cx + dx, cy + dy)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (cx + dx, cy + dy) not in obs
        ]
        items_each = total_items // len(patch)
        remainder = total_items % len(patch)
        for i, tile in enumerate(patch):
            amt = items_each + (1 if i < remainder else 0)
            if amt > 0:
                food[tile] = food.get(tile, 0) + amt
    return food


def build_nav(stations: list[Tile], obstacles: set[Tile] | None = None) -> NavMap:
    """Shortest-path arrows to the nearest station (multi-source BFS)."""
    obs = obstacles or set()
    dist: dict[Tile, int] = {s: 0 for s in stations}
    src: dict[Tile, Tile] = {s: s for s in stations}
    queue = deque(stations)
    while queue:
        u = queue.popleft()
        for dc, dr in DIRECTIONS:
            v = (u[0] + dc, u[1] + dr)
            if (
                not (0 <= v[0] < GRID_SIZE and 0 <= v[1] < GRID_SIZE)
                or v in dist
                or v in obs
            ):
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
            and src.get((col + dc, row + dr)) == (sx, sy)
        ]
        want = cardinal(sx - col, sy - row)
        arrow[row][col] = want if want in steps else (steps[0] if steps else (0, 0))
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
    world.tick_obstacles(now, dt)
    step_timer += dt
    while step_timer >= STEP_INTERVAL:
        step_timer -= STEP_INTERVAL
        world.tick(agents, trail, now)
    for agent in agents:
        if agent.state == "walk":
            trail.mark(agent.col, agent.row, now)
        if (
            agent.has_food
            and agent.food_trail_valid
            and agent.state not in ("seek", "charge")
        ):
            if (agent.col, agent.row) != getattr(agent, "last_food_mark_pos", None):
                world.food_trail.mark(agent.col, agent.row, now)
                agent.last_food_mark_pos = (agent.col, agent.row)
    for agent in agents:
        if agent.state == "charge":
            agent.charge_up(dt)
    return step_timer
