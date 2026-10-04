import math
import random
from collections import deque
from dataclasses import dataclass

import pygame

GRID_SIZE = 100  # tiles per side
TILE_SIZE = 50  # px per tile
WORLD_SIZE = GRID_SIZE * TILE_SIZE
CAM_SPEED = 1000  # px/s

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
AGENT_COUNT = 100
SPEED = 8.0  # tiles/s, constant
MASS = 1.0  # kg
STEP_INTERVAL = 1 / SPEED  # s per one-tile move
MOVE_COST = round(MASS * SPEED**2)  # U per tile (m*v^2)
MAX_CHARGE = 100000  # U
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
AGENT_COLOR = (220, 60, 60)
CHARGE_RATE = 10000  # U/s while on a station
LOW_CHARGE = MAX_CHARGE * 0.25  # below this the agent heads for a station
MARK_VALUE = 100  # value an agent writes to the tile it stands on
MARK_DECAY = 0.9  # per-second multiplier on written values
MARK_FLOOR = 0.1  # relative walk weight of a tile at full value (lowest bias)
MARK_TINT = (255, 200, 120)  # tile tint at full value
SEEK_RING = (255, 200, 0)
CHARGE_RING = (60, 220, 90)

STATION_COUNT = 20
STATION_COLOR = (190, 235, 200)
LANE_COLOR = (235, 245, 238)  # wall tiles beside a station: exit-only
STATION_DOT = (120, 255, 160)
ARROW_COLOR = (30, 80, 160)
TEXT_COLOR = (40, 40, 40)
VALUE_COLOR = (170, 90, 20)

BP_BG = (20, 60, 120)  # blueprint blue
BP_LINE = (200, 225, 255)  # pale chalk-blue lines
MINIMAP_SIZE = 100  # px


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


class Agent:
    def __init__(self, ident: int, col: int, row: int) -> None:
        self.id = ident
        self.col = col
        self.row = row
        self.charge: float = 6000  # U
        self.state = "walk"  # walk | seek | charge

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

    def _can_move(
        self, dc: int, dr: int, occupied: set[Tile], nav: NavMap
    ) -> bool:
        col, row = self.col + dc, self.row + dr
        return (
            0 <= col < GRID_SIZE
            and 0 <= row < GRID_SIZE
            and (col, row) not in occupied
            and ((self.col, self.row), (col, row)) not in nav.blocked
        )

    def step(
        self,
        occupied: set[Tile],
        nav: NavMap,
        trail: Trail,
        now: float,
    ) -> None:
        """One tile move per step; a completed move costs MOVE_COST charge.

        Walking is a random cardinal walk over the legal neighbours, weighted
        toward tiles with a lower trail value. Below LOW_CHARGE the agent
        seeks: it follows the arrow on its tile to the nearest station and
        charges there. A blocked seeker just retries on the next step.
        """
        if self.state == "charge" or self.charge < MOVE_COST:
            return
        if self.state == "walk" and self.charge < LOW_CHARGE:
            self.state = "seek"

        if self.state == "seek":
            if nav.offset[self.row][self.col] == (0, 0):
                self.state = "charge"
                return
            dc, dr = nav.arrow[self.row][self.col]
            if not self._can_move(dc, dr, occupied, nav):
                return
        else:
            options = [
                d for d in DIRECTIONS if self._can_move(d[0], d[1], occupied, nav)
            ]
            if not options:
                return
            weights = [
                1 - (1 - MARK_FLOOR)
                * trail.value(self.col + dc, self.row + dr, now) / MARK_VALUE
                for dc, dr in options
            ]
            dc, dr = random.choices(options, weights)[0]

        col, row = self.col + dc, self.row + dr
        occupied.remove((self.col, self.row))
        occupied.add((col, row))
        self.col, self.row = col, row
        self.charge -= MOVE_COST
        if self.state == "seek" and nav.offset[row][col] == (0, 0):
            self.state = "charge"


def draw_agents(
    screen: pygame.Surface,
    agents: list[Agent],
    labels: list[pygame.Surface],
    charge_labels: list[pygame.Surface],
    cam_x: float,
    cam_y: float,
) -> None:
    w, h = screen.get_size()
    radius = TILE_SIZE // 2 - 2
    for agent in agents:
        cx = agent.col * TILE_SIZE + TILE_SIZE // 2 - int(cam_x)
        cy = agent.row * TILE_SIZE + TILE_SIZE // 2 - int(cam_y)
        if cx + radius < 0 or cy + radius < 0 or cx - radius > w or cy - radius > h:
            continue
        pygame.draw.circle(screen, agent.color, (cx, cy), radius)
        if agent.state != "walk":
            ring = SEEK_RING if agent.state == "seek" else CHARGE_RING
            pygame.draw.circle(screen, ring, (cx, cy), radius, 2)
        label = labels[agent.id]
        screen.blit(label, label.get_rect(midbottom=(cx, cy + 1)))
        charge = charge_labels[int(agent.charge)]
        screen.blit(charge, charge.get_rect(midtop=(cx, cy + 1)))


@dataclass
class NavMap:
    offset: list[list[Tile]]  # (dx, dy) to the target station
    arrow: list[list[Tile]]  # cardinal unit step an agent should take
    blocked: set[tuple[Tile, Tile]]  # forbidden (from_tile, to_tile) moves
    lanes: set[Tile]  # exit-only wall tiles beside stations


def make_stations() -> list[Tile]:
    """STATION_COUNT charging points, equally spaced along the walls.

    Offset by half a spacing so none lands on a corner (a corner tile has
    no inward neighbour to enter from).
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


def inward(station: Tile) -> Tile:
    """Direction from a wall station into the room."""
    x, y = station
    if y == 0:
        return 0, 1
    if y == GRID_SIZE - 1:
        return 0, -1
    return (1, 0) if x == 0 else (-1, 0)


def build_nav(stations: list[Tile]) -> NavMap:
    """Entry only from the room side; the two wall tiles beside a station
    are exit-only lanes (nobody may step onto them except from the station,
    and nobody may step from them onto the station). Arrows follow shortest
    paths that respect those rules.
    """
    blocked: set[tuple[Tile, Tile]] = set()
    lanes: set[Tile] = set()
    for sx, sy in stations:
        nx, ny = inward((sx, sy))
        wx, wy = abs(ny), abs(nx)  # along the wall
        for lane in ((sx + wx, sy + wy), (sx - wx, sy - wy)):
            lanes.add(lane)
            blocked.add((lane, (sx, sy)))
            for dc, dr in DIRECTIONS:
                v = (lane[0] + dc, lane[1] + dr)
                if v != (sx, sy):
                    blocked.add((v, lane))

    # Multi-source BFS backwards over allowed moves: shortest legal path.
    dist: dict[Tile, int] = {s: 0 for s in stations}
    src: dict[Tile, Tile] = {s: s for s in stations}
    queue = deque(stations)
    while queue:
        u = queue.popleft()
        for dc, dr in DIRECTIONS:
            v = (u[0] + dc, u[1] + dr)
            if not (0 <= v[0] < GRID_SIZE and 0 <= v[1] < GRID_SIZE):
                continue
            if v in dist or (v, u) in blocked:
                continue
            dist[v] = dist[u] + 1
            src[v] = src[u]
            queue.append(v)

    offset = [[(0, 0)] * GRID_SIZE for _ in range(GRID_SIZE)]
    arrow = [[(0, 0)] * GRID_SIZE for _ in range(GRID_SIZE)]
    for (col, row), d in dist.items():
        sx, sy = src[(col, row)]
        offset[row][col] = (sx - col, sy - row)
        if d == 0:
            continue
        # Legal steps that are one closer to the same station.
        steps = [
            (dc, dr)
            for dc, dr in DIRECTIONS
            if dist.get((col + dc, row + dr)) == d - 1
            and src[(col + dc, row + dr)] == (sx, sy)
            and ((col, row), (col + dc, row + dr)) not in blocked
        ]
        want = cardinal(sx - col, sy - row)
        arrow[row][col] = want if want in steps else steps[0]
    return NavMap(offset, arrow, blocked, lanes)


def draw_arrow(screen: pygame.Surface, cx: int, cy: int, dx: int, dy: int) -> None:
    """Arrow centred on (cx, cy) pointing along (dx, dy)."""
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    half = 10
    tail = (cx - ux * half, cy - uy * half)
    tip = (cx + ux * half, cy + uy * half)
    pygame.draw.line(screen, ARROW_COLOR, tail, tip, 2)
    px, py = -uy, ux  # perpendicular
    head = [
        tip,
        (tip[0] - ux * 7 + px * 4, tip[1] - uy * 7 + py * 4),
        (tip[0] - ux * 7 - px * 4, tip[1] - uy * 7 - py * 4),
    ]
    pygame.draw.polygon(screen, ARROW_COLOR, head)


def draw_grid(
    screen: pygame.Surface,
    cam_x: float,
    cam_y: float,
    nav: NavMap,
    station_set: set[tuple[int, int]],
    font: pygame.font.Font,
    text_cache: dict[tuple[int, int], pygame.Surface],
    trail: Trail,
    now: float,
    value_labels: list[pygame.Surface],
) -> None:
    w, h = screen.get_size()
    # Only tiles intersecting the viewport.
    first_col = max(0, int(cam_x) // TILE_SIZE)
    first_row = max(0, int(cam_y) // TILE_SIZE)
    last_col = min(GRID_SIZE - 1, (int(cam_x) + w) // TILE_SIZE)
    last_row = min(GRID_SIZE - 1, (int(cam_y) + h) // TILE_SIZE)

    for row in range(first_row, last_row + 1):
        for col in range(first_col, last_col + 1):
            rect = pygame.Rect(
                col * TILE_SIZE - int(cam_x),
                row * TILE_SIZE - int(cam_y),
                TILE_SIZE,
                TILE_SIZE,
            )
            is_station = (col, row) in station_set
            value = trail.value(col, row, now)
            if is_station:
                fill = STATION_COLOR
            elif (col, row) in nav.lanes:
                fill = LANE_COLOR
            else:  # white, tinted toward MARK_TINT by the tile's value
                k = value / MARK_VALUE
                fill = tuple(round(255 + (t - 255) * k) for t in MARK_TINT)
            pygame.draw.rect(screen, fill, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            dx, dy = nav.offset[row][col]
            if dx or dy:
                ax, ay = nav.arrow[row][col]
                draw_arrow(screen, rect.centerx, rect.top + 18, ax, ay)
            text = text_cache.get((dx, dy))
            if text is None:
                text = text_cache[(dx, dy)] = font.render(
                    f"({dx},{dy})", True, TEXT_COLOR
                )
            screen.blit(text, text.get_rect(midbottom=(rect.centerx, rect.bottom - 3)))
            screen.blit(value_labels[round(value)], (rect.left + 3, rect.top + 2))


def draw_minimap(
    screen: pygame.Surface,
    cam_x: float,
    cam_y: float,
    agents: list[Agent],
    stations: list[tuple[int, int]],
) -> None:
    scale = MINIMAP_SIZE / WORLD_SIZE
    w, h = screen.get_size()
    pygame.draw.rect(screen, BP_BG, (0, 0, MINIMAP_SIZE, MINIMAP_SIZE))
    view = pygame.Rect(
        round(cam_x * scale),
        round(cam_y * scale),
        max(1, round(min(w, WORLD_SIZE) * scale)),
        max(1, round(min(h, WORLD_SIZE) * scale)),
    )
    for sx, sy in stations:
        pygame.draw.rect(
            screen,
            STATION_DOT,
            (sx * MINIMAP_SIZE // GRID_SIZE - 1, sy * MINIMAP_SIZE // GRID_SIZE - 1, 3, 3),
        )
    for agent in agents:
        screen.set_at((agent.col * MINIMAP_SIZE // GRID_SIZE,
                       agent.row * MINIMAP_SIZE // GRID_SIZE), WHITE)
    pygame.draw.rect(screen, BP_LINE, view, 1)
    pygame.draw.rect(screen, BP_LINE, (0, 0, MINIMAP_SIZE, MINIMAP_SIZE), 1)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("apeks")
    clock = pygame.time.Clock()
    cam_x = cam_y = 0.0

    font = pygame.font.Font(None, 16)
    # Unique random start tiles: one agent per tile.
    starts = random.sample(range(GRID_SIZE * GRID_SIZE), AGENT_COUNT)
    agents = [Agent(i, t % GRID_SIZE, t // GRID_SIZE) for i, t in enumerate(starts)]
    occupied = {(a.col, a.row) for a in agents}
    labels = [font.render(f"#{i:03d}", True, WHITE) for i in range(AGENT_COUNT)]
    charge_labels = [font.render(f"{c}U", True, WHITE) for c in range(MAX_CHARGE + 1)]
    step_timer = 0.0
    now = 0.0
    trail = Trail()
    value_font = pygame.font.Font(None, 14)
    value_labels = [
        value_font.render(str(v), True, VALUE_COLOR) for v in range(MARK_VALUE + 1)
    ]

    stations = make_stations()
    station_set = set(stations)
    nav = build_nav(stations)
    tile_font = pygame.font.Font(None, 15)
    text_cache: dict[tuple[int, int], pygame.Surface] = {}

    running = True
    while running:
        dt = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                cam_x -= event.rel[0]
                cam_y -= event.rel[1]

        now += dt
        step_timer += dt
        while step_timer >= STEP_INTERVAL:
            step_timer -= STEP_INTERVAL
            random.shuffle(agents)  # no agent gets move priority
            for agent in agents:
                agent.step(occupied, nav, trail, now)
            agents.sort(key=lambda a: a.id)
        for agent in agents:
            trail.mark(agent.col, agent.row, now)

        for agent in agents:
            if agent.state == "charge":
                agent.charge_up(dt)

        keys = pygame.key.get_pressed()
        cam_x += (keys[pygame.K_RIGHT] or keys[pygame.K_d]) * CAM_SPEED * dt
        cam_x -= (keys[pygame.K_LEFT] or keys[pygame.K_a]) * CAM_SPEED * dt
        cam_y += (keys[pygame.K_DOWN] or keys[pygame.K_s]) * CAM_SPEED * dt
        cam_y -= (keys[pygame.K_UP] or keys[pygame.K_w]) * CAM_SPEED * dt

        w, h = screen.get_size()
        cam_x = max(0, min(cam_x, WORLD_SIZE - w))
        cam_y = max(0, min(cam_y, WORLD_SIZE - h))

        screen.fill(BLACK)
        draw_grid(
            screen, cam_x, cam_y, nav, station_set, tile_font, text_cache,
            trail, now, value_labels,
        )
        draw_agents(screen, agents, labels, charge_labels, cam_x, cam_y)
        draw_minimap(screen, cam_x, cam_y, agents, stations)
        pygame.display.flip()

    pygame.quit()
