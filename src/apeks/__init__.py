import math
import random

import pygame

GRID_SIZE = 100  # tiles per side
TILE_SIZE = 50  # px per tile
WORLD_SIZE = GRID_SIZE * TILE_SIZE
CAM_SPEED = 1000  # px/s

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
AGENT_COUNT = 100
SPEED = 5.0  # tiles/s, constant
MASS = 1.0  # kg
STEP_INTERVAL = 1 / SPEED  # s per one-tile move
MOVE_COST = round(MASS * SPEED**2)  # U per tile (m*v^2)
MAX_CHARGE = 10000  # U
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
AGENT_COLOR = (220, 60, 60)
CHARGE_RATE = 1000  # U/s while on a station
LOW_CHARGE = MAX_CHARGE * 0.25  # below this the agent heads for a station
SEEK_RING = (255, 200, 0)
CHARGE_RING = (60, 220, 90)

STATION_COUNT = 20
STATION_COLOR = (190, 235, 200)
STATION_DOT = (120, 255, 160)
ARROW_COLOR = (30, 80, 160)
TEXT_COLOR = (40, 40, 40)

BP_BG = (20, 60, 120)  # blueprint blue
BP_LINE = (200, 225, 255)  # pale chalk-blue lines
MINIMAP_SIZE = 100  # px


def cardinal(dx: int, dy: int) -> tuple[int, int]:
    """Unit step along the axis with the longer remaining distance."""
    if abs(dx) >= abs(dy):
        return (1 if dx > 0 else -1), 0
    return 0, (1 if dy > 0 else -1)


class Agent:
    def __init__(self, ident: int, col: int, row: int) -> None:
        self.id = ident
        self.col = col
        self.row = row
        self.charge: float = MAX_CHARGE  # U
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

    def step(
        self,
        occupied: set[tuple[int, int]],
        nav: list[list[tuple[int, int]]],
    ) -> None:
        """One tile move per step; a completed move costs MOVE_COST charge.

        Walking is a random cardinal walk. Below LOW_CHARGE the agent seeks:
        it follows the arrow on its tile to the nearest station and charges
        there. A move that leaves the grid or hits a taken tile is skipped
        (and free); a seeking agent just retries on the next step.
        """
        if self.state == "charge" or self.charge < MOVE_COST:
            return
        if self.state == "walk" and self.charge < LOW_CHARGE:
            self.state = "seek"

        if self.state == "seek":
            dx, dy = nav[self.row][self.col]
            if dx == 0 and dy == 0:
                self.state = "charge"
                return
            dc, dr = cardinal(dx, dy)
        else:
            dc, dr = random.choice(DIRECTIONS)

        col, row = self.col + dc, self.row + dr
        if not (0 <= col < GRID_SIZE and 0 <= row < GRID_SIZE):
            return
        if (col, row) in occupied:
            return
        occupied.remove((self.col, self.row))
        occupied.add((col, row))
        self.col, self.row = col, row
        self.charge -= MOVE_COST
        if self.state == "seek" and nav[row][col] == (0, 0):
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


def make_stations() -> list[tuple[int, int]]:
    """STATION_COUNT charging points, equally spaced along the wall tiles."""
    last = GRID_SIZE - 1
    ring = (
        [(c, 0) for c in range(GRID_SIZE)]
        + [(last, r) for r in range(1, GRID_SIZE)]
        + [(c, last) for c in range(last - 1, -1, -1)]
        + [(0, r) for r in range(last - 1, 0, -1)]
    )
    return [ring[round(i * len(ring) / STATION_COUNT)] for i in range(STATION_COUNT)]


def build_nav(
    stations: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    """nav[row][col] = (dx, dy) to the nearest station (Manhattan distance)."""
    nav = []
    for row in range(GRID_SIZE):
        line = []
        for col in range(GRID_SIZE):
            sx, sy = min(stations, key=lambda s: abs(s[0] - col) + abs(s[1] - row))
            line.append((sx - col, sy - row))
        nav.append(line)
    return nav


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
    nav: list[list[tuple[int, int]]],
    station_set: set[tuple[int, int]],
    font: pygame.font.Font,
    text_cache: dict[tuple[int, int], pygame.Surface],
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
            pygame.draw.rect(screen, STATION_COLOR if is_station else WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            dx, dy = nav[row][col]
            if dx or dy:
                # Cardinal arrow along the axis with the longer remaining distance.
                ax, ay = cardinal(dx, dy)
                draw_arrow(screen, rect.centerx, rect.top + 18, ax, ay)
            text = text_cache.get((dx, dy))
            if text is None:
                text = text_cache[(dx, dy)] = font.render(
                    f"({dx},{dy})", True, TEXT_COLOR
                )
            screen.blit(text, text.get_rect(midbottom=(rect.centerx, rect.bottom - 3)))


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

        step_timer += dt
        while step_timer >= STEP_INTERVAL:
            step_timer -= STEP_INTERVAL
            random.shuffle(agents)  # no agent gets move priority
            for agent in agents:
                agent.step(occupied, nav)
            agents.sort(key=lambda a: a.id)

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
        draw_grid(screen, cam_x, cam_y, nav, station_set, tile_font, text_cache)
        draw_agents(screen, agents, labels, charge_labels, cam_x, cam_y)
        draw_minimap(screen, cam_x, cam_y, agents, stations)
        pygame.display.flip()

    pygame.quit()
