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

BP_BG = (20, 60, 120)  # blueprint blue
BP_LINE = (200, 225, 255)  # pale chalk-blue lines
MINIMAP_SIZE = 100  # px


class Agent:
    def __init__(self, ident: int, col: int, row: int) -> None:
        self.id = ident
        self.col = col
        self.row = row
        self.charge = MAX_CHARGE  # U

    @property
    def color(self) -> tuple[int, int, int]:
        """AGENT_COLOR darkened in proportion to the remaining charge."""
        f = self.charge / MAX_CHARGE
        return tuple(round(c * f) for c in AGENT_COLOR)

    def step(self, occupied: set[tuple[int, int]]) -> None:
        """Move one tile in a random cardinal direction (Manhattan walk).

        A completed move costs MOVE_COST charge; at 0 the agent stops. The
        move is skipped (and free) if it leaves the grid or the tile is taken.
        """
        if self.charge < MOVE_COST:
            return
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
        label = labels[agent.id]
        screen.blit(label, label.get_rect(midbottom=(cx, cy + 1)))
        charge = charge_labels[agent.charge]
        screen.blit(charge, charge.get_rect(midtop=(cx, cy + 1)))


def draw_grid(screen: pygame.Surface, cam_x: float, cam_y: float) -> None:
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
            pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)


def draw_minimap(
    screen: pygame.Surface, cam_x: float, cam_y: float, agents: list[Agent]
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
                agent.step(occupied)
            agents.sort(key=lambda a: a.id)

        keys = pygame.key.get_pressed()
        cam_x += (keys[pygame.K_RIGHT] or keys[pygame.K_d]) * CAM_SPEED * dt
        cam_x -= (keys[pygame.K_LEFT] or keys[pygame.K_a]) * CAM_SPEED * dt
        cam_y += (keys[pygame.K_DOWN] or keys[pygame.K_s]) * CAM_SPEED * dt
        cam_y -= (keys[pygame.K_UP] or keys[pygame.K_w]) * CAM_SPEED * dt

        w, h = screen.get_size()
        cam_x = max(0, min(cam_x, WORLD_SIZE - w))
        cam_y = max(0, min(cam_y, WORLD_SIZE - h))

        screen.fill(BLACK)
        draw_grid(screen, cam_x, cam_y)
        draw_agents(screen, agents, labels, charge_labels, cam_x, cam_y)
        draw_minimap(screen, cam_x, cam_y, agents)
        pygame.display.flip()

    pygame.quit()
