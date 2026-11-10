import random

import pygame

from apeks.config import (
    AGENT_COUNT,
    BLACK,
    CAM_SPEED,
    GRID_SIZE,
    MARK_VALUE,
    MAX_CHARGE,
    VALUE_COLOR,
    WHITE,
    WORLD_SIZE,
)
from apeks.render import draw_agents, draw_grid, draw_minimap
from apeks.sim import Agent, Trail, World, advance, build_nav, make_stations


def move_camera(
    cam_x: float,
    cam_y: float,
    axis_x: int,
    axis_y: int,
    dt: float,
    view: tuple[int, int],
) -> tuple[float, float]:
    """Pan by the held-key axes (-1, 0, 1), clamped to the world."""
    cam_x += axis_x * CAM_SPEED * dt
    cam_y += axis_y * CAM_SPEED * dt
    return (
        max(0, min(cam_x, WORLD_SIZE - view[0])),
        max(0, min(cam_y, WORLD_SIZE - view[1])),
    )


def main() -> None:  # pragma: no cover - opens a window and runs the event loop
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("apeks")
    clock = pygame.time.Clock()
    cam_x = cam_y = 0.0

    font = pygame.font.Font(None, 16)
    # Unique random start tiles: one agent per tile.
    starts = random.sample(range(GRID_SIZE * GRID_SIZE), AGENT_COUNT)
    agents = [Agent(i, t % GRID_SIZE, t // GRID_SIZE) for i, t in enumerate(starts)]
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
    world = World(nav, agents)
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
        step_timer = advance(world, agents, trail, now, dt, step_timer)

        keys = pygame.key.get_pressed()
        w, h = screen.get_size()
        cam_x, cam_y = move_camera(
            cam_x,
            cam_y,
            (keys[pygame.K_RIGHT] or keys[pygame.K_d])
            - (keys[pygame.K_LEFT] or keys[pygame.K_a]),
            (keys[pygame.K_DOWN] or keys[pygame.K_s])
            - (keys[pygame.K_UP] or keys[pygame.K_w]),
            dt,
            (w, h),
        )

        screen.fill(BLACK)
        draw_grid(
            screen,
            cam_x,
            cam_y,
            nav,
            station_set,
            tile_font,
            text_cache,
            trail,
            now,
            value_labels,
        )
        draw_agents(screen, agents, labels, charge_labels, cam_x, cam_y)
        draw_minimap(screen, cam_x, cam_y, agents, stations)
        pygame.display.flip()

    pygame.quit()
