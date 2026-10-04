import math

import pygame

from apeks.config import (
    ARROW_COLOR,
    BLACK,
    BP_BG,
    BP_LINE,
    CHARGE_RING,
    GRID_SIZE,
    MARK_TINT,
    MARK_VALUE,
    MINIMAP_SIZE,
    SEEK_RING,
    STATION_COLOR,
    STATION_DOT,
    TEXT_COLOR,
    TILE_SIZE,
    WHITE,
    WORLD_SIZE,
)
from apeks.sim import Agent, NavMap, Trail


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
            (
                sx * MINIMAP_SIZE // GRID_SIZE - 1,
                sy * MINIMAP_SIZE // GRID_SIZE - 1,
                3,
                3,
            ),
        )
    for agent in agents:
        screen.set_at(
            (
                agent.col * MINIMAP_SIZE // GRID_SIZE,
                agent.row * MINIMAP_SIZE // GRID_SIZE,
            ),
            WHITE,
        )
    pygame.draw.rect(screen, BP_LINE, view, 1)
    pygame.draw.rect(screen, BP_LINE, (0, 0, MINIMAP_SIZE, MINIMAP_SIZE), 1)
