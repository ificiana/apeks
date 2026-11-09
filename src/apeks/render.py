import math
from typing import Any

import pygame

from apeks.config import (
    ARROW_COLOR,
    BARRIER_COLOR,
    BARRIER_DOT,
    BLACK,
    BP_BG,
    BP_LINE,
    CHARGE_RING,
    FOOD_COLOR,
    FOOD_DOT,
    FOOD_RING,
    FOOD_TINT,
    GRID_SIZE,
    HAZARD_COLOR,
    HAZARD_DOT,
    MARK_TINT,
    MARK_VALUE,
    MINIMAP_SIZE,
    NEST_COLOR,
    NEST_DOT,
    OBSTACLE_COLOR,
    OBSTACLE_DOT,
    SEEK_RING,
    STATION_COLOR,
    STATION_DOT,
    TEXT_COLOR,
    TILE_SIZE,
    TOKEN_BADGE,
    WHITE,
    WORLD_SIZE,
)
from apeks.sim import Agent, MovingBarrier, NavMap, Trail


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
            ring = (
                SEEK_RING
                if agent.state == "seek"
                else (CHARGE_RING if agent.state == "charge" else FOOD_RING)
            )
            pygame.draw.circle(screen, ring, (cx, cy), radius, 2)
        if agent.has_food:
            pygame.draw.circle(screen, FOOD_RING, (cx, cy), 4)
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
    nest_set: set[tuple[int, int]] | None = None,
    food: dict[tuple[int, int], int] | None = None,
    food_trail: Trail | None = None,
    obstacles: set[tuple[int, int]] | None = None,
    dynamic_hazards: dict[tuple[int, int], float] | set[tuple[int, int]] | None = None,
    tokens: dict[tuple[int, int], str] | None = None,
    barrier_coords: set[tuple[int, int]] | None = None,
    barriers: list[MovingBarrier] | None = None,
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
            is_nest = nest_set is not None and (col, row) in nest_set
            has_food = food is not None and (col, row) in food
            is_barrier = barrier_coords is not None and (col, row) in barrier_coords
            is_hazard = dynamic_hazards is not None and (col, row) in dynamic_hazards
            is_obs = obstacles is not None and (col, row) in obstacles
            value = trail.value(col, row, now)
            food_val = food_trail.value(col, row, now) if food_trail is not None else 0.0

            if is_station:
                fill = STATION_COLOR
            elif is_nest:
                fill = NEST_COLOR
            elif has_food:
                fill = FOOD_COLOR
            elif is_barrier:
                fill = BARRIER_COLOR
            elif is_hazard:
                fill = HAZARD_COLOR
            elif is_obs:
                fill = OBSTACLE_COLOR
            else:
                k = value / MARK_VALUE
                kf = min(1.0, food_val / MARK_VALUE)
                r = max(0, min(255, round(255 + (MARK_TINT[0] - 255) * k + (FOOD_TINT[0] - 255) * kf)))
                g = max(0, min(255, round(255 + (MARK_TINT[1] - 255) * k + (FOOD_TINT[1] - 255) * kf)))
                b = max(0, min(255, round(255 + (MARK_TINT[2] - 255) * k + (FOOD_TINT[2] - 255) * kf)))
                fill = (r, g, b)
            pygame.draw.rect(screen, fill, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            if is_obs or is_hazard or is_barrier:
                inner = rect.inflate(-4, -4)
                border_color = (
                    (255, 185, 90)
                    if is_barrier
                    else (225, 100, 100)
                    if is_hazard
                    else (100, 115, 135)
                )
                pygame.draw.rect(screen, border_color, inner, 1)

            if tokens and (col, row) in tokens:
                token_text = tokens[(col, row)]
                if "ALERT" in token_text or "HAZARD" in token_text:
                    token_color = (255, 90, 90)
                elif "DEPLETED" in token_text:
                    token_color = (255, 120, 80)
                elif "STATION" in token_text:
                    token_color = (255, 220, 50)
                elif "GUIDE" in token_text:
                    token_color = (80, 220, 255)
                elif "FOOD" in token_text:
                    token_color = (90, 255, 120)
                else:
                    token_color = TOKEN_BADGE
                pygame.draw.circle(
                    screen, token_color, (rect.left + 8, rect.bottom - 8), 4
                )

            if not is_station and not is_nest and not has_food and not is_obs and not is_hazard and not is_barrier and food_val > 0.5:
                pip_radius = max(2, min(5, round(5 * (food_val / MARK_VALUE))))
                pygame.draw.circle(screen, FOOD_RING, rect.center, pip_radius)

            if has_food and food:
                fl = font.render(f"x{food[(col, row)]}", True, BLACK)
                screen.blit(fl, fl.get_rect(center=rect.center))
            elif is_nest:
                nl = font.render("NEST", True, WHITE)
                screen.blit(nl, nl.get_rect(center=rect.center))
            elif is_barrier:
                lbl = "MOVE"
                if barriers:
                    for b in barriers:
                        if (col, row) in b.coords and b.expiry != float("inf"):
                            lbl = f"M:{max(0, round(b.expiry - now))}s"
                            break
                bl = font.render(lbl, True, WHITE)
                screen.blit(bl, bl.get_rect(center=rect.center))
            elif is_hazard:
                rem_text = "HAZ"
                if isinstance(dynamic_hazards, dict) and (col, row) in dynamic_hazards:
                    rem_sec = max(0, round(dynamic_hazards[(col, row)] - now))
                    rem_text = f"{rem_sec}s"
                hl = font.render(rem_text, True, WHITE)
                screen.blit(hl, hl.get_rect(center=rect.center))
            elif is_obs:
                pass
            else:
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
                if food_val > 0.5:
                    fl_val = font.render(str(round(food_val)), True, (130, 45, 0))
                    screen.blit(
                        fl_val,
                        fl_val.get_rect(topright=(rect.right - 3, rect.top + 2)),
                    )


def draw_minimap(
    screen: pygame.Surface,
    cam_x: float,
    cam_y: float,
    agents: list[Agent],
    stations: list[tuple[int, int]],
    nest_tiles: list[tuple[int, int]] | None = None,
    food_tiles: list[tuple[int, int]] | None = None,
    obstacles: set[tuple[int, int]] | None = None,
    barrier_coords: set[tuple[int, int]] | None = None,
    dynamic_hazards: set[tuple[int, int]] | None = None,
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
    if nest_tiles:
        for nx, ny in nest_tiles:
            pygame.draw.rect(
                screen,
                NEST_DOT,
                (
                    nx * MINIMAP_SIZE // GRID_SIZE - 1,
                    ny * MINIMAP_SIZE // GRID_SIZE - 1,
                    3,
                    3,
                ),
            )
    if food_tiles:
        for fx, fy in food_tiles:
            pygame.draw.rect(
                screen,
                FOOD_DOT,
                (
                    fx * MINIMAP_SIZE // GRID_SIZE - 1,
                    fy * MINIMAP_SIZE // GRID_SIZE - 1,
                    2,
                    2,
                ),
            )
    if obstacles:
        for ox, oy in obstacles:
            screen.set_at(
                (
                    ox * MINIMAP_SIZE // GRID_SIZE,
                    oy * MINIMAP_SIZE // GRID_SIZE,
                ),
                OBSTACLE_DOT,
            )
    if barrier_coords:
        for bx, by in barrier_coords:
            screen.set_at(
                (
                    bx * MINIMAP_SIZE // GRID_SIZE,
                    by * MINIMAP_SIZE // GRID_SIZE,
                ),
                BARRIER_DOT,
            )
    if dynamic_hazards:
        for hx, hy in dynamic_hazards:
            screen.set_at(
                (
                    hx * MINIMAP_SIZE // GRID_SIZE,
                    hy * MINIMAP_SIZE // GRID_SIZE,
                ),
                HAZARD_DOT,
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


def draw_hud(
    screen: pygame.Surface,
    font: pygame.font.Font,
    food_collected: int,
    food_remaining: int,
    agents: list[Agent],
    max_food_trail: float = 0.0,
    obstacle_count: int = 0,
    barrier_count: int = 0,
    hazard_count: int = 0,
    mode: str = "classical",
    llm_stats: dict[str, int] | None = None,
    latest_thought: str = "",
) -> None:
    states = {"walk": 0, "return_nest": 0, "seek": 0, "charge": 0}
    for a in agents:
        states[a.state] = states.get(a.state, 0) + 1

    mode_name = mode.upper()
    if mode == "coarse_fine":
        mode_color = (255, 220, 80)
        mode_desc = "Approach 1: LLM Waypoints + 60Hz Kinematics"
    elif mode == "global_contextual":
        mode_color = (240, 140, 255)
        mode_desc = "Approach 2: Contextual Roles + Stigmergy Blackboard"
    elif mode == "pure_llm":
        mode_color = (180, 160, 255)
        mode_desc = "Direct LLM Swarm Control"
    else:
        mode_color = (100, 255, 200)
        mode_desc = "Baseline: Ant Colony Optimization"

    line1 = f"SWARM MODE [Press 1-4]:  {mode_name}  ({mode_desc})"
    peak_str = f" | Peak Trail: {round(max_food_trail)}" if max_food_trail > 0.5 else ""
    line2 = (
        f"Food: {food_collected}  Remaining: {food_remaining} | "
        f"Agents [Walk: {states['walk']}  Return: {states['return_nest']}  Seek: {states['seek']}  Charge: {states['charge']}]{peak_str}"
    )

    stats = llm_stats or {}
    stats_str = (
        f" | LLM: {stats.get('decisions', 0)} dec, {stats.get('waypoints_assigned', 0)} wp, {stats.get('roles_switched', 0)} roles, {stats.get('tokens_written', 0)} tokens"
        if mode != "classical"
        else ""
    )
    thought_str = (
        f" | CoT: {latest_thought[:40]}..."
        if (latest_thought and mode != "classical")
        else ""
    )
    line3 = (
        f"Movers: {barrier_count}  Hazards: {hazard_count}  Static: {obstacle_count}{stats_str}{thought_str}"
    )

    surf1 = font.render(line1, True, mode_color)
    surf2 = font.render(line2, True, WHITE)
    surf3 = font.render(line3, True, (200, 220, 240))

    line_h = surf1.get_height()
    pad = 8
    total_w = max(surf1.get_width(), surf2.get_width(), surf3.get_width()) + 2 * pad
    total_h = 3 * line_h + 2 * pad + 6

    h = screen.get_height()
    bg_rect = pygame.Rect(10, h - total_h - 10, total_w, total_h)
    pygame.draw.rect(screen, (20, 20, 30), bg_rect, border_radius=6)
    pygame.draw.rect(screen, (80, 100, 140), bg_rect, 1, border_radius=6)

    screen.blit(surf1, (bg_rect.x + pad, bg_rect.y + pad))
    screen.blit(surf2, (bg_rect.x + pad, bg_rect.y + pad + line_h + 3))
    screen.blit(surf3, (bg_rect.x + pad, bg_rect.y + pad + 2 * (line_h + 3)))


def draw_bench_hud(
    screen: pygame.Surface,
    font: pygame.font.Font,
    instances: list[Any],
    active_idx: int,
    quad_view: bool,
) -> None:
    w, h = screen.get_size()
    active_inst = instances[active_idx]

    view_tag = "[TAB] 2x2 QUAD-VIEW (ACTIVE)" if quad_view else "[TAB] TOGGLE 2x2 QUAD-VIEW"
    line1 = (
        f"{view_tag}  |  "
        f"[1] CLASSICAL   [2] COARSE-FINE   [3] CONTEXTUAL   [4] PURE-LLM   -->  VIEWING: [{active_idx+1}] {active_inst.name}"
    )

    max_food = max((x.world.food_collected for x in instances), default=0)
    score_parts = []
    for i, inst in enumerate(instances):
        is_lead = (inst.world.food_collected == max_food and max_food > 0)
        tag = " ★" if is_lead else ""
        score_parts.append(f"[{i+1}] {inst.name}: {inst.world.food_collected}{tag}")
    line2 = f"BENCHMARK SCOREBOARD:  {'  |  '.join(score_parts)}"

    states = {"walk": 0, "return_nest": 0, "seek": 0, "charge": 0}
    for a in active_inst.agents:
        states[a.state] = states.get(a.state, 0) + 1

    stats = active_inst.orchestrator.stats
    stats_str = (
        f" | LLM: {stats.get('decisions', 0)} dec, {stats.get('waypoints_assigned', 0)} wp, {stats.get('roles_switched', 0)} roles, {stats.get('tokens_written', 0)} tokens"
        if active_inst.mode != "classical"
        else ""
    )
    thought_str = (
        f" | CoT: {active_inst.orchestrator.latest_thought[:42]}..."
        if (active_inst.orchestrator.latest_thought and active_inst.mode != "classical")
        else ""
    )
    line3 = (
        f"Active #{active_idx+1}: Food: {active_inst.world.food_collected} (Rem: {sum(active_inst.world.food.values())}) | "
        f"Agents [W:{states['walk']} R:{states['return_nest']} S:{states['seek']} C:{states['charge']}] | "
        f"Movers: {len(active_inst.world.moving_barriers)} Hazards: {len(active_inst.world.dynamic_hazards)}{stats_str}{thought_str}"
    )

    surf1 = font.render(line1, True, active_inst.color)
    surf2 = font.render(line2, True, (255, 235, 140))
    surf3 = font.render(line3, True, (210, 225, 245))

    line_h = surf1.get_height()
    pad = 8
    total_w = max(surf1.get_width(), surf2.get_width(), surf3.get_width()) + 2 * pad
    total_h = 3 * line_h + 2 * pad + 6

    bg_rect = pygame.Rect(10, h - total_h - 8, total_w, total_h)
    pygame.draw.rect(screen, (16, 18, 28), bg_rect, border_radius=6)
    pygame.draw.rect(screen, (80, 100, 140), bg_rect, 1, border_radius=6)

    screen.blit(surf1, (bg_rect.x + pad, bg_rect.y + pad))
    screen.blit(surf2, (bg_rect.x + pad, bg_rect.y + pad + line_h + 3))
    screen.blit(surf3, (bg_rect.x + pad, bg_rect.y + pad + 2 * (line_h + 3)))


def draw_quad_view(
    screen: pygame.Surface,
    instances: list[Any],
    active_idx: int,
    cam_x: float,
    cam_y: float,
    nav: NavMap,
    station_set: set[tuple[int, int]],
    tile_font: pygame.font.Font,
    banner_font: pygame.font.Font,
    text_cache: dict[tuple[int, int], pygame.Surface],
    value_labels: list[pygame.Surface],
    labels: list[pygame.Surface],
    charge_labels: list[pygame.Surface],
    nest_set: set[tuple[int, int]],
    now: float,
    hud_h: int = 80,
) -> None:
    w, h = screen.get_size()
    avail_h = max(100, h - hud_h)
    qw = w // 2
    qh = avail_h // 2

    for i, inst in enumerate(instances):
        col = i % 2
        row = i // 2
        qx = col * qw
        qy = row * qh
        rect = pygame.Rect(qx, qy, qw, qh)
        sub = screen.subsurface(rect)
        sub.fill((0, 0, 0))

        draw_grid(
            sub,
            cam_x,
            cam_y,
            nav,
            station_set,
            tile_font,
            text_cache,
            inst.trail,
            now,
            value_labels,
            nest_set=nest_set,
            food=inst.world.food,
            food_trail=inst.food_trail,
            obstacles=inst.world.obstacles,
            dynamic_hazards=inst.world.dynamic_hazards,
            tokens=inst.world.active_tokens(now),
            barrier_coords=inst.world.barrier_coords,
            barriers=inst.world.moving_barriers,
        )
        draw_agents(sub, inst.agents, labels, charge_labels, cam_x, cam_y)

        banner_rect = pygame.Rect(0, 0, qw, 22)
        pygame.draw.rect(sub, (20, 24, 36), banner_rect)
        pygame.draw.line(sub, inst.color, (0, 22), (qw, 22), 1)

        sel_tag = "  [FOCUSED - Click to Zoom]" if i == active_idx else "  [Click to Focus]"
        title_text = f"[{i+1}] {inst.name}  |  Food: {inst.world.food_collected}  |  Agents: {len(inst.agents)}{sel_tag}"
        title_surf = banner_font.render(title_text, True, inst.color)
        sub.blit(title_surf, (8, 3))

        border_color = (255, 230, 80) if i == active_idx else (50, 60, 80)
        border_width = 2 if i == active_idx else 1
        pygame.draw.rect(screen, border_color, rect, border_width)

