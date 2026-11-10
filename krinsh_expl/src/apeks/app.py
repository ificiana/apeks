import random

import pygame

from apeks.config import (
    AGENT_COUNT,
    BLACK,
    CAM_SPEED,
    MARK_VALUE,
    MAX_CHARGE,
    VALUE_COLOR,
    WHITE,
    WORLD_SIZE,
)
from apeks.llm import ParallelSwarmBench
from apeks.render import (
    draw_agents,
    draw_bench_hud,
    draw_grid,
    draw_minimap,
    draw_quad_view,
)
from apeks.sim import (
    make_nest,
    make_stations,
)


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
    pygame.display.set_caption("apeks — [1] CLASSICAL")
    clock = pygame.time.Clock()
    cam_x = cam_y = 0.0

    font = pygame.font.Font(None, 16)
    banner_font = pygame.font.Font(None, 18)
    hud_font = pygame.font.Font(None, 20)

    stations = make_stations()
    station_set = set(stations)
    nest_tiles = make_nest()
    nest_set = set(nest_tiles)

    bench = ParallelSwarmBench(seed=42, stations=stations, nest_tiles=nest_tiles)
    nav = bench.instances[0].world.nav

    labels = [font.render(f"#{i:03d}", True, WHITE) for i in range(AGENT_COUNT)]
    charge_labels = [font.render(f"{c}U", True, WHITE) for c in range(MAX_CHARGE + 1)]
    now = 0.0
    value_font = pygame.font.Font(None, 14)
    value_labels = [
        value_font.render(str(v), True, VALUE_COLOR) for v in range(MARK_VALUE + 1)
    ]

    tile_font = pygame.font.Font(None, 15)
    text_cache: dict[tuple[int, int], pygame.Surface] = {}

    print("=" * 72)
    print(" APEKS PARALLEL MULTI-SWARM BENCHMARK ENGINE STARTED")
    print(" View Controls:")
    print("   [1] Full-Screen Classical  |  [2] Full-Screen Coarse-Fine Hybrid")
    print("   [3] Full-Screen Contextual |  [4] Full-Screen Pure-LLM")
    print("   [TAB] / [Q] Toggle 2x2 Quad-View (All 4 Worlds Side-by-Side)")
    print("   [Click on any Quadrant in Quad-View] to zoom in!")
    print("   Pan Camera: W/A/S/D, Arrow Keys, or Left-Click Drag")
    print("=" * 72)

    last_printed_thought = ""
    next_food_time = random.uniform(4.0, 8.0)

    running = True
    while running:
        dt = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if bench.quad_view:
                    w, h = screen.get_size()
                    avail_h = max(100, h - 80)
                    qw, qh = w // 2, avail_h // 2
                    mx, my = event.pos
                    if my < avail_h:
                        col = 0 if mx < qw else 1
                        row = 0 if my < qh else 1
                        clicked_idx = row * 2 + col
                        bench.set_active_idx(clicked_idx)
                        pygame.display.set_caption(
                            f"apeks — [{bench.active_idx + 1}] "
                            f"{bench.active_instance.name}"
                        )
                        print(
                            f"\n[VIEWPORT] Zoomed into [{bench.active_idx + 1}] "
                            f"{bench.active_instance.name}"
                        )
            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                cam_x -= event.rel[0]
                cam_y -= event.rel[1]
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    bench.set_active_idx(0)
                    pygame.display.set_caption("apeks — [1] CLASSICAL")
                    print("\n[VIEWPORT] Switched to [1] CLASSICAL")
                elif event.key == pygame.K_2:
                    bench.set_active_idx(1)
                    pygame.display.set_caption("apeks — [2] COARSE-FINE HYBRID")
                    print("\n[VIEWPORT] Switched to [2] COARSE-FINE")
                elif event.key == pygame.K_3:
                    bench.set_active_idx(2)
                    pygame.display.set_caption("apeks — [3] GLOBAL-CONTEXTUAL HYBRID")
                    print("\n[VIEWPORT] Switched to [3] GLOBAL-CONTEXTUAL")
                elif event.key == pygame.K_4:
                    bench.set_active_idx(3)
                    pygame.display.set_caption("apeks — [4] PURE-LLM")
                    print("\n[VIEWPORT] Switched to [4] PURE-LLM")
                elif event.key in (pygame.K_TAB, pygame.K_q):
                    is_quad = bench.toggle_quad_view()
                    cap = (
                        "apeks — 2x2 QUAD-VIEW (Parallel Swarm Benchmark)"
                        if is_quad
                        else (
                            f"apeks — [{bench.active_idx + 1}] "
                            f"{bench.active_instance.name}"
                        )
                    )
                    pygame.display.set_caption(cap)
                    print(
                        "\n[VIEWPORT] 2x2 Quad-View: "
                        f"{'ACTIVATED' if is_quad else 'DEACTIVATED'}"
                    )

        now += dt
        bench.step(now, dt)

        # Synchronized food spawn across all 4 benchmark worlds
        if now >= next_food_time:
            if len(bench.active_instance.world.food) < 160:
                bench.spawn_synchronized_food()
            next_food_time = now + random.uniform(3.0, 7.0)

        # Log CoT thoughts from active instance
        active = bench.active_instance
        if (
            active.mode != "classical"
            and active.orchestrator.latest_thought
            and active.orchestrator.latest_thought != last_printed_thought
        ):
            print(f"[LLM CoT | {active.name}] {active.orchestrator.latest_thought}")
            last_printed_thought = active.orchestrator.latest_thought

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
        if bench.quad_view:
            draw_quad_view(
                screen,
                bench.instances,
                bench.active_idx,
                cam_x,
                cam_y,
                nav,
                station_set,
                tile_font,
                banner_font,
                text_cache,
                value_labels,
                labels,
                charge_labels,
                nest_set,
                now,
            )
        else:
            inst = bench.active_instance
            draw_grid(
                screen,
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
            draw_agents(screen, inst.agents, labels, charge_labels, cam_x, cam_y)
            draw_minimap(
                screen,
                cam_x,
                cam_y,
                inst.agents,
                stations,
                nest_tiles=nest_tiles,
                food_tiles=list(inst.world.food.keys()),
                obstacles=inst.world.static_obstacles,
                barrier_coords=inst.world.barrier_coords,
                dynamic_hazards=set(inst.world.dynamic_hazards.keys()),
            )

        draw_bench_hud(
            screen,
            hud_font,
            bench.instances,
            bench.active_idx,
            bench.quad_view,
        )
        pygame.display.flip()

    pygame.quit()
