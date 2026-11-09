import pygame
import pytest

from apeks.config import (
    BARRIER_COLOR,
    BARRIER_DOT,
    CHARGE_RING,
    FOOD_COLOR,
    FOOD_DOT,
    FOOD_RING,
    HAZARD_COLOR,
    HAZARD_DOT,
    MARK_VALUE,
    MAX_CHARGE,
    MINIMAP_SIZE,
    NEST_COLOR,
    NEST_DOT,
    OBSTACLE_COLOR,
    OBSTACLE_DOT,
    SEEK_RING,
    STATION_COLOR,
    TILE_SIZE,
    TOKEN_BADGE,
    WHITE,
    WORLD_SIZE,
)
from apeks.llm import ParallelSwarmBench
from apeks.render import (
    draw_agents,
    draw_arrow,
    draw_bench_hud,
    draw_grid,
    draw_hud,
    draw_minimap,
    draw_quad_view,
)
from apeks.sim import Agent, MovingBarrier, Trail, build_nav, make_stations

pygame.font.init()

CLEAR = pygame.Surface((1, 1), pygame.SRCALPHA)
STATIONS = make_stations()
NAV = build_nav(STATIONS)


@pytest.fixture
def screen():
    return pygame.Surface((300, 200))


def agent_at(col, row, state="walk"):
    a = Agent(0, col, row)
    a.state = state
    a.charge = MAX_CHARGE
    return a


def draw_agent(screen, a, cam=(0, 0)):
    draw_agents(screen, [a], [CLEAR], [CLEAR] * (MAX_CHARGE + 1), *cam)


def test_arrow_draws_in_its_direction(screen):
    screen.fill((0, 0, 0))
    draw_arrow(screen, 100, 100, 1, 0)
    assert screen.get_at((105, 100)) != (0, 0, 0, 255)
    assert screen.get_at((100, 130)) == (0, 0, 0, 255)


def test_agent_drawn_as_filled_circle(screen):
    screen.fill((0, 0, 0))
    a = agent_at(1, 1)
    draw_agent(screen, a)
    centre = (TILE_SIZE + TILE_SIZE // 2, TILE_SIZE + TILE_SIZE // 2)
    assert screen.get_at(centre)[:3] == a.color
    assert screen.get_at((TILE_SIZE + TILE_SIZE // 2, TILE_SIZE - 5))[:3] == (0, 0, 0)


@pytest.mark.parametrize(
    ("state", "ring"),
    [("seek", SEEK_RING), ("charge", CHARGE_RING), ("return_nest", FOOD_RING)],
)
def test_agent_ring_marks_non_walking_states(screen, state, ring):
    screen.fill((0, 0, 0))
    draw_agent(screen, agent_at(1, 1, state))
    cx = TILE_SIZE + TILE_SIZE // 2
    cy = TILE_SIZE + TILE_SIZE // 2
    radius = TILE_SIZE // 2 - 2
    assert screen.get_at((cx + radius - 1, cy))[:3] == ring


def test_agent_has_food_draws_dot(screen):
    screen.fill((0, 0, 0))
    a = agent_at(1, 1, "walk")
    a.has_food = True
    draw_agent(screen, a)
    cx = TILE_SIZE + TILE_SIZE // 2
    cy = TILE_SIZE + TILE_SIZE // 2
    assert screen.get_at((cx, cy))[:3] == FOOD_RING


def test_agent_outside_viewport_is_skipped(screen):
    screen.fill((0, 0, 0))
    draw_agent(screen, agent_at(50, 50))
    assert screen.get_at((10, 10))[:3] == (0, 0, 0)
    assert (
        pygame.mask.from_threshold(screen, (0, 0, 0), (1, 1, 1, 255)).count()
        == 300 * 200
    )


def test_agent_labels_are_blitted(screen):
    screen.fill((0, 0, 0))
    label = pygame.Surface((8, 4))
    label.fill(WHITE)
    draw_agents(screen, [agent_at(1, 1)], [label], [label] * (MAX_CHARGE + 1), 0, 0)
    assert (
        screen.get_at((TILE_SIZE + TILE_SIZE // 2, TILE_SIZE + TILE_SIZE // 2 - 1))[:3]
        == WHITE
    )


def grid_args():
    font = pygame.font.Font(None, 15)
    labels = [pygame.Surface((1, 1), pygame.SRCALPHA) for _ in range(MARK_VALUE + 1)]
    return font, {}, labels


def test_grid_tints_by_trail_and_colours_stations(screen):
    font, cache, labels = grid_args()
    trail = Trail()
    trail.mark(1, 0, 0.0)
    cam_x = STATIONS[0][0] * TILE_SIZE - 100
    draw_grid(screen, 0, 0, NAV, set(STATIONS), font, cache, trail, 0.0, labels)
    assert screen.get_at((5, 30))[:3] == WHITE
    hot = screen.get_at((TILE_SIZE + 45, 30))[:3]
    assert hot != WHITE
    station_screen = pygame.Surface((300, 200))
    draw_grid(
        station_screen, cam_x, 0, NAV, set(STATIONS), font, cache, Trail(), 0.0, labels
    )
    assert station_screen.get_at((100 + 45, 30))[:3] == STATION_COLOR


def test_grid_reuses_cached_offset_text(screen):
    font, cache, labels = grid_args()
    draw_grid(screen, 0, 0, NAV, set(STATIONS), font, cache, Trail(), 0.0, labels)
    size = len(cache)
    assert size > 0
    draw_grid(screen, 0, 0, NAV, set(STATIONS), font, cache, Trail(), 0.0, labels)
    assert len(cache) == size


def test_grid_clamps_viewport_to_the_world(screen):
    font, cache, labels = grid_args()
    far = WORLD_SIZE - 100
    draw_grid(screen, far, far, NAV, set(STATIONS), font, cache, Trail(), 0.0, labels)
    assert screen.get_at((5, 5)) != (0, 0, 0, 255)


def test_minimap_draws_frame_agents_and_view(screen):
    screen.fill((0, 0, 0))
    draw_minimap(screen, 0, 0, [agent_at(10, 10)], STATIONS)
    assert screen.get_at((0, 0))[:3] != (0, 0, 0)
    assert screen.get_at((10, 10))[:3] == WHITE
    assert screen.get_at((MINIMAP_SIZE + 5, 5))[:3] == (0, 0, 0)


def test_grid_renders_nest_and_food_and_trail(screen):
    font, cache, labels = grid_args()
    food_trail = Trail()
    food_trail.mark(2, 0, 0.0)
    draw_grid(
        screen,
        0,
        0,
        NAV,
        set(STATIONS),
        font,
        cache,
        Trail(),
        0.0,
        labels,
        nest_set={(0, 0)},
        food={(1, 0): 5},
        food_trail=food_trail,
    )
    # Check nest tile
    assert screen.get_at((10, 10))[:3] == NEST_COLOR
    # Check food tile
    assert screen.get_at((TILE_SIZE + 10, 10))[:3] == FOOD_COLOR
    # Check food trail tinted tile
    tinted = screen.get_at((2 * TILE_SIZE + 10, 10))[:3]
    assert tinted != WHITE
    # Check golden pip at center of food trail tile
    assert screen.get_at((2 * TILE_SIZE + TILE_SIZE // 2, TILE_SIZE // 2))[:3] == FOOD_RING



def test_minimap_draws_nest_and_food(screen):
    screen.fill((0, 0, 0))
    draw_minimap(
        screen,
        0,
        0,
        [agent_at(10, 10)],
        STATIONS,
        nest_tiles=[(50, 50)],
        food_tiles=[(20, 20)],
    )
    # Nest dot
    assert screen.get_at((50 * MINIMAP_SIZE // 100, 50 * MINIMAP_SIZE // 100))[:3] == NEST_DOT
    # Food dot
    assert screen.get_at((20 * MINIMAP_SIZE // 100, 20 * MINIMAP_SIZE // 100))[:3] == FOOD_DOT


def test_draw_hud(screen):
    font = pygame.font.Font(None, 16)
    agents = [
        agent_at(0, 0, "walk"),
        agent_at(1, 1, "return_nest"),
        agent_at(2, 2, "seek"),
        agent_at(3, 3, "charge"),
    ]
    draw_hud(screen, font, 10, 90, agents)
    # Bottom HUD area should have non-black pixels from background and text
    assert screen.get_at((15, screen.get_height() - 15))[:3] != (0, 0, 0)
    # Test with max_food_trail, obstacles, barriers, and hazards
    screen.fill((0, 0, 0))
    draw_hud(
        screen,
        font,
        10,
        90,
        agents,
        max_food_trail=350.0,
        obstacle_count=5,
        barrier_count=4,
        hazard_count=2,
    )
    assert screen.get_at((15, screen.get_height() - 15))[:3] != (0, 0, 0)

    # Test coarse_fine mode with stats and thought
    screen.fill((0, 0, 0))
    draw_hud(
        screen,
        font,
        15,
        85,
        agents,
        mode="coarse_fine",
        llm_stats={"decisions": 5, "waypoints_assigned": 2, "roles_switched": 1, "tokens_written": 0},
        latest_thought="Exploring coldest quadrant",
    )
    assert screen.get_at((15, screen.get_height() - 15))[:3] != (0, 0, 0)

    # Test global_contextual mode with empty thought
    screen.fill((0, 0, 0))
    draw_hud(
        screen,
        font,
        15,
        85,
        agents,
        mode="global_contextual",
        llm_stats={"decisions": 10},
        latest_thought="",
    )
    assert screen.get_at((15, screen.get_height() - 15))[:3] != (0, 0, 0)

    # Test pure_llm mode
    screen.fill((0, 0, 0))
    draw_hud(
        screen,
        font,
        15,
        85,
        agents,
        mode="pure_llm",
    )
    assert screen.get_at((15, screen.get_height() - 15))[:3] != (0, 0, 0)


def test_grid_renders_obstacles_and_hazards_and_tokens(screen):
    font, cache, labels = grid_args()
    draw_grid(
        screen,
        0,
        0,
        NAV,
        set(STATIONS),
        font,
        cache,
        Trail(),
        0.0,
        labels,
        obstacles={(1, 0)},
        dynamic_hazards={(2, 0): 25.0},
        tokens={
            (1, 0): "ALERT: Trap",
            (2, 0): "[FOOD] 4u",
            (3, 0): "INFO",
            (4, 0): "[DEPLETED] Source empty",
            (5, 0): "[GUIDE] 5 steps East",
            (6, 0): "[STATION] Charging depot at (10, 10)",
        },
        barrier_coords={(3, 0)},
        barriers=[
            MovingBarrier(
                coords=[(3, 0)],
                axis="x",
                direction=1,
                min_pos=0,
                max_pos=10,
                expiry=30.0,
            )
        ],
    )
    # Check static obstacle fill
    assert screen.get_at((TILE_SIZE + 10, 10))[:3] == OBSTACLE_COLOR
    # Check dynamic hazard fill
    assert screen.get_at((2 * TILE_SIZE + 10, 10))[:3] == HAZARD_COLOR
    # Check moving barrier fill
    assert screen.get_at((3 * TILE_SIZE + 10, 10))[:3] == BARRIER_COLOR
    # Check token badge on (1, 0)
    # Test with dynamic_hazards as a set (fallback to 'HAZ' label) and barrier with inf expiry
    screen.fill((0, 0, 0))
    draw_grid(
        screen,
        0,
        0,
        NAV,
        set(STATIONS),
        font,
        cache,
        Trail(),
        0.0,
        labels,
        tokens={(3, 0): "HAZARD: Deep pit"},
        dynamic_hazards={(2, 0)},
        barrier_coords={(3, 0)},
        barriers=[
            MovingBarrier(
                coords=[(3, 0)],
                axis="x",
                direction=1,
                min_pos=0,
                max_pos=10,
                expiry=float("inf"),
            )
        ],
    )
    assert screen.get_at((2 * TILE_SIZE + 10, 10))[:3] == HAZARD_COLOR
    assert screen.get_at((3 * TILE_SIZE + 10, 10))[:3] == BARRIER_COLOR

    # Test with barriers=None
    screen.fill((0, 0, 0))
    draw_grid(
        screen,
        0,
        0,
        NAV,
        set(STATIONS),
        font,
        cache,
        Trail(),
        0.0,
        labels,
        barrier_coords={(3, 0)},
        barriers=None,
    )
    assert screen.get_at((3 * TILE_SIZE + 10, 10))[:3] == BARRIER_COLOR


def test_minimap_draws_obstacles(screen):
    screen.fill((0, 0, 0))
    draw_minimap(
        screen,
        0,
        0,
        [agent_at(10, 10)],
        STATIONS,
        obstacles={(30, 30)},
        barrier_coords={(40, 40)},
        dynamic_hazards={(50, 50)},
    )
    assert (
        screen.get_at((30 * MINIMAP_SIZE // 100, 30 * MINIMAP_SIZE // 100))[:3]
        == OBSTACLE_DOT
    )
    assert (
        screen.get_at((40 * MINIMAP_SIZE // 100, 40 * MINIMAP_SIZE // 100))[:3]
        == BARRIER_DOT
    )
    assert (
        screen.get_at((50 * MINIMAP_SIZE // 100, 50 * MINIMAP_SIZE // 100))[:3]
        == HAZARD_DOT
    )


def test_draw_bench_hud():
    screen = pygame.Surface((1280, 720))
    font = pygame.font.Font(None, 18)
    bench = ParallelSwarmBench(seed=42)

    # 1. Classical active, quad_view False, 0 food collected
    screen.fill((0, 0, 0))
    draw_bench_hud(screen, font, bench.instances, active_idx=0, quad_view=False)
    assert screen.get_at((20, 700))[:3] != (0, 0, 0)

    # 2. Coarse-fine active, quad_view True, with food collected and latest_thought
    bench.instances[1].world.food_collected = 15
    bench.instances[1].orchestrator.latest_thought = "Exploring coldest quadrant"
    screen.fill((0, 0, 0))
    draw_bench_hud(screen, font, bench.instances, active_idx=1, quad_view=True)
    assert screen.get_at((20, 700))[:3] != (0, 0, 0)

    # 3. Global-contextual active with empty thought
    bench.instances[2].orchestrator.latest_thought = ""
    screen.fill((0, 0, 0))
    draw_bench_hud(screen, font, bench.instances, active_idx=2, quad_view=False)
    assert screen.get_at((20, 700))[:3] != (0, 0, 0)


def test_draw_quad_view():
    screen = pygame.Surface((1280, 720))
    tile_font, cache, value_labels = grid_args()
    banner_font = pygame.font.Font(None, 18)
    agent_labels = [pygame.Surface((1, 1), pygame.SRCALPHA) for _ in range(50)]
    charge_labels = [
        pygame.Surface((1, 1), pygame.SRCALPHA) for _ in range(MAX_CHARGE + 1)
    ]

    bench = ParallelSwarmBench(seed=42)
    screen.fill((0, 0, 0))
    draw_quad_view(
        screen,
        bench.instances,
        active_idx=0,
        cam_x=0.0,
        cam_y=0.0,
        nav=NAV,
        station_set=set(STATIONS),
        tile_font=tile_font,
        banner_font=banner_font,
        text_cache=cache,
        value_labels=value_labels,
        labels=agent_labels,
        charge_labels=charge_labels,
        nest_set={(25, 25)},
        now=1.0,
        hud_h=80,
    )
    # Check banner pixels in top-left, top-right, bottom-left, bottom-right quadrants
    assert screen.get_at((50, 10))[:3] != (0, 0, 0)
    assert screen.get_at((640 + 50, 10))[:3] != (0, 0, 0)
    assert screen.get_at((50, 330))[:3] != (0, 0, 0)
    assert screen.get_at((640 + 50, 330))[:3] != (0, 0, 0)


