import pygame
import pytest
from apeks.sim import Agent, Trail, build_nav, make_stations

from apeks.config import (
    CHARGE_RING,
    MARK_VALUE,
    MAX_CHARGE,
    MINIMAP_SIZE,
    SEEK_RING,
    STATION_COLOR,
    TILE_SIZE,
    WHITE,
    WORLD_SIZE,
)
from apeks.render import draw_agents, draw_arrow, draw_grid, draw_minimap

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
    ("state", "ring"), [("seek", SEEK_RING), ("charge", CHARGE_RING)]
)
def test_agent_ring_marks_non_walking_states(screen, state, ring):
    screen.fill((0, 0, 0))
    draw_agent(screen, agent_at(1, 1, state))
    cx = TILE_SIZE + TILE_SIZE // 2
    cy = TILE_SIZE + TILE_SIZE // 2
    radius = TILE_SIZE // 2 - 2
    assert screen.get_at((cx + radius - 1, cy))[:3] == ring


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
