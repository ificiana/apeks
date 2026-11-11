from pathlib import Path

import pygame

from apeks import scenario
from apeks.app import SPEEDS, change_speed, parse_args
from apeks.experiment import ExperimentConfig, build_world
from apeks.render import draw, hud_lines

ARMS = Path(__file__).parent.parent / "experiments"


def test_parse_args_defaults_and_loads_config() -> None:
    assert parse_args([]) == ExperimentConfig()
    assert parse_args([str(ARMS / "llm.toml")]).routing == "llm"


def test_speed_steps_are_clamped() -> None:
    assert change_speed(0, -1) == 0
    assert change_speed(0, 1) == 1
    assert change_speed(len(SPEEDS) - 1, 1) == len(SPEEDS) - 1


def test_hud_shows_metrics_closures_and_llm_stats() -> None:
    world = build_world(ExperimentConfig(routing="llm"))
    world.step()
    world.closed = {"W2>W3"}
    lines = hud_lines(world, "llm")
    assert lines[0] == "llm  t=1"
    assert "closed W2>W3" in lines
    assert any(line.startswith("llm routing: 1 calls") for line in lines)


def test_draw_paints_storms_closures_and_vessels() -> None:
    pygame.init()
    surface = pygame.Surface((scenario.WIDTH, scenario.HEIGHT))
    font = pygame.font.Font(None, 14)
    world = build_world(ExperimentConfig())
    world.t = 2000.0
    world.closed = {"W2>W3"}
    draw(surface, world, "classical", font)
    vessel = world.vessels["east-start.0"].pos
    assert surface.get_at((int(vessel[0]), int(vessel[1])))[:3] == (255, 170, 60)
    pygame.quit()
