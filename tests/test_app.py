from pathlib import Path

from apeks.app import advance, parse_args, start
from apeks.experiment import ExperimentConfig
from apeks.render import SPEEDS, layout

ARMS = Path(__file__).parent.parent / "experiments"


def test_parse_args_defaults_and_loads_config() -> None:
    assert parse_args([]) == ExperimentConfig()
    assert parse_args([str(ARMS / "llm.toml")]).routing == "llm"


def test_start_fits_the_whole_map_in_the_map_view() -> None:
    world, ui, cam, history = start(ExperimentConfig(seed=24008), size=(1200, 700))
    geo = world.geography
    assert ui.world_size == (geo.width, geo.height)
    assert cam.view == tuple(layout((1200, 700))["map"])
    x0, y0 = cam.to_screen((0.0, 0.0))
    x1, y1 = cam.to_screen((geo.width, geo.height))
    assert x0 >= 0 and y0 >= 0 and x1 <= cam.view[2] + 1e-9 and y1 <= cam.view[3] + 1e-9
    assert list(history.ticks) == [0.0]


def test_advance_steps_by_speed_until_paused_or_done() -> None:
    world, ui, _, history = start(ExperimentConfig(map="classic"))
    ui.speed = 2
    advance(world, ui, history, ticks=100)
    assert world.t == SPEEDS[2]
    ui.paused = True
    advance(world, ui, history, ticks=100)
    assert world.t == SPEEDS[2]
    ui.paused, ui.speed = False, len(SPEEDS) - 1
    advance(world, ui, history, ticks=7)
    assert world.t == 7


def test_headless_runner_does_not_load_pygame() -> None:
    import subprocess
    import sys

    code = "import sys, apeks.runner; print('pygame' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False"


def test_stepper_steps_in_the_background_and_holds_the_lock_while_busy() -> None:
    import threading

    from apeks.app import Stepper

    world, ui, _, history = start(ExperimentConfig(map="classic"))
    gate, entered = threading.Event(), threading.Event()
    real = world.step

    def slow_step() -> None:  # stands in for a blocking LLM call
        entered.set()
        gate.wait(5)
        real()

    world.step = slow_step
    stepper = Stepper(world, ui, history, ticks=3, period=0.001)
    stepper.start()
    assert entered.wait(5)
    stepper.stop(wait=False)  # returns at once though a step is in flight
    assert stepper.thread.is_alive()
    stepper._stop.clear()
    assert stepper.busy and not stepper.lock.acquire(blocking=False)
    gate.set()
    for _ in range(500):
        if world.t == 3:
            break
        threading.Event().wait(0.01)
    assert world.t == 3
    stepper.stop()
    assert not stepper.thread.is_alive() and not stepper.busy


def test_draw_waiting_paints_a_banner(fonts) -> None:
    import pygame

    from apeks.render import BG, draw_waiting

    surface = pygame.Surface((400, 300))
    surface.fill(BG)
    draw_waiting(surface, fonts)
    assert any(
        surface.get_at((x, y))[:3] != BG for x in range(400) for y in range(0, 60)
    )
