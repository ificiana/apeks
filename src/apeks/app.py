"""Interactive dashboard: ``apeks [config.toml]``.

Drag or WASD pans, the wheel or +/- zooms, F fits the map, space pauses,
up/down change speed, Esc quits.
"""

import argparse
import threading
from pathlib import Path

from apeks.experiment import ExperimentConfig, build_world, load_config
from apeks.render import SPEEDS, Camera, History, Ui, layout
from apeks.world import World


def parse_args(argv: list[str] | None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(prog="apeks", description=__doc__)
    parser.add_argument("config", nargs="?", type=Path)
    args = parser.parse_args(argv)
    return load_config(args.config) if args.config else ExperimentConfig()


def start(
    cfg: ExperimentConfig, size: tuple[int, int] = (1600, 900)
) -> tuple[World, Ui, Camera, History]:
    """Build the world and a camera that fits its whole map."""
    world = build_world(cfg)
    geo = world.geography
    ui = Ui(size=size, world_size=(geo.width, geo.height))
    cam = Camera(view=tuple(layout(size)["map"]))
    cam.fit(*ui.world_size)
    history = History()
    history.record(world)
    return world, ui, cam, history


def advance(world: World, ui: Ui, history: History, ticks: int) -> None:
    """Step the world by the current speed unless paused or finished."""
    if ui.paused:
        return
    for _ in range(SPEEDS[ui.speed]):
        if world.t >= ticks:
            return
        world.step()
        history.record(world)


class Stepper:
    """Steps the world on a worker thread so the window stays responsive.

    ``world.step`` can block for seconds on an LLM call. The event loop draws
    only when it can take ``lock`` without waiting, and otherwise shows a
    banner over the last frame, so the OS never sees the window hang.
    """

    def __init__(
        self, world: World, ui: Ui, history: History, ticks: int, period: float = 1 / 30
    ) -> None:
        self.world, self.ui, self.history = world, ui, history
        self.ticks, self.period = ticks, period
        self.lock = threading.Lock()
        self.busy = False
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self, wait: bool = True) -> None:
        """Ask the worker to finish; ``wait=False`` returns at once, so
        quitting never waits on an LLM call in flight."""
        self._stop.set()
        if wait:
            self.thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self.lock:
                self.busy = True
                advance(self.world, self.ui, self.history, self.ticks)
                self.busy = False
            self._stop.wait(self.period)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    # Opens a real window and runs the event loop until closed; the pieces it
    # uses (parse_args, start, advance, handle_event, draw) are tested directly.
    import pygame

    from apeks.render import draw, draw_waiting, handle_event

    cfg = parse_args(argv)
    world, ui, cam, history = start(cfg)
    pygame.init()
    screen = pygame.display.set_mode(ui.size, pygame.RESIZABLE)
    pygame.display.set_caption(f"apeks: {cfg.name}")
    fonts = {
        k: pygame.font.SysFont("dejavusansmono,menlo,consolas,monospace", n)
        for k, n in (("s", 13), ("m", 17), ("l", 24))
    }
    clock = pygame.time.Clock()
    stepper = Stepper(world, ui, history, cfg.ticks)
    stepper.start()
    running = True
    while running:
        for event in pygame.event.get():
            running = running and handle_event(event, ui, cam)
        if stepper.lock.acquire(blocking=False):
            try:
                draw(screen, world, cfg.name, fonts, cam, history, ui)
            finally:
                stepper.lock.release()
        else:
            draw_waiting(screen, fonts)
        pygame.display.flip()
        clock.tick(30)
    stepper.stop(wait=False)
    pygame.quit()
