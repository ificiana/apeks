"""Interactive dashboard: ``apeks [config.toml]``.

The left column edits a draft of the config (run, map, modules, LLM, news and
engine parameters) and reruns it. Drag or WASD pans, the wheel or +/- zooms,
F fits the map, space pauses, up/down change speed, R applies the draft and
reruns, N reruns with the next seed, . steps one tick, Esc quits.
"""

import argparse
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import pygame

from apeks.controls import get, nudge, parse, put, show, specs
from apeks.experiment import ExperimentConfig, build_world, load_config, to_toml
from apeks.render import (
    SPEEDS,
    Camera,
    ControlsView,
    History,
    Ui,
    controls_view,
    draw,
    draw_waiting,
    layout,
)
from apeks.runner import timestamp
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


def advance(
    world: World,
    ui: Ui,
    history: History,
    ticks: int,
    lock: AbstractContextManager[Any] | None = None,
) -> None:
    """Step the world by the current speed unless paused or finished.

    ``lock`` is held around each tick, not the whole batch, so the window can
    draw between ticks.
    """
    for _ in range(SPEEDS[ui.speed]):
        if ui.paused or world.t >= ticks:
            return
        with lock or nullcontext():
            world.step()
            history.record(world)


def present(
    screen: pygame.Surface,
    frame: pygame.Surface | None,
    lock: threading.Lock,
    paint: Callable[[pygame.Surface], object],
    fonts: dict[str, pygame.font.Font],
    wait: float = 0.1,
) -> pygame.Surface:
    """Show a fresh frame, or the last one with a banner while a step is slow.

    Painting goes to an offscreen ``frame`` that is shown every time, so a
    busy world never makes the window blink. A step that keeps ``lock`` longer
    than ``wait`` (an LLM call) counts as busy.
    """
    if frame is None or frame.get_size() != screen.get_size():
        frame = pygame.Surface(screen.get_size())
    if lock.acquire(timeout=wait):
        try:
            paint(frame)
        finally:
            lock.release()
        screen.blit(frame, (0, 0))
    else:
        screen.blit(frame, (0, 0))
        draw_waiting(screen, fonts)
    return frame


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
            self.busy = True
            advance(self.world, self.ui, self.history, self.ticks, self.lock)
            self.busy = False
            self._stop.wait(self.period)


class Session:
    """The dashboard's state: the live config and world, and a draft to edit.

    Edits change only ``draft``; Apply builds a new world from it. Each world
    steps on its own ``Stepper`` thread when ``threaded``.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        size: tuple[int, int] = (1600, 900),
        threaded: bool = False,
    ) -> None:
        self.threaded = threaded
        self.save_dir = Path("logs/dashboard")
        self.ui = Ui(size=size)
        self.frame: pygame.Surface | None = None
        self.draft = cfg
        self._build(cfg)

    def _build(self, cfg: ExperimentConfig) -> None:
        world = build_world(cfg)
        if hasattr(self, "stepper"):
            self.stepper.stop(wait=False)  # never wait on an LLM call in flight
        geo = world.geography
        self.live, self.world = cfg, world
        self.ui.world_size = (geo.width, geo.height)
        self.cam = Camera(view=tuple(layout(self.ui.size)["map"]))
        self.cam.fit(*self.ui.world_size)
        self.history = History()
        self.history.record(world)
        self.stepper = Stepper(world, self.ui, self.history, cfg.ticks)
        if self.threaded:
            self.stepper.start()

    def rerun(self, cfg: ExperimentConfig) -> None:
        try:
            self._build(cfg)
        except (ValueError, RuntimeError) as err:
            self.ui.message = f"not rerun: {err}"
            return
        self.ui.message = f"rerun {cfg.name}, seed {cfg.seed}"

    def view(self) -> ControlsView:
        return controls_view(self.ui, self.draft)

    def process(self) -> None:
        actions, self.ui.actions = self.ui.actions, []
        for action in actions:
            self.act(action)

    def act(self, action: tuple[Any, ...]) -> None:
        kind = action[0]
        if kind == "button":
            self._button(action[1])
            return
        key = action[1]
        spec = next(s for s in specs(self.draft) if s.key == key)
        try:
            if kind == "nudge":
                self.draft = nudge(self.draft, spec, action[2])
            elif kind == "edit":
                self.ui.editing, self.ui.text = key, show(spec, get(self.draft, key))
            else:  # commit
                self.draft = put(self.draft, key, parse(spec, action[2]))
        except ValueError as err:
            self.ui.message = f"{key}: {err}"

    def _button(self, name: str) -> None:
        if name == "apply":
            self.rerun(self.draft)
        elif name == "rerun":
            self.rerun(self.live)
        elif name == "seed":
            self.draft = put(self.draft, "seed", self.draft.seed + 1)
            self.rerun(self.draft)
        elif name == "pause":
            self.ui.paused = not self.ui.paused
        elif name == "step":
            self._step()
        elif name == "revert":
            self.draft = self.live
        elif name == "save":
            self.save_dir.mkdir(parents=True, exist_ok=True)
            path = self.save_dir / f"{self.draft.name}_{timestamp()}.toml"
            path.write_text(to_toml(self.draft))
            self.ui.message = f"saved {path}"
        else:  # fit
            self.cam.fit(*self.ui.world_size)

    def _step(self) -> None:
        """Pause and step one tick, unless a slow step is still running."""
        self.ui.paused = True
        if not self.stepper.lock.acquire(blocking=False):
            self.ui.message = "busy: a step (an LLM call) is still running"
            return
        try:
            self.world.step()
            self.history.record(self.world)
        finally:
            self.stepper.lock.release()

    def paint(self, surface: pygame.Surface, fonts: dict[str, Any]) -> None:
        draw(
            surface,
            self.world,
            self.live.name,
            fonts,
            self.cam,
            self.history,
            self.ui,
            self.draft,
            self.live,
        )

    def close(self) -> None:
        self.stepper.stop(wait=False)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    # Opens a real window and runs the event loop until closed; the pieces it
    # uses (parse_args, Session, present, handle_event) are tested directly.
    from apeks.render import handle_event

    session = Session(parse_args(argv), threaded=True)
    pygame.init()
    screen = pygame.display.set_mode(session.ui.size, pygame.RESIZABLE)
    pygame.display.set_caption("apeks")
    fonts = {
        k: pygame.font.SysFont("dejavusansmono,menlo,consolas,monospace", n)
        for k, n in (("s", 13), ("m", 17), ("l", 24))
    }
    clock = pygame.time.Clock()
    running = True
    while running:
        view = session.view()
        for event in pygame.event.get():
            running = running and handle_event(event, session.ui, session.cam, view)
        session.process()
        session.frame = present(
            screen,
            session.frame,
            session.stepper.lock,
            lambda f: session.paint(f, fonts),
            fonts,
        )
        pygame.display.flip()
        clock.tick(30)
    session.close()
    pygame.quit()
