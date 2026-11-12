"""Replay a recorded run: ``replay [logs]``.

Given one ``trace_*.jsonl`` (or its ``out_*.json``), the run opens at once.
Given a folder, the window lists every ``trace_*.jsonl`` under it; pick one with the
arrows and Enter, or a click. The run is simulated again from the config in
the trace's ``meta`` record. Every LLM call is answered at once with the reply
recorded in the trace, so the replay never touches a model and matches the
original run. Each tick that has a recorded snapshot is compared with it, and
the dashboard says whether the replay still matches.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import pygame

from apeks.experiment import ExperimentConfig, NewsSettings, build_world
from apeks.trace import _plain
from apeks.world import World

SNAPSHOT_PREFIX = '"kind": "snapshot"'


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    known = {f.name for f in fields(ExperimentConfig)}
    values = {k: v for k, v in data.items() if k in known}
    if "news" in values:
        values["news"] = NewsSettings(**values["news"])
    return ExperimentConfig(**values)


@dataclass
class ReplayClient:
    """Answers each LLM call with the next recorded request, instantly.

    A recorded failure is raised again. A prompt that differs from the
    recorded one counts as a mismatch, since the run has left the recording.
    """

    remaining: list[dict[str, Any]]
    mismatches: int = 0
    usage: dict[str, float] = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    )

    def complete(self, prompt: str, schema: dict[str, Any]) -> str:
        if not self.remaining:
            self.mismatches += 1
            raise OSError("no recorded reply")
        entry = self.remaining.pop(0)
        if entry.get("prompt") != prompt:
            self.mismatches += 1
        if "reply" in entry:
            return entry["reply"]
        raise OSError(entry.get("error", "recorded failure"))


def _digest(line: str) -> str:
    return hashlib.sha1(line.encode()).hexdigest()


@dataclass
class SnapshotCheck:
    """A tracer that compares the replay's snapshots with the recorded ones.

    It keeps only a digest of each recorded snapshot line, so a large trace
    costs little memory.
    """

    expected: dict[float, str]
    checked: int = 0
    diverged_at: float | None = None

    def wants_snapshot(self, t: float) -> bool:
        return round(t, 3) in self.expected

    def emit(self, t: float, kind: str, **data: Any) -> None:
        if kind != "snapshot":
            return
        t = round(t, 3)
        line = json.dumps({"t": t, "kind": kind, **_plain(data)})
        self.checked += 1
        if _digest(line) != self.expected[t] and self.diverged_at is None:
            self.diverged_at = t


@dataclass
class Recording:
    cfg: ExperimentConfig
    requests: list[dict[str, Any]]
    snapshots: dict[float, str]
    last: tuple[ReplayClient, SnapshotCheck] | None = None

    def world(self, cfg: ExperimentConfig) -> World:
        """A fresh world for ``cfg`` that replays the recorded LLM replies."""
        client = ReplayClient(list(self.requests))
        check = SnapshotCheck(self.snapshots)
        world = build_world(cfg, client)
        world.tracer = check  # type: ignore[assignment]
        self.last = (client, check)
        return world


def load(path: Path) -> Recording:
    """Read a trace, parsing only the meta and LLM records."""
    cfg: ExperimentConfig | None = None
    requests: list[dict[str, Any]] = []
    snapshots: dict[float, str] = {}
    with path.open() as f:
        for raw in f:
            line = raw.rstrip("\n")
            head = line[:64]
            if SNAPSHOT_PREFIX in head:
                t = float(json.loads(line[: line.index(",")] + "}")["t"])
                snapshots[t] = _digest(line)
            elif '"kind": "policy_request"' in head:
                requests.append(json.loads(line))
            elif '"kind": "meta"' in head:
                cfg = config_from_dict(json.loads(line)["config"])
    if cfg is None:
        raise ValueError(f"{path} has no meta record")
    return Recording(cfg, requests, snapshots)


def status(client: ReplayClient, check: SnapshotCheck) -> str:
    if check.diverged_at is not None:
        return f"replay diverged at t={check.diverged_at:.0f}"
    if client.mismatches:
        return f"replay diverged: {client.mismatches} LLM prompts differ"
    return f"replay matches ({check.checked} snapshots)"


def find_traces(root: Path) -> list[Path]:
    """Every trace under ``root``, newest report folder first."""
    return sorted(
        root.rglob("trace_*.jsonl"),
        key=lambda p: (str(p.parent), p.name),
        reverse=True,
    )


def given_trace(path: Path) -> Path | None:
    """The trace to open at once when ``path`` names one run, else ``None``.

    An ``out_*.json`` file stands for the trace saved next to it.
    """
    if path.is_file() and path.name.startswith("out_") and path.suffix == ".json":
        path = path.with_name("trace_" + path.name[4:] + "l")
    return path if path.is_file() else None


@dataclass
class Picker:
    """A scrolling list of trace files."""

    TOP = 60
    ROW = 22

    items: list[Path]
    root: Path
    rows: int = 30
    index: int = 0
    top: int = 0

    def move(self, delta: int) -> None:
        if not self.items:
            return
        self.index = max(0, min(len(self.items) - 1, self.index + delta))
        if self.index < self.top:
            self.top = self.index
        elif self.index >= self.top + self.rows:
            self.top = self.index - self.rows + 1


def picker_event(picker: Picker, event: pygame.event.Event) -> Path | None:
    """Handle one event; return the chosen trace, if any."""
    if event.type == pygame.KEYDOWN:
        steps = {
            pygame.K_UP: -1,
            pygame.K_DOWN: 1,
            pygame.K_PAGEUP: -picker.rows,
            pygame.K_PAGEDOWN: picker.rows,
            pygame.K_HOME: -len(picker.items),
            pygame.K_END: len(picker.items),
        }
        if event.key in steps:
            picker.move(steps[event.key])
        elif event.key == pygame.K_RETURN and picker.items:
            return picker.items[picker.index]
    elif event.type == pygame.MOUSEWHEEL:
        picker.move(-event.y)
    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        y = event.pos[1] - Picker.TOP
        row = picker.top + y // Picker.ROW
        if y >= 0 and row < len(picker.items):
            return picker.items[row]
    return None


def draw_picker(surface: pygame.Surface, fonts: dict[str, Any], picker: Picker) -> None:
    picker.rows = max(1, (surface.get_height() - Picker.TOP) // Picker.ROW)
    surface.fill((14, 22, 34))
    title = "pick a trace to replay  (arrows, Enter or click; Esc quits)"
    surface.blit(fonts["l"].render(title, True, (230, 236, 244)), (16, 16))
    if not picker.items:
        hint = f"no trace_*.jsonl under {picker.root}"
        surface.blit(fonts["m"].render(hint, True, (240, 180, 90)), (16, Picker.TOP))
        return
    shown = picker.items[picker.top : picker.top + picker.rows]
    for i, path in enumerate(shown, start=picker.top):
        y = Picker.TOP + (i - picker.top) * Picker.ROW
        if i == picker.index:
            pygame.draw.rect(
                surface, (40, 70, 110), (8, y, surface.get_width() - 16, Picker.ROW)
            )
        text = str(path.relative_to(picker.root))
        surface.blit(fonts["m"].render(text, True, (220, 228, 238)), (16, y + 2))


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    # Opens a real window and runs the event loop; the pieces it uses (load,
    # Recording.world, status, Picker, picker_event, draw_picker, Session) are
    # tested directly.
    from apeks.app import Session, present
    from apeks.render import handle_event

    parser = argparse.ArgumentParser(prog="replay", description=__doc__)
    parser.add_argument(
        "logs", nargs="?", type=Path, default=Path("logs"), help="folder or trace"
    )
    root = parser.parse_args(argv).logs
    pygame.init()
    screen = pygame.display.set_mode((1600, 900), pygame.RESIZABLE)
    pygame.display.set_caption("apeks replay")
    fonts = {
        k: pygame.font.SysFont("inter,helvetica,arial,dejavusans,sans", n)
        for k, n in (("s", 14), ("m", 18), ("l", 26))
    }
    clock = pygame.time.Clock()
    chosen = given_trace(root)
    picker = Picker([] if chosen else find_traces(root), root)
    while chosen is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                return
            chosen = chosen or picker_event(picker, event)
        draw_picker(screen, fonts, picker)
        pygame.display.flip()
        clock.tick(30)
    draw_picker(screen, fonts, picker)
    screen.blit(fonts["l"].render("loading...", True, (240, 180, 90)), (16, 860))
    pygame.display.flip()
    rec = load(chosen)
    session = Session(
        rec.cfg, size=screen.get_size(), threaded=True, world_factory=rec.world
    )
    pygame.display.set_caption(f"apeks replay: {chosen.name}")
    running = True
    while running:
        view = session.view()
        for event in pygame.event.get():
            running = running and handle_event(event, session.ui, session.cam, view)
        session.process()
        if rec.last is not None:
            session.ui.message = status(*rec.last)
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
