"""The dashboard: run controls, a pan-and-zoom map, a panel of figures, a bar."""

import math
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pygame

from apeks.controls import Spec, get, show, specs
from apeks.experiment import ExperimentConfig
from apeks.geometry import Vec, add, normalize, scale
from apeks.mapgen import Geography
from apeks.world import World

# Panel colours come from the theme (see THEMES); the map keeps its own.
BG = (246, 247, 249)
PANEL = (255, 255, 255)
CARD = (242, 244, 247)
RULE = (226, 230, 235)
TEXT = (28, 33, 42)
MUTED = (112, 120, 132)
DEEP = (16, 40, 66)
OPEN_SEA = (19, 34, 52)  # beyond the world's edge
SHALLOW = (28, 82, 108)
LAND = (54, 60, 50)
COAST = (104, 108, 86)
LANE = (128, 156, 186)
CLOSED = (232, 93, 93)
NODE = (128, 156, 186)
PORT = (240, 244, 248)
CURRENT = (110, 168, 210)
STORM = (168, 120, 220)
LADEN = (247, 178, 72)
EMPTY = (104, 214, 204)
MAP_TEXT = (226, 233, 241)  # labels drawn over the sea
GOOD, WARN, BAD = (34, 150, 96), (204, 132, 20), (214, 64, 64)
SERIES = {"delivered": (34, 150, 96), "backlog": (214, 120, 40)}

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "BG": BG,
        "PANEL": PANEL,
        "CARD": CARD,
        "RULE": RULE,
        "TEXT": TEXT,
        "MUTED": MUTED,
        "GOOD": GOOD,
        "WARN": WARN,
        "BAD": BAD,
        "SERIES": SERIES,
    },
    "dark": {
        "BG": (11, 15, 21),
        "PANEL": (18, 23, 31),
        "CARD": (27, 33, 43),
        "RULE": (44, 52, 64),
        "TEXT": (226, 232, 240),
        "MUTED": (136, 146, 160),
        "GOOD": (74, 198, 132),
        "WARN": (240, 176, 72),
        "BAD": (244, 108, 108),
        "SERIES": {"delivered": (74, 198, 132), "backlog": (244, 146, 72)},
    },
}
"""Panel colours by theme name; ``use_theme`` makes one current."""


def use_theme(name: str) -> None:
    """Point the panel colour names at theme ``name``."""
    globals().update(THEMES[name])


PANEL_W = 400
CONTROLS_W = 330
ROW_H = 22
BUTTONS = (
    ("apply", "Apply & rerun"),
    ("rerun", "Rerun"),
    ("seed", "New seed"),
    ("pause", "Pause"),
    ("step", "Step"),
    ("revert", "Revert edits"),
    ("save", "Save TOML"),
    ("fit", "Fit map"),
)
"""Control buttons, by the action name the session runs."""
BAR_H = 30
DRAWER_W = 44  # width of the controls column when shut
SPEEDS = (1, 2, 5, 10, 25, 50)
PAN_STEP = 60.0
ZOOM_STEP = 1.15


@dataclass
class Camera:
    """World-to-screen transform for the map viewport ``view`` (x, y, w, h)."""

    view: tuple[int, int, int, int]
    zoom: float = 1.0
    offset: Vec = (0.0, 0.0)  # screen position of world (0, 0) inside the view
    MIN_ZOOM = 0.1
    MAX_ZOOM = 12.0

    def to_screen(self, p: Vec) -> Vec:
        return (
            self.view[0] + self.offset[0] + p[0] * self.zoom,
            self.view[1] + self.offset[1] + p[1] * self.zoom,
        )

    def to_world(self, s: Vec) -> Vec:
        return (
            (s[0] - self.view[0] - self.offset[0]) / self.zoom,
            (s[1] - self.view[1] - self.offset[1]) / self.zoom,
        )

    def fit(self, width: float, height: float) -> None:
        self.zoom = min(self.view[2] / width, self.view[3] / height)
        self.offset = (
            (self.view[2] - width * self.zoom) / 2,
            (self.view[3] - height * self.zoom) / 2,
        )

    def zoom_at(self, anchor: Vec, factor: float) -> None:
        """Zoom by ``factor``, keeping the world point under ``anchor`` still."""
        fixed = self.to_world(anchor)
        self.zoom = min(self.MAX_ZOOM, max(self.MIN_ZOOM, self.zoom * factor))
        self.offset = (
            anchor[0] - self.view[0] - fixed[0] * self.zoom,
            anchor[1] - self.view[1] - fixed[1] * self.zoom,
        )

    def pan(self, dx: float, dy: float) -> None:
        self.offset = (self.offset[0] + dx, self.offset[1] + dy)


@dataclass
class History:
    """Deliveries and backlog sampled every ``every`` ticks, last ``size`` kept."""

    every: int = 10
    size: int = 600
    ticks: deque[float] = field(init=False)
    series: dict[str, deque[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.ticks = deque(maxlen=self.size)
        self.series = {k: deque(maxlen=self.size) for k in SERIES}

    def record(self, world: World) -> None:
        if world.t % self.every or (self.ticks and self.ticks[-1] == world.t):
            return
        self.ticks.append(world.t)
        self.series["delivered"].append(world.metrics["deliveries"])
        self.series["backlog"].append(sum(len(w) for w in world.waiting.values()))


@dataclass
class Ui:
    """Window state the event loop changes."""

    size: tuple[int, int] = (1600, 900)
    world_size: tuple[float, float] = (1200.0, 800.0)
    speed: int = 0
    paused: bool = False
    dragging: bool = False
    mouse: tuple[int, int] = (0, 0)
    scroll: int = 0  # pixels the control list is scrolled down
    editing: str | None = None  # key of the field being typed into
    text: str = ""
    message: str = ""
    selected: str | None = None  # vessel whose dialog is open
    drawer: bool = False  # controls column open
    theme: str = "dark"  # a key of THEMES; T switches it
    press: tuple[int, int] | None = None  # where the left button went down
    actions: list[tuple[Any, ...]] = field(default_factory=list)
    """Control actions from input, for the session to run."""


def layout(size: tuple[int, int], drawer: bool = True) -> dict[str, pygame.Rect]:
    w, h = size
    panel = min(PANEL_W, w // 3)
    left = min(CONTROLS_W, w // 4) if drawer else DRAWER_W
    return {
        "controls": pygame.Rect(0, 0, left, h - BAR_H),
        "map": pygame.Rect(left, 0, w - panel - left, h - BAR_H),
        "panel": pygame.Rect(w - panel, 0, panel, h - BAR_H),
        "bar": pygame.Rect(0, h - BAR_H, w, BAR_H),
    }


type Action = tuple[Any, ...]


@dataclass
class ControlsView:
    """Where every button and field row of the controls column sits."""

    buttons: list[tuple[pygame.Rect, str, str]]
    rows: list[tuple[pygame.Rect, str | Spec]]  # section headers and fields
    list_area: pygame.Rect
    max_scroll: int
    targets: list[tuple[pygame.Rect, Action]]


def controls_view(ui: Ui, draft: ExperimentConfig) -> ControlsView:
    """Lay out the controls for ``draft``; clamps ``ui.scroll`` to the list."""
    rect = layout(ui.size, ui.drawer)["controls"]
    if not ui.drawer:
        toggle = pygame.Rect(rect.x + 8, rect.y + 10, 28, 28)
        return ControlsView([], [], pygame.Rect(0, 0, 0, 0), 0, [(toggle, ("drawer",))])
    toggle = pygame.Rect(rect.right - 36, rect.y + 8, 28, 28)
    x, w, gap = rect.x + 10, rect.width - 20, 6
    bw, bh = (w - gap) // 2, 26
    y = rect.y + 40
    buttons: list[tuple[pygame.Rect, str, str]] = []
    targets: list[tuple[pygame.Rect, Action]] = [(toggle, ("drawer",))]
    for i, (name, label) in enumerate(BUTTONS):
        r = pygame.Rect(x + (i % 2) * (bw + gap), y + (i // 2) * (bh + gap), bw, bh)
        buttons.append((r, name, label))
        targets.append((r, ("button", name)))
    top = y + (len(BUTTONS) + 1) // 2 * (bh + gap) + 6
    area = pygame.Rect(rect.x, top, rect.width, rect.bottom - top - ROW_H - 6)
    entries: list[str | Spec] = []
    for spec in specs(draft):
        if not entries or spec.section != _section(entries[-1]):
            entries.append(spec.section)
        entries.append(spec)
    max_scroll = max(0, len(entries) * ROW_H - area.height)
    ui.scroll = min(ui.scroll, max_scroll)
    rows = []
    for i, entry in enumerate(entries):
        r = pygame.Rect(x, area.y + i * ROW_H - ui.scroll, w, ROW_H - 2)
        if not area.contains(r):
            continue
        rows.append((r, entry))
        if isinstance(entry, Spec):
            targets += _row_targets(r, entry)
    return ControlsView(buttons, rows, area, max_scroll, targets)


def _section(entry: str | Spec) -> str:
    return entry.section if isinstance(entry, Spec) else entry


def _row_targets(r: pygame.Rect, spec: Spec) -> list[tuple[pygame.Rect, Action]]:
    value = pygame.Rect(r.x + r.width * 11 // 20, r.y, r.width * 9 // 20, r.height)
    if spec.kind == "text":
        return [(value, ("edit", spec.key))]
    minus = pygame.Rect(value.x, r.y, 18, r.height)
    plus = pygame.Rect(value.right - 18, r.y, 18, r.height)
    box = pygame.Rect(minus.right + 2, r.y, plus.x - minus.right - 4, r.height)
    return [
        (minus, ("nudge", spec.key, -1)),
        (box, ("edit", spec.key)),
        (plus, ("nudge", spec.key, 1)),
    ]


def hit(targets: list[tuple[pygame.Rect, Action]], pos: Vec) -> Action | None:
    return next((a for r, a in targets if r.collidepoint(pos)), None)


def handle_event(
    event: pygame.event.Event, ui: Ui, cam: Camera, view: ControlsView | None = None
) -> bool:
    """Apply one input event; return False when the window should close.

    Clicks on controls and control hotkeys go to ``ui.actions``.
    """
    targets = view.targets if view else []
    if event.type == pygame.QUIT:
        return False
    if event.type == pygame.VIDEORESIZE:
        ui.size = (event.w, event.h)
        cam.view = tuple(layout(ui.size, ui.drawer)["map"])
    elif event.type == pygame.MOUSEWHEEL:
        if view and view.list_area.collidepoint(ui.mouse):
            ui.scroll = max(0, ui.scroll - event.y * ROW_H)
        else:
            cam.zoom_at(ui.mouse, ZOOM_STEP**event.y)
    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        ui.editing = None
        action = hit(targets, event.pos)
        if action == ("drawer",):
            _toggle(ui, cam)
        elif action:
            ui.actions.append(action)
        else:
            ui.dragging, ui.press = True, event.pos
    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
        if ui.press and math.dist(ui.press, event.pos) <= CLICK_SLOP:
            ui.actions.append(("pick", event.pos))
        ui.dragging, ui.press = False, None
    elif event.type == pygame.MOUSEMOTION:
        ui.mouse = event.pos
        if ui.dragging:
            cam.pan(*event.rel)
    elif event.type == pygame.KEYDOWN and ui.editing:
        _type(event, ui)
    elif event.type == pygame.KEYDOWN:
        return _key(event.key, ui, cam)
    return True


def _type(event: pygame.event.Event, ui: Ui) -> None:
    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
        ui.actions.append(("commit", ui.editing, ui.text))
        ui.editing = None
    elif event.key == pygame.K_ESCAPE:
        ui.editing = None
    elif event.key == pygame.K_BACKSPACE:
        ui.text = ui.text[:-1]
    elif event.unicode.isprintable():
        ui.text += event.unicode


def _toggle(ui: Ui, cam: Camera) -> None:
    """Open or shut the controls column; the map takes up the difference."""
    ui.drawer = not ui.drawer
    cam.view = tuple(layout(ui.size, ui.drawer)["map"])


HOTKEYS = {pygame.K_r: "apply", pygame.K_n: "seed", pygame.K_PERIOD: "step"}


def _key(key: int, ui: Ui, cam: Camera) -> bool:
    centre = (cam.view[0] + cam.view[2] / 2, cam.view[1] + cam.view[3] / 2)
    pans = {
        pygame.K_a: (PAN_STEP, 0.0),
        pygame.K_d: (-PAN_STEP, 0.0),
        pygame.K_w: (0.0, PAN_STEP),
        pygame.K_s: (0.0, -PAN_STEP),
    }
    if key == pygame.K_ESCAPE and ui.selected:
        ui.selected = None
    elif key == pygame.K_ESCAPE:
        return False
    if key in HOTKEYS:
        ui.actions.append(("button", HOTKEYS[key]))
    elif key in pans:
        cam.pan(*pans[key])
    elif key == pygame.K_c:
        _toggle(ui, cam)
    elif key == pygame.K_f:
        cam.fit(*ui.world_size)
    elif key == pygame.K_t:
        ui.theme = "light" if ui.theme == "dark" else "dark"
    elif key == pygame.K_SPACE:
        ui.paused = not ui.paused
    elif key in (pygame.K_UP, pygame.K_DOWN):
        step = 1 if key == pygame.K_UP else -1
        ui.speed = max(0, min(len(SPEEDS) - 1, ui.speed + step))
    elif key in (pygame.K_EQUALS, pygame.K_MINUS):
        cam.zoom_at(centre, ZOOM_STEP if key == pygame.K_EQUALS else 1 / ZOOM_STEP)
    return True


def tiles(world: World) -> list[tuple[str, str]]:
    s = world.summary()
    calls = sum(m["calls"] for m in s["llm"].values())
    latency = s["mean_latency"]
    return [
        ("delivered", f"{s['deliveries']} / {s['generated']}"),
        ("backlog", str(s["backlog"])),
        ("latency", "-" if latency is None else f"{latency:.0f}"),
        ("near misses", str(s["near_misses"])),
        ("congestion", f"{s['congestion_ticks']:.0f}"),
        ("llm calls", str(calls)),
    ]


def port_rows(world: World) -> list[tuple[str, int, int, int]]:
    """Per port: name, packages waiting, ships in port, convoys queued."""
    rows = []
    for port in world.network.ports:
        here = [
            c for c in world.convoys.values() if c.node == port and c.state != "sailing"
        ]
        ships = sum(len(c.vessels) for c in here)
        queued = len(world.queues.get(port, []))
        rows.append((port, len(world.waiting.get(port, [])), ships, queued))
    return rows


def draw(
    surface: pygame.Surface,
    world: World,
    name: str,
    fonts: dict[str, pygame.font.Font],
    cam: Camera,
    history: History,
    ui: Ui,
    draft: ExperimentConfig | None = None,
    live: ExperimentConfig | None = None,
) -> None:
    """Paint the whole window; the controls show ``draft`` against ``live``."""
    use_theme(ui.theme)
    lay = layout(surface.get_size(), ui.drawer)
    surface.fill(BG)
    surface.set_clip(lay["map"])
    _draw_map(surface, world, fonts, cam, ui)
    surface.set_clip(None)
    surface.fill(PANEL, lay["controls"])
    pygame.draw.line(
        surface, RULE, lay["controls"].topright, lay["controls"].bottomright
    )
    if not ui.drawer:
        _draw_toggle(surface, controls_view(ui, ExperimentConfig()).targets[0][0])
    elif draft is not None and live is not None:
        _draw_controls(surface, lay["controls"], draft, live, fonts, ui)
    _draw_panel(surface, lay["panel"], world, name, fonts, history)
    _draw_bar(surface, lay["bar"], world, fonts, cam, ui)


@lru_cache(maxsize=4)
def _terrain(geo: Geography) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Colour of every shallow-water and land cell; deep water is left out."""
    out = {}
    for r in range(geo.rows):
        for c in range(geo.cols):
            pos = ((c + 0.5) * geo.cell, (r + 0.5) * geo.cell)
            if not geo.land[r][c]:
                if geo.near_land(pos):
                    out[r, c] = SHALLOW
                continue
            inland = all(
                geo.is_land((pos[0] + dx * geo.cell, pos[1] + dy * geo.cell))
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
            )
            out[r, c] = LAND if inland else COAST
    return out


def _cells(
    geo: Geography,
    cam: Camera,
    view: pygame.Rect,
    pick: Callable[[int, int], tuple[int, int, int] | None],
) -> Iterator[tuple[tuple[int, int, int], pygame.Rect]]:
    """Yield the colour and screen rect of each visible cell ``pick`` colours."""
    x0, y0 = cam.to_world(view.topleft)
    x1, y1 = cam.to_world(view.bottomright)
    c0, c1 = max(0, int(x0 // geo.cell)), min(geo.cols, int(x1 // geo.cell) + 1)
    r0, r1 = max(0, int(y0 // geo.cell)), min(geo.rows, int(y1 // geo.cell) + 1)
    size = geo.cell * cam.zoom
    for r in range(r0, r1):
        for c in range(c0, c1):
            colour = pick(r, c)
            if colour is not None:
                x, y = cam.to_screen((c * geo.cell, r * geo.cell))
                yield (
                    colour,
                    pygame.Rect(
                        math.floor(x),
                        math.floor(y),
                        math.ceil(x + size) - math.floor(x),
                        math.ceil(y + size) - math.floor(y),
                    ),
                )


def _arrow(overlay, cam: Camera, centre: Vec, heading: Vec, length: float) -> None:
    """A current: a thin arrow through ``centre`` along ``heading``."""
    tail = cam.to_screen(add(centre, scale(heading, -length)))
    tip = cam.to_screen(add(centre, scale(heading, length)))
    back, side = scale(heading, -10.0), (-heading[1] * 5.0, heading[0] * 5.0)
    wings = [add(add(tip, back), side), add(add(tip, back), scale(side, -1.0))]
    colour = (*CURRENT, 150)
    pygame.draw.line(overlay, colour, tail, tip, 2)
    pygame.draw.polygon(overlay, colour, [tip, *wings])


def _draw_map(surface, world: World, fonts, cam: Camera, ui: Ui) -> None:
    view = pygame.Rect(cam.view)
    surface.fill(OPEN_SEA, view)
    geo = world.geography or Geography.open_sea(*ui.world_size)
    tl, br = cam.to_screen((0.0, 0.0)), cam.to_screen((geo.width, geo.height))
    pygame.draw.rect(surface, DEEP, pygame.Rect(tl, (br[0] - tl[0], br[1] - tl[1])))

    terrain = _terrain(geo)

    def shallow(r, c):
        colour = terrain.get((r, c))
        return colour if colour == SHALLOW else None

    for colour, rect in _cells(geo, cam, view, shallow):
        surface.fill(colour, rect)
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for cur in world.ocean.currents:
        _arrow(overlay, cam, cur.centre, normalize(cur.velocity), cur.radius * 0.5)
    for storm in world.ocean.active_storms(world.t):
        centre, radius = cam.to_screen(storm.centre), storm.radius * cam.zoom
        pygame.draw.circle(overlay, (*STORM, 70), centre, radius)
        pygame.draw.circle(overlay, (*STORM, 170), centre, radius, 2)
    surface.blit(overlay, (0, 0))

    def land(r, c):
        colour = terrain.get((r, c))
        return None if colour in (None, SHALLOW) else colour

    for colour, rect in _cells(geo, cam, view, land):
        surface.fill(colour, rect)

    net = world.network
    lane_w = max(1, round(1.2 * cam.zoom))
    for lane in net.lanes.values():
        if lane.a > lane.b and lane.b + ">" + lane.a in net.lanes:
            continue
        shut = lane.id in world.closed or f"{lane.b}>{lane.a}" in world.closed
        a, b = cam.to_screen(net.nodes[lane.a]), cam.to_screen(net.nodes[lane.b])
        pygame.draw.line(surface, CLOSED if shut else LANE, a, b, lane_w + 2 * shut)
    small = fonts["s"]
    for node, pos in net.nodes.items():
        if node not in net.ports:
            pygame.draw.circle(
                surface, NODE, cam.to_screen(pos), max(2, cam.zoom * 1.5)
            )
    for port, waiting, _, _ in port_rows(world):
        pos = cam.to_screen(net.nodes[port])
        ring = GOOD if waiting < 10 else WARN if waiting < 30 else BAD
        r = max(5.0, 6 * cam.zoom)
        pygame.draw.circle(surface, ring, pos, r + 3)
        pygame.draw.circle(surface, PORT, pos, r)
        label = small.render(port, True, MAP_TEXT)
        surface.blit(label, (pos[0] + r + 5, pos[1] - label.get_height() / 2))
    for v in world.vessels.values():
        if world.convoys[v.convoy].state != "sailing":
            continue
        colour = LADEN if v.cargo else EMPTY
        r = max(4.0, cam.zoom * 3.5)
        pygame.draw.circle(surface, OPEN_SEA, cam.to_screen(v.pos), r + 1.5)
        pygame.draw.circle(surface, colour, cam.to_screen(v.pos), r)
    _draw_minimap(surface, world, geo, cam, view)
    _draw_legend(surface, view, small)
    if ui.selected in world.vessels:
        _draw_dialog(surface, world, ui.selected, fonts, cam)


LEGEND = (
    ("ship, laden", LADEN),
    ("ship, empty", EMPTY),
    ("port (ring: backlog)", PORT),
    ("lane", LANE),
    ("lane closed", CLOSED),
    ("storm", STORM),
    ("current", CURRENT),
)


def legend(
    view: pygame.Rect, font: pygame.font.Font
) -> list[tuple[str, tuple[int, int, int], pygame.Rect]]:
    """Label, colour and swatch rect of each map mark, top left of ``view``."""
    line = font.get_height() + 6
    size = font.get_height() - 4
    return [
        (label, colour, pygame.Rect(view.x + 22, view.y + 22 + i * line, size, size))
        for i, (label, colour) in enumerate(LEGEND)
    ]


def _draw_legend(surface, view: pygame.Rect, font) -> None:
    entries = legend(view, font)
    width = max(font.size(label)[0] for label, _, _ in entries)
    first, last = entries[0][2], entries[-1][2]
    box = pygame.Rect(
        first.x - 10, first.y - 10, first.width + width + 30, last.bottom - first.y + 20
    )
    shade = pygame.Surface(box.size, pygame.SRCALPHA)
    shade.fill((*OPEN_SEA, 200))
    surface.blit(shade, box)
    for label, colour, r in entries:
        surface.fill(colour, r)
        text = font.render(label, True, MAP_TEXT)
        surface.blit(text, (r.right + 10, r.centery - text.get_height() / 2))


CLICK_SLOP = 4  # pixels a click may move and still count as a click
PICK_RADIUS = 10  # pixels around a ship that select it


def pick(world: World, cam: Camera, pos: Vec) -> str | None:
    """The sailing ship nearest ``pos`` on screen, if within reach."""
    best, near = None, PICK_RADIUS
    for v in world.vessels.values():
        if world.convoys[v.convoy].state != "sailing":
            continue
        d = math.dist(cam.to_screen(v.pos), pos)
        if d <= near:
            best, near = v.id, d
    return best


def ship_info(world: World, vid: str) -> list[str]:
    """Dialog lines for vessel ``vid``: where it is going and what it carries."""
    v = world.vessels.get(vid)
    if v is None:
        return []
    convoy = world.convoys[v.convoy]
    lines = [f"{v.id}  convoy {convoy.id}  {convoy.state}"]
    if convoy.lane:
        lane = world.network.lanes[convoy.lane]
        lines.append(f"on {lane.a} > {lane.b}, next {lane.b}")
    else:
        lines.append(f"at {convoy.node}")
    if convoy.destination:
        lines.append(f"bound for {convoy.destination}")
    lines.append(f"cargo {len(v.cargo)}/{v.capacity}")
    drops: dict[str, int] = {}
    for p in v.cargo:
        drops[p.destination] = drops.get(p.destination, 0) + 1
    lines += [f"  {n} for {port}" for port, n in sorted(drops.items())] or ["  (empty)"]
    return lines


def _draw_dialog(surface, world: World, vid: str, fonts, cam: Camera) -> None:
    font = fonts["s"]
    x, y = cam.to_screen(world.vessels[vid].pos)
    pygame.draw.circle(surface, MAP_TEXT, (x, y), PICK_RADIUS, 2)
    rows = [font.render(t, True, TEXT) for t in ship_info(world, vid)]
    lh = font.get_linesize()
    box = pygame.Rect(0, 0, max(r.get_width() for r in rows) + 20, lh * len(rows) + 16)
    box.topleft = (x + 12, y + 12)
    box.clamp_ip(pygame.Rect(cam.view))
    surface.fill(PANEL, box)
    pygame.draw.rect(surface, RULE, box, 1)
    for i, r in enumerate(rows):
        surface.blit(r, (box.x + 10, box.y + 8 + i * lh))


@lru_cache(maxsize=4)
def _minimap(geo: Geography, w: int, h: int) -> pygame.Surface:
    img = pygame.Surface((geo.cols, geo.rows))
    img.fill(DEEP)
    for r in range(geo.rows):
        for c in range(geo.cols):
            if geo.land[r][c]:
                img.set_at((c, r), LAND)
    return pygame.transform.scale(img, (w, h))


def _draw_minimap(surface, world: World, geo: Geography, cam: Camera, view) -> None:
    w = min(220, view.width // 4)
    h = round(w * geo.height / geo.width)
    box = pygame.Rect(view.right - w - 12, view.bottom - h - 12, w, h)
    k = w / geo.width
    surface.blit(_minimap(geo, w, h), box)
    for port in world.network.ports:
        x, y = world.network.nodes[port]
        pygame.draw.circle(surface, PORT, (box.x + x * k, box.y + y * k), 2)
    a, b = cam.to_world(view.topleft), cam.to_world(view.bottomright)
    seen = pygame.Rect(
        box.x + a[0] * k, box.y + a[1] * k, (b[0] - a[0]) * k, (b[1] - a[1]) * k
    )
    pygame.draw.rect(surface, MAP_TEXT, seen.clip(box), 1)
    pygame.draw.rect(surface, RULE, box, 1)


def _draw_panel(
    surface, rect, world: World, name: str, fonts, history: History
) -> None:
    surface.fill(PANEL, rect)
    pygame.draw.line(surface, RULE, rect.topleft, rect.bottomleft)
    s, m, big = fonts["s"], fonts["m"], fonts["l"]
    x, y, w = rect.x + 14, rect.y + 12, rect.width - 28
    surface.blit(big.render(name, True, TEXT), (x, y))
    y += big.get_height() + 8
    cols, gap = 2, 8
    tw, th = (w - gap) // cols, m.get_height() + s.get_height() + 14
    for i, (label, value) in enumerate(tiles(world)):
        card = pygame.Rect(
            x + (i % cols) * (tw + gap), y + (i // cols) * (th + gap), tw, th
        )
        pygame.draw.rect(surface, CARD, card, border_radius=6)
        surface.blit(s.render(label.upper(), True, MUTED), (card.x + 8, card.y + 6))
        surface.blit(
            m.render(value, True, TEXT), (card.x + 8, card.y + 8 + s.get_height())
        )
    y += -(-len(tiles(world)) // cols) * (th + gap) + 4
    for key, colour in SERIES.items():
        chart = pygame.Rect(x, y, w, 58)
        _sparkline(surface, chart, list(history.series[key]), colour, key, s)
        y += chart.height + gap
    y += 4
    cols_x = (x, x + w * 0.36, x + w * 0.58, x + w * 0.78)
    for cx, head in zip(cols_x, ("PORT", "WAITING", "SHIPS", "QUEUED"), strict=True):
        surface.blit(s.render(head, True, MUTED), (cx, y))
    y += s.get_height() + 6
    rows = port_rows(world)
    room = max(0, (rect.bottom - y - 140) // (s.get_height() + 2))
    for port, waiting, ships, queued in rows[:room]:
        colour = GOOD if waiting < 10 else WARN if waiting < 30 else BAD
        cells = (port, waiting, ships, queued)
        for cx, cell, ink in zip(
            cols_x, cells, (TEXT, colour, TEXT, TEXT), strict=True
        ):
            surface.blit(s.render(str(cell), True, ink), (cx, y))
        y += s.get_height() + 4
    if len(rows) > room:
        surface.blit(
            s.render(f"... {len(rows) - room} more ports", True, MUTED), (x, y)
        )
        y += s.get_height() + 2
    y += 8
    surface.blit(s.render("EVENTS", True, MUTED), (x, y))
    y += s.get_height() + 4
    for line in tail(world.log, rect.bottom - y - 6, s.get_height() + 2):
        surface.blit(s.render(line[:56], True, TEXT), (x, y))
        y += s.get_height() + 2


def tail(lines: list[str], height: float, line_height: float) -> list[str]:
    """The last lines of ``lines`` that fit in ``height``."""
    n = max(0, int(height // line_height))
    return lines[len(lines) - n :] if n else []


def _sparkline(surface, rect, values: list[float], colour, label: str, font) -> None:
    pygame.draw.rect(surface, CARD, rect, border_radius=6)
    top = max(values, default=0)
    now = f"{values[-1]:.0f}" if values else "-"
    surface.blit(
        font.render(f"{label.upper()}  {now}", True, MUTED), (rect.x + 8, rect.y + 5)
    )
    if len(values) < 2:
        return
    inner = rect.inflate(-16, -font.get_height() - 14).move(
        0, font.get_height() // 2 + 3
    )
    span = max(top, 1)
    points = [
        (
            inner.x + inner.width * i / (len(values) - 1),
            inner.bottom - inner.height * v / span,
        )
        for i, v in enumerate(values)
    ]
    pygame.draw.lines(surface, colour, False, points, 2)


def _draw_toggle(surface, r: pygame.Rect) -> None:
    """Three bars: the button that opens the controls."""
    pygame.draw.rect(surface, CARD, r, border_radius=6)
    for dy in (-6, 0, 6):
        y = r.centery + dy
        pygame.draw.line(surface, TEXT, (r.x + 7, y), (r.right - 8, y), 2)


def _draw_controls(
    surface, rect, draft: ExperimentConfig, live: ExperimentConfig, fonts, ui: Ui
) -> None:
    s, m = fonts["s"], fonts["m"]
    view = controls_view(ui, draft)
    surface.blit(m.render("Controls", True, TEXT), (rect.x + 10, rect.y + 12))
    _draw_toggle(surface, view.targets[0][0])
    edited = draft != live
    for r, name, label in view.buttons:
        if name == "pause" and ui.paused:
            label = "Resume"
        pygame.draw.rect(surface, CARD, r, border_radius=5)
        if name == "apply" and edited:
            pygame.draw.rect(surface, WARN, r, 2, border_radius=5)
        text = s.render(label, True, TEXT)
        surface.blit(text, text.get_rect(center=r.center))
    for r, entry in view.rows:
        if isinstance(entry, str):
            surface.blit(s.render(entry.upper(), True, MUTED), (r.x, r.y + 6))
            continue
        value = get(draft, entry.key)
        colour = WARN if value != get(live, entry.key) else TEXT
        label = entry.key.split(".")[-1]
        surface.blit(s.render(label, True, colour), (r.x + 8, r.y + 4))
        for box, action in _row_targets(r, entry):
            pygame.draw.rect(surface, CARD, box, border_radius=4)
            if action[0] == "nudge":
                glyph = "<" if action[2] < 0 else ">"
            elif ui.editing == entry.key:
                pygame.draw.rect(surface, TEXT, box, 1, border_radius=4)
                glyph = ui.text + "|"
            else:
                glyph = show(entry, value)
            text = s.render(glyph[-24:], True, colour)
            surface.blit(text, text.get_rect(center=box.center))
    if ui.message:
        line = s.render(ui.message[:48], True, WARN)
        surface.blit(line, (rect.x + 10, rect.bottom - ROW_H))


def _draw_bar(surface, rect, world: World, fonts, cam: Camera, ui: Ui) -> None:
    surface.fill(PANEL, rect)
    pygame.draw.line(surface, RULE, rect.topleft, rect.topright)
    state = "PAUSED" if ui.paused else f"x{SPEEDS[ui.speed]}"
    text = (
        f"t {world.t:.0f}  ·  {state}  ·  {len(world.network.ports)} ports"
        "        space pause  ·  C controls  ·  F fit  ·  R rerun  ·  N new seed"
        "  ·  T theme"
    )
    label = fonts["s"].render(text, True, MUTED)
    surface.blit(label, (rect.x + 12, rect.centery - label.get_height() / 2))


def draw_waiting(surface: pygame.Surface, fonts: dict[str, pygame.font.Font]) -> None:
    """Banner over the last frame while the world is busy (an LLM call)."""
    label = fonts["m"].render("waiting for the LLM...", True, TEXT)
    box = label.get_rect(midtop=(surface.get_width() // 2, 12)).inflate(24, 12)
    pygame.draw.rect(surface, CARD, box, border_radius=6)
    surface.blit(label, label.get_rect(center=box.center))
