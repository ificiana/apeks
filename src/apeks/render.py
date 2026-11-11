"""The dashboard: a pan-and-zoom map, a side panel of figures and a status bar."""

import math
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache

import pygame

from apeks.geometry import Vec, add, normalize, scale
from apeks.mapgen import Geography
from apeks.world import World

BG = (11, 17, 27)
PANEL = (19, 27, 40)
CARD = (27, 38, 55)
RULE = (44, 58, 80)
TEXT = (226, 232, 240)
MUTED = (134, 150, 172)
DEEP = (16, 42, 68)
SHALLOW = (30, 72, 104)
LAND = (58, 74, 52)
COAST = (150, 136, 98)
LANE = (98, 150, 196)
CLOSED = (232, 72, 72)
NODE = (150, 178, 206)
PORT = (240, 228, 190)
CURRENT = (70, 130, 170)
STORM = (150, 90, 190)
LADEN = (255, 178, 60)
EMPTY = (120, 220, 230)
GOOD, WARN, BAD = (92, 200, 120), (236, 190, 70), (232, 72, 72)
SERIES = {"delivered": (92, 200, 120), "backlog": (236, 150, 70)}

PANEL_W = 400
BAR_H = 30
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


def layout(size: tuple[int, int]) -> dict[str, pygame.Rect]:
    w, h = size
    panel = min(PANEL_W, w // 2)
    return {
        "map": pygame.Rect(0, 0, w - panel, h - BAR_H),
        "panel": pygame.Rect(w - panel, 0, panel, h - BAR_H),
        "bar": pygame.Rect(0, h - BAR_H, w, BAR_H),
    }


def handle_event(event: pygame.event.Event, ui: Ui, cam: Camera) -> bool:
    """Apply one input event; return False when the window should close."""
    if event.type == pygame.QUIT:
        return False
    if event.type == pygame.VIDEORESIZE:
        ui.size = (event.w, event.h)
        cam.view = tuple(layout(ui.size)["map"])
    elif event.type == pygame.MOUSEWHEEL:
        cam.zoom_at(ui.mouse, ZOOM_STEP**event.y)
    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        ui.dragging = True
    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
        ui.dragging = False
    elif event.type == pygame.MOUSEMOTION:
        ui.mouse = event.pos
        if ui.dragging:
            cam.pan(*event.rel)
    elif event.type == pygame.KEYDOWN:
        return _key(event.key, ui, cam)
    return True


def _key(key: int, ui: Ui, cam: Camera) -> bool:
    centre = (cam.view[0] + cam.view[2] / 2, cam.view[1] + cam.view[3] / 2)
    pans = {
        pygame.K_a: (PAN_STEP, 0.0),
        pygame.K_d: (-PAN_STEP, 0.0),
        pygame.K_w: (0.0, PAN_STEP),
        pygame.K_s: (0.0, -PAN_STEP),
    }
    if key == pygame.K_ESCAPE:
        return False
    if key in pans:
        cam.pan(*pans[key])
    elif key == pygame.K_f:
        cam.fit(*ui.world_size)
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
    at_sea = sum(len(c.vessels) for c in world.convoys.values() if c.state == "sailing")
    calls = sum(m["calls"] for m in s["llm"].values())
    latency = s["mean_latency"]
    return [
        ("delivered", f"{s['deliveries']} / {s['generated']}"),
        ("latency", "-" if latency is None else f"{latency:.0f}"),
        ("backlog", str(s["backlog"])),
        ("near misses", str(s["near_misses"])),
        ("at sea", str(at_sea)),
        ("closed lanes", str(len(world.closed))),
        ("llm calls", str(calls)),
        ("congestion", f"{s['congestion_ticks']:.0f}"),
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
) -> None:
    lay = layout(surface.get_size())
    surface.fill(BG)
    surface.set_clip(lay["map"])
    _draw_map(surface, world, fonts, cam, ui)
    surface.set_clip(None)
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


def _draw_map(surface, world: World, fonts, cam: Camera, ui: Ui) -> None:
    view = pygame.Rect(cam.view)
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
        centre = cam.to_screen(cur.centre)
        pygame.draw.circle(overlay, (*CURRENT, 50), centre, cur.radius * cam.zoom)
        tip = add(cur.centre, scale(normalize(cur.velocity), cur.radius * 0.6))
        pygame.draw.line(overlay, (*CURRENT, 180), centre, cam.to_screen(tip), 2)
    for storm in world.ocean.active_storms(world.t):
        centre, radius = cam.to_screen(storm.centre), storm.radius * cam.zoom
        pygame.draw.circle(overlay, (*STORM, 90), centre, radius)
        pygame.draw.circle(overlay, (*STORM, 200), centre, radius, 2)
    surface.blit(overlay, (0, 0))

    def land(r, c):
        colour = terrain.get((r, c))
        return None if colour in (None, SHALLOW) else colour

    for colour, rect in _cells(geo, cam, view, land):
        surface.fill(colour, rect)

    net = world.network
    lane_w = max(1, round(1.5 * cam.zoom))
    for lane in net.lanes.values():
        if lane.a > lane.b and lane.b + ">" + lane.a in net.lanes:
            continue
        shut = lane.id in world.closed or f"{lane.b}>{lane.a}" in world.closed
        a, b = cam.to_screen(net.nodes[lane.a]), cam.to_screen(net.nodes[lane.b])
        pygame.draw.line(surface, CLOSED if shut else LANE, a, b, lane_w + 2 * shut)
    small = fonts["s"]
    for node, pos in net.nodes.items():
        if node not in net.ports:
            pygame.draw.circle(surface, NODE, cam.to_screen(pos), max(2, cam.zoom * 2))
    for port, waiting, _, _ in port_rows(world):
        pos = cam.to_screen(net.nodes[port])
        ring = GOOD if waiting < 10 else WARN if waiting < 30 else BAD
        r = max(5.0, 7 * cam.zoom)
        pygame.draw.circle(surface, ring, pos, r + 3)
        pygame.draw.circle(surface, PORT, pos, r)
        label = small.render(f"{port}  {waiting}", True, TEXT)
        surface.blit(label, (pos[0] + r + 5, pos[1] - label.get_height() / 2))
    for v in world.vessels.values():
        if world.convoys[v.convoy].state != "sailing":
            continue
        colour = LADEN if v.cargo else EMPTY
        pygame.draw.circle(
            surface, colour, cam.to_screen(v.pos), max(2.5, cam.zoom * 3)
        )
    _draw_minimap(surface, world, geo, cam, view)


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
    pygame.draw.rect(surface, TEXT, seen.clip(box), 1)
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
    y += 4 * (th + gap) + 4
    for key, colour in SERIES.items():
        chart = pygame.Rect(x, y, w, 58)
        _sparkline(surface, chart, list(history.series[key]), colour, key, s)
        y += chart.height + gap
    y += 4
    surface.blit(s.render("PORT     WAITING   SHIPS   QUEUED", True, MUTED), (x, y))
    y += s.get_height() + 4
    rows = port_rows(world)
    room = max(0, (rect.bottom - y - 140) // (s.get_height() + 2))
    for port, waiting, ships, queued in rows[:room]:
        colour = GOOD if waiting < 10 else WARN if waiting < 30 else BAD
        surface.blit(
            s.render(
                f"{port:<8} {waiting:>7}   {ships:>5}   {queued:>6}", True, colour
            ),
            (x, y),
        )
        y += s.get_height() + 2
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


def _draw_bar(surface, rect, world: World, fonts, cam: Camera, ui: Ui) -> None:
    surface.fill(PANEL, rect)
    pygame.draw.line(surface, RULE, rect.topleft, rect.topright)
    state = "PAUSED" if ui.paused else f"x{SPEEDS[ui.speed]}"
    text = (
        f"t {world.t:.0f}   {state}   zoom {cam.zoom:.2f}   "
        f"ports {len(world.network.ports)}   nodes {len(world.network.nodes)}   "
        "drag/WASD pan · wheel/+- zoom · F fit · space pause · up/down speed"
    )
    label = fonts["s"].render(text, True, MUTED)
    surface.blit(label, (rect.x + 12, rect.centery - label.get_height() / 2))


def draw_waiting(surface: pygame.Surface, fonts: dict[str, pygame.font.Font]) -> None:
    """Banner over the last frame while the world is busy (an LLM call)."""
    label = fonts["m"].render("waiting for the LLM...", True, TEXT)
    box = label.get_rect(midtop=(surface.get_width() // 2, 12)).inflate(24, 12)
    pygame.draw.rect(surface, CARD, box, border_radius=6)
    surface.blit(label, label.get_rect(center=box.center))
