import pygame
import pytest

from apeks.experiment import ExperimentConfig, build_world
from apeks.mapgen import Geography
from apeks.render import (
    CLOSED,
    EMPTY,
    LADEN,
    LAND,
    OPEN_SEA,
    PANEL,
    STORM,
    TEXT,
    Camera,
    History,
    Ui,
    draw,
    handle_event,
    layout,
    pick,
    port_rows,
    ship_info,
    tail,
    tiles,
)


def test_camera_maps_both_ways_and_zooms_about_the_cursor() -> None:
    cam = Camera(view=(10, 20, 400, 300))
    cam.fit(800.0, 600.0)
    assert cam.zoom == 0.5
    assert cam.to_screen((0.0, 0.0)) == (10.0, 20.0)
    assert cam.to_world(cam.to_screen((123.0, 45.0))) == pytest.approx((123.0, 45.0))
    anchor = (110.0, 70.0)
    before = cam.to_world(anchor)
    cam.zoom_at(anchor, 2.0)
    assert cam.zoom == 1.0 and cam.to_world(anchor) == pytest.approx(before)
    cam.zoom_at(anchor, 1000.0)
    assert cam.zoom == Camera.MAX_ZOOM
    cam.zoom_at(anchor, 1e-6)
    assert cam.zoom == Camera.MIN_ZOOM


def test_camera_fit_centres_the_wider_axis_and_pan_moves_by_pixels() -> None:
    cam = Camera(view=(0, 0, 400, 400))
    cam.fit(800.0, 400.0)
    assert cam.zoom == 0.5 and cam.to_screen((0.0, 200.0)) == (0.0, 200.0)
    assert cam.to_screen((0.0, 0.0)) == (0.0, 100.0)
    cam.pan(10.0, -20.0)
    assert cam.to_screen((0.0, 0.0)) == (10.0, 80.0)


def test_history_samples_every_n_ticks_and_keeps_a_window() -> None:
    world = build_world(ExperimentConfig(map="classic"))
    hist = History(every=2, size=3)
    for _ in range(10):
        world.step()
        hist.record(world)
    assert list(hist.ticks) == [6.0, 8.0, 10.0]
    assert len(hist.series["backlog"]) == 3 and len(hist.series["delivered"]) == 3
    hist.record(world)  # same tick again: no new sample
    assert list(hist.ticks) == [6.0, 8.0, 10.0]


def test_layout_puts_the_map_left_panel_right_and_bar_below() -> None:
    lay = layout((1600, 900))
    assert lay["controls"].left == 0 and lay["map"].left == lay["controls"].right
    assert lay["map"].right == lay["panel"].left
    assert lay["panel"].right == 1600 and lay["bar"].bottom == 900
    assert lay["map"].bottom == lay["bar"].top == lay["panel"].bottom
    small = layout((500, 300))
    assert small["map"].width > 0 and small["panel"].width > 0
    assert small["controls"].width > 0


def test_tiles_and_port_rows_summarise_the_world() -> None:
    world = build_world(ExperimentConfig(map="classic", routing="llm", llm="stub"))
    world.step()
    world.closed = {"W2>W3"}
    t = dict(tiles(world))
    s = world.summary()
    assert t["delivered"] == f"{s['deliveries']} / {s['generated']}"
    assert t["closed lanes"] == "1"
    assert set(t) >= {"latency", "backlog", "near misses", "at sea", "llm calls"}
    assert t["llm calls"] == "1"
    rows = port_rows(world)
    assert [r[0] for r in rows] == ["P1", "P2", "P3", "P4", "P5"]
    assert sum(r[2] for r in rows) + int(t["at sea"]) == 15


def test_events_pan_zoom_fit_pause_speed_and_quit() -> None:
    pygame.init()
    cam = Camera(view=(0, 0, 400, 300))
    cam.fit(400.0, 300.0)
    ui = Ui(world_size=(400.0, 300.0))
    ev = pygame.event.Event
    assert handle_event(
        ev(pygame.MOUSEMOTION, pos=(100, 100), rel=(0, 0), buttons=(0, 0, 0)), ui, cam
    )
    assert handle_event(ev(pygame.MOUSEWHEEL, x=0, y=1), ui, cam)
    assert cam.zoom > 1.0 and cam.to_world((100, 100)) == pytest.approx((100.0, 100.0))
    handle_event(ev(pygame.MOUSEWHEEL, x=0, y=-1), ui, cam)
    assert cam.zoom == pytest.approx(1.0)
    handle_event(ev(pygame.MOUSEBUTTONDOWN, pos=(100, 100), button=1), ui, cam)
    handle_event(
        ev(pygame.MOUSEMOTION, pos=(110, 90), rel=(10, -10), buttons=(1, 0, 0)), ui, cam
    )
    handle_event(ev(pygame.MOUSEBUTTONUP, pos=(110, 90), button=1), ui, cam)
    handle_event(
        ev(pygame.MOUSEMOTION, pos=(120, 90), rel=(10, 0), buttons=(0, 0, 0)), ui, cam
    )
    assert cam.to_screen((0.0, 0.0)) == pytest.approx((10.0, -10.0))
    for key, check in (
        (pygame.K_d, lambda: cam.to_screen((0.0, 0.0))[0] < 10.0),
        (pygame.K_a, lambda: cam.to_screen((0.0, 0.0))[0] == pytest.approx(10.0)),
        (pygame.K_s, lambda: cam.to_screen((0.0, 0.0))[1] < -10.0),
        (pygame.K_w, lambda: cam.to_screen((0.0, 0.0))[1] == pytest.approx(-10.0)),
        (pygame.K_f, lambda: cam.to_screen((0.0, 0.0)) == (0.0, 0.0)),
        (pygame.K_SPACE, lambda: ui.paused),
        (pygame.K_UP, lambda: ui.speed == 1),
        (pygame.K_DOWN, lambda: ui.speed == 0),
        (pygame.K_EQUALS, lambda: cam.zoom > 1.0),
        (pygame.K_MINUS, lambda: cam.zoom == pytest.approx(1.0)),
        (pygame.K_x, lambda: True),
    ):
        assert handle_event(ev(pygame.KEYDOWN, key=key), ui, cam)
        assert check(), key
    assert handle_event(ev(pygame.VIDEORESIZE, w=800, h=600, size=(800, 600)), ui, cam)
    assert ui.size == (800, 600) and cam.view == tuple(layout((800, 600))["map"])
    assert handle_event(ev(pygame.MOUSEBUTTONDOWN, pos=(1, 1), button=3), ui, cam)
    assert handle_event(ev(pygame.ACTIVEEVENT, gain=1, state=1), ui, cam)
    assert not ui.dragging
    assert not handle_event(ev(pygame.KEYDOWN, key=pygame.K_ESCAPE), ui, cam)
    assert not handle_event(ev(pygame.QUIT), ui, cam)
    pygame.quit()


def test_draw_paints_land_ocean_lanes_storms_and_vessels(fonts) -> None:
    world = build_world(ExperimentConfig(seed=24008))
    hist = History(every=50)
    for _ in range(300):
        world.step()
        hist.record(world)
    surface = pygame.Surface((1600, 900))
    ui = Ui(world_size=(world.geography.width, world.geography.height))
    cam = Camera(view=tuple(layout(ui.size)["map"]))
    cam.fit(*ui.world_size)
    lane = next(iter(world.network.lanes.values()))
    world.closed = {lane.id}
    draw(surface, world, "classical", fonts, cam, hist, ui)

    def px(p):
        x, y = cam.to_screen(p)
        return surface.get_at((int(x), int(y)))[:3]

    geo = world.geography
    inland = [
        ((c + 0.5) * geo.cell, (r + 0.5) * geo.cell)
        for r in range(geo.rows)
        for c in range(geo.cols)
        if all(
            geo.land[min(geo.rows - 1, max(0, r + dr))][
                min(geo.cols - 1, max(0, c + dc))
            ]
            for dc in (-1, 0, 1)
            for dr in (-1, 0, 1)
        )
    ]
    view = pygame.Rect(cam.view)
    inland = [p for p in inland if view.collidepoint(cam.to_screen(p))]
    painted = [p for p in inland if px(p) == LAND]
    assert len(painted) > 0.8 * len(inland) > 0  # overlays cover a few
    net = world.network
    a, b = net.nodes[lane.a], net.nodes[lane.b]
    assert px(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)) == CLOSED
    sailing = [
        v for v in world.vessels.values() if world.convoys[v.convoy].state == "sailing"
    ]
    assert sailing and px(sailing[0].pos) in (LADEN, EMPTY)

    storm = world.ocean.storms[0]
    calm = [
        (
            storm.centre[0] + dx * storm.radius / 4,
            storm.centre[1] + dy * storm.radius / 4,
        )
        for dx in range(-3, 4)
        for dy in range(-3, 4)
    ]
    calm = [p for p in calm if pygame.Rect(cam.view).collidepoint(cam.to_screen(p))]
    before = {p: px(p) for p in calm}
    world.t = (storm.start + storm.end) / 2
    draw(surface, world, "classical", fonts, cam, hist, ui)
    changed = [p for p in calm if px(p) != before[p]]
    assert changed and all(not geo.is_land(p) for p in changed)
    assert STORM != LAND


def test_draw_handles_an_open_sea_without_history(fonts) -> None:
    world = build_world(ExperimentConfig(map="classic"))
    world.geography = None
    surface = pygame.Surface((900, 420))
    ui = Ui(size=(900, 420), world_size=(1200.0, 800.0), paused=True)
    cam = Camera(view=tuple(layout(ui.size)["map"]))
    cam.fit(*ui.world_size)
    world.log = [f"event {i}" for i in range(50)]
    draw(surface, world, "classical", fonts, cam, History(), ui)
    panel = layout(ui.size)["panel"]
    below = pygame.Rect(panel.x, panel.bottom, panel.width, 30)
    assert all(surface.get_at(p)[:3] != TEXT for p in _pixels(below))
    tall = pygame.Surface((900, 1400))
    ui.size = (900, 1400)
    cam.view = tuple(layout(ui.size)["map"])
    draw(tall, world, "classical", fonts, cam, History(), ui)
    panel = layout(ui.size)["panel"]
    assert any(tall.get_at(p)[:3] == TEXT for p in _pixels(panel))
    assert isinstance(Geography.open_sea(1.0, 1.0), Geography)


def test_draw_paints_the_map_beyond_the_world_as_open_sea(fonts) -> None:
    world = build_world(ExperimentConfig(map="classic"))
    ui = Ui(size=(900, 420), world_size=(1200.0, 800.0), paused=True)
    cam = Camera(view=tuple(layout(ui.size)["map"]))
    cam.fit(*ui.world_size)
    cam.pan(0, 2000)  # push the world off the bottom of the view
    surface = pygame.Surface(ui.size)
    draw(surface, world, "classical", fonts, cam, History(), ui)
    view = pygame.Rect(cam.view)
    assert surface.get_at((view.centerx, view.top + 5))[:3] == OPEN_SEA


def _pixels(rect):
    return [
        (x, y)
        for x in range(rect.left, rect.right)
        for y in range(rect.top, rect.bottom)
    ]


def test_tail_shows_only_the_lines_that_fit() -> None:
    log = ["a", "b", "c"]
    assert tail(log, 0, 10) == []
    assert tail(log, -5, 10) == []
    assert tail(log, 25, 10) == ["b", "c"]
    assert tail(log, 100, 10) == log


def test_controls_view_lays_out_buttons_and_scrolls_field_rows() -> None:
    from apeks.controls import specs
    from apeks.experiment import ExperimentConfig
    from apeks.render import BUTTONS, controls_view, hit

    ui = Ui(size=(1600, 900))
    cfg = ExperimentConfig()
    view = controls_view(ui, cfg)
    names = [a[1] for _, a in view.targets if a[0] == "button"]
    assert names == [b for b, _ in BUTTONS]
    keys = [s.key for s in specs(cfg)]
    shown = [a[1] for _, a in view.targets if a[0] == "edit"]
    assert shown == keys[: len(shown)] and shown
    seed_box = next(r for r, a in view.targets if a == ("edit", "seed"))
    assert hit(view.targets, seed_box.center) == ("edit", "seed")
    assert hit(view.targets, (5000, 5000)) is None
    name_row = [a for _, a in view.targets if len(a) > 1 and a[1] == "name"]
    assert name_row == [("edit", "name")]  # text has no arrows
    ui.scroll = 10**6
    scrolled = controls_view(ui, cfg)
    assert ui.scroll == scrolled.max_scroll > 0
    last = [a[1] for _, a in scrolled.targets if a[0] == "edit"]
    assert last[-1] == keys[-1] and "seed" not in last
    for rect, action in scrolled.targets:
        if action[0] != "button":
            assert scrolled.list_area.contains(rect)


def test_events_click_targets_scroll_controls_and_type_values() -> None:
    from apeks.experiment import ExperimentConfig
    from apeks.render import controls_view

    pygame.init()
    ui = Ui(size=(1600, 900))
    cam = Camera(view=tuple(layout(ui.size)["map"]))
    view = controls_view(ui, ExperimentConfig())
    ev = pygame.event.Event
    seed_box = next(r for r, a in view.targets if a == ("edit", "seed"))
    handle_event(
        ev(pygame.MOUSEBUTTONDOWN, pos=seed_box.center, button=1), ui, cam, view
    )
    assert ui.actions == [("edit", "seed")] and not ui.dragging
    ui.actions.clear()
    ui.editing, ui.text = "seed", "1"
    zoom = cam.zoom
    for key, uni in (
        (pygame.K_2, "2"),
        (pygame.K_d, "d"),
        (pygame.K_TAB, "\t"),
        (pygame.K_BACKSPACE, ""),
    ):
        assert handle_event(ev(pygame.KEYDOWN, key=key, unicode=uni), ui, cam, view)
    assert ui.text == "12" and cam.zoom == zoom  # typing is not a hotkey
    handle_event(ev(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r"), ui, cam, view)
    assert ui.actions == [("commit", "seed", "12")] and ui.editing is None
    ui.editing = "seed"
    assert handle_event(ev(pygame.KEYDOWN, key=pygame.K_ESCAPE, unicode=""), ui, cam)
    assert ui.editing is None  # Esc cancels the edit, not the window
    ui.editing = "seed"
    handle_event(ev(pygame.MOUSEBUTTONDOWN, pos=cam.view[:2], button=1), ui, cam, view)
    assert ui.editing is None and ui.dragging
    ui.actions.clear()
    for key, action in (
        (pygame.K_r, ("button", "apply")),
        (pygame.K_n, ("button", "seed")),
        (pygame.K_PERIOD, ("button", "step")),
    ):
        handle_event(ev(pygame.KEYDOWN, key=key, unicode=""), ui, cam)
        assert ui.actions[-1] == action
    ui.mouse = view.list_area.center
    handle_event(ev(pygame.MOUSEWHEEL, x=0, y=-2), ui, cam, view)
    assert ui.scroll > 0 and cam.zoom == zoom
    handle_event(ev(pygame.MOUSEWHEEL, x=0, y=50), ui, cam, view)
    assert ui.scroll == 0
    pygame.quit()


def test_draw_shows_controls_with_edits_highlighted(fonts) -> None:
    from apeks.controls import put
    from apeks.experiment import ExperimentConfig
    from apeks.render import WARN, controls_view

    live = ExperimentConfig(map="classic")
    world = build_world(live)
    surface = pygame.Surface((1600, 900))
    ui = Ui(world_size=(1200.0, 800.0))
    cam = Camera(view=tuple(layout(ui.size)["map"]))
    view = controls_view(ui, live)
    seed_box = next(r for r, a in view.targets if a == ("edit", "seed"))
    row = pygame.Rect(
        view.list_area.x, seed_box.y, seed_box.x - view.list_area.x, seed_box.h
    )

    def colours():
        return {surface.get_at(p)[:3] for p in _pixels(row)}

    draw(surface, world, "classic", fonts, cam, History(), ui, live, live)
    assert WARN not in colours()
    draft = put(live, "seed", 5)
    ui.message, ui.paused = "saved", True
    draw(surface, world, "classic", fonts, cam, History(), ui, draft, live)
    assert WARN in colours()
    ui.editing, ui.text = "seed", "77"
    draw(surface, world, "classic", fonts, cam, History(), ui, draft, live)


def _sailing_world():
    world = build_world(ExperimentConfig(map="classic", seed=3))
    for _ in range(400):
        world.step()
    ui = Ui(world_size=(1200.0, 800.0))
    cam = Camera(view=tuple(layout(ui.size)["map"]))
    cam.fit(*ui.world_size)
    ship = next(
        v
        for v in world.vessels.values()
        if world.convoys[v.convoy].state == "sailing" and v.cargo
    )
    return world, ui, cam, ship


def test_pick_finds_the_sailing_ship_under_the_cursor() -> None:
    world, _, cam, ship = _sailing_world()
    x, y = cam.to_screen(ship.pos)
    assert pick(world, cam, (x + 2, y + 1)) == ship.id
    assert pick(world, cam, (-500, -500)) is None


def test_ship_info_lists_cargo_and_next_destinations() -> None:
    world, _, _, ship = _sailing_world()
    convoy = world.convoys[ship.convoy]
    lines = ship_info(world, ship.id)
    assert lines[0].startswith(ship.id)
    text = "\n".join(lines)
    assert f"cargo {len(ship.cargo)}/{ship.capacity}" in text
    assert f"bound for {convoy.destination}" in text
    for dest in {p.destination for p in ship.cargo}:
        assert dest in text
    ship.cargo = []
    convoy.lane, convoy.destination, convoy.state = None, None, "docked"
    text = "\n".join(ship_info(world, ship.id))
    assert "cargo 0/" in text and "(empty)" in text and "at " in text
    assert ship_info(world, "nope") == []


def test_click_without_drag_picks_and_escape_closes_the_dialog() -> None:
    ui, cam = Ui(), Camera(view=tuple(layout((1600, 900))["map"]))
    down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(600, 400))
    up = pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(601, 400))
    handle_event(down, ui, cam)
    handle_event(up, ui, cam)
    assert ui.actions == [("pick", (601, 400))]
    ui.actions = []
    handle_event(down, ui, cam)
    handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(700, 400)), ui, cam
    )
    assert ui.actions == []  # a drag is not a click
    ui.selected = "v1"
    esc = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, unicode="")
    assert handle_event(esc, ui, cam) is True and ui.selected is None
    assert handle_event(esc, ui, cam) is False


def test_draw_shows_the_ship_dialog(fonts) -> None:
    world, ui, cam, ship = _sailing_world()
    surface = pygame.Surface(ui.size)
    draw(surface, world, "classical", fonts, cam, History(), ui)
    before = surface.copy()
    ui.selected = ship.id
    draw(surface, world, "classical", fonts, cam, History(), ui)
    x, y = (int(c) for c in cam.to_screen(ship.pos))
    box = pygame.Rect(x + 12, y + 12, 40, 20).clip(pygame.Rect(cam.view))
    assert any(surface.get_at(p) != before.get_at(p) for p in _pixels(box))
    assert any(surface.get_at(p)[:3] == PANEL for p in _pixels(pygame.Rect(cam.view)))
    ui.selected = "gone"  # a ship from an old world draws nothing
    draw(surface, world, "classical", fonts, cam, History(), ui)
