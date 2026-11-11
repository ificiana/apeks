import threading
from pathlib import Path

from apeks.app import advance, parse_args, start
from apeks.experiment import ExperimentConfig, load_config
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
    vx, vy, vw, vh = cam.view
    assert x0 >= vx and y0 >= vy
    assert x1 <= vx + vw + 1e-9 and y1 <= vy + vh + 1e-9
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


def test_present_redraws_when_free_and_holds_the_frame_with_a_banner_when_busy(
    fonts,
) -> None:
    import threading

    import pygame

    from apeks.app import present

    screen = pygame.Surface((300, 200))
    lock = threading.Lock()
    red, blue = (200, 0, 0), (0, 0, 200)
    frame = present(screen, None, lock, lambda s: s.fill(red), fonts, wait=0.01)
    assert screen.get_at((150, 190))[:3] == red and screen.get_at((150, 20))[:3] == red
    lock.acquire()  # a slow step (an LLM call) is running
    frame = present(screen, frame, lock, lambda s: s.fill(blue), fonts, wait=0.01)
    assert screen.get_at((150, 190))[:3] == red  # last frame kept, no blink
    assert screen.get_at((150, 20))[:3] != red  # banner on top
    lock.release()
    frame = present(screen, frame, lock, lambda s: s.fill(blue), fonts, wait=0.01)
    assert screen.get_at((150, 20))[:3] == blue  # banner cleared
    bigger = pygame.Surface((400, 300))
    frame = present(bigger, frame, lock, lambda s: s.fill(red), fonts, wait=0.01)
    assert frame.get_size() == (400, 300) and bigger.get_at((399, 299))[:3] == red


def test_advance_takes_the_lock_per_tick() -> None:
    world, ui, _, history = start(ExperimentConfig(map="classic"))
    entries = []

    class Spy:
        def __enter__(self):
            entries.append(world.t)

        def __exit__(self, *exc):
            return False

    ui.speed = 2
    advance(world, ui, history, ticks=100, lock=Spy())
    assert entries == [float(t) for t in range(SPEEDS[2])]


def _session(**kw):
    from apeks.app import Session

    return Session(ExperimentConfig(map="classic", **kw), size=(1600, 900))


def test_session_edits_a_draft_and_reruns_it(tmp_path) -> None:
    s = _session()
    s.world.step()
    s.ui.actions += [("nudge", "seed", 1), ("nudge", "engine.aco.alpha", 1)]
    s.process()
    assert s.draft.seed == 1 and s.live.seed == 0 and s.world.t == 1
    s.ui.actions.append(("edit", "ticks"))
    s.process()
    assert s.ui.editing == "ticks" and s.ui.text == "6000"
    s.ui.actions.append(("commit", "ticks", "900"))
    s.process()
    assert s.draft.ticks == 900
    old = s.world
    s.act(("button", "apply"))
    assert s.live == s.draft and s.world is not old and s.world.t == 0
    assert s.world.routing.params.alpha == 1.1 and s.stepper.ticks == 900
    assert list(s.history.ticks) == [0.0] and "rerun" in s.ui.message


def test_session_buttons(tmp_path) -> None:
    s = _session()
    s.save_dir = tmp_path / "saved"
    s.act(("button", "seed"))
    assert s.live.seed == 1 and s.draft.seed == 1
    s.act(("nudge", "ticks", 1))
    s.act(("button", "rerun"))
    assert s.live.ticks == 6000 and s.draft.ticks == 6500  # draft kept
    s.act(("button", "revert"))
    assert s.draft == s.live
    s.act(("button", "pause"))
    assert s.ui.paused
    s.act(("button", "step"))
    s.act(("button", "step"))
    assert s.world.t == 2 and s.ui.paused
    s.stepper.lock.acquire()
    s.act(("button", "step"))
    assert s.world.t == 2 and "busy" in s.ui.message
    s.stepper.lock.release()
    s.act(("button", "save"))
    (saved,) = (tmp_path / "saved").glob("*.toml")
    assert load_config(saved) == s.draft and str(saved) in s.ui.message
    s.cam.pan(500, 500)
    s.act(("button", "fit"))
    fitted = s.cam.offset
    s.cam.fit(*s.ui.world_size)
    assert s.cam.offset == fitted


def test_session_reports_bad_edits_and_failed_builds(monkeypatch) -> None:
    import apeks.app

    s = _session()
    s.act(("commit", "seed", "abc"))
    assert s.draft == s.live and "seed" in s.ui.message
    s.act(("commit", "llm", "(none)"))  # no llm module, so allowed
    assert s.draft.llm == ""
    s.act(("nudge", "routing", 1))  # llm routing needs a backend: picks stub
    assert s.draft.routing == "llm" and s.draft.llm == "stub"
    s.act(("commit", "llm", "(none)"))
    assert s.draft.llm == "stub" and "llm" in s.ui.message
    world = s.world

    def broken(cfg):
        raise RuntimeError("no connected map")

    monkeypatch.setattr(apeks.app, "build_world", broken)
    s.act(("button", "apply"))
    assert s.world is world and "no connected map" in s.ui.message


def test_session_paints_and_runs_its_stepper_thread(fonts) -> None:
    import pygame

    from apeks.app import Session

    s = Session(ExperimentConfig(map="classic", ticks=5), threaded=True)
    for _ in range(500):
        if s.world.t == 5:
            break
        threading.Event().wait(0.01)
    assert s.world.t == 5
    first = s.stepper
    s.act(("button", "rerun"))
    assert s.stepper is not first and first._stop.is_set()
    screen = pygame.Surface(s.ui.size)
    s.paint(screen, fonts)
    assert s.view().targets
    s.close()
    assert s.stepper._stop.is_set()
