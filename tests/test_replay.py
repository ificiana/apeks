import json
from pathlib import Path

import pygame
import pytest

from apeks.app import Session
from apeks.experiment import ExperimentConfig, NewsSettings
from apeks.llm import StubClient
from apeks.replay import (
    Picker,
    Recording,
    ReplayClient,
    SnapshotCheck,
    config_from_dict,
    draw_picker,
    find_traces,
    load,
    picker_event,
    status,
)
from apeks.runner import run
from apeks.trace import Tracer

LLM = ExperimentConfig(
    name="t", routing="llm", llm="stub", model="", map="classic", ticks=120
)
REPLY = '{"avoid": []}'


def record(tmp_path: Path, cfg: ExperimentConfig, client=None) -> Path:
    path = tmp_path / f"trace_{cfg.name}.jsonl"
    with path.open("w") as f:
        run(cfg, client=client, tracer=Tracer(f, 10))
    return path


def replay(path: Path, ticks: int) -> tuple[Recording, ReplayClient, SnapshotCheck]:
    rec = load(path)
    session = Session(rec.cfg, world_factory=rec.world)
    for _ in range(ticks):
        session.world.step()
    return rec, rec.last[0], rec.last[1]


def test_config_from_dict_round_trips_news_and_engine() -> None:
    from dataclasses import asdict

    cfg = ExperimentConfig(
        news=NewsSettings(delay=5.0), engine={"aco": {"alpha": 2.0}}, llm=""
    )
    assert config_from_dict(json.loads(json.dumps(asdict(cfg)))) == cfg
    assert config_from_dict({"seed": 3, "extra": 1}) == ExperimentConfig(seed=3)


def test_load_rejects_a_trace_without_meta(tmp_path: Path) -> None:
    path = tmp_path / "trace_x.jsonl"
    path.write_text('{"t": 0.0, "kind": "event", "text": "x"}\n')
    with pytest.raises(ValueError, match="no meta record"):
        load(path)


def test_replay_of_an_llm_run_matches_every_snapshot(tmp_path: Path) -> None:
    path = record(tmp_path, LLM, StubClient([REPLY] * 50))
    rec, client, check = replay(path, LLM.ticks)
    assert rec.requests, "fixture must make LLM calls"
    assert client.mismatches == 0 and not client.remaining
    assert check.checked > 0 and check.diverged_at is None
    assert status(client, check) == f"replay matches ({check.checked} snapshots)"


def test_replay_of_a_classical_run_matches(tmp_path: Path) -> None:
    cfg = ExperimentConfig(name="c", llm="", model="", map="classic", ticks=40)
    _, client, check = replay(record(tmp_path, cfg), cfg.ticks)
    assert client.mismatches == 0 and check.diverged_at is None


def test_a_different_world_is_flagged_as_diverged(tmp_path: Path) -> None:
    path = record(tmp_path, LLM, StubClient([REPLY] * 50))
    rec = load(path)

    def other(cfg: ExperimentConfig):
        from dataclasses import replace

        return rec.world(replace(cfg, seed=cfg.seed + 1, demand=3.0))

    session = Session(rec.cfg, world_factory=other)
    for _ in range(LLM.ticks):
        session.world.step()
    client, check = rec.last
    assert check.diverged_at is not None
    assert status(client, check).startswith("replay diverged at t=")


def test_replay_client_reproduces_errors_and_counts_mismatches() -> None:
    client = ReplayClient(
        [{"prompt": "a", "reply": "{}"}, {"prompt": "b", "error": "timeout"}]
    )
    assert client.complete("a", {}) == "{}"
    with pytest.raises(OSError, match="timeout"):
        client.complete("x", {})
    assert client.mismatches == 1
    with pytest.raises(OSError, match="no recorded reply"):
        client.complete("c", {})
    assert client.mismatches == 2
    assert status(client, SnapshotCheck({})) == "replay diverged: 2 LLM prompts differ"


def test_snapshot_check_wants_only_recorded_ticks() -> None:
    check = SnapshotCheck({10.0: "x"})
    assert check.wants_snapshot(10.0) and not check.wants_snapshot(11.0)
    check.emit(11.0, "route_choice", convoy="c")
    assert check.checked == 0


def test_find_traces_lists_newest_reports_first(tmp_path: Path) -> None:
    for name in ("r1/trace_a.jsonl", "r2/trace_b.jsonl", "r2/out_b.json"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text("")
    assert find_traces(tmp_path) == [
        tmp_path / "r2/trace_b.jsonl",
        tmp_path / "r1/trace_a.jsonl",
    ]


def key(k: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=k)


def test_picker_moves_scrolls_and_chooses(tmp_path: Path) -> None:
    items = [tmp_path / f"trace_{i}.jsonl" for i in range(30)]
    picker = Picker(items, root=tmp_path, rows=5)
    assert picker_event(picker, key(pygame.K_UP)) is None and picker.index == 0
    for _ in range(7):
        picker_event(picker, key(pygame.K_DOWN))
    assert picker.index == 7 and picker.top == 3
    picker_event(picker, key(pygame.K_PAGEDOWN))
    assert picker.index == 12
    picker_event(picker, key(pygame.K_END))
    assert picker.index == 29
    picker_event(picker, key(pygame.K_DOWN))
    assert picker.index == 29
    picker_event(picker, key(pygame.K_HOME))
    assert (picker.index, picker.top) == (0, 0)
    picker_event(picker, key(pygame.K_PAGEUP))
    assert picker.index == 0
    assert picker_event(picker, key(pygame.K_RETURN)) == items[0]
    wheel = pygame.event.Event(pygame.MOUSEWHEEL, y=-2)
    picker_event(picker, wheel)
    assert picker.index == 2
    click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(50, 0))
    assert picker_event(picker, click) is None  # header row
    row = Picker.TOP + 3 * Picker.ROW + 1
    click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(50, row))
    assert picker_event(picker, click) == items[3]
    far = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(50, 10_000))
    assert picker_event(picker, far) is None
    right = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(50, row))
    assert picker_event(picker, right) is None
    assert picker_event(picker, key(pygame.K_a)) is None


def test_empty_picker_chooses_nothing(tmp_path: Path) -> None:
    picker = Picker([], root=tmp_path)
    assert picker_event(picker, key(pygame.K_RETURN)) is None
    assert picker_event(picker, key(pygame.K_DOWN)) is None


def test_draw_picker_draws_rows_and_empty_hint(tmp_path: Path) -> None:
    pygame.font.init()
    fonts = {k: pygame.font.Font(None, 16) for k in "sml"}
    surface = pygame.Surface((800, 600))
    picker = Picker(
        [tmp_path / "trace_a.jsonl", tmp_path / "trace_b.jsonl"], root=tmp_path
    )
    draw_picker(surface, fonts, picker)
    assert picker.rows == (600 - Picker.TOP) // Picker.ROW
    draw_picker(surface, fonts, Picker([], root=tmp_path))
