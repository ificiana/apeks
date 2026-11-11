import json
from pathlib import Path

import pytest

from apeks import runner, scenario
from apeks.experiment import ExperimentConfig, build_world, load_config, make_client
from apeks.formation import BoidsFormation, SlotFormation
from apeks.llm import OllamaClient, StubClient
from apeks.routing import AcoRouting, ScoredRouting
from apeks.trace import read_trace

ARMS = Path(__file__).parent.parent / "experiments"


def test_scenario_has_strait_and_every_convoy_can_reach_its_ports() -> None:
    net = scenario.network()
    assert net.lanes[scenario.STRAIT].capacity == 1
    ocean = scenario.ocean()
    for _, orders, _ in scenario.CONVOYS:
        for a, b in zip(orders, orders[1:] + orders[:1], strict=True):
            assert net.times_to(b, ocean, 0.0, 1.8)[a] < float("inf")
    assert [e.kind for e in scenario.events()] == ["close", "open"]


@pytest.mark.parametrize(
    ("routing", "formation", "types", "authors"),
    [
        ("aco", "boids", (AcoRouting, BoidsFormation), []),
        ("llm", "llm", (ScoredRouting, SlotFormation), ["formation", "routing"]),
    ],
)
def test_build_world_plugs_engines_per_module(
    routing, formation, types, authors
) -> None:
    world = build_world(ExperimentConfig(routing=routing, formation=formation))
    assert isinstance(world.routing, types[0])
    assert isinstance(world.formation, types[1])
    assert sorted(world.authors) == authors
    assert len(world.vessels) == 16


def test_config_validation_and_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(routing="dijkstra")
    bad = tmp_path / "bad.toml"
    bad.write_text('colour = "red"\n')
    with pytest.raises(ValueError, match="colour"):
        load_config(bad)
    arms = {load_config(p).name for p in ARMS.glob("*.toml")}
    assert arms == {"classical", "llm", "llm_routing", "llm_formation"}


def test_make_client_picks_backend() -> None:
    assert isinstance(make_client(ExperimentConfig()), StubClient)
    client = make_client(ExperimentConfig(llm="ollama", model="m", url="http://h"))
    assert isinstance(client, OllamaClient) and client.url == "http://h"


def test_run_returns_summary_and_llm_log() -> None:
    stub = StubClient(['{"shape": "wedge"}'])
    result = runner.run(ExperimentConfig(formation="llm", ticks=30), stub)
    assert result["config"]["formation"] == "llm"
    assert result["llm_log"]["formation"][0]["trigger"] == "mission start"
    assert result["summary"]["ticks"] == 30


def test_cli_saves_out_and_trace_per_arm_under_logs(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "timestamp", lambda: "20261111-070000")
    args = [str(ARMS / "classical.toml"), str(ARMS / "llm.toml")]
    assert (
        runner.main([*args, "--ticks", "3", "--llm", "stub", "--snapshot-every", "2"])
        == 0
    )
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [x["name"] for x in lines] == ["classical", "llm"]
    logs = tmp_path / "logs"
    assert sorted(p.name for p in logs.iterdir()) == [
        "out_classical_20261111-070000.json",
        "out_llm_20261111-070000.json",
        "trace_classical_20261111-070000.jsonl",
        "trace_llm_20261111-070000.jsonl",
    ]
    saved = json.loads((logs / "out_llm_20261111-070000.json").read_text())
    assert saved["config"]["ticks"] == 3
    records = read_trace(logs / "trace_classical_20261111-070000.jsonl")
    assert records[0]["kind"] == "meta" and records[0]["config"]["ticks"] == 3
    assert [r["t"] for r in records if r["kind"] == "snapshot"] == [0.0, 2.0]


def test_cli_logs_dir_and_no_trace(tmp_path, capsys) -> None:
    short = tmp_path / "short.toml"
    short.write_text('name = "short"\nticks = 2\n')
    out = tmp_path / "elsewhere"
    assert runner.main([str(short), "--logs", str(out), "--no-trace"]) == 0
    assert json.loads(capsys.readouterr().out)["ticks"] == 2
    [saved] = list(out.iterdir())
    assert saved.name.startswith("out_short_") and saved.suffix == ".json"


def test_timestamp_is_sortable_local_time() -> None:
    import re

    assert re.fullmatch(r"\d{8}-\d{6}", runner.timestamp())


def test_make_client_claude_backend() -> None:
    from apeks.llm import ClaudeCodeClient

    client = make_client(ExperimentConfig(llm="claude", model="claude-haiku-5-5"))
    assert isinstance(client, ClaudeCodeClient) and client.model == "claude-haiku-5-5"


def test_cli_model_override(tmp_path, capsys, monkeypatch) -> None:
    seen = {}
    real = runner.run

    def spy(cfg, **kw):
        seen["model"] = cfg.model
        return real(cfg, **kw)

    monkeypatch.setattr(runner, "run", spy)
    args = [str(ARMS / "classical.toml"), "--ticks", "1", "--no-trace"]
    runner.main([*args, "--logs", str(tmp_path), "--model", "x"])
    assert seen["model"] == "x"
