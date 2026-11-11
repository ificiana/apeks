import json
from pathlib import Path

import pytest

from apeks import config, runner, scenario
from apeks.experiment import (
    ExperimentConfig,
    build_world,
    load_config,
    load_configs,
    make_client,
)
from apeks.formation import BoidsFormation, SlotFormation
from apeks.llm import OllamaClient, StubClient
from apeks.routing import AcoRouting, ScoredRouting
from apeks.trace import read_trace

ARMS = Path(__file__).parent.parent / "experiments"


def test_scenario_has_strait_and_every_port_reaches_every_other() -> None:
    net = scenario.network()
    assert net.lanes[scenario.STRAIT].capacity == 1
    ocean = scenario.ocean()
    for a in net.ports:
        for b in net.ports:
            assert net.times_to(b, ocean, 0.0, 1.8)[a] < float("inf")
    assert [e.kind for e in scenario.events()] == ["close", "open"]
    assert sum(scenario.FLEET.values()) == 15 and set(scenario.FLEET) == set(net.ports)


def test_world_starts_docked_with_seeded_demand(tmp_path: Path) -> None:
    world = build_world(ExperimentConfig(seed=3))
    assert len(world.vessels) == 15
    assert {c.state for c in world.convoys.values()} == {"docked"}
    assert world.packages == build_world(ExperimentConfig(seed=3)).packages
    assert world.packages != build_world(ExperimentConfig(seed=4)).packages
    rate = config.DEMAND_RATE * len(world.network.ports) * 6000
    assert 0.7 * rate < len(world.packages) < 1.3 * rate
    double = build_world(ExperimentConfig(seed=3, demand=2.0))
    assert len(double.packages) > 1.5 * len(world.packages)
    path = tmp_path / "d.toml"
    path.write_text('name = "d"\ndemand = 0.5\n')
    assert load_config(path).demand == 0.5


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
    assert len(world.vessels) == 15


def test_config_validation_and_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(routing="dijkstra")
    bad = tmp_path / "bad.toml"
    bad.write_text('colour = "red"\n')
    with pytest.raises(ValueError, match="colour"):
        load_config(bad)
    arms = {load_config(p).name for p in ARMS.glob("*.toml")}
    assert arms >= {"classical", "llm", "llm_routing", "llm_formation"}


def test_seed_list_expands_to_one_config_per_seed(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep.toml"
    sweep.write_text('name = "arm"\nseed = [0, 2]\nticks = 5\n')
    cfgs = load_configs(sweep)
    assert [(c.name, c.seed, c.ticks) for c in cfgs] == [
        ("arm_s0", 0, 5),
        ("arm_s2", 2, 5),
    ]
    assert load_config(sweep) == cfgs[0]
    single = tmp_path / "single.toml"
    single.write_text('name = "one"\nseed = 3\n')
    assert [(c.name, c.seed) for c in load_configs(single)] == [("one", 3)]


def test_sweep_arms_cover_seeds_0_to_49() -> None:
    for name in ["classical", "llm", "llm_routing", "llm_formation"]:
        cfgs = load_configs(ARMS / f"{name}_sweep.toml")
        assert [c.seed for c in cfgs] == list(range(50))
        assert cfgs[0].name == f"{name}_s0"
        base = load_config(ARMS / f"{name}.toml")
        assert cfgs[0] == base.__class__(**{**base.__dict__, "name": f"{name}_s0"})


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
        runner.main(
            [*args, "--ticks", "3", "--llm", "stub", "--snapshot-every", "2", "-j1"]
        )
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


def test_cli_runs_sweep_in_parallel_in_seed_order(tmp_path, capsys) -> None:
    sweep = tmp_path / "sweep.toml"
    sweep.write_text('name = "sw"\nseed = [0, 1, 2]\nticks = 2\n')
    out = tmp_path / "logs"
    args = [str(sweep), "--logs", str(out), "--no-trace"]
    assert runner.main([*args, "--jobs", "3"]) == 0
    parallel = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [x["name"] for x in parallel] == ["sw_s0", "sw_s1", "sw_s2"]
    assert len(list(out.iterdir())) == 3
    assert runner.main([*args, "--jobs", "1"]) == 0
    serial = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert all(x.pop("timing")["wall_s"] >= 0 for x in serial + parallel)
    assert serial == parallel


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


def test_empty_backend_allowed_only_without_llm_modules() -> None:
    world = build_world(ExperimentConfig(llm="", model=""))
    assert world.authors == {}
    with pytest.raises(ValueError, match="needs an llm backend"):
        ExperimentConfig(routing="llm", llm="")


def test_news_table_configures_the_feed(tmp_path: Path) -> None:
    from apeks.news import NewsSettings

    good = tmp_path / "news.toml"
    good.write_text(
        'name = "n"\nticks = 5000\n[news]\nmiss_rate = 1.0\nfalse_alarm_rate = 0.0\n'
    )
    cfg = load_config(good)
    assert cfg.news == NewsSettings(miss_rate=1.0, false_alarm_rate=0.0)
    assert build_world(cfg).news == []
    feed = build_world(ExperimentConfig(seed=1)).news
    assert feed == build_world(ExperimentConfig(seed=1)).news
    assert len([i for i in feed if i.truth]) == 4
    bad = tmp_path / "bad.toml"
    bad.write_text("[news]\nloudness = 2\n")
    with pytest.raises(ValueError, match="loudness"):
        load_config(bad)


def test_world_is_built_on_a_generated_or_the_classic_sea() -> None:
    world = build_world(ExperimentConfig(seed=24008, ports=10, adjacency=2))
    assert len(world.network.ports) == 10
    assert len(world.vessels) == 10 * config.SHIPS_PER_PORT
    assert world.geography.width > scenario.WIDTH
    assert {e.kind for e in world.events} == {"close", "open"}
    classic = build_world(ExperimentConfig(map="classic"))
    assert scenario.STRAIT in classic.network.lanes
    assert not classic.geography.is_land((500.0, 350.0))
    with pytest.raises(ValueError, match="atlantis"):
        ExperimentConfig(map="atlantis")


def test_port_and_adjacency_lists_expand_with_seeds(tmp_path: Path) -> None:
    sweep = tmp_path / "scale.toml"
    sweep.write_text('name = "c"\nports = [5, 10]\nadjacency = [2, 3]\nseed = [7]\n')
    cfgs = load_configs(sweep)
    assert [(c.name, c.ports, c.adjacency, c.seed) for c in cfgs] == [
        ("c_p5_k2_s7", 5, 2, 7),
        ("c_p5_k3_s7", 5, 3, 7),
        ("c_p10_k2_s7", 10, 2, 7),
        ("c_p10_k3_s7", 10, 3, 7),
    ]


def test_run_times_each_phase_and_describes_the_map() -> None:
    result = runner.run(ExperimentConfig(ticks=20, ports=10))
    timing = result["timing"]
    assert set(timing["phases"]) == set(runner.PHASES)
    assert all(v >= 0 for v in timing["phases"].values())
    assert timing["wall_s"] >= sum(timing["phases"].values())
    assert timing["build_s"] >= 0 and timing["per_tick_ms"] > 0
    m = result["map"]
    assert m["ports"] == 10 and m["nodes"] >= 20 and m["lanes"] > 0
    assert m["adjacency"] == 3 and m["kind"] == "generated"
