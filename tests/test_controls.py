import pytest

from apeks.controls import get, nudge, parse, put, specs
from apeks.experiment import ExperimentConfig


def by_key(cfg):
    return {s.key: s for s in specs(cfg)}


def test_specs_cover_run_map_llm_news_and_the_active_engines() -> None:
    keys = by_key(ExperimentConfig())
    for k in ("seed", "ticks", "map", "ports", "adjacency", "demand", "routing"):
        assert k in keys
    assert "news.delay" in keys and "engine.aco.alpha" in keys
    assert "url" not in keys and keys["ollama_port"].kind == "int"
    assert keys["capacity"].kind == "int" and keys["capacity"].lo == 1
    assert keys["effort"].options == ("", "low", "medium", "high", "xhigh", "max")
    assert "engine.boids.w_sep" in keys and "engine.scored.w_report" not in keys
    llm = by_key(ExperimentConfig(routing="llm", formation="llm"))
    assert "engine.scored.w_report" in llm and "engine.slot.shape" in llm
    assert "engine.scored.avoid" not in llm  # lists are not editable here
    classic = by_key(ExperimentConfig(map="classic"))
    assert "ports" not in classic and "adjacency" not in classic
    sections = [s.section for s in specs(ExperimentConfig())]
    assert sections == sorted(sections, key=sections.index)  # grouped


def test_get_and_put_reach_nested_fields_and_engine_defaults() -> None:
    cfg = ExperimentConfig()
    assert get(cfg, "seed") == 0 and get(cfg, "news.delay") == 60.0
    assert get(cfg, "engine.aco.alpha") == 1.0
    cfg2 = put(cfg, "engine.aco.alpha", 2.5)
    assert get(cfg2, "engine.aco.alpha") == 2.5 and cfg.engine == {}
    assert get(put(cfg2, "engine.aco.beta", 1.0), "engine.aco.alpha") == 2.5
    assert put(cfg, "news.delay", 5.0).news.delay == 5.0
    assert put(cfg, "seed", 9).seed == 9
    with pytest.raises(ValueError):
        put(cfg, "map", "nowhere")


def test_put_keeps_the_config_valid_when_switching_to_llm() -> None:
    cfg = ExperimentConfig(llm="")
    assert put(cfg, "routing", "llm").llm == "stub"
    assert put(ExperimentConfig(llm="claude"), "routing", "llm").llm == "claude"


def test_switching_backend_picks_its_default_model() -> None:
    cfg = ExperimentConfig()
    assert cfg.llm == "claude" and cfg.model == "claude-haiku-5-5"
    ollama = put(cfg, "llm", "ollama")
    assert ollama.model == "lfm3"
    assert put(ollama, "llm", "claude").model == "claude-haiku-5-5"
    assert put(put(cfg, "model", "x"), "llm", "stub").model == "x"


def test_nudge_steps_numbers_within_bounds_and_cycles_choices() -> None:
    cfg = ExperimentConfig()
    s = by_key(cfg)
    assert nudge(cfg, s["ports"], 1).ports == 6
    assert nudge(cfg, s["ticks"], -1).ticks == 5500
    low = put(cfg, "ports", 2)
    assert nudge(low, s["ports"], -1).ports == 2
    a = nudge(cfg, s["engine.aco.alpha"], 1)
    assert get(a, "engine.aco.alpha") == pytest.approx(1.1)
    top = put(cfg, "engine.aco.alpha", 5.0)
    assert get(nudge(top, s["engine.aco.alpha"], 1), "engine.aco.alpha") == 5.0
    assert nudge(cfg, s["map"], 1).map == "classic"
    assert nudge(cfg, s["map"], -1).map == "classic"
    assert nudge(cfg, s["name"], 1) == cfg  # text does not step


def test_parse_reads_typed_text_by_kind() -> None:
    s = by_key(ExperimentConfig())
    assert parse(s["seed"], " 42 ") == 42
    assert parse(s["ports"], "999") == 200  # clamped
    assert parse(s["demand"], "1.25") == 1.25
    assert parse(s["news.correction_delay"], "inf") == float("inf")
    assert parse(s["name"], "run a") == "run a"
    assert parse(s["map"], "classic") == "classic"
    for spec, text in ((s["seed"], "x"), (s["map"], "moon"), (s["demand"], "")):
        with pytest.raises(ValueError):
            parse(spec, text)


def test_show_formats_values_compactly() -> None:
    from apeks.controls import show

    s = by_key(ExperimentConfig())
    assert show(s["demand"], 1.5) == "1.5" and show(s["seed"], 3) == "3"
    assert show(s["engine.aco.memory"], 400.0) == "400"
    assert show(s["llm"], "") == "(none)" and show(s["name"], "x") == "x"
