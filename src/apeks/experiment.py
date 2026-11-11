"""Experiment configs: which engine runs each module, and how to build the world."""

import itertools
import json
import math
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from apeks import config, mapgen, scenario
from apeks.cargo import generate_demand
from apeks.formation import BoidsFormation, BoidsParams, SlotFormation, SlotParams
from apeks.hq import Hq
from apeks.llm import (
    ClaudeCodeClient,
    LlmClient,
    OllamaClient,
    PolicyAuthor,
    StubClient,
)
from apeks.news import NewsSettings, generate
from apeks.policy import coerce
from apeks.radio import Radio
from apeks.routing import AcoParams, AcoRouting, ScoredRouteParams, ScoredRouting
from apeks.seeds import derive
from apeks.world import World, make_fleet

ROUTING = ("aco", "llm")
FORMATION = ("boids", "llm")
BACKENDS = ("stub", "ollama", "claude")
MAPS = ("generated", "classic")
SWEPT = (("ports", "p"), ("adjacency", "k"), ("seed", "s"))
"""Keys a TOML may give as lists, in name order, with their name suffix."""
ENGINES: dict[str, type] = {
    "aco": AcoParams,
    "scored": ScoredRouteParams,
    "boids": BoidsParams,
    "slot": SlotParams,
}
"""Policy parameter classes, by the key of their ``[engine.<key>]`` TOML table."""

ROUTING_PURPOSE = (
    "At every waypoint a convoy scores its open lanes and takes the lowest score: "
    "w_estimate * estimated lane time + w_cost_to_go * time from the lane end to "
    "the destination + w_report * (recent reported transit - estimate), plus "
    "avoid_penalty for lanes listed in avoid. Lane ids look like 'W2>W3'. "
    "A convoy never enters a lane it believes closed. Destinations are set by "
    "the cargo the ships carry, not by this policy."
)
FORMATION_PURPOSE = (
    "Each vessel steers toward its slot in a convoy shape around the convoy "
    "centre while heading to the next waypoint, keeping a minimum gap to others. "
    "Narrow lanes compress the shape sideways."
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "classical"
    routing: str = "aco"
    formation: str = "boids"
    seed: int = 0
    ticks: int = 6000
    llm: str = "stub"
    model: str = "lfm3"
    url: str = "http://localhost:11434"
    latency: float = config.LLM_LATENCY
    budget: int = config.LLM_BUDGET
    news: NewsSettings = NewsSettings()
    demand: float = 1.0
    map: str = "generated"
    ports: int = 5
    adjacency: int = 3
    engine: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Starting policy overrides per engine key; values are clamped to bounds."""

    def __post_init__(self) -> None:
        for value, allowed in [
            (self.routing, ROUTING),
            (self.formation, FORMATION),
            (self.llm, ("", *BACKENDS)),
            (self.map, MAPS),
        ]:
            if value not in allowed:
                raise ValueError(f"{value!r} is not one of {allowed}")
        if not self.llm and "llm" in (self.routing, self.formation):
            raise ValueError(f"{self.name}: an llm module needs an llm backend")


def engine_keys(cfg: ExperimentConfig) -> tuple[str, str]:
    """The ``ENGINES`` keys of the routing and formation policies ``cfg`` runs."""
    return (
        "aco" if cfg.routing == "aco" else "scored",
        "boids" if cfg.formation == "boids" else "slot",
    )


def engine_params(cfg: ExperimentConfig, key: str) -> Any:
    cls = ENGINES[key]
    return coerce(cls, cfg.engine.get(key, {}), cls())


def _toml_value(value: Any) -> str:
    if isinstance(value, float) and math.isinf(value):
        return str(value)  # "inf" or "-inf", as TOML spells them
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return json.dumps(value)


def to_toml(cfg: ExperimentConfig) -> str:
    """A TOML file that ``load_config`` reads back as ``cfg``."""
    lines = [
        f"{f.name} = {_toml_value(getattr(cfg, f.name))}"
        for f in fields(cfg)
        if f.name not in ("news", "engine")
    ]
    lines += ["", "[news]"]
    lines += [
        f"{f.name} = {_toml_value(getattr(cfg.news, f.name))}" for f in fields(cfg.news)
    ]
    for key, values in sorted(cfg.engine.items()):
        lines += ["", f"[engine.{key}]"]
        lines += [f"{k} = {_toml_value(v)}" for k, v in sorted(values.items())]
    return "\n".join(lines) + "\n"


def load_configs(path: Path) -> list[ExperimentConfig]:
    """One config per run.

    Lists of ports, adjacency or seeds expand to every combination, named
    ``<name>_p<ports>_k<adjacency>_s<seed>`` for the keys given as lists.
    """
    data: dict[str, Any] = tomllib.loads(path.read_text())
    known = {f.name for f in fields(ExperimentConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    if "news" in data:
        news_keys = {f.name for f in fields(NewsSettings)}
        bad = set(data["news"]) - news_keys
        if bad:
            raise ValueError(f"unknown [news] keys in {path}: {sorted(bad)}")
        data["news"] = NewsSettings(**data["news"])
    for key, values in data.get("engine", {}).items():
        names = {f.name for f in fields(ENGINES[key])} if key in ENGINES else set()
        if not names or set(values) - names:
            raise ValueError(f"unknown [engine.{key}] table or keys in {path}")
    swept = [(k, tag, data.pop(k)) for k, tag in SWEPT if isinstance(data.get(k), list)]
    base = ExperimentConfig(**data)
    runs = []
    for values in itertools.product(*(v for _, _, v in swept)):
        chosen = {k: v for (k, _, _), v in zip(swept, values, strict=True)}
        name = base.name + "".join(
            f"_{tag}{v}" for (_, tag, _), v in zip(swept, values, strict=True)
        )
        runs.append(replace(base, name=name, **chosen))
    return runs


def load_config(path: Path) -> ExperimentConfig:
    """The first run of a config file."""
    return load_configs(path)[0]


def make_client(cfg: ExperimentConfig) -> LlmClient:
    if cfg.llm == "ollama":
        return OllamaClient(model=cfg.model, url=cfg.url)
    if cfg.llm == "claude":
        return ClaudeCodeClient(model=cfg.model)
    return StubClient()


def build_world(cfg: ExperimentConfig, client: LlmClient | None = None) -> World:
    sea = (
        scenario.sea()
        if cfg.map == "classic"
        else mapgen.generate(cfg.seed, cfg.ports, cfg.adjacency, cfg.ticks)
    )
    net, ocean, events = sea.network, sea.ocean, list(sea.events)
    news = generate(
        events,
        ocean.storms,
        net.nodes,
        sorted(net.lanes),
        cfg.news,
        derive(cfg.seed, "news"),
        cfg.ticks,
    )
    fleet = [(f"{port}-start", port, n, None) for port, n in sea.fleet.items()]
    vessels, convoys = make_fleet(net, fleet)
    packages = generate_demand(
        sorted(net.ports), config.DEMAND_RATE * cfg.demand, cfg.seed, cfg.ticks
    )
    authors: dict[str, PolicyAuthor] = {}
    client = client or make_client(cfg)

    def author(module: str, purpose: str, params_cls: type) -> None:
        authors[module] = PolicyAuthor(
            module, purpose, params_cls, client, cfg.budget, cfg.latency
        )

    if cfg.routing == "aco":
        routing: Any = AcoRouting(engine_params(cfg, "aco"), derive(cfg.seed, "aco"))
    else:
        routing = ScoredRouting(engine_params(cfg, "scored"))
        author("routing", ROUTING_PURPOSE, ScoredRouteParams)
    if cfg.formation == "boids":
        formation: Any = BoidsFormation(engine_params(cfg, "boids"))
    else:
        formation = SlotFormation(engine_params(cfg, "slot"))
        author("formation", FORMATION_PURPOSE, SlotParams)
    return World(
        network=net,
        ocean=ocean,
        radio=Radio(
            derive(cfg.seed, "radio"),
            range=config.RADIO_RANGE,
            loss=config.RADIO_LOSS,
            bandwidth=config.RADIO_BANDWIDTH,
            noise=config.RADIO_NOISE,
        ),
        vessels=vessels,
        convoys=convoys,
        routing=routing,
        formation=formation,
        events=events,
        authors=authors,
        news=news,
        hq=Hq(config.UPLINK_DELAY),
        packages=packages,
        rng=derive(cfg.seed, "reports"),
        geography=sea.geography,
    )
