"""Experiment configs: which engine runs each module, and how to build the world."""

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from apeks import config, scenario
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
from apeks.radio import Radio
from apeks.routing import AcoParams, AcoRouting, ScoredRouteParams, ScoredRouting
from apeks.seeds import derive
from apeks.world import World, make_fleet

ROUTING = ("aco", "llm")
FORMATION = ("boids", "llm")
BACKENDS = ("stub", "ollama", "claude")

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

    def __post_init__(self) -> None:
        for value, allowed in [
            (self.routing, ROUTING),
            (self.formation, FORMATION),
            (self.llm, ("", *BACKENDS)),
        ]:
            if value not in allowed:
                raise ValueError(f"{value!r} is not one of {allowed}")
        if not self.llm and "llm" in (self.routing, self.formation):
            raise ValueError(f"{self.name}: an llm module needs an llm backend")


def load_configs(path: Path) -> list[ExperimentConfig]:
    """One config per run; a list of seeds expands to ``<name>_s<seed>`` runs."""
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
    seeds = data.get("seed", 0)
    if not isinstance(seeds, list):
        return [ExperimentConfig(**data)]
    base = ExperimentConfig(**{k: v for k, v in data.items() if k != "seed"})
    return [replace(base, seed=seed, name=f"{base.name}_s{seed}") for seed in seeds]


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
    net = scenario.network()
    ocean = scenario.ocean()
    events = scenario.events()
    news = generate(
        events,
        ocean.storms,
        net.nodes,
        sorted(net.lanes),
        cfg.news,
        derive(cfg.seed, "news"),
        cfg.ticks,
    )
    fleet = [(f"{port}-start", port, n, None) for port, n in scenario.FLEET.items()]
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
        routing: Any = AcoRouting(AcoParams(), derive(cfg.seed, "aco"))
    else:
        routing = ScoredRouting(ScoredRouteParams())
        author("routing", ROUTING_PURPOSE, ScoredRouteParams)
    if cfg.formation == "boids":
        formation: Any = BoidsFormation(BoidsParams())
    else:
        formation = SlotFormation(SlotParams())
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
    )
