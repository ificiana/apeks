"""Experiment configs: which engine runs each module, and how to build the world."""

import random
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from apeks import config, scenario
from apeks.formation import BoidsFormation, BoidsParams, SlotFormation, SlotParams
from apeks.llm import (
    ClaudeCodeClient,
    LlmClient,
    OllamaClient,
    PolicyAuthor,
    StubClient,
)
from apeks.radio import Radio
from apeks.routing import AcoParams, AcoRouting, ScoredRouteParams, ScoredRouting
from apeks.world import World, make_fleet

ROUTING = ("aco", "llm")
FORMATION = ("boids", "llm")
BACKENDS = ("stub", "ollama", "claude")

ROUTING_PURPOSE = (
    "At every waypoint a convoy scores its open lanes and takes the lowest score: "
    "w_estimate * estimated lane time + w_cost_to_go * time from the lane end to "
    "the destination + w_report * (recent reported transit - estimate), plus "
    "avoid_penalty for lanes listed in avoid. Lane ids look like 'W2>W3'."
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

    def __post_init__(self) -> None:
        for value, allowed in [
            (self.routing, ROUTING),
            (self.formation, FORMATION),
            (self.llm, BACKENDS),
        ]:
            if value not in allowed:
                raise ValueError(f"{value!r} is not one of {allowed}")


def load_config(path: Path) -> ExperimentConfig:
    data: dict[str, Any] = tomllib.loads(path.read_text())
    known = {f.name for f in fields(ExperimentConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    return ExperimentConfig(**data)


def make_client(cfg: ExperimentConfig) -> LlmClient:
    if cfg.llm == "ollama":
        return OllamaClient(model=cfg.model, url=cfg.url)
    if cfg.llm == "claude":
        return ClaudeCodeClient(model=cfg.model)
    return StubClient()


def build_world(cfg: ExperimentConfig, client: LlmClient | None = None) -> World:
    rng = random.Random(cfg.seed)
    net = scenario.network()
    vessels, convoys = make_fleet(net, scenario.CONVOYS)
    authors: dict[str, PolicyAuthor] = {}
    client = client or make_client(cfg)

    def author(module: str, purpose: str, params_cls: type) -> None:
        authors[module] = PolicyAuthor(
            module, purpose, params_cls, client, cfg.budget, cfg.latency
        )

    if cfg.routing == "aco":
        routing: Any = AcoRouting(AcoParams(), random.Random(rng.random()))
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
        ocean=scenario.ocean(),
        radio=Radio(
            random.Random(rng.random()),
            range=config.RADIO_RANGE,
            loss=config.RADIO_LOSS,
            bandwidth=config.RADIO_BANDWIDTH,
            noise=config.RADIO_NOISE,
        ),
        vessels=vessels,
        convoys=convoys,
        routing=routing,
        formation=formation,
        events=scenario.events(),
        authors=authors,
    )
