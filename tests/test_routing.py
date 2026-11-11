import math
import random

import pytest

from apeks.routing import (
    AcoParams,
    AcoRouting,
    Report,
    RouteObservation,
    RouteOption,
    ScoredRouteParams,
    ScoredRouting,
    pheromone,
)


def opt(lane, est, ctg=0.0, reports=()) -> RouteOption:
    return RouteOption(lane, lane[-1], est, ctg, tuple(reports))


def obs(*options, now=100.0) -> RouteObservation:
    return RouteObservation(now, "A", "B", tuple(options))


def test_pheromone_rises_with_fast_recent_reports_and_fades() -> None:
    p = AcoParams(tau0=0.1, memory=100.0)
    assert pheromone(opt("x", 10.0), 0.0, p) == 0.1
    fresh = pheromone(opt("x", 10.0, reports=[Report(5.0, 50.0)]), 50.0, p)
    assert fresh == pytest.approx(2.1)
    old = pheromone(opt("x", 10.0, reports=[Report(5.0, 50.0)]), 150.0, p)
    assert old == pytest.approx(0.1 + 2 * math.exp(-1))


def test_aco_prefers_reinforced_lane_probabilistically() -> None:
    engine = AcoRouting(AcoParams(alpha=1.0, beta=1.0), random.Random(0))
    o = obs(
        opt("A>X", 50.0, 50.0, [Report(40.0, 99.0)] * 3),
        opt("A>Y", 50.0, 50.0),
    )
    picks = [engine.choose(o) for _ in range(200)]
    assert picks.count("A>X") > 150
    assert "A>Y" in picks


def test_aco_skips_impassable_and_returns_none_without_options() -> None:
    engine = AcoRouting(AcoParams(), random.Random(0))
    assert engine.choose(obs(opt("A>X", math.inf), opt("A>Y", 10.0))) == "A>Y"
    assert engine.choose(obs(opt("A>X", math.inf))) is None
    assert engine.choose(obs()) is None


def test_aco_falls_back_to_last_option_on_rounding() -> None:
    class Max(random.Random):
        def random(self) -> float:
            return 1.0

    engine = AcoRouting(AcoParams(), Max())
    assert engine.choose(obs(opt("A>X", 10.0), opt("A>Y", 10.0))) == "A>Y"


def test_scored_routing_is_deterministic_and_obeys_avoid_list() -> None:
    engine = ScoredRouting(ScoredRouteParams())
    o = obs(opt("A>X", 10.0, 10.0), opt("A>Y", 12.0, 10.0))
    assert engine.choose(o) == "A>X"
    engine.params = ScoredRouteParams(avoid=("A>X",))
    assert engine.choose(o) == "A>Y"
    assert engine.choose(obs(opt("A>X", math.inf))) is None


def test_scored_routing_uses_recent_reports_only() -> None:
    engine = ScoredRouting(ScoredRouteParams(report_horizon=50.0))
    slow = opt("A>X", 10.0, 10.0, [Report(40.0, 90.0)])
    stale = opt("A>X", 10.0, 10.0, [Report(40.0, 10.0)])
    other = opt("A>Y", 15.0, 10.0)
    assert engine.choose(obs(slow, other)) == "A>Y"
    assert engine.choose(obs(stale, other)) == "A>X"
