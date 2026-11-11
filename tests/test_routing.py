import math
import random
from dataclasses import replace

import pytest

from apeks.routing import (
    AcoParams,
    AcoRouting,
    Report,
    ReportLog,
    RouteObservation,
    RouteOption,
    ScoredRouteParams,
    ScoredRouting,
    pheromone,
)


def opt(lane, est, ctg=0.0, reports=()) -> RouteOption:
    reports = reports if isinstance(reports, ReportLog) else tuple(reports)
    return RouteOption(lane, lane[-1], est, ctg, reports)


def choose(engine, o: RouteObservation) -> str | None:
    """Choose with the engine's own parameters, as a convoy holding them would."""
    return engine.choose(replace(o, params=engine.params))


def obs(*options, now=100.0, params=None) -> RouteObservation:
    return RouteObservation(now, "A", "B", tuple(options), params)


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
    picks = [choose(engine, o) for _ in range(200)]
    assert picks.count("A>X") > 150
    assert "A>Y" in picks


def test_aco_skips_impassable_and_returns_none_without_options() -> None:
    engine = AcoRouting(AcoParams(), random.Random(0))
    assert choose(engine, obs(opt("A>X", math.inf), opt("A>Y", 10.0))) == "A>Y"
    assert choose(engine, obs(opt("A>X", math.inf))) is None
    assert choose(engine, obs()) is None


def test_aco_falls_back_to_last_option_on_rounding() -> None:
    class Max(random.Random):
        def random(self) -> float:
            return 1.0

    engine = AcoRouting(AcoParams(), Max())
    assert choose(engine, obs(opt("A>X", 10.0), opt("A>Y", 10.0))) == "A>Y"


def test_scored_routing_is_deterministic_and_obeys_avoid_list() -> None:
    engine = ScoredRouting(ScoredRouteParams())
    o = obs(opt("A>X", 10.0, 10.0), opt("A>Y", 12.0, 10.0))
    assert choose(engine, o) == "A>X"
    engine.params = ScoredRouteParams(avoid=("A>X",))
    assert choose(engine, o) == "A>Y"
    assert choose(engine, obs(opt("A>X", math.inf))) is None


def test_scored_routing_uses_recent_reports_only() -> None:
    engine = ScoredRouting(ScoredRouteParams(report_horizon=50.0))
    slow = opt("A>X", 10.0, 10.0, [Report(40.0, 90.0)])
    stale = opt("A>X", 10.0, 10.0, [Report(40.0, 10.0)])
    other = opt("A>Y", 15.0, 10.0)
    assert choose(engine, obs(slow, other)) == "A>Y"
    assert choose(engine, obs(stale, other)) == "A>X"


def test_report_log_fades_incrementally_like_the_full_sum() -> None:
    rng = random.Random(3)
    log, plain = ReportLog(), []
    p = AcoParams(memory=50.0)
    now = 0.0
    for _ in range(200):
        now += rng.uniform(0.0, 20.0)
        r = Report(rng.uniform(5.0, 50.0), now - rng.uniform(0.0, 100.0))
        log.append(r)
        plain.append(r)
        for m in (50.0, 400.0):
            full = sum(math.exp(-(now - x.finished_at) / m) / x.transit for x in plain)
            assert log.faded(now, m) == pytest.approx(full, rel=1e-9)
        a = pheromone(opt("x", 10.0, reports=log), now, p)
        assert a == pytest.approx(pheromone(opt("x", 10.0, reports=plain), now, p))
    assert ReportLog().faded(5.0, 10.0) == 0.0


def test_report_log_recomputes_if_asked_about_the_past() -> None:
    log = ReportLog([Report(10.0, 0.0)])
    assert log.faded(100.0, 50.0) == pytest.approx(math.exp(-2) / 10)
    assert log.faded(50.0, 50.0) == pytest.approx(math.exp(-1) / 10)
