"""Routing engines: which lane a convoy takes next from a node.

Both engines read the same ``RouteObservation``, which carries the policy the
convoy holds (``params``; ``self.params`` is only the fleet's starting policy),
and return a lane id (or
``None`` when no lane is passable). ``AcoRouting`` is ant colony optimisation:
lane reports from other convoys act as pheromone that fades with age, mixed
with a travel-time heuristic. ``ScoredRouting`` is the policy shape an LLM
authors: a deterministic weighted score plus an explicit avoid-list.
"""

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from apeks.policy import bounded, names


@dataclass(frozen=True)
class Report:
    transit: float
    finished_at: float


@dataclass(frozen=True)
class RouteOption:
    lane: str
    to_node: str
    estimate: float
    cost_to_go: float
    reports: Sequence[Report] = ()


@dataclass(frozen=True)
class RouteObservation:
    now: float
    node: str
    destination: str
    options: tuple[RouteOption, ...]
    params: Any  # the policy this convoy currently holds


def _viable(obs: RouteObservation) -> list[RouteOption]:
    return [o for o in obs.options if math.isfinite(o.estimate + o.cost_to_go)]


@dataclass(frozen=True)
class AcoParams:
    alpha: float = bounded(1.0, 0.0, 5.0, "weight of lane reports (pheromone)")
    beta: float = bounded(2.0, 0.0, 5.0, "weight of the travel-time heuristic")
    tau0: float = bounded(0.2, 0.01, 5.0, "base pheromone on unreported lanes")
    memory: float = bounded(400.0, 10.0, 5000.0, "time for a report to fade")


class ReportLog(list[Report]):
    """Lane reports in arrival order, with the faded sum kept up to date.

    ``faded(now, m)`` is the sum over reports of exp(-(now - finished_at) / m)
    / transit. For each ``m`` it keeps that sum at the time it was last asked
    and how many reports it covers, so a later call only decays the total and
    adds the reports that arrived since: O(new reports), not O(all reports).
    """

    def __init__(self, reports: Iterable[Report] = ()) -> None:
        super().__init__(reports)
        self._fade: dict[float, tuple[int, float, float]] = {}

    def faded(self, now: float, memory: float) -> float:
        count, at, total = self._fade.get(memory, (0, now, 0.0))
        if now < at:
            count, at, total = 0, now, 0.0
        total *= math.exp(-(now - at) / memory)
        for r in self[count:]:
            total += math.exp(-(now - r.finished_at) / memory) / r.transit
        self._fade[memory] = (len(self), now, total)
        return total


def pheromone(option: RouteOption, now: float, p: AcoParams) -> float:
    """Base level plus each report's speed-up over the estimate, fading with age."""
    reports = option.reports
    if isinstance(reports, ReportLog):
        faded = reports.faded(now, p.memory)
    else:
        faded = sum(
            math.exp(-(now - r.finished_at) / p.memory) / r.transit for r in reports
        )
    return p.tau0 + option.estimate * faded


class AcoRouting:
    def __init__(self, params: AcoParams, rng: random.Random) -> None:
        self.params = params
        self.rng = rng

    def choose(self, obs: RouteObservation) -> str | None:
        viable = _viable(obs)
        if not viable:
            return None
        p = obs.params
        weights = [
            pheromone(o, obs.now, p) ** p.alpha
            * (1.0 / (o.estimate + o.cost_to_go)) ** p.beta
            for o in viable
        ]
        pick = self.rng.random() * sum(weights)
        for o, w in zip(viable, weights, strict=True):
            pick -= w
            if pick < 0:
                return o.lane
        return viable[-1].lane


@dataclass(frozen=True)
class ScoredRouteParams:
    w_estimate: float = bounded(1.0, 0.0, 5.0, "weight of this lane's estimated time")
    w_cost_to_go: float = bounded(
        1.0, 0.0, 5.0, "weight of estimated time from the lane's end to destination"
    )
    w_report: float = bounded(
        1.0, 0.0, 5.0, "weight of recent reported delay over the estimate"
    )
    report_horizon: float = bounded(
        400.0, 10.0, 5000.0, "how old a report may be and still count"
    )
    avoid: tuple[str, ...] = names("lane ids to avoid unless there is no other way")
    avoid_penalty: float = bounded(
        10000.0, 0.0, 100000.0, "score added to avoided lanes"
    )


class ScoredRouting:
    def __init__(self, params: ScoredRouteParams) -> None:
        self.params = params

    def _score(self, o: RouteOption, now: float, p: ScoredRouteParams) -> float:
        recent = [
            r.transit for r in o.reports if now - r.finished_at <= p.report_horizon
        ]
        delay = sum(recent) / len(recent) - o.estimate if recent else 0.0
        score = p.w_estimate * o.estimate + p.w_cost_to_go * o.cost_to_go
        score += p.w_report * delay
        return score + (p.avoid_penalty if o.lane in p.avoid else 0.0)

    def choose(self, obs: RouteObservation) -> str | None:
        viable = _viable(obs)
        if not viable:
            return None
        return min(viable, key=lambda o: self._score(o, obs.now, obs.params)).lane
