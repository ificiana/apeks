"""Formation engines: the velocity a vessel asks its propulsion for.

Both engines read the same ``FormationObservation``, built from the vessel's
own state and from radio beacons (neighbour positions are estimates). Boids
gives emergent cohesion from separation, alignment and cohesion rules.
``SlotFormation`` is the policy shape an LLM authors: an explicit geometric
formation whose shape and spacing are chosen, plus a minimum-gap safety rule.
"""

from dataclasses import dataclass

from apeks.geometry import (
    Vec,
    add,
    clamp_length,
    length,
    normalize,
    perp,
    scale,
    sub,
)
from apeks.policy import bounded, choice


@dataclass(frozen=True)
class Neighbour:
    rel_pos: Vec
    velocity: Vec
    same_convoy: bool


@dataclass(frozen=True)
class FormationObservation:
    velocity: Vec
    neighbours: tuple[Neighbour, ...]
    goal: Vec
    lane_offset: Vec
    half_width: float
    max_speed: float
    slot_index: int
    convoy_size: int
    holding: bool


def _goal_velocity(obs: FormationObservation) -> Vec:
    speed = obs.max_speed
    if obs.holding:
        speed *= min(1.0, length(obs.goal) / 30.0)
    return scale(normalize(obs.goal), speed)


def _separation(obs: FormationObservation, radius: float) -> Vec:
    push = (0.0, 0.0)
    for n in obs.neighbours:
        d = length(n.rel_pos)
        if d < radius:
            push = sub(push, scale(normalize(n.rel_pos), (radius - d) / radius))
    return push


@dataclass(frozen=True)
class BoidsParams:
    neighbour_radius: float = bounded(80.0, 5.0, 300.0, "how far convoy mates count")
    separation: float = bounded(12.0, 1.0, 60.0, "distance below which to push apart")
    w_sep: float = bounded(2.0, 0.0, 5.0, "weight of separation")
    w_ali: float = bounded(0.5, 0.0, 5.0, "weight of matching mates' velocity")
    w_coh: float = bounded(0.3, 0.0, 5.0, "weight of moving to mates' centre")
    w_goal: float = bounded(1.0, 0.0, 5.0, "weight of heading to the waypoint")
    w_lane: float = bounded(0.8, 0.0, 5.0, "weight of staying inside the lane")


class BoidsFormation:
    def __init__(self, params: BoidsParams) -> None:
        self.params = params

    def steer(self, obs: FormationObservation) -> Vec:
        p = self.params
        mates = [
            n
            for n in obs.neighbours
            if n.same_convoy and length(n.rel_pos) < p.neighbour_radius
        ]
        ali = coh = (0.0, 0.0)
        if mates:
            k = 1.0 / len(mates)
            mean_vel = scale(
                (sum(n.velocity[0] for n in mates), sum(n.velocity[1] for n in mates)),
                k,
            )
            ali = sub(mean_vel, obs.velocity)
            centre = scale(
                (sum(n.rel_pos[0] for n in mates), sum(n.rel_pos[1] for n in mates)), k
            )
            coh = scale(
                normalize(centre), min(1.0, length(centre) / p.neighbour_radius)
            )
        lane = (0.0, 0.0)
        off = length(obs.lane_offset)
        core = obs.half_width * 0.5
        if off > core:
            lane = scale(normalize(obs.lane_offset), (off - core) / obs.half_width)
        steer = scale(_goal_velocity(obs), p.w_goal)
        push = add(scale(_separation(obs, p.separation), p.w_sep), scale(coh, p.w_coh))
        push = add(push, scale(lane, p.w_lane))
        steer = add(steer, scale(push, obs.max_speed))
        steer = add(steer, scale(ali, p.w_ali))
        return clamp_length(steer, obs.max_speed)


SHAPES = ("column", "line", "wedge")


def slot_offset(
    shape: str, index: int, size: int, heading: Vec, spacing: float, lateral: float
) -> Vec:
    """Offset of slot ``index`` from the formation centre."""
    k = index - (size - 1) / 2
    side = perp(heading)
    if shape == "column":
        return scale(heading, -k * spacing)
    if shape == "line":
        return scale(side, k * lateral)
    return add(scale(heading, -abs(k) * spacing), scale(side, k * lateral))


@dataclass(frozen=True)
class SlotParams:
    shape: str = choice("column", SHAPES, "formation shape relative to heading")
    spacing: float = bounded(15.0, 4.0, 60.0, "distance between neighbouring slots")
    w_slot: float = bounded(0.05, 0.0, 1.0, "gain toward the assigned slot")
    w_goal: float = bounded(1.0, 0.0, 5.0, "weight of heading to the waypoint")
    w_sep: float = bounded(2.0, 0.0, 5.0, "weight of the minimum-gap safety push")
    min_gap: float = bounded(8.0, 1.0, 40.0, "distance below which to push apart")


class SlotFormation:
    def __init__(self, params: SlotParams) -> None:
        self.params = params

    def steer(self, obs: FormationObservation) -> Vec:
        p = self.params
        mates = [n.rel_pos for n in obs.neighbours if n.same_convoy]
        k = 1.0 / (len(mates) + 1)
        centre = scale((sum(m[0] for m in mates), sum(m[1] for m in mates)), k)
        heading = normalize(obs.goal) if length(obs.goal) else normalize(obs.velocity)
        if heading == (0.0, 0.0):
            heading = (1.0, 0.0)
        lateral = min(p.spacing, 2 * obs.half_width / max(1, obs.convoy_size))
        offset = slot_offset(
            p.shape, obs.slot_index, obs.convoy_size, heading, p.spacing, lateral
        )
        to_slot = add(centre, offset)
        steer = scale(_goal_velocity(obs), p.w_goal)
        steer = add(steer, scale(to_slot, p.w_slot))
        steer = add(steer, scale(_separation(obs, p.min_gap), p.w_sep * obs.max_speed))
        return clamp_length(steer, obs.max_speed)
