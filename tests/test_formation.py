import math

import pytest

from apeks.formation import (
    BoidsFormation,
    BoidsParams,
    FormationObservation,
    Neighbour,
    SlotFormation,
    SlotParams,
    slot_offset,
)
from apeks.geometry import length


def observe(**kw) -> FormationObservation:
    base = dict(
        velocity=(0.0, 0.0),
        neighbours=(),
        goal=(100.0, 0.0),
        lane_offset=(0.0, 0.0),
        half_width=50.0,
        max_speed=2.0,
        slot_index=0,
        convoy_size=1,
        holding=False,
    )
    base.update(kw)
    return FormationObservation(**base)


def test_boids_alone_heads_for_goal_at_full_speed() -> None:
    v = BoidsFormation(BoidsParams()).steer(observe())
    assert v == pytest.approx((2.0, 0.0))


def test_boids_holding_slows_near_goal() -> None:
    v = BoidsFormation(BoidsParams()).steer(observe(goal=(15.0, 0.0), holding=True))
    assert v == pytest.approx((1.0, 0.0))


def test_boids_separates_from_close_vessels_of_any_convoy() -> None:
    p = BoidsParams(w_goal=0.0, w_ali=0.0, w_coh=0.0, w_lane=0.0)
    near = Neighbour(rel_pos=(3.0, 0.0), velocity=(0.0, 0.0), same_convoy=False)
    v = BoidsFormation(p).steer(observe(neighbours=(near,)))
    assert v[0] < 0 and v[1] == 0


def test_boids_aligns_and_coheres_with_own_convoy_only() -> None:
    p = BoidsParams(w_goal=0.0, w_sep=0.0, w_lane=0.0, w_ali=1.0, w_coh=0.0)
    mate = Neighbour(rel_pos=(30.0, 0.0), velocity=(0.0, 1.0), same_convoy=True)
    other = Neighbour(rel_pos=(30.0, 0.0), velocity=(0.0, -1.0), same_convoy=False)
    far = Neighbour(rel_pos=(500.0, 0.0), velocity=(1.0, 0.0), same_convoy=True)
    assert BoidsFormation(p).steer(observe(neighbours=(mate, other, far))) == (
        pytest.approx((0.0, 1.0))
    )
    p = BoidsParams(w_goal=0.0, w_sep=0.0, w_lane=0.0, w_ali=0.0, w_coh=1.0)
    v = BoidsFormation(p).steer(observe(neighbours=(mate,)))
    assert v[0] > 0 and v[1] == 0


def test_boids_returns_to_lane_when_outside_its_core() -> None:
    p = BoidsParams(w_goal=0.0)
    assert BoidsFormation(p).steer(
        observe(lane_offset=(0.0, 4.0), half_width=10.0)
    ) == ((0.0, 0.0))
    v = BoidsFormation(p).steer(observe(lane_offset=(0.0, 9.0), half_width=10.0))
    assert v[1] > 0


@pytest.mark.parametrize(
    ("shape", "index", "expected"),
    [
        ("column", 0, (10.0, 0.0)),
        ("column", 2, (-10.0, 0.0)),
        ("line", 0, (0.0, -10.0)),
        ("line", 2, (0.0, 10.0)),
        ("wedge", 0, (-10.0, -10.0)),
        ("wedge", 1, (0.0, 0.0)),
    ],
)
def test_slot_offsets(shape, index, expected) -> None:
    got = slot_offset(shape, index, 3, (1.0, 0.0), spacing=10.0, lateral=10.0)
    assert got == pytest.approx(expected)


def test_slot_formation_steers_toward_slot_and_compresses_in_narrow_lane() -> None:
    p = SlotParams(shape="line", spacing=20.0, w_slot=1.0, w_goal=0.0, w_sep=0.0)
    mate = Neighbour(rel_pos=(0.0, 20.0), velocity=(0.0, 0.0), same_convoy=True)
    wide = SlotFormation(p).steer(observe(neighbours=(mate,), convoy_size=2))
    assert wide == pytest.approx((0.0, 0.0))
    narrow = SlotFormation(p).steer(
        observe(neighbours=(mate,), convoy_size=2, half_width=5.0)
    )
    assert narrow[1] > 0


def test_slot_formation_keeps_min_gap_and_speed_limit() -> None:
    p = SlotParams(w_goal=0.0, w_slot=0.0)
    close = Neighbour(rel_pos=(0.0, 2.0), velocity=(0.0, 0.0), same_convoy=False)
    v = SlotFormation(p).steer(observe(neighbours=(close,)))
    assert v[1] < 0
    v = SlotFormation(SlotParams()).steer(observe())
    assert length(v) <= 2.0 + 1e-9


def test_slot_heading_falls_back_to_velocity_then_east() -> None:
    p = SlotParams(shape="column", spacing=10.0, w_slot=1.0, w_goal=0.0, w_sep=0.0)
    mate = Neighbour(rel_pos=(0.0, 0.0), velocity=(0.0, 0.0), same_convoy=True)
    moving = SlotFormation(p).steer(
        observe(goal=(0.0, 0.0), velocity=(0.0, 1.0), neighbours=(mate,), convoy_size=2)
    )
    assert moving == pytest.approx((0.0, 2.0))
    still = SlotFormation(p).steer(
        observe(goal=(0.0, 0.0), neighbours=(mate,), convoy_size=2)
    )
    assert math.isclose(still[0], 2.0) and math.isclose(still[1], 0.0, abs_tol=1e-9)
