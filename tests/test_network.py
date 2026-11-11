import math

import pytest

from apeks.network import Lane, SeaNetwork
from apeks.ocean import Ocean


def make_net() -> SeaNetwork:
    net = SeaNetwork()
    net.add_port("A", (0.0, 0.0), berths=2)
    net.add_node("W", (100.0, 0.0))
    net.add_port("B", (200.0, 0.0), berths=1)
    net.add_node("X", (100.0, 100.0))
    net.add_channel("A", "W", width=20.0, capacity=2)
    net.add_channel("W", "B", width=6.0, capacity=1)
    net.add_channel("A", "X", width=20.0, capacity=2)
    net.add_channel("X", "B", width=20.0, capacity=2)
    return net


def test_channels_create_both_directions() -> None:
    net = make_net()
    assert set(net.lanes) == {
        "A>W",
        "W>A",
        "W>B",
        "B>W",
        "A>X",
        "X>A",
        "X>B",
        "B>X",
    }
    lane = net.lanes["W>B"]
    assert lane == Lane("W>B", "W", "B", 6.0, 1)
    assert net.ports == {"A": 2, "B": 1}
    assert net.length(lane) == 100.0


def test_out_lanes_respects_exclusions() -> None:
    net = make_net()
    assert [lane.id for lane in net.out_lanes("A")] == ["A>W", "A>X"]
    assert [lane.id for lane in net.out_lanes("A", {"A>W"})] == ["A>X"]


def test_transit_estimate_is_directional_under_a_current() -> None:
    net = make_net()
    still = Ocean()
    assert net.transit_estimate(net.lanes["A>W"], still, 0.0, 2.0) == 50.0
    flowing = Ocean(base=(1.0, 0.0))
    assert net.transit_estimate(net.lanes["A>W"], flowing, 0.0, 2.0) == pytest.approx(
        100 / 3
    )
    assert net.transit_estimate(net.lanes["W>A"], flowing, 0.0, 2.0) == 100.0


def test_transit_estimate_with_cross_drift_requires_crabbing() -> None:
    net = make_net()
    ocean = Ocean(base=(0.0, 1.0))
    expected = 100 / math.sqrt(3)
    assert net.transit_estimate(net.lanes["A>W"], ocean, 0.0, 2.0) == pytest.approx(
        expected
    )


def test_transit_impossible_when_drift_overpowers() -> None:
    net = make_net()
    assert net.transit_estimate(net.lanes["A>W"], Ocean(base=(0.0, 3.0)), 0, 2.0) == (
        math.inf
    )
    assert net.transit_estimate(net.lanes["W>A"], Ocean(base=(3.0, 0.0)), 0, 2.0) == (
        math.inf
    )


def test_times_to_destination_uses_shortest_open_route() -> None:
    net = make_net()
    times = net.times_to("B", Ocean(), 0.0, 2.0)
    assert times["B"] == 0.0
    assert times["W"] == 50.0
    assert times["A"] == 100.0
    assert times["X"] == pytest.approx(math.hypot(100, 100) / 2)
    detour = net.times_to("B", Ocean(), 0.0, 2.0, exclude={"W>B"})
    assert detour["A"] == pytest.approx(math.hypot(100, 100))
    assert detour["W"] == pytest.approx(50.0 + math.hypot(100, 100))


def test_times_to_improves_a_node_found_first_by_a_slow_route() -> None:
    net = SeaNetwork()
    net.add_node("D", (0.0, 0.0))
    net.add_node("Q", (10.0, 0.0))
    net.add_node("P", (20.0, 0.0))
    net.add_channel("P", "D", width=5.0, capacity=1)
    net.add_channel("P", "Q", width=5.0, capacity=1)
    net.add_channel("Q", "D", width=5.0, capacity=1)
    costs = {"P>D": 100.0, "Q>D": 1.0, "P>Q": 1.0}
    net.transit_estimate = lambda lane, *_: costs.get(lane.id, 1000.0)
    assert net.times_to("D", Ocean(), 0.0, 1.0)["P"] == 2.0


def test_times_to_skips_unreachable_and_impassable() -> None:
    net = make_net()
    net.add_node("Z", (500.0, 500.0))
    times = net.times_to("B", Ocean(base=(0.0, 5.0)), 0.0, 2.0)
    assert "Z" not in times
    assert times == {"B": 0.0}
