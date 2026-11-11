import random

import pytest

from apeks import config, mapgen
from apeks.mapgen import Geography, generate, is_connected, segment_clear


def small(ports: int = 5, adjacency: int = 3, seed: int = 24008) -> mapgen.Sea:
    return generate(seed, ports, adjacency, ticks=6000)


def test_geography_reads_land_by_position() -> None:
    geo = Geography(40.0, 20.0, 10.0, ((False, True, False, False),) * 2)
    assert geo.cols == 4 and geo.rows == 2
    assert geo.is_land((15.0, 5.0)) and not geo.is_land((5.0, 15.0))
    assert geo.is_land((-1.0, 5.0)) and geo.is_land((5.0, 25.0))  # off the map
    assert segment_clear(geo, (5.0, 5.0), (5.0, 15.0))
    assert not segment_clear(geo, (5.0, 5.0), (35.0, 5.0))


def test_open_sea_has_no_land() -> None:
    geo = Geography.open_sea(100.0, 50.0)
    assert not geo.is_land((99.0, 49.0)) and geo.cols == 2


def test_generation_is_seeded() -> None:
    a, b = small(), small()
    assert a.network.nodes == b.network.nodes and a.geography == b.geography
    assert small(seed=1).network.nodes != a.network.nodes


@pytest.mark.parametrize("ports", [5, 10, 20])
@pytest.mark.parametrize("adjacency", [2, 3, 4])
def test_ports_on_shore_lanes_at_sea_and_map_connected(ports, adjacency) -> None:
    sea = small(ports, adjacency)
    net, geo = sea.network, sea.geography
    assert list(net.ports) == [f"P{i}" for i in range(1, ports + 1)]
    assert set(net.ports.values()) == {config.BERTHS}
    assert sea.fleet == dict.fromkeys(net.ports, config.SHIPS_PER_PORT)
    for name, pos in net.nodes.items():
        assert not geo.is_land(pos)
        if name in net.ports:
            assert geo.near_land(pos)
    for lane in net.lanes.values():
        assert segment_clear(geo, net.nodes[lane.a], net.nodes[lane.b])
    assert is_connected(net.nodes, {(lane.a, lane.b) for lane in net.lanes.values()})


def test_denser_adjacency_gives_more_lanes_and_area_grows_with_ports() -> None:
    assert len(small(10, 4).network.lanes) > len(small(10, 2).network.lanes)
    five, twenty = small(5), small(20)
    assert twenty.geography.width == pytest.approx(2 * five.geography.width)


def test_events_close_and_reopen_lanes_without_cutting_the_map() -> None:
    sea = small(10, 2)
    net = sea.network
    edges = {(lane.a, lane.b) for lane in net.lanes.values()}
    closes = [e for e in sea.events if e.kind == "close"]
    assert len(closes) == len(sea.events) // 2 >= 1
    for e in closes:
        a, b = e.lane.split(">")
        assert is_connected(net.nodes, edges - {(a, b), (b, a)})
        assert any(
            o.kind == "open" and o.lane == e.lane and o.at > e.at for o in sea.events
        )
        assert 0 < e.at < 6000
    assert sea.events == sorted(sea.events, key=lambda e: e.at)


def test_storms_and_currents_sit_in_the_ocean() -> None:
    sea = small(10)
    assert len(sea.ocean.storms) == 2 and len(sea.ocean.currents) == 4
    for s in sea.ocean.storms:
        assert not sea.geography.is_land(s.centre) and 0 < s.start < s.end < 6000
    for c in sea.ocean.currents:
        assert not sea.geography.is_land(c.centre)


def test_is_connected() -> None:
    assert is_connected({"a": (0, 0), "b": (1, 0)}, {("a", "b")})
    assert not is_connected({"a": (0, 0), "b": (1, 0)}, set())
    assert is_connected({}, set())


def test_generation_retries_until_the_map_connects(monkeypatch) -> None:
    calls = []
    real = mapgen._attempt

    def flaky(seed, ports, adjacency, ticks, attempt):
        calls.append(attempt)
        return None if attempt == 0 else real(seed, ports, adjacency, ticks, attempt)

    monkeypatch.setattr(mapgen, "_attempt", flaky)
    assert small().network.ports
    assert calls == [0, 1]
    monkeypatch.setattr(mapgen, "_attempt", lambda *a: None)
    with pytest.raises(RuntimeError):
        small()


def test_a_sea_without_shore_has_no_ports(monkeypatch) -> None:
    monkeypatch.setattr(mapgen, "LAND_SHARE", 0.0)
    assert mapgen._attempt(1, 5, 3, 6000, 0) is None


def test_bridges_parts_out_of_sight_with_waypoints_on_water() -> None:
    wall = tuple(tuple(c == 5 and r < 8 for c in range(10)) for r in range(10))
    geo = Geography(100.0, 100.0, 10.0, wall)
    nodes = {"P1": (25.0, 15.0), "P2": (85.0, 15.0)}
    edges = mapgen._lanes(geo, nodes, 3)
    assert len(nodes) > 2 and is_connected(nodes, edges)
    for a, b in edges:
        assert segment_clear(geo, nodes[a], nodes[b])


def test_parts_in_sight_join_by_their_shortest_lane() -> None:
    geo = Geography.open_sea(400.0, 100.0)
    nodes = {
        "A": (10.0, 10.0),
        "B": (30.0, 10.0),
        "C": (300.0, 10.0),
        "D": (320.0, 10.0),
    }
    edges = mapgen._lanes(geo, nodes, 1)
    assert edges == {("A", "B"), ("C", "D"), ("B", "C")}


def test_no_closure_when_every_lane_is_a_bridge() -> None:
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0)}
    edges = {("A", "B"), ("B", "C")}
    assert mapgen._closures(random.Random(0), nodes, edges, 2, 6000) == []


def test_rejects_too_few_ports_or_lanes() -> None:
    with pytest.raises(ValueError):
        generate(1, 1, 3, 6000)
    with pytest.raises(ValueError):
        generate(1, 5, 0, 6000)
