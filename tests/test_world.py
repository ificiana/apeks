import random

import pytest

from apeks import config
from apeks.formation import BoidsFormation, BoidsParams
from apeks.llm import PolicyAuthor, StubClient
from apeks.network import SeaNetwork
from apeks.ocean import Ocean, Storm
from apeks.radio import LaneReport, Notice, Radio
from apeks.routing import Report, ScoredRouteParams, ScoredRouting
from apeks.world import Convoy, Event, Vessel, World, make_fleet, reverse


def radio(loss: float = 0.0) -> Radio:
    return Radio(random.Random(0), range=500.0, loss=loss, bandwidth=200, noise=0.0)


def line_net() -> SeaNetwork:
    net = SeaNetwork()
    net.add_port("P", (0.0, 0.0), berths=1)
    net.add_port("Q", (200.0, 0.0), berths=1)
    net.add_channel("P", "Q", width=40.0, capacity=1)
    return net


def detour_net() -> SeaNetwork:
    net = line_net()
    net.add_node("M", (100.0, 120.0))
    net.add_channel("P", "M", width=40.0, capacity=2)
    net.add_channel("M", "Q", width=40.0, capacity=2)
    return net


def make_world(net, specs, *, ocean=None, loss=0.0, events=(), authors=None) -> World:
    vessels, convoys = make_fleet(net, specs)
    return World(
        network=net,
        ocean=ocean or Ocean(),
        radio=radio(loss),
        vessels=vessels,
        convoys=convoys,
        routing=ScoredRouting(ScoredRouteParams()),
        formation=BoidsFormation(BoidsParams()),
        events=list(events),
        authors=authors or {},
    )


def run(world: World, ticks: int) -> None:
    for _ in range(ticks):
        world.step()


def test_vessel_thrust_is_acceleration_limited_and_drift_is_free() -> None:
    v = Vessel("a", "c", (0.0, 0.0))
    v.propel((5.0, 0.0), (0.0, 0.0), 1.0)
    assert v.vel == pytest.approx((config.MAX_ACCEL, 0.0))
    assert v.pos == pytest.approx((config.MAX_ACCEL, 0.0))
    assert v.energy == pytest.approx(config.MAX_ACCEL)
    drifting = Vessel("b", "c", (0.0, 0.0))
    drifting.propel((1.0, 0.0), (1.0, 0.0), 1.0)
    assert drifting.vel == (1.0, 0.0) and drifting.energy == 0.0


def test_convoy_hears_notices_and_dedupes_reports() -> None:
    c = Convoy("c", ["c.0"], ("P", "Q"), node="P")
    c.hear(Notice("P", "P>Q", True, "closed"))
    assert c.known_closed == {"P>Q"}
    c.hear(Notice("P", "P>Q", False, "open"))
    assert c.known_closed == set()
    report = LaneReport("x.0", "P>Q", 50.0, 10.0)
    c.hear(report)
    c.hear(report)
    assert c.reports == {"P>Q": [Report(50.0, 10.0)]}
    assert c.destination == "Q"


def test_reverse_lane_id() -> None:
    assert reverse("P>Q") == "Q>P"


def test_make_fleet_lines_vessels_up_at_first_port() -> None:
    vessels, convoys = make_fleet(line_net(), [("c", ("Q", "P"), 3)], spacing=10.0)
    assert [vessels[v].pos for v in convoys["c"].vessels] == [
        (190.0, 0.0),
        (200.0, 0.0),
        (210.0, 0.0),
    ]
    assert convoys["c"].node == "Q" and convoys["c"].destination == "P"


def test_convoy_sails_delivers_and_shares_a_lane_report() -> None:
    world = make_world(line_net(), [("c", ("P", "Q"), 2)])
    run(world, 400)
    s = world.summary()
    assert s["deliveries"] >= 1
    assert s["mean_trip_time"] > 0
    assert s["energy"] > 0
    assert world.convoys["c"].reports["P>Q"][0].transit > 0
    assert s["radio"]["sent"] > 0
    assert s["llm"] == {}


def test_blocked_lane_is_discovered_when_no_notice_arrives() -> None:
    closing = [Event(0.0, "close", "P>Q", "grounded vessel")]
    world = make_world(detour_net(), [("c", ("P", "Q"), 1)], loss=1.0, events=closing)
    world.step()
    assert world.closed == {"P>Q", "Q>P"}
    assert world.metrics["blocked_discoveries"] == 1
    assert world.convoys["c"].lane is None
    world.step()
    assert world.convoys["c"].lane == "P>M"
    assert "t=0 grounded vessel" in world.log


def test_port_notices_announce_closure_and_reopening() -> None:
    events = [
        Event(0.0, "close", "P>Q", "closed"),
        Event(5.0, "open", "P>Q", "reopened"),
    ]
    world = make_world(detour_net(), [("c", ("P", "Q"), 1)], events=events)
    world.step()
    c = world.convoys["c"]
    assert {"P>Q", "Q>P"} <= c.known_closed
    assert c.lane == "P>M"
    assert world.metrics["blocked_discoveries"] == 0
    run(world, config.NOTICE_EVERY)
    assert world.reopened == {"P>Q", "Q>P"} and world.closed == set()
    assert c.known_closed == set()


def test_convoy_with_no_open_lane_holds() -> None:
    world = make_world(line_net(), [("c", ("P", "Q"), 1)])
    world.convoys["c"].known_closed.add("P>Q")
    world.step()
    assert world.convoys["c"].lane is None


def test_single_berth_makes_second_convoy_wait() -> None:
    net = line_net()
    world = make_world(net, [("a", ("Q", "P", "Q"), 1), ("b", ("Q", "P", "Q"), 1)])
    for c in world.convoys.values():
        c.order_idx = 0
    world.step()
    assert world.convoys["a"].state == "berthed"
    assert world.convoys["b"].state == "queued"
    run(world, config.SERVICE_TICKS)
    assert world.metrics["deliveries"] == 1
    assert world.convoys["a"].state == "sailing"
    assert world.convoys["b"].state == "berthed"
    assert world.metrics["wait_ticks"] == config.SERVICE_TICKS


def test_overfull_lane_counts_congestion_and_slows() -> None:
    world = make_world(line_net(), [("a", ("P", "Q"), 1), ("b", ("P", "Q"), 1)])
    run(world, 30)
    assert world.metrics["congestion_ticks"] > 0
    single = make_world(line_net(), [("a", ("P", "Q"), 1)])
    run(single, 30)
    assert world.vessels["a.0"].pos[0] < single.vessels["a.0"].pos[0]


def test_near_miss_counts_once_per_encounter() -> None:
    world = make_world(line_net(), [("a", ("P", "Q"), 1), ("b", ("P", "Q"), 1)])
    world.step()
    world.step()
    assert world.metrics["near_misses"] == 1


def test_llm_author_rewrites_policy_on_triggers_after_latency() -> None:
    stub = StubClient(['{"avoid": ["P>Q"]}'] * 3)
    author = PolicyAuthor("routing", "Pick lanes.", ScoredRouteParams, stub, 5, 2.0)
    ocean = Ocean(storms=(Storm((500.0, 500.0), 10.0, 1.0, 1.0, 3.0),))
    world = make_world(
        line_net(), [("c", ("P", "Q"), 1)], ocean=ocean, authors={"routing": author}
    )
    run(world, 4)
    assert world.routing.params.avoid == ("P>Q",)
    assert author.calls == 3
    assert any("new routing policy" in line for line in world.log)
    assert any("storm forming" in line for line in world.log)
    assert any("cleared" in line for line in world.log)
    assert '"trigger": "mission start"' in stub.prompts[0]
    assert world.summary()["llm"]["routing"] == {
        "calls": 3,
        "failures": 0,
        "skipped": 0,
    }


def test_formation_author_targets_formation_engine() -> None:
    stub = StubClient(['{"w_sep": 4}'])
    author = PolicyAuthor("formation", "Keep shape.", BoidsParams, stub, 5, 0.0)
    world = make_world(
        line_net(), [("c", ("P", "Q"), 1)], authors={"formation": author}
    )
    world.step()
    assert world.formation.params.w_sep == 4.0


def test_situation_reports_storms_closures_and_delays() -> None:
    ocean = Ocean(storms=(Storm((50.0, 50.0), 20.0, 1.0, 0.0, 9.0),))
    world = make_world(line_net(), [("c", ("P", "Q"), 1)], ocean=ocean)
    world.closed = {"P>Q"}
    world.convoys["c"].reports = {
        "P>Q": [Report(500.0, 0.0)],
        "Q>P": [Report(9.0, -900.0)],
    }
    s = world.situation("storm")
    assert s["trigger"] == "storm" and s["closed_lanes"] == ["P>Q"]
    assert s["storms"] == [{"centre": (50.0, 50.0), "radius": 20.0, "until": 9.0}]
    assert s["reported_delay_by_lane"]["P>Q"] > 0
    assert "Q>P" not in s["reported_delay_by_lane"]
    assert s["lanes"]["P>Q"] == {"width": 40.0, "capacity": 1}


def test_summary_without_trips_has_no_mean() -> None:
    world = make_world(line_net(), [("c", ("P", "Q"), 1)])
    assert world.summary()["mean_trip_time"] is None


def test_tracer_records_decisions_events_and_snapshots() -> None:
    import io
    import json

    from apeks.trace import Tracer

    buf = io.StringIO()
    events = [
        Event(0.0, "close", "P>Q", "closed"),
        Event(30.0, "open", "P>Q", "reopened"),
    ]
    stub = StubClient(['{"w_report": 2}', "oops"])
    author = PolicyAuthor("routing", "Pick.", ScoredRouteParams, stub, 1, 1.0)
    world = make_world(
        detour_net(),
        [("a", ("P", "Q"), 2), ("b", ("P", "Q"), 1)],
        events=events,
        authors={"routing": author},
    )
    world.tracer = Tracer(buf, snapshot_every=10)
    run(world, 400)
    records = [json.loads(line) for line in buf.getvalue().splitlines()]
    kinds = {r["kind"] for r in records}
    assert {
        "event",
        "policy_request",
        "policy_applied",
        "notice_heard",
        "route_choice",
        "lane_start",
        "lane_end",
        "queued",
        "berthed",
        "delivered",
        "near_miss",
        "snapshot",
    } <= kinds
    snaps = [r for r in records if r["kind"] == "snapshot"]
    assert [s["t"] for s in snaps[:2]] == [0.0, 10.0]
    assert snaps[0]["vessels"][0]["id"] == "a.0"
    assert snaps[0]["routing_params"]["w_report"] == 1.0
    assert "prompt" in next(r for r in records if r["kind"] == "policy_request")
    static = world.static()
    assert static["convoys"][0] == {
        "id": "a",
        "orders": ("P", "Q"),
        "vessels": ["a.0", "a.1"],
    }


def test_tracer_records_blocked_lane() -> None:
    import io

    from apeks.trace import Tracer

    buf = io.StringIO()
    closing = [Event(0.0, "close", "P>Q", "x")]
    world = make_world(detour_net(), [("c", ("P", "Q"), 1)], loss=1.0, events=closing)
    world.tracer = Tracer(buf, snapshot_every=0)
    world.step()
    assert '"kind": "blocked"' in buf.getvalue()
