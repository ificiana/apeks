import random

import pytest

from apeks import config
from apeks.cargo import Package
from apeks.formation import BoidsFormation, BoidsParams
from apeks.hq import Hq
from apeks.llm import PolicyAuthor, StubClient
from apeks.network import SeaNetwork
from apeks.news import NewsItem
from apeks.ocean import Ocean, Storm
from apeks.radio import Backlog, Hazard, LaneReport, PolicyMsg, Radio
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


def make_world(
    net,
    specs,
    *,
    ocean=None,
    loss=0.0,
    events=(),
    authors=None,
    news=(),
    delay=0.0,
    range_=500.0,
    packages=(),
) -> World:
    """``specs`` are ``(id, (start, destination), size)``; no destination docks."""
    fleet = [
        (cid, ports[0], n, ports[1] if len(ports) > 1 else None)
        for cid, ports, n in specs
    ]
    vessels, convoys = make_fleet(net, fleet)
    world = World(
        network=net,
        ocean=ocean or Ocean(),
        radio=radio(loss),
        vessels=vessels,
        convoys=convoys,
        routing=ScoredRouting(ScoredRouteParams()),
        formation=BoidsFormation(BoidsParams()),
        events=list(events),
        authors=authors or {},
        news=list(news),
        hq=Hq(delay),
        packages=list(packages),
    )
    world.radio.range = range_
    return world


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


def test_convoy_hears_hazards_policies_and_dedupes_reports() -> None:
    c = Convoy("c", ["c.0"], "P", "Q")
    closed = Hazard("x.0", "P>Q", True, 10.0, "x")
    assert c.hear(closed, 10.0) == "changed"
    assert c.known_closed == {"P>Q", "Q>P"}
    assert c.outbox == [Hazard("c.0", "P>Q", True, 10.0, "x", hops=1)]
    assert c.hear(closed, 11.0) is None
    assert c.hear(Hazard("x.0", "P>Q", False, 20.0, "y"), 20.0 + 1000) == "changed"
    assert c.known_closed == set() and len(c.outbox) == 1
    assert c.hear(Hazard("x.0", "P>Q", True, 30.0, "x", hops=3), 30.0) == "changed"
    assert len(c.outbox) == 1
    policy = PolicyMsg("port:P", "routing", 2, "params")
    assert c.hear(policy, 0.0) == "adopted"
    assert c.policies["routing"] == (2, "params")
    assert c.hear(PolicyMsg("port:P", "routing", 1, "old"), 0.0) is None
    report = LaneReport("x.0", "P>Q", 50.0, 10.0)
    assert c.hear(report, 0.0) is None
    c.hear(report, 0.0)
    assert c.reports == {"P>Q": [Report(50.0, 10.0)]}
    assert c.destination == "Q"


def test_reverse_lane_id() -> None:
    assert reverse("P>Q") == "Q>P"


def test_make_fleet_lines_vessels_up_at_first_port() -> None:
    vessels, convoys = make_fleet(line_net(), [("c", "Q", 3, "P")], spacing=10.0)
    assert [vessels[v].pos for v in convoys["c"].vessels] == [
        (190.0, 0.0),
        (200.0, 0.0),
        (210.0, 0.0),
    ]
    assert convoys["c"].node == "Q" and convoys["c"].destination == "P"


def test_convoy_sails_delivers_and_shares_a_lane_report() -> None:
    world = make_world(line_net(), [("c", ("P", "Q"), 2)])
    world.vessels["c.0"].cargo = [Package("P#0", "P", "Q", 0.0)]
    run(world, 400)
    s = world.summary()
    assert s["deliveries"] == 1
    assert s["mean_latency"] > 0
    assert s["energy"] > 0
    assert world.convoys["c"].reports["P>Q"][0].transit > 0
    assert s["radio"]["sent"] > 0
    assert s["llm"] == {}


def test_convoy_sees_a_closure_where_it_stands_and_tells_others() -> None:
    closing = [Event(0.0, "close", "P>Q", "grounded vessel")]
    world = make_world(detour_net(), [("c", ("P", "Q"), 1)], events=closing)
    world.step()
    c = world.convoys["c"]
    assert world.closed == {"P>Q", "Q>P"} and c.known_closed == {"P>Q", "Q>P"}
    assert world.metrics["blocked_discoveries"] == 1
    assert c.lane == "P>M"
    assert world.radio.by_type["Hazard"] == 1
    assert "t=0 grounded vessel" in world.log
    world.step()
    assert world.stations["P"].hazards.closed() == {"P>Q", "Q>P"}
    assert world.metrics["blocked_discoveries"] == 1


def test_nobody_knows_a_closure_nobody_has_seen() -> None:
    closing = [Event(0.0, "close", "M>Q", "x")]
    world = make_world(detour_net(), [("c", ("P", "Q"), 1)], events=closing)
    run(world, 3)
    assert world.convoys["c"].known_closed == set()
    assert world.stations["P"].hazards.closed() == set()
    assert world.metrics["belief_missed"] == 6
    assert world.metrics["hq_belief_missed"] == 6


def test_closure_spreads_by_port_station_and_reaches_hq() -> None:
    net = detour_net()
    net.add_port("R", (400.0, 0.0), berths=1)
    net.add_channel("Q", "R", width=40.0, capacity=1)
    closing = [Event(0.0, "close", "P>Q", "x")]
    world = make_world(
        net,
        [("a", ("P", "Q"), 1), ("b", ("R", "Q"), 1)],
        events=closing,
        delay=5.0,
        range_=250.0,
    )
    world.step()
    world.step()
    assert world.convoys["b"].known_closed == set()
    assert world.stations["Q"].hazards.closed() == {"P>Q", "Q>P"}
    run(world, config.NOTICE_EVERY - 1)
    assert world.convoys["b"].known_closed == {"P>Q", "Q>P"}
    assert world.hq.hazards.closed() == {"P>Q", "Q>P"}
    assert world.hq.hazards.entries["P>Q"].sources == {"a"}


def test_reopening_is_seen_spread_and_then_no_longer_repeated() -> None:
    events = [Event(0.0, "close", "P>Q", "x"), Event(2.0, "open", "P>Q", "y")]
    world = make_world(
        detour_net(), [("c", ("P", "Q"), 1), ("d", ("Q", "P"), 1)], events=events
    )
    world.convoys["c"].state = "berthed"
    world.convoys["c"].service_until = 1e9
    world.step()
    world.step()
    assert world.convoys["c"].known_closed == {"P>Q", "Q>P"}
    world.convoys["c"].state = "sailing"
    run(world, 2)
    assert world.convoys["c"].known_closed == set()
    assert world.convoys["c"].lane == "P>Q"
    run(world, config.NOTICE_EVERY)
    station = world.stations["P"]
    assert station.hazards.closed() == set()
    assert station.hazards.fresh(world.t, config.NOTICE_MAX_AGE)
    assert not station.hazards.fresh(world.t + 1e4, config.NOTICE_MAX_AGE)


def test_convoy_with_no_open_lane_holds() -> None:
    world = make_world(line_net(), [("c", ("P", "Q"), 1)])
    world.convoys["c"].hazards.update("P>Q", True, 0.0, "x")
    world.step()
    assert world.convoys["c"].lane is None


def test_lane_reports_are_noisy_on_air_but_exact_for_the_sender() -> None:
    world = make_world(
        line_net(), [("a", ("P", "Q"), 1), ("b", ("P", "Q", "P"), 1)], delay=0.0
    )
    world.convoys["b"].state = "berthed"
    world.convoys["b"].service_until = 1e9
    run(world, 200)
    own = world.convoys["a"].reports["P>Q"][0]
    heard = world.convoys["b"].reports["P>Q"][0]
    assert own.finished_at == heard.finished_at
    assert own.transit != heard.transit
    assert world.hq.reports["P>Q"][0].transit == heard.transit


def test_single_berth_makes_second_convoy_wait() -> None:
    net = line_net()
    world = make_world(net, [("a", ("P", "P"), 1), ("b", ("P", "P"), 1)])
    world.vessels["a.0"].cargo = [Package("Q#0", "Q", "P", 0.0)]
    world.step()
    assert world.convoys["a"].state == "berthed"
    assert world.convoys["b"].state == "queued"
    service = config.SERVICE_TICKS + config.HANDLING_TIME
    assert world.convoys["a"].service_until == service
    run(world, service)
    assert world.metrics["deliveries"] == 1
    assert world.convoys["a"].state == "docked"
    assert world.convoys["b"].state == "berthed"
    assert world.metrics["wait_ticks"] == service
    assert world.metrics["arrivals"] == 2


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


def test_hq_writes_policy_on_news_and_ports_hand_it_to_convoys() -> None:
    stub = StubClient(['{"avoid": ["P>Q"]}'] * 3)
    author = PolicyAuthor("routing", "Pick lanes.", ScoredRouteParams, stub, 5, 2.0)
    news = [
        NewsItem(3.0, "Wire", "Gale near Q.", "storm"),
        NewsItem(3.0, "Radio", "Rumour.", None),
        NewsItem(1e9, "Wire", "Never read.", None),
    ]
    world = make_world(
        line_net(),
        [("c", ("P", "Q"), 1), ("d", ("Q", "P"), 1)],
        authors={"routing": author},
        news=news,
        delay=1.0,
    )
    run(world, 2 * config.NOTICE_EVERY + 1)
    assert author.calls == 2
    assert '"mission start"' not in stub.prompts[0]
    assert "Trigger: mission start" in stub.prompts[0]
    assert "Trigger: news: Gale near Q.; news: Rumour." in stub.prompts[1]
    assert "[t=3, 0 ticks ago, Wire] Gale near Q." in stub.prompts[1]
    assert world.policies["routing"][0] == 2
    assert world.convoys["c"].policies["routing"][1].avoid == ("P>Q",)
    assert world.routing.params.avoid == ()
    assert any("new routing policy v1" in line for line in world.log)
    s = world.summary()
    assert s["llm"]["routing"]["calls"] == 2
    assert s["llm"]["routing"]["input_tokens"] == 0
    first = s["policies"][0]
    assert first["module"] == "routing" and first["version"] == 1
    assert first["adopted_by"] == 0 and first["to_all"] is None
    second = s["policies"][1]
    assert second["adopted_by"] == 2 and second["to_all"] == second["to_half"]
    assert world.radio.stats["relayed"] > 0


def test_fleet_report_reaching_hq_triggers_a_new_policy() -> None:
    stub = StubClient()
    author = PolicyAuthor("formation", "Keep shape.", BoidsParams, stub, 5, 0.0)
    closing = [Event(0.0, "close", "P>Q", "x")]
    world = make_world(
        detour_net(),
        [("c", ("P", "Q"), 1)],
        authors={"formation": author},
        events=closing,
        delay=2.0,
    )
    run(world, 4)
    assert author.calls == 2
    assert "Trigger: fleet report: P>Q closed" in stub.prompts[1]
    assert world.convoys["c"].policies["formation"][0] == 0


def test_station_ignores_beacons_and_policies() -> None:
    world = make_world(line_net(), [("c", ("P", "Q"), 1)])
    world.convoys["c"].outbox.append(PolicyMsg("c.0", "routing", 1, None, hops=1))
    world.step()
    assert world.stations["P"].policies == {}
    assert all(isinstance(m, Backlog) for _, m in world.hq._inbox)


def test_situation_is_built_from_what_reached_hq() -> None:
    world = make_world(line_net(), [("c", ("P", "Q"), 1)])
    world.closed = {"P>Q"}
    world.convoys["c"].reports = {"P>Q": [Report(500.0, 0.0)]}
    world.hq.hazards.update("Q>P", True, 0.0, "c")
    world.hq.reports = {"P>Q": [Report(500.0, 0.0)], "Q>P": [Report(9.0, -900.0)]}
    world.hq.news = [NewsItem(0.0, "Wire", "Hello.", "x")]
    world.t = 10.0
    s = world.situation("storm")
    assert s["trigger"] == "storm"
    assert s["fleet_reports"]["hazards"] == [
        {"lane": "Q>P", "state": "closed", "age": 10, "confirmations": 1}
    ]
    assert s["fleet_reports"]["reported_delay_by_lane"]["P>Q"] > 0
    assert "Q>P" not in s["fleet_reports"]["reported_delay_by_lane"]
    assert s["map"]["lanes"]["P>Q"] == {"width": 40.0, "capacity": 1}
    assert s["map"]["ports"] == ["P", "Q"]
    assert s["news"] == [
        {"published": 0.0, "age": 10, "source": "Wire", "text": "Hello."}
    ]
    assert "truth" not in str(s)


def test_summary_without_trips_has_no_mean() -> None:
    world = make_world(line_net(), [("c", ("P", "Q"), 1)])
    s = world.summary()
    assert s["mean_latency"] is None and s["energy_per_delivery"] is None
    assert s["beliefs"]["belief_false"] == 0


def test_summary_energy_per_delivery() -> None:
    world = make_world(line_net(), [("c", ("P", "Q"), 2)])
    world.vessels["c.1"].cargo = [Package("P#0", "P", "Q", 0.0)]
    run(world, 400)
    s = world.summary()
    assert s["energy_per_delivery"] == round(s["energy"] / s["deliveries"], 1)
    assert s["radio"]["by_type"]["PositionReport"] > 0


def test_false_belief_is_counted() -> None:
    world = make_world(line_net(), [("c", ("P", "Q"), 1)])
    world.convoys["c"].hazards.update("Q>P", True, 0.0, "x")
    world.hq.hazards.update("Q>P", True, 0.0, "x")
    world.convoys["c"].state = "queued"
    world.step()
    assert world.metrics["belief_false"] == 1
    assert world.metrics["hq_belief_false"] == 1


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
        news=[NewsItem(5.0, "Wire", "Strait closed.", "close")],
    )
    world.tracer = Tracer(buf, snapshot_every=10)
    run(world, 400)
    records = [json.loads(line) for line in buf.getvalue().splitlines()]
    kinds = {r["kind"] for r in records}
    assert {
        "event",
        "policy_request",
        "hazard_seen",
        "news",
        "policy_decided",
        "policy_at_ports",
        "route_choice",
        "lane_start",
        "lane_end",
        "queued",
        "berthed",
        "delivered",
        "snapshot",
    } <= kinds
    snaps = [r for r in records if r["kind"] == "snapshot"]
    assert [s["t"] for s in snaps[:2]] == [0.0, 10.0]
    assert snaps[0]["vessels"][0]["id"] == "a.0"
    assert snaps[0]["policies"]["routing"]["params"]["w_report"] == 1.0
    assert snaps[0]["convoys"][0]["policy_versions"] == {"routing": 0, "formation": 0}
    assert "prompt" in next(r for r in records if r["kind"] == "policy_request")
    static = world.static()
    assert static["convoys"][0] == {
        "id": "Q-0",
        "port": "Q",
        "vessels": ["a.0", "a.1", "b.0"],
    }


def test_tracer_records_blocked_lane() -> None:
    import io

    from apeks.trace import Tracer

    buf = io.StringIO()
    closing = [Event(0.0, "close", "P>Q", "x")]
    world = make_world(detour_net(), [("c", ("P", "Q"), 1)], loss=1.0, events=closing)
    world.tracer = Tracer(buf, snapshot_every=0)
    world.step()
    assert '"kind": "hazard_seen"' in buf.getvalue()


def test_storms_forming_and_clearing_are_logged() -> None:
    ocean = Ocean(storms=(Storm((0.0, 0.0), 10.0, 1.0, 1.0, 2.0),))
    world = make_world(line_net(), [("c", ("P", "Q"), 1)], ocean=ocean)
    run(world, 4)
    assert any("storm forming" in line for line in world.log)
    assert any("cleared" in line for line in world.log)


def test_regroup_merges_docks_loads_and_sends_full_ships() -> None:
    pkgs = [Package(f"P#{i}", "P", "Q", 0.0) for i in range(8)]
    world = make_world(
        line_net(),
        [("a", ("P",), 1), ("b", ("P",), 1), ("c", ("P",), 1)],
        packages=pkgs,
    )
    world.convoys["b"].policies["routing"] = (3, world.policies["routing"][1])
    world.adoption[("routing", 3)] = {"decided": 0.0, "adopted": {"b.0": 0.0}}
    world.convoys["a"].backlogs["P"] = (2, 1.0)
    world.step()
    assert set(world.convoys) == {"P-0", "P-1"}
    gone, rest = world.convoys["P-0"], world.convoys["P-1"]
    assert gone.vessels == ["a.0", "b.0"] and gone.state == "loading"
    assert gone.destination == "Q" and gone.service_until == 4 * config.HANDLING_TIME
    assert rest.vessels == ["c.0"] and rest.state == "docked"
    assert world.vessels["c.0"].cargo == [] and world.waiting["P"] == []
    assert gone.policies["routing"][0] == 3 and gone.backlogs["P"] == (2, 1.0)
    assert world.adoption[("routing", 3)]["adopted"] == {
        "a.0": 0.0,
        "b.0": 0.0,
        "c.0": 0.0,
    }
    run(world, int(gone.service_until))
    assert gone.state == "sailing"


def test_regroup_after_max_wait_sends_everyone_and_counts_empty_sailing() -> None:
    world = make_world(line_net(), [("a", ("P",), 2)])
    world.convoys["a"].backlogs["Q"] = (3, 1e6)
    world.step()
    assert set(world.convoys) == {"a"}
    run(world, config.MAX_WAIT)
    (convoy,) = world.convoys.values()
    assert convoy.vessels == ["a.0", "a.1"] and convoy.destination == "Q"
    run(world, 20)
    s = world.summary()
    assert s["empty_distance"] > 0 and s["load_factor"] == 0.0
    assert s["arrivals"] == 0 and s["backlog"] == 0


def test_lone_docked_convoy_with_nowhere_to_go_stays() -> None:
    world = make_world(line_net(), [("a", ("P",), 1)])
    run(world, config.MAX_WAIT + 1)
    assert list(world.convoys) == ["a"]


def test_convoy_absorbs_knowledge_and_hears_backlogs() -> None:
    a, b = Convoy("a", ["a.0"], "P"), Convoy("b", ["b.0"], "P")
    b.hazards.update("P>Q", True, 5.0, "x")
    b._add_report(("x", "P>Q", 5.0), Report(9.0, 5.0))
    b.policies["routing"] = (2, "p")
    a.policies["routing"] = (4, "q")
    assert a.hear(Backlog("port:Q", "Q", 3, 2.0), 2.0) is None
    assert a.hear(Backlog("port:Q", "Q", 1, 1.0), 2.0) is None
    b.backlogs = {"Q": (7, 9.0), "R": (1, 0.0)}
    a.absorb(b)
    a.absorb(b)
    assert a.known_closed == {"P>Q"}
    assert a.reports == {"P>Q": [Report(9.0, 5.0)]}
    assert a.policies["routing"] == (4, "q")
    assert a.backlogs == {"Q": (7, 9.0), "R": (1, 0.0)}
    k = a.knowledge()
    assert k["backlogs"] == a.backlogs and k["backlogs"] is not a.backlogs
