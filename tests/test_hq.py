from apeks.hq import Hq
from apeks.radio import Backlog, Hazard, LaneReport


def test_uplinked_reports_arrive_after_the_delay_and_trigger_on_change() -> None:
    hq = Hq(delay=10.0)
    hazard = Hazard("port:P", "A>B", True, 3.0, "c1")
    hq.send(5.0, hazard)
    hq.send(5.0, Hazard("port:Q", "A>B", True, 3.0, "c2"))
    hq.send(5.0, LaneReport("c1.0", "A>B", 40.0, 2.0))
    hq.send(6.0, LaneReport("c1.0", "A>B", 40.0, 2.0))
    assert hq.receive(14.0) == []
    assert hq.hazards.closed() == set()
    assert hq.receive(15.0) == ["fleet report: A>B closed"]
    assert hq.hazards.closed() == {"A>B", "B>A"}
    assert hq.hazards.entries["A>B"].sources == {"c1", "c2"}
    assert len(hq.reports["A>B"]) == 1
    hq.receive(16.0)
    assert len(hq.reports["A>B"]) == 1
    hq.send(20.0, Hazard("port:P", "A>B", False, 18.0, "c3"))
    assert hq.receive(30.0) == ["fleet report: A>B open"]


def test_hq_keeps_the_newest_backlog_per_port_without_a_trigger() -> None:
    hq = Hq(2.0)
    hq.send(0.0, Backlog("port:P", "P", 5, 0.0))
    hq.send(1.0, Backlog("port:P", "P", 3, 1.0))
    hq.send(1.0, Backlog("port:P", "P", 9, -5.0))
    assert hq.receive(1.0) == []
    assert hq.receive(3.0) == []
    assert hq.backlogs == {"P": (3, 1.0)}
