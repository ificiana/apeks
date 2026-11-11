import random

from apeks.radio import LaneReport, Notice, PositionReport, Radio, Received


def test_broadcast_reaches_receivers_in_range_but_not_the_sender() -> None:
    radio = Radio(random.Random(1), range=100.0, loss=0.0, bandwidth=10, noise=0.0)
    msg = PositionReport(sender="v1", convoy="c1", velocity=(1.0, 0.0))
    radio.broadcast(msg, (0.0, 0.0))
    got = radio.deliver({"v1": (0.0, 0.0), "v2": (30.0, 40.0), "v3": (200.0, 0.0)})
    assert got == {"v2": [Received(msg, (-30.0, -40.0))]}
    assert radio.stats == {"sent": 1, "delivered": 1, "lost": 0, "overflow": 0}
    assert radio.deliver({"v2": (0.0, 0.0)}) == {}


def test_signal_position_estimate_is_noisy_with_distance() -> None:
    radio = Radio(random.Random(1), range=100.0, loss=0.0, bandwidth=10, noise=0.1)
    radio.broadcast(LaneReport("v1", "A>B", 12.0, 3.0), (0.0, 0.0))
    (rx,) = radio.deliver({"v2": (50.0, 0.0)})["v2"]
    assert rx.rel_pos != (-50.0, 0.0)
    assert abs(rx.rel_pos[0] + 50.0) < 25.0


def test_packets_can_be_lost_or_overflow_the_channel() -> None:
    radio = Radio(random.Random(3), range=100.0, loss=0.5, bandwidth=2, noise=0.0)
    for i in range(3):
        radio.broadcast(Notice(f"s{i}", "A>B", True, "closed"), (0.0, 0.0))
    got = radio.deliver({f"v{i}": (1.0, 0.0) for i in range(20)})
    received = sum(len(v) for v in got.values())
    assert radio.stats["overflow"] == 1
    assert radio.stats["sent"] == 3
    assert received == radio.stats["delivered"]
    assert radio.stats["lost"] + received == 40
    assert 0 < radio.stats["lost"] < 40
