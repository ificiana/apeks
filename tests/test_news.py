import math
import random

from apeks.news import (
    NewsSettings,
    channel,
    forecast,
    generate,
    place_name,
    severity,
)
from apeks.ocean import Storm
from apeks.world import Event

PLACES = {"P": (0.0, 0.0), "Q": (100.0, 0.0), "R": (0.0, 300.0)}
LANES = ["P>Q", "Q>P"]


def exact(**kw) -> NewsSettings:
    base = dict(
        delay=0.0,
        location_error=0.0,
        severity_error=0.0,
        forecast_error=0.0,
        false_alarm_rate=0.0,
        miss_rate=0.0,
        correction_delay=0.0,
    )
    return NewsSettings(**{**base, **kw})


def test_helpers_describe_places_strength_durations_and_channels() -> None:
    rng = random.Random(0)
    assert place_name((10.0, 0.0), PLACES, rng, 0.0) == "P"
    assert place_name((10.0, 0.0), PLACES, random.Random(0), 1.0) == "Q"
    assert severity(0.1, rng, 0.0) == "squall"
    assert severity(0.8, rng, 0.0) == "severe storm"
    assert severity(5.0, rng, 0.0) == "violent storm"
    noisy = {severity(0.5, random.Random(i), 1.0) for i in range(50)}
    assert len(noisy) > 1
    assert forecast(1000.0, rng, 0.0) == 1000
    assert forecast(1000.0, random.Random(1), 0.5) != 1000
    assert channel("W2>W3") == "the W2-W3 channel"


def test_exact_news_reports_every_event_at_once() -> None:
    storms = (Storm((5.0, 0.0), 50.0, 0.8, 100.0, 300.0),)
    events = [
        Event(400.0, "close", "P>Q", "grounded vessel"),
        Event(600.0, "open", "P>Q", "reopened"),
    ]
    items = generate(events, storms, PLACES, LANES, exact(), random.Random(0), 1000)
    assert [i.published for i in items] == [100.0, 300.0, 400.0, 600.0]
    assert all(i.truth for i in items)
    assert "near P" in items[0].text and "200" in items[0].text
    assert "P-Q channel" in items[2].text and "200" in items[2].text
    assert "grounded vessel" in items[2].text
    assert {i.source for i in items} <= {
        "Coastguard bulletin",
        "Shipping wire",
        "Local radio",
    }


def test_delays_misses_false_alarms_and_missing_corrections() -> None:
    storms = (Storm((5.0, 0.0), 50.0, 0.8, 100.0, 300.0),)
    events = [Event(400.0, "close", "P>Q", "x")]
    late = generate(
        events,
        storms,
        PLACES,
        LANES,
        exact(delay=50.0, correction_delay=math.inf),
        random.Random(0),
        1000,
    )
    assert len(late) == 2 and late[0].published > 100.0
    assert (
        generate(
            events, storms, PLACES, LANES, exact(miss_rate=1.0), random.Random(0), 1000
        )
        == []
    )
    alarms = generate(
        events,
        storms,
        PLACES,
        LANES,
        exact(false_alarm_rate=1.0),
        random.Random(3),
        1000,
    )
    fakes = [i for i in alarms if i.truth is None]
    assert len(fakes) == 2
    assert all(0.0 <= i.published <= 1000.0 for i in fakes)
    kinds = set()
    for seed in range(20):
        for i in generate(
            [],
            storms,
            PLACES,
            LANES,
            exact(false_alarm_rate=1.0),
            random.Random(seed),
            1000,
        ):
            if i.truth is None:
                kinds.add("channel" in i.text)
    assert kinds == {True, False}
