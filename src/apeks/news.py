"""A free-text news feed about the world, as HQ would read it.

``generate`` turns the scenario's storms and lane events into news items from
templates: published late, placed by the nearest named place (sometimes the
wrong one), with worded severity and rough forecasts. Some events are missed,
some items are false alarms, and an all-clear may come late or never.
``truth`` names the real event behind an item (``None`` for a false alarm); it
is kept for analysis and never shown to HQ.
"""

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

from apeks.geometry import Vec, dist
from apeks.ocean import Storm

SOURCES = ("Coastguard bulletin", "Shipping wire", "Local radio")
SEVERITY = ("squall", "gale", "severe storm", "violent storm")

STORM = (
    "A {sev} is forming near {place}; forecasters expect it to clear in about "
    "{n} ticks.",
    "Mariners warned of a {sev} building near {place}, lasting roughly {n} ticks.",
    "{Sev} reported near {place}. Expected duration about {n} ticks.",
)
STORM_CLEAR = (
    "The storm near {place} has cleared.",
    "All clear near {place}: the storm has passed.",
)
CLOSE = (
    "{Chan} is closed: {text}. Expected to reopen in about {n} ticks.",
    "Traffic halted in {chan} after reports of: {text}. Closure expected to "
    "last roughly {n} ticks.",
)
CLOSE_OPEN_ENDED = (
    "{Chan} is closed: {text}. No reopening time given.",
    "Traffic halted in {chan} after reports of: {text}.",
)
OPEN = (
    "{Chan} has reopened to traffic.",
    "Authorities lifted the closure of {chan}.",
)


@dataclass(frozen=True)
class NewsSettings:
    delay: float = 60.0  # mean publication delay, ticks
    location_error: float = 0.2  # chance a storm is placed at the 2nd nearest place
    severity_error: float = 0.2  # noise on the severity wording, in levels / 4
    forecast_error: float = 0.3  # log-normal sd on forecast durations
    false_alarm_rate: float = 0.2  # chance per real event of a false item
    miss_rate: float = 0.0  # chance a real event is never reported
    correction_delay: float = 60.0  # delay of all-clear items; inf means never


@dataclass(frozen=True)
class NewsItem:
    published: float
    source: str
    text: str
    truth: str | None


def place_name(
    pos: Vec, places: dict[str, Vec], rng: random.Random, error: float
) -> str:
    ranked = sorted(places, key=lambda name: dist(places[name], pos))
    return ranked[1] if rng.random() < error else ranked[0]


def severity(strength: float, rng: random.Random, error: float) -> str:
    level = strength * (len(SEVERITY) - 1) + rng.gauss(0.0, error * len(SEVERITY))
    return SEVERITY[max(0, min(len(SEVERITY) - 1, int(level)))]


def forecast(duration: float, rng: random.Random, error: float) -> int:
    return int(round(duration * math.exp(rng.gauss(0.0, error)), -1))


def channel(lane: str) -> str:
    return f"the {lane.replace('>', '-')} channel"


def _fill(template: str, **kw: object) -> str:
    caps = {k.capitalize(): str(v)[:1].upper() + str(v)[1:] for k, v in kw.items()}
    return template.format(**kw, **caps)


def generate(
    events: Iterable,
    storms: Iterable[Storm],
    places: dict[str, Vec],
    lanes: list[str],
    settings: NewsSettings,
    rng: random.Random,
    horizon: float,
) -> list[NewsItem]:
    """Every news item of a run, in publication order."""
    s = settings
    items: list[NewsItem] = []

    def publish(at: float, text: str, truth: str | None, clear: bool) -> None:
        if rng.random() < s.miss_rate:
            return
        if clear:
            delay = s.correction_delay
        else:
            delay = rng.expovariate(1.0 / s.delay) if s.delay > 0 else 0.0
        if math.isfinite(delay):
            items.append(NewsItem(at + delay, rng.choice(SOURCES), text, truth))

    real = 0
    for st in storms:
        real += 1
        place = place_name(st.centre, places, rng, s.location_error)
        n = forecast(st.end - st.start, rng, s.forecast_error)
        sev = severity(st.strength, rng, s.severity_error)
        text = _fill(rng.choice(STORM), sev=sev, place=place, n=n)
        publish(st.start, text, f"storm at {st.centre} from t={st.start:.0f}", False)
        text = _fill(rng.choice(STORM_CLEAR), place=place)
        publish(st.end, text, f"storm at {st.centre} cleared t={st.end:.0f}", True)
    events = list(events)
    for e in events:
        real += 1
        chan = channel(e.lane)
        if e.kind == "close":
            reopen = [o.at for o in events if o.kind == "open" and o.lane == e.lane]
            later = [at for at in reopen if at > e.at]
            if later:
                n = forecast(min(later) - e.at, rng, s.forecast_error)
                text = _fill(rng.choice(CLOSE), chan=chan, text=e.text, n=n)
            else:
                text = _fill(rng.choice(CLOSE_OPEN_ENDED), chan=chan, text=e.text)
            publish(e.at, text, f"{e.lane} closed t={e.at:.0f}", False)
        else:
            text = _fill(rng.choice(OPEN), chan=chan)
            publish(e.at, text, f"{e.lane} opened t={e.at:.0f}", True)
    for _ in range(real):
        if rng.random() >= s.false_alarm_rate:
            continue
        at = rng.uniform(0.0, horizon)
        if rng.random() < 0.5:
            place = rng.choice(sorted(places))
            n = forecast(500.0, rng, s.forecast_error)
            text = _fill(rng.choice(STORM), sev=rng.choice(SEVERITY), place=place, n=n)
        else:
            text = _fill(
                rng.choice(CLOSE_OPEN_ENDED),
                chan=channel(rng.choice(lanes)),
                text="unconfirmed reports of an obstruction",
            )
        items.append(NewsItem(at, rng.choice(SOURCES), text, None))
    return sorted(items, key=lambda i: i.published)
