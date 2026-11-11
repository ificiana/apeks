"""The fleet's operations centre (HQ) and what has reached it.

HQ is ashore. It hears ships only through port stations, which forward what
they receive over a land link after ``delay`` ticks. It also reads the news.
"""

from dataclasses import dataclass, field

from apeks.hazards import HazardBook, observe_channel
from apeks.news import NewsItem
from apeks.radio import Backlog, Hazard, LaneReport
from apeks.routing import Report


@dataclass
class Hq:
    delay: float
    hazards: HazardBook = field(default_factory=HazardBook)
    reports: dict[str, list[Report]] = field(default_factory=dict)
    news: list[NewsItem] = field(default_factory=list)
    backlogs: dict[str, tuple[int, float]] = field(default_factory=dict)
    _inbox: list[tuple[float, Hazard | LaneReport | Backlog]] = field(
        default_factory=list
    )
    _seen: set[tuple[str, str, float]] = field(default_factory=set)

    def send(self, now: float, msg: Hazard | LaneReport | Backlog) -> None:
        """A port station forwards ``msg``; it arrives after the uplink delay."""
        self._inbox.append((now + self.delay, msg))

    def receive(self, now: float) -> list[str]:
        """Take in arrived messages; return a trigger per changed hazard."""
        ready = [m for at, m in self._inbox if at <= now]
        self._inbox = [(at, m) for at, m in self._inbox if at > now]
        triggers = []
        for msg in ready:
            if isinstance(msg, Hazard):
                got = observe_channel(
                    self.hazards, msg.lane, msg.closed, msg.observed_at, msg.origin
                )
                if got == "changed":
                    state = "closed" if msg.closed else "open"
                    triggers.append(f"fleet report: {msg.lane} {state}")
                continue
            if isinstance(msg, Backlog):
                if msg.observed_at > self.backlogs.get(msg.port, (0, -1.0))[1]:
                    self.backlogs[msg.port] = (msg.waiting, msg.observed_at)
                continue
            key = (msg.sender, msg.lane, msg.finished_at)
            if key not in self._seen:
                self._seen.add(key)
                self.reports.setdefault(msg.lane, []).append(
                    Report(msg.transit, msg.finished_at)
                )
        return triggers
