"""Beliefs about lane closures, held by convoys, port stations and HQ.

Each lane maps to the latest observation heard of it: closed or open, when it
was observed and which convoys observed that state. A newer observation
replaces an older one; an older one is ignored. Nobody but ``World`` holds the
true closures.
"""

from dataclasses import dataclass, field


@dataclass
class Belief:
    closed: bool
    observed_at: float
    sources: set[str]


@dataclass
class HazardBook:
    entries: dict[str, Belief] = field(default_factory=dict)

    def update(
        self, lane: str, closed: bool, observed_at: float, source: str
    ) -> str | None:
        """Merge one observation.

        Returns ``"changed"`` for a new or flipped state, ``"refreshed"`` for a
        newer observation of the same state, ``None`` for old or repeated news.
        """
        old = self.entries.get(lane)
        if old is None or (old.closed != closed and observed_at > old.observed_at):
            self.entries[lane] = Belief(closed, observed_at, {source})
            return "changed"
        if old.closed == closed:
            old.sources.add(source)
            if observed_at > old.observed_at:
                old.observed_at = observed_at
                return "refreshed"
        return None

    def closed(self) -> set[str]:
        return {lane for lane, b in self.entries.items() if b.closed}

    def fresh(self, now: float, max_age: float) -> list[tuple[str, Belief]]:
        """Entries worth repeating: every closure, and recent reopenings."""
        return [
            (lane, b)
            for lane, b in sorted(self.entries.items())
            if b.closed or now - b.observed_at <= max_age
        ]


def reverse(lane_id: str) -> str:
    a, b = lane_id.split(">")
    return f"{b}>{a}"


def observe_channel(
    book: HazardBook, lane: str, closed: bool, observed_at: float, source: str
) -> str | None:
    """A closure blocks the channel both ways: update the lane and its reverse."""
    book.update(reverse(lane), closed, observed_at, source)
    return book.update(lane, closed, observed_at, source)
