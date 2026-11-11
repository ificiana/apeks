"""Packages, port demand and the fixed heuristics that give ships a purpose.

Every arm uses these same rules: which waiting packages a ship loads, which
port it sails for next, and how ships at a port group into convoys.
"""

import math
from collections import Counter
from dataclasses import dataclass

from apeks.seeds import derive


@dataclass(frozen=True)
class Package:
    id: str
    origin: str
    destination: str
    created: float


def generate_demand(
    ports: list[str], rate: float, seed: int, horizon: float
) -> list[Package]:
    """Poisson arrivals at each port, one random stream per port.

    Destinations are uniform over the other ports. Adding a port never shifts
    another port's packages.
    """
    out = []
    for port in ports:
        if rate <= 0:
            continue
        rng = derive(seed, f"demand:{port}")
        others = [p for p in ports if p != port]
        t, n = rng.expovariate(rate), 0
        while t < horizon:
            out.append(Package(f"{port}#{n}", port, rng.choice(others), t))
            t, n = t + rng.expovariate(rate), n + 1
    return sorted(out, key=lambda p: (p.created, p.id))


def mode_port(ports: list[str], times: dict[str, float]) -> str | None:
    """The most common port; ties go to the nearest by charted time, then name."""
    if not ports:
        return None
    counts = Counter(ports)
    return min(counts, key=lambda p: (-counts[p], times.get(p, math.inf), p))


def load(
    cargo: list[Package],
    capacity: int,
    waiting: list[Package],
    times: dict[str, float],
) -> list[Package]:
    """Move packages from ``waiting`` into ``cargo``, oldest first.

    First those bound where the cargo already favours, then the most common
    waiting destination, until the ship is full or nothing waits.
    """
    loaded: list[Package] = []
    while len(cargo) < capacity and waiting:
        target = mode_port([p.destination for p in cargo], times)
        if target is None or not any(p.destination == target for p in waiting):
            target = mode_port([p.destination for p in waiting], times)
        for p in [p for p in waiting if p.destination == target]:
            if len(cargo) == capacity:
                break
            waiting.remove(p)
            cargo.append(p)
            loaded.append(p)
    return loaded


def next_port(
    cargo: list[Package],
    here: str,
    times: dict[str, float],
    backlogs: dict[str, int],
) -> str | None:
    """Where a ship sails next: its cargo's port, else the nearest known demand."""
    if cargo:
        return mode_port([p.destination for p in cargo], times)
    wanted = [
        p
        for p, n in backlogs.items()
        if n > 0 and p != here and times.get(p, math.inf) < math.inf
    ]
    return min(wanted, key=lambda p: (times[p], p)) if wanted else None


def group(
    ships: list[tuple[str, str | None]], max_size: int
) -> list[tuple[str, list[str]]]:
    """Ships with the same next port, in chunks of at most ``max_size``."""
    by: dict[str, list[str]] = {}
    for ship, dest in ships:
        if dest is not None:
            by.setdefault(dest, []).append(ship)
    return [
        (dest, members[i : i + max_size])
        for dest, members in sorted(by.items())
        for i in range(0, len(members), max_size)
    ]
