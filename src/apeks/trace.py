"""Structured run traces: one JSON object per line, for replay and plotting.

Every record has ``t`` (sim time) and ``kind``. A trace starts with a ``meta``
record holding the config and the static world (map, ocean, events, fleet), so
a run can be reconstructed from the trace alone. Record kinds:

- ``meta``: config, nodes, ports, lanes, currents, storms, events, convoys
- ``event``: closure, reopening, storm forming or clearing (``text``)
- ``policy_request``: module, trigger, prompt, reply, policy or error
- ``policy_applied``: module, params
- ``route_choice``: convoy, node, destination, options, choice
- ``blocked``: convoy found ``lane`` closed at its entrance
- ``notice_heard``: convoy, lane, closed
- ``lane_start`` / ``lane_end``: convoy, lane (``lane_end`` adds transit)
- ``queued`` / ``berthed`` / ``delivered``: convoy, port (``delivered`` adds trip)
- ``near_miss``: vessels a and b, distance
- ``snapshot``: vessels, convoys, queues, closures, metrics, radio, params
"""

import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, TextIO


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [_plain(v) for v in items]
    if isinstance(value, float):
        return round(value, 3) if math.isfinite(value) else None
    return value


@dataclass
class Tracer:
    out: TextIO
    snapshot_every: int = 1
    records: int = 0

    def emit(self, t: float, kind: str, **data: Any) -> None:
        record = {"t": round(t, 3), "kind": kind, **_plain(data)}
        self.out.write(json.dumps(record) + "\n")
        self.records += 1

    def wants_snapshot(self, t: float) -> bool:
        return self.snapshot_every > 0 and int(t) % self.snapshot_every == 0


def read_trace(path: Path, kind: str | None = None) -> list[dict[str, Any]]:
    """All records of a trace file, optionally only those of one ``kind``."""
    with path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [r for r in records if kind is None or r["kind"] == kind]
