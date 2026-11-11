import io
from dataclasses import dataclass

from apeks.trace import Tracer, read_trace


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def test_emit_writes_plain_rounded_json_lines(tmp_path) -> None:
    buf = io.StringIO()
    tracer = Tracer(buf)
    tracer.emit(1.23456, "x", p=Point(1.0, 2.00049), s={"b", "a"}, d={1: (0.5,)})
    tracer.emit(2.0, "y", n=float("inf"))
    path = tmp_path / "t.jsonl"
    path.write_text(buf.getvalue() + "\n")
    records = read_trace(path)
    assert records[0] == {
        "t": 1.235,
        "kind": "x",
        "p": {"x": 1.0, "y": 2.0},
        "s": ["a", "b"],
        "d": {"1": [0.5]},
    }
    assert read_trace(path, "y")[0]["n"] is None
    assert tracer.records == 2


def test_snapshot_cadence() -> None:
    every3 = Tracer(io.StringIO(), snapshot_every=3)
    assert [every3.wants_snapshot(t) for t in (0.0, 1.0, 3.0)] == [True, False, True]
    assert not Tracer(io.StringIO(), snapshot_every=0).wants_snapshot(0.0)
