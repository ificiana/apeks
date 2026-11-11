import math

from apeks.cargo import Package, generate_demand, group, load, mode_port, next_port

TIMES = {"A": 0.0, "B": 10.0, "C": 20.0, "D": math.inf}


def pkg(dest: str, n: int = 0, origin: str = "A") -> Package:
    return Package(f"{origin}#{dest}{n}", origin, dest, 0.0)


def test_demand_is_poisson_per_port_reproducible_and_independent() -> None:
    one = generate_demand(["A", "B", "C"], 0.05, 7, 1000.0)
    again = generate_demand(["A", "B", "C"], 0.05, 7, 1000.0)
    assert one == again
    assert [p.created for p in one] == sorted(p.created for p in one)
    assert all(p.destination != p.origin for p in one)
    assert all(p.created < 1000.0 for p in one)
    assert {p.destination for p in one} == {"A", "B", "C"}
    from_a = [p for p in one if p.origin == "A"]
    assert 25 < len(from_a) < 80
    assert from_a[0].id == "A#0"
    only_a = generate_demand(["A", "B"], 0.05, 7, 1000.0)
    a_alone = [p for p in only_a if p.origin == "A"]
    assert [p.created for p in a_alone] == [p.created for p in from_a]
    assert generate_demand(["A", "B"], 0.0, 7, 1000.0) == []


def test_mode_port_breaks_ties_by_time_then_name() -> None:
    assert mode_port(["C", "C", "B"], TIMES) == "C"
    assert mode_port(["C", "B"], TIMES) == "B"
    assert mode_port(["D", "C"], TIMES) == "C"
    assert mode_port(["Y", "X"], {}) == "X"
    assert mode_port([], TIMES) is None


def test_load_prefers_cargo_destination_then_most_common_waiting() -> None:
    waiting = [pkg("C", 0), pkg("B", 0), pkg("C", 1), pkg("B", 1), pkg("B", 2)]
    cargo = [pkg("C", 9)]
    got = load(cargo, 4, waiting, TIMES)
    assert [p.destination for p in got] == ["C", "C", "B"]
    assert [p.destination for p in cargo] == ["C", "C", "C", "B"]
    assert [p.id for p in waiting] == ["A#B1", "A#B2"]
    empty: list[Package] = []
    assert [p.destination for p in load(empty, 2, waiting, TIMES)] == ["B", "B"]
    assert waiting == []
    assert load(empty, 2, waiting, TIMES) == []
    full = [pkg("B", i) for i in range(2)]
    assert load(full, 2, [pkg("C")], TIMES) == []


def test_next_port_from_cargo_or_known_backlog() -> None:
    assert next_port([pkg("C"), pkg("C"), pkg("B")], "A", TIMES, {}) == "C"
    backlogs = {"A": 5, "B": 0, "C": 2, "D": 9}
    assert next_port([], "A", TIMES, backlogs) == "C"
    assert next_port([], "A", TIMES, {"B": 1, "C": 1}) == "B"
    assert next_port([], "A", TIMES, {"A": 3, "D": 4}) is None


def test_group_by_destination_in_chunks() -> None:
    ships = [("s1", "B"), ("s2", "C"), ("s3", "B"), ("s4", "B"), ("s5", None)]
    assert group(ships, 2) == [("B", ["s1", "s3"]), ("B", ["s4"]), ("C", ["s2"])]
