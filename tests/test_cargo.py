import math

from apeks.cargo import (
    Package,
    detours,
    generate_demand,
    group,
    load,
    mode_port,
    next_port,
    target_port,
)

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


def test_target_is_cargo_mode_else_most_common_waiting() -> None:
    assert target_port([pkg("C"), pkg("B"), pkg("C")], [pkg("B")], TIMES) == "C"
    assert target_port([], [pkg("C"), pkg("B"), pkg("C")], TIMES) == "C"
    assert target_port([], [], TIMES) is None


def test_detours_measure_extra_time_via_each_port() -> None:
    here = {"A": 0.0, "B": 10.0, "C": 4.0, "D": 20.0}
    to_b = {"A": 10.0, "B": 0.0, "C": 6.0, "D": 15.0}
    assert detours(here, to_b, "B") == {"A": 0.0, "B": 0.0, "C": 0.0, "D": 25.0}
    assert detours(here, {"A": 10.0}, "B")["C"] == math.inf
    cut_off = {"A": 0.0, "B": math.inf, "C": 4.0}
    assert detours(cut_off, to_b, "B") == {"A": math.inf, "B": 0.0, "C": math.inf}


def test_load_fills_target_then_cheapest_detours_with_no_cut_off() -> None:
    waiting = [pkg("D", 0), pkg("C", 0), pkg("B", 0), pkg("D", 1), pkg("E", 0)]
    det = {"B": 0.0, "C": 1.0, "D": 1.0, "E": 50.0}
    cargo = [pkg("B", 9)]
    got = load(cargo, 4, waiting, "B", det)
    assert [p.destination for p in got] == ["B", "D", "D"]
    assert [p.id for p in waiting] == ["A#C0", "A#E0"]
    roomy: list[Package] = []
    got = load(roomy, 9, waiting, "B", det)
    assert [p.destination for p in got] == ["C", "E"]  # a long detour still fills
    assert waiting == []
    assert load([pkg("B", i) for i in range(2)], 2, [pkg("B")], "B", det) == []
    assert load([], 2, [pkg("Z")], "B", det) == []  # no detour known
    stranded = [pkg("Y")]
    assert load([], 2, stranded, "B", {"B": 0.0, "Y": math.inf}) == []
    assert stranded == [pkg("Y")]  # unreachable ports are never loaded


def test_next_port_from_cargo_or_known_backlog() -> None:
    assert next_port([pkg("C"), pkg("C"), pkg("B")], "A", TIMES, {}) == "C"
    backlogs = {"A": 5, "B": 0, "C": 2, "D": 9}
    assert next_port([], "A", TIMES, backlogs) == "C"
    assert next_port([], "A", TIMES, {"B": 1, "C": 1}) == "B"
    assert next_port([], "A", TIMES, {"A": 3, "D": 4}) is None


def test_group_by_destination_in_chunks() -> None:
    ships = [("s1", "B"), ("s2", "C"), ("s3", "B"), ("s4", "B"), ("s5", None)]
    assert group(ships, 2) == [("B", ["s1", "s3"]), ("B", ["s4"]), ("C", ["s2"])]
