import pytest

from apeks import sim
from apeks.config import (
    AGENT_COLOR,
    CHARGE_RATE,
    DIRECTIONS,
    GRID_SIZE,
    LOW_CHARGE,
    MARK_DECAY,
    MARK_VALUE,
    MAX_CHARGE,
    MOVE_COST,
    PRESSURE_LIMIT,
    PUSH_REACH,
    STATION_COUNT,
    STEP_INTERVAL,
)
from apeks.sim import (
    CONES,
    Agent,
    Trail,
    World,
    advance,
    build_nav,
    cardinal,
    make_stations,
)

STATIONS = make_stations()
NAV = build_nav(STATIONS)
MID = (50, 50)
MIXED = (21, 20)  # some neighbours farther from a station, some nearer


def agent(ident, col, row, state="walk", charge=50000):
    a = Agent(ident, col, row)
    a.state = state
    a.charge = charge
    return a


def world(*agents):
    return World(NAV, list(agents))


def ring(col, row, ident_start=100, state="charge"):
    """Non-pushable agents on every in-grid neighbour of (col, row)."""
    out = []
    for i, (dc, dr) in enumerate(DIRECTIONS):
        c, r = col + dc, row + dr
        if 0 <= c < GRID_SIZE and 0 <= r < GRID_SIZE:
            out.append(agent(ident_start + i, c, r, state=state))
    return out


def arrow_target(col, row):
    dc, dr = NAV.arrow[row][col]
    return col + dc, row + dr


# --- cardinal ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("dx", "dy", "step"),
    [
        (3, 1, (1, 0)),
        (-3, 1, (-1, 0)),
        (1, 3, (0, 1)),
        (1, -3, (0, -1)),
        (2, 2, (1, 0)),
    ],
)
def test_cardinal_steps_along_longer_axis(dx, dy, step):
    assert cardinal(dx, dy) == step


# --- Trail ------------------------------------------------------------------


def test_trail_unwritten_tile_is_zero():
    assert Trail().value(3, 4, 10.0) == 0.0


def test_trail_value_decays_with_time():
    trail = Trail()
    trail.mark(3, 4, 2.0)
    assert trail.value(3, 4, 2.0) == MARK_VALUE
    assert trail.value(3, 4, 3.0) == pytest.approx(MARK_VALUE * MARK_DECAY)


def test_cone_mean_averages_in_grid_tiles():
    trail = Trail()
    trail.mark(1, 0, 0.0)
    assert trail.cone_mean(0, 0, [(1, 0), (2, 0)], 0.0) == MARK_VALUE / 2


def test_cone_mean_ignores_tiles_outside_grid():
    trail = Trail()
    trail.mark(1, 0, 0.0)
    assert trail.cone_mean(0, 0, [(1, 0), (-1, 0), (0, -1)], 0.0) == MARK_VALUE


def test_cone_mean_of_fully_outside_cone_is_zero():
    assert Trail().cone_mean(0, 0, [(-1, 0)], 0.0) == 0.0


# --- cones ------------------------------------------------------------------


def test_cones_cover_tiles_ahead_within_reach():
    for (dc, dr), cone in CONES.items():
        assert len(cone) == 13
        assert (dc, dr) in cone
        for x, y in cone:
            assert abs(x) + abs(y) <= PUSH_REACH
            assert x * dc + y * dr > 0


# --- Agent ------------------------------------------------------------------


def test_color_darkens_with_charge():
    assert agent(0, 0, 0, charge=MAX_CHARGE).color == AGENT_COLOR
    assert agent(0, 0, 0, charge=0).color == (0, 0, 0)


def test_charge_up_keeps_charging_until_full():
    a = agent(0, 0, 0, state="charge", charge=0)
    a.charge_up(0.5)
    assert a.charge == CHARGE_RATE * 0.5
    assert a.state == "charge"
    assert not a.released


def test_charge_up_releases_when_full():
    a = agent(0, 0, 0, state="charge", charge=MAX_CHARGE - 1)
    a.charge_up(1.0)
    assert a.charge == MAX_CHARGE
    assert a.state == "walk"
    assert a.released


def test_slack_is_charge_moves_minus_distance():
    a = agent(0, *MID, charge=MOVE_COST * 100)
    assert a.slack(NAV) == 100 - NAV.dist[MID[1]][MID[0]]


def test_priority_released_first():
    a = agent(7, *MID)
    a.released = True
    assert a.priority(NAV) == (4, 0, 7)


def test_priority_urgent_seeker():
    a = agent(7, *MID, state="seek", charge=MOVE_COST)
    assert a.priority(NAV) == (3, -a.slack(NAV), 7)


def test_priority_queued_seeker_orders_by_age():
    a = agent(7, *MID, state="seek")
    a.age = 5
    assert a.priority(NAV) == (2, 5, -7)


def test_priority_walker_orders_by_slack():
    a = agent(7, *MID)
    assert a.priority(NAV) == (1, -a.slack(NAV), 7)


def test_pushable_requires_lower_priority_and_slack():
    walker = agent(1, *MID)
    claim = (2, 0, 0)
    assert walker.pushable(NAV, claim)
    assert not walker.pushable(NAV, (0, 0, 0))
    assert not agent(2, *MID, state="charge").pushable(NAV, claim)
    assert not agent(3, *MID, charge=MOVE_COST).pushable(NAV, claim)


def test_pushable_seeker_needs_more_slack_than_walker():
    dist = NAV.dist[MID[1]][MID[0]]
    seeker = agent(1, *MID, state="seek", charge=MOVE_COST * (dist + 10))
    walker = agent(2, *MID, charge=MOVE_COST * (dist + 10))
    assert not seeker.pushable(NAV, (9, 0, 0))
    assert walker.pushable(NAV, (9, 0, 0))
    seeker.charge = MOVE_COST * (dist + 100)
    assert seeker.pushable(NAV, (9, 0, 0))


# --- Agent.step: early exits ------------------------------------------------


@pytest.mark.parametrize("state", ["charge"])
def test_step_does_nothing_while_charging(state):
    a = agent(0, *MID, state=state)
    world(a).tick([a], Trail(), 0.0)
    assert (a.col, a.row) == MID


def test_step_does_nothing_when_already_moved():
    a = agent(0, *MID)
    w = world(a)
    a.moved = True
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == MID


def test_step_does_nothing_without_charge_for_a_move():
    a = agent(0, *MID, charge=MOVE_COST - 1)
    w = world(a)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == MID
    assert a.state == "walk"


# --- Agent.step: walking ----------------------------------------------------


def test_walk_moves_one_tile_and_pays_for_it():
    a = agent(0, *MID)
    w = world(a)
    a.step(w, Trail(), 0.0)
    assert abs(a.col - 50) + abs(a.row - 50) == 1
    assert a.charge == 50000 - MOVE_COST
    assert a.moved


def test_walk_heads_for_coldest_cone():
    a = agent(0, *MID)
    w = world(a)
    trail = Trail()
    for dc, dr in CONES[(1, 0)] + CONES[(0, 1)] + CONES[(0, -1)]:
        trail.mark(50 + dc, 50 + dr, 0.0)
    a.step(w, trail, 0.0)
    assert (a.col, a.row) == (49, 50)


def test_walk_below_low_charge_turns_seeker_and_follows_arrow():
    a = agent(0, *MID, charge=LOW_CHARGE - 1)
    w = world(a)
    a.step(w, Trail(), 0.0)
    assert a.state == "seek"
    assert (a.col, a.row) == arrow_target(*MID)
    assert w.pressure[a.row][a.col] == 1
    assert sum(map(sum, w.pressure)) == 2 * PUSH_REACH * (PUSH_REACH + 1) + 1


def test_walk_retreats_from_pressure_to_farther_tiles():
    a = agent(0, *MIXED)
    w = world(a)
    w.pressure[MIXED[1]][MIXED[0]] = PRESSURE_LIMIT
    here = NAV.dist[MIXED[1]][MIXED[0]]
    a.step(w, Trail(), 0.0)
    assert NAV.dist[a.row][a.col] > here


def test_walk_retreat_falls_back_when_no_farther_tile_is_free():
    a = agent(0, *MIXED)
    col, row = MIXED
    here = NAV.dist[row][col]
    farther = [
        (col + dc, row + dr)
        for dc, dr in DIRECTIONS
        if NAV.dist[row + dr][col + dc] > here
    ]
    assert farther
    blockers = [
        agent(100 + i, c, r, state="charge") for i, (c, r) in enumerate(farther)
    ]
    w = world(a, *blockers)
    w.pressure[row][col] = PRESSURE_LIMIT
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) != MIXED
    assert NAV.dist[a.row][a.col] < here


def test_walk_avoids_pressured_tiles():
    a = agent(0, *MID)
    w = world(a)
    for c, r in [(51, 50), (50, 51), (50, 49)]:
        w.pressure[r][c] = PRESSURE_LIMIT
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (49, 50)


def test_walk_stays_put_when_every_option_is_pressured():
    a = agent(0, *MID)
    w = world(a)
    for dc, dr in DIRECTIONS:
        w.pressure[50 + dr][50 + dc] = PRESSURE_LIMIT
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == MID
    assert a.charge == 50000


def test_released_agent_boxed_in_pushes_a_neighbour_aside():
    a = agent(0, *MID)
    a.released = True
    neighbours = ring(*MID, state="walk")
    w = world(a, *neighbours)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) != MID
    assert len(w.occ) == 5


def test_released_agent_stays_when_nobody_can_be_pushed():
    a = agent(0, 0, 0)
    a.released = True
    w = world(a, *ring(0, 0))
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (0, 0)


def test_unreleased_agent_boxed_in_waits():
    a = agent(0, *MID)
    w = world(a, *ring(*MID, state="walk"))
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == MID


# --- Agent.step: seeking ----------------------------------------------------


def test_seeker_on_station_starts_charging():
    sx, sy = STATIONS[0]
    a = agent(0, sx, sy, state="seek", charge=1000)
    w = world(a)
    a.step(w, Trail(), 0.0)
    assert a.state == "charge"
    assert (a.col, a.row) == (sx, sy)


def test_seeker_follows_arrow_to_free_tile():
    a = agent(0, *MID, state="seek", charge=20000)
    w = world(a)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == arrow_target(*MID)


def test_seeker_pushes_lower_priority_blocker_aside():
    a = agent(0, *MID, state="seek", charge=20000)
    target = arrow_target(*MID)
    blocker = agent(1, *target)
    w = world(a, blocker)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == target
    assert (blocker.col, blocker.row) != target
    assert w.occ[target] is a


def test_seeker_waits_behind_unpushable_blocker():
    a = agent(0, *MID, state="seek", charge=20000)
    blocker = agent(1, *arrow_target(*MID), state="charge")
    w = world(a, blocker)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == MID


def test_seeker_waits_when_push_fails(monkeypatch):
    a = agent(0, *MID, state="seek", charge=20000)
    blocker = agent(1, *arrow_target(*MID))
    w = world(a, blocker)
    monkeypatch.setattr(w, "push", lambda *args: False)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == MID
    assert (blocker.col, blocker.row) == arrow_target(*MID)


# --- World ------------------------------------------------------------------


def test_free_rejects_occupied_and_out_of_grid_tiles():
    w = world(agent(0, *MID))
    assert not w.free(MID)
    assert not w.free((-1, 5))
    assert not w.free((5, GRID_SIZE))
    assert w.free((51, 50))


def test_is_station():
    w = world()
    assert w.is_station(STATIONS[0])
    assert not w.is_station(MID)


def test_press_adds_diamond_and_clips_at_edges():
    w = world()
    w.press((0, 0), 1)
    assert w.pressure[0][0] == 1
    assert w.pressure[0][PUSH_REACH] == 1
    assert w.pressure[0][PUSH_REACH + 1] == 0
    assert w.pressure[PUSH_REACH][1] == 0
    w.press((0, 0), -1)
    assert sum(map(sum, w.pressure)) == 0


def test_move_updates_occupancy_and_pays_cost():
    a = agent(0, *MID)
    w = world(a)
    w.move(a, (51, 50))
    assert MID not in w.occ
    assert w.occ[(51, 50)] is a
    assert (a.col, a.row) == (51, 50)
    assert a.charge == 50000 - MOVE_COST
    assert a.moved


def test_move_clears_released_when_leaving_station():
    sx, sy = STATIONS[0]
    a = agent(0, sx, sy + 1)
    a.released = True
    w = world(a)
    w.move(a, (sx, sy))
    assert a.released
    w.move(a, (sx, sy + 1))
    assert not a.released


def test_seeker_move_shifts_pressure_and_starts_charging_on_station():
    sx, sy = STATIONS[0]
    a = agent(0, sx, sy + 1, state="seek")
    w = world(a)
    w.press((sx, sy + 1), 1)
    w.move(a, (sx, sy))
    assert a.state == "charge"
    assert sum(map(sum, w.pressure)) == 0


def test_seeker_move_to_plain_tile_presses_new_tile():
    a = agent(0, *MID, state="seek")
    w = world(a)
    w.press(MID, 1)
    w.move(a, (51, 50))
    assert a.state == "seek"
    assert w.pressure[50][51] == 1
    assert w.pressure[50][50] == 1


# --- World.push -------------------------------------------------------------


def test_push_moves_blocker_to_free_neighbour():
    blocker = agent(1, *MID)
    w = world(blocker)
    assert w.push(blocker, (9, 0, 0), (49, 50))
    assert (blocker.col, blocker.row) != MID
    assert (blocker.col, blocker.row) != (49, 50)


def test_push_shifts_a_chain_like_a_train():
    b1 = agent(1, 50, 50)
    b2 = agent(2, 51, 50)
    guards = [agent(3, 50, 49, state="charge"), agent(4, 50, 51, state="charge")]
    w = world(b1, b2, *guards)
    assert w.push(b1, (9, 0, 0), (49, 50))
    assert (b1.col, b1.row) == (51, 50)
    assert (b2.col, b2.row) != (51, 50)
    assert 50 not in {c for c, r in w.occ if r == 50 and c < 51 and c > 49}


def test_push_fails_when_boxed_in_by_edges_and_unpushables():
    blocker = agent(1, 0, 0)
    w = world(blocker, agent(2, 0, 1, state="charge"))
    assert not w.push(blocker, (9, 0, 0), (1, 0))
    assert (blocker.col, blocker.row) == (0, 0)


def test_push_never_ends_on_a_station():
    sx, sy = STATIONS[0]
    blocker = agent(1, sx, sy + 1)
    guards = [
        agent(2, sx - 1, sy + 1, state="charge"),
        agent(3, sx + 1, sy + 1, state="charge"),
    ]
    w = world(blocker, *guards)
    assert not w.push(blocker, (9, 0, 0), (sx, sy + 2))
    assert (blocker.col, blocker.row) == (sx, sy + 1)


def test_push_fails_when_the_blocker_cannot_yield():
    blocker = agent(1, *MID, state="charge")
    w = world(blocker)
    assert not w.push(blocker, (9, 0, 0), (49, 50))
    assert (blocker.col, blocker.row) == MID


def test_push_gives_up_beyond_reach(monkeypatch):
    monkeypatch.setattr(sim, "PUSH_REACH", 1)
    blocker = agent(1, *MID)
    helper = agent(2, 51, 50)
    guards = [agent(3, 50, 49, state="charge"), agent(4, 50, 51, state="charge")]
    w = world(blocker, helper, *guards)
    assert not w.push(blocker, (9, 0, 0), (49, 50))
    assert (helper.col, helper.row) == (51, 50)


# --- World.tick -------------------------------------------------------------


def test_tick_resets_moved_flags_and_ages_seekers():
    walker = agent(0, 10, 10)
    walker.moved = True
    seeker = agent(1, *MID, state="seek", charge=20000)
    blocker = agent(2, *arrow_target(*MID), state="charge")
    w = world(walker, seeker, blocker)
    w.tick([walker, seeker, blocker], Trail(), 0.0)
    assert seeker.age == 1
    assert walker.age == 0
    assert (seeker.col, seeker.row) == MID


def test_tick_moves_highest_priority_first():
    low = agent(0, *MID)
    high = agent(1, 49, 50, state="seek", charge=20000)
    w = world(low, high)
    w.tick([low, high], Trail(), 0.0)
    assert (high.col, high.row) == arrow_target(49, 50)


# --- stations and nav -------------------------------------------------------


def test_stations_are_spaced_on_walls_away_from_corners():
    assert len(STATIONS) == STATION_COUNT == len(set(STATIONS))
    last = GRID_SIZE - 1
    corners = {(0, 0), (last, 0), (0, last), (last, last)}
    assert not corners & set(STATIONS)
    for c, r in STATIONS:
        assert c in (0, last) or r in (0, last)


def test_nav_arrows_lead_to_the_nearest_station():
    for sx, sy in STATIONS:
        assert NAV.dist[sy][sx] == 0
        assert NAV.offset[sy][sx] == (0, 0)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if NAV.dist[row][col] == 0:
                continue
            c, r = arrow_target(col, row)
            assert NAV.dist[r][c] == NAV.dist[row][col] - 1
            dx, dy = NAV.offset[row][col]
            assert abs(dx) + abs(dy) == NAV.dist[row][col]


# --- advance ----------------------------------------------------------------


def test_advance_keeps_leftover_time_without_ticking():
    a = agent(0, *MID)
    w = world(a)
    left = advance(w, [a], Trail(), 0.0, STEP_INTERVAL / 2, 0.0)
    assert left == pytest.approx(STEP_INTERVAL / 2)
    assert (a.col, a.row) == MID


def test_advance_ticks_once_per_owed_step_and_marks_trail():
    a = agent(0, *MID)
    w = world(a)
    trail = Trail()
    left = advance(w, [a], trail, 1.0, STEP_INTERVAL * 2.5, 0.0)
    assert left == pytest.approx(STEP_INTERVAL / 2)
    assert a.charge == 50000 - 2 * MOVE_COST
    assert trail.value(a.col, a.row, 1.0) == MARK_VALUE


def test_advance_charges_agents_on_stations():
    sx, sy = STATIONS[0]
    a = agent(0, sx, sy, state="charge", charge=0)
    w = world(a)
    advance(w, [a], Trail(), 0.0, 0.1, 0.0)
    assert a.charge == pytest.approx(CHARGE_RATE * 0.1)
