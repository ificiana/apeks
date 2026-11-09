import pytest

from apeks import sim
from apeks.config import (
    AGENT_COLOR,
    CHARGE_RATE,
    DIRECTIONS,
    FOOD_CLUSTER_COUNT,
    FOOD_PER_CLUSTER,
    GRID_SIZE,
    LOW_CHARGE,
    MARK_DECAY,
    MARK_VALUE,
    MAX_CHARGE,
    MAX_MOVING_BARRIERS,
    MOVE_COST,
    NEST_TILES,
    PRESSURE_LIMIT,
    PUSH_REACH,
    STATION_COUNT,
    STEP_INTERVAL,
)
from apeks.sim import (
    CONES,
    Agent,
    MovingBarrier,
    Trail,
    World,
    advance,
    build_nav,
    cardinal,
    format_token_line,
    make_food_clusters,
    make_moving_barriers,
    make_nest,
    make_stations,
    make_static_obstacles,
    parse_token_line,
    prune_token_lines,
    update_token_text,
)

STATIONS = make_stations()
NAV = build_nav(STATIONS)
NEST = make_nest()
NEST_NAV = build_nav(NEST)
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


def test_charge_up_resumes_return_nest_when_holding_food():
    a = agent(0, 0, 0, state="charge", charge=MAX_CHARGE - 1)
    a.has_food = True
    a.charge_up(1.0)
    assert a.charge == MAX_CHARGE
    assert a.state == "return_nest"
    assert a.released


def test_slack_is_charge_moves_minus_distance():
    a = agent(0, *MID, charge=MOVE_COST * 100)
    assert a.slack(NAV) == 100 - NAV.dist[MID[1]][MID[0]]


def test_priority_released_first():
    a = agent(7, *MID)
    a.released = True
    assert a.priority(NAV) == (5, 0, 7)


def test_priority_urgent_seeker():
    a = agent(7, *MID, state="seek", charge=MOVE_COST)
    assert a.priority(NAV) == (4, -a.slack(NAV), 7)


def test_priority_queued_seeker_orders_by_age():
    a = agent(7, *MID, state="seek")
    a.age = 5
    assert a.priority(NAV) == (3, 5, -7)


def test_priority_return_nest():
    a = agent(7, *MID, state="return_nest")
    assert a.priority(NAV) == (2, -a.slack(NAV), 7)


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


# --- foraging and nest ------------------------------------------------------


def test_make_nest_returns_nest_tiles():
    assert make_nest() == list(NEST_TILES)


def test_make_food_clusters_generates_food_and_matches_totals():
    food = make_food_clusters(cluster_count=FOOD_CLUSTER_COUNT, food_per_cluster=FOOD_PER_CLUSTER, seed=42)
    assert len(food) > 0
    assert sum(food.values()) == FOOD_CLUSTER_COUNT * FOOD_PER_CLUSTER
    rand_food = make_food_clusters()
    assert len(rand_food) > 0
    assert sum(rand_food.values()) > 0
    small = make_food_clusters(cluster_count=1, food_per_cluster=3, seed=1)
    assert sum(small.values()) == 3


def test_spawn_random_food():
    w = World(NAV, [])
    assert len(w.food) == 0
    w.spawn_random_food(seed=42)
    assert len(w.food) > 0
    assert sum(w.food.values()) > 0


def test_is_nest():
    w = World(NAV, [], nest_set=set(NEST))
    assert w.is_nest(NEST[0])
    assert not w.is_nest((0, 0))


def test_move_into_nest_deposits_food():
    nx, ny = NEST[0]
    a = agent(0, nx - 1, ny, state="return_nest")
    a.has_food = True
    w = World(NAV, [a], nest_nav=NEST_NAV, nest_set=set(NEST))
    w.move(a, (nx, ny))
    assert not a.has_food
    assert a.state == "walk"
    assert w.food_collected == 1


def test_move_into_food_picks_up_food():
    target = (10, 10)
    a = agent(0, 9, 10, state="walk")
    w = World(NAV, [a], food={target: 2})
    w.move(a, target)
    assert a.has_food
    assert a.state == "return_nest"
    assert a.last_food_tile == target
    assert w.food[target] == 1


def test_move_into_food_depletes_last_item():
    target = (10, 10)
    a = agent(0, 9, 10, state="walk")
    w = World(NAV, [a], food={target: 1})
    w.move(a, target)
    assert target not in w.food
    assert a.has_food


def test_return_nest_on_nest_tile_deposits_food():
    nx, ny = NEST[0]
    a = agent(0, nx, ny, state="return_nest")
    a.has_food = True
    w = World(NAV, [a], nest_nav=NEST_NAV, nest_set=set(NEST))
    a.step(w, Trail(), 0.0)
    assert not a.has_food
    assert a.state == "walk"
    assert w.food_collected == 1


def test_return_nest_moves_along_nest_nav():
    a = agent(0, 10, 10, state="return_nest")
    a.has_food = True
    w = World(NAV, [a], nest_nav=NEST_NAV, nest_set=set(NEST))
    dc, dr = NEST_NAV.arrow[10][10]
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (10 + dc, 10 + dr)
    assert a.moved


def test_return_nest_pushes_blocker():
    a = agent(0, 10, 10, state="return_nest")
    a.has_food = True
    dc, dr = NEST_NAV.arrow[10][10]
    target = (10 + dc, 10 + dr)
    blocker = agent(1, *target, state="walk")
    w = World(NAV, [a, blocker], nest_nav=NEST_NAV, nest_set=set(NEST))
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == target
    assert (blocker.col, blocker.row) != target


def test_return_nest_waits_behind_unpushable_blocker():
    a = agent(0, 10, 10, state="return_nest")
    a.has_food = True
    dc, dr = NEST_NAV.arrow[10][10]
    target = (10 + dc, 10 + dr)
    blocker = agent(1, *target, state="charge")
    w = World(NAV, [a, blocker], nest_nav=NEST_NAV, nest_set=set(NEST))
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (10, 10)


def test_return_nest_released_agent_boxed_in_pushes_aside():
    sx, sy = STATIONS[0]
    a = agent(0, sx, sy, state="return_nest")
    a.has_food = True
    a.released = True
    dc, dr = NEST_NAV.arrow[sy][sx]
    target = (sx + dc, sy + dr)
    target_blocker = agent(1, *target, state="charge")
    other_blockers = [b for b in ring(sx, sy, state="walk") if (b.col, b.row) != target]
    w = World(NAV, [a, target_blocker, *other_blockers], nest_nav=NEST_NAV, nest_set=set(NEST))
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) != (sx, sy)


def test_return_nest_released_stays_when_nobody_can_be_pushed():
    sx, sy = STATIONS[0]
    a = agent(0, sx, sy, state="return_nest")
    a.has_food = True
    a.released = True
    blockers = ring(sx, sy, state="charge")
    w = World(NAV, [a, *blockers], nest_nav=NEST_NAV, nest_set=set(NEST))
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (sx, sy)


def test_return_nest_below_low_charge_becomes_seeker():
    a = agent(0, 10, 10, state="return_nest", charge=LOW_CHARGE - 1)
    a.has_food = True
    w = World(NAV, [a], nest_nav=NEST_NAV, nest_set=set(NEST))
    a.step(w, Trail(), 0.0)
    assert a.state == "seek"
    assert a.has_food


def test_walk_standing_on_food_picks_up():
    here = (10, 10)
    a = agent(0, *here, state="walk")
    w = World(NAV, [a], food={here: 3})
    a.step(w, Trail(), 0.0)
    assert a.has_food
    assert a.state == "return_nest"
    assert a.last_food_tile == here
    assert w.food[here] == 2


def test_walk_standing_on_food_depletes():
    here = (10, 10)
    a = agent(0, *here, state="walk")
    w = World(NAV, [a], food={here: 1})
    a.step(w, Trail(), 0.0)
    assert here not in w.food
    assert a.has_food
    assert a.state == "return_nest"


def test_walk_prefers_adjacent_food_tile():
    a = agent(0, 10, 10, state="walk")
    food_tile = (11, 10)
    w = World(NAV, [a], food={food_tile: 5})
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == food_tile
    assert a.has_food
    assert a.state == "return_nest"


def test_walk_site_fidelity_follows_previous_food_tile():
    a = agent(0, 10, 10, state="walk")
    food_tile = (15, 10)
    a.last_food_tile = food_tile
    w = World(NAV, [a], food={food_tile: 3})
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (11, 10)


def test_walk_site_fidelity_falls_back_when_preferred_blocked():
    a = agent(0, 10, 10, state="walk")
    food_tile = (15, 10)
    a.last_food_tile = food_tile
    blocker = agent(1, 11, 10, state="charge")
    w = World(NAV, [a, blocker], food={food_tile: 3})
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) != (11, 10)
    assert (a.col, a.row) != (10, 10)


def test_walk_site_fidelity_clears_when_tile_depleted():
    a = agent(0, 10, 10, state="walk")
    depleted = (15, 10)
    a.last_food_tile = depleted
    w = World(NAV, [a], food={})
    a.step(w, Trail(), 0.0)
    assert a.last_food_tile is None


def test_walk_follows_food_trail():
    a = agent(0, 10, 10, state="walk")
    food_trail = Trail()
    for dc, dr in CONES[(1, 0)]:
        food_trail.mark(10 + dc, 10 + dr, 0.0)
    w = World(NAV, [a], food_trail=food_trail)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (11, 10)


def test_advance_marks_food_trail_when_holding_food():
    a = agent(0, *MID)
    a.has_food = True
    w = World(NAV, [a])
    trail = Trail()
    advance(w, [a], trail, 1.0, STEP_INTERVAL * 0.5, 0.0)
    assert w.food_trail.value(a.col, a.row, 1.0) == MARK_VALUE


def test_walk_avoids_immediate_uturn_to_prev_tile():
    a = agent(0, 10, 10, state="walk")
    a.prev_tile = (9, 10)
    w = world(a)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) != (9, 10)
    assert a.prev_tile == (10, 10)


def test_walk_broken_food_trail_falls_back_to_exploration():
    a = agent(0, 10, 10, state="walk")
    food_trail = Trail()
    food_trail.mark(9, 10, 0.0)
    a.prev_tile = (9, 10)
    w = World(NAV, [a], food_trail=food_trail)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) != (9, 10)


def test_agent_default_charge_is_max_charge():
    a = Agent(0, 5, 5)
    assert a.charge == MAX_CHARGE


def test_walk_on_station_steps_away_into_interior():
    sx, sy = STATIONS[0]
    a = agent(0, sx, sy, state="walk")
    a.released = True
    w = world(a)
    a.step(w, Trail(), 0.0)
    assert NAV.dist[a.row][a.col] > 0


def test_walk_on_station_released_pushes_when_boxed_in():
    sx, sy = STATIONS[0]
    a = agent(0, sx, sy, state="walk")
    a.released = True
    blockers = [
        agent(10 + i, sx + dc, sy + dr, state="walk")
        for i, (dc, dr) in enumerate(DIRECTIONS)
        if 0 <= sx + dc < GRID_SIZE and 0 <= sy + dr < GRID_SIZE
    ]
    w = world(a, *blockers)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) != (sx, sy)


def test_advance_does_not_mark_food_trail_when_charging():
    sx, sy = STATIONS[0]
    a = agent(0, sx, sy, state="charge", charge=MAX_CHARGE / 2)
    a.has_food = True
    w = World(NAV, [a])
    advance(w, [a], Trail(), 1.0, STEP_INTERVAL * 0.5, 0.0)
    assert w.food_trail.value(sx, sy, 1.0) == 0.0


def test_walk_prefers_unvisited_immediate_tile_over_marked_tile():
    a = agent(0, 10, 10, state="walk")
    w = world(a)
    trail = Trail()
    # Mark (11, 10), (10, 11), (10, 9) heavily
    trail.mark(11, 10, 0.0)
    trail.mark(10, 11, 0.0)
    trail.mark(10, 9, 0.0)
    # (9, 10) is completely unvisited (value == 0)
    a.step(w, trail, 0.0)
    assert (a.col, a.row) == (9, 10)


def test_walk_prefers_free_three_block_corridor():
    a = agent(0, 20, 20, state="walk")
    w = world(a)
    trail = Trail()
    # Left corridor (19, 20), (18, 20), (17, 20) is free of pheromones (value == 0)
    # But lateral cone cells on left have some marks
    trail.mark(18, 21, 0.0)
    trail.mark(17, 21, 0.0)
    # Right, top, bottom have marks right in the corridor
    trail.mark(21, 20, 0.0)
    trail.mark(20, 19, 0.0)
    trail.mark(20, 21, 0.0)
    a.step(w, trail, 0.0)
    # Should choose Left (19, 20)
    assert (a.col, a.row) == (19, 20)


def test_walk_prefers_food_trail_over_direct_cardinal_site_fidelity():
    a = agent(0, 50, 50, state="walk")
    a.last_food_tile = (55, 50)  # Direct cardinal would be East (51, 50)
    food_trail = Trail()
    # Pheromone trail actually leads South (50, 51)
    for dc, dr in CONES[(0, 1)]:
        food_trail.mark(50 + dc, 50 + dr, 0.0)
    w = World(NAV, [a], food_trail=food_trail, food={(55, 50): 10})
    a.step(w, Trail(), 0.0)
    # Follows the pheromone trail South instead of blindly taking the direct cardinal vector East
    assert (a.col, a.row) == (50, 51)


@pytest.mark.parametrize("free_dir", DIRECTIONS)
def test_walk_always_chooses_unvisited_side(free_dir):
    a = agent(0, 50, 50, state="walk")
    w = world(a)
    trail = Trail()
    # Mark all neighbours except the free direction
    for dc, dr in DIRECTIONS:
        if (dc, dr) != free_dir:
            trail.mark(50 + dc, 50 + dr, 0.0)
    a.step(w, trail, 0.0)
    # Must step into the unvisited tile on that side
    assert (a.col, a.row) == (50 + free_dir[0], 50 + free_dir[1])


def test_walk_on_wall_reflects_inward():
    a = agent(0, 0, 10, state="walk")
    w = world(a)
    trail = Trail()
    a.step(w, trail, 0.0)
    assert a.col == 1 and a.row == 10


def test_walk_on_wall_with_no_inward_moves_falls_back():
    a = agent(0, 0, 10, state="walk")
    blocker = agent(1, 1, 10, state="walk")
    w = world(a, blocker)
    trail = Trail()
    a.step(w, trail, 0.0)
    # Inward (1, 10) is blocked, so inward_moves is empty; it falls back to along wall
    assert a.col == 0 and a.row in (9, 11)


def test_walk_momentum_favors_same_heading():
    a = agent(0, 50, 50, state="walk")
    a.prev_tile = (49, 50)  # heading East (1, 0)
    w = world(a)
    trail = Trail()
    a.step(w, trail, 0.0)
    # Momentum keeps agent traveling East
    assert (a.col, a.row) == (51, 50)


def test_walk_ignores_ghost_food_scent_below_threshold():
    a = agent(0, 10, 10, state="walk")
    food_trail = Trail()
    food_trail.mark(11, 10, 0.0)
    w = World(NAV, [a], food_trail=food_trail)
    # At t=100.0, the mark has decayed far below 0.5
    a.step(w, Trail(), 100.0)
    # Ignored decayed ghost scent; does not treat it as a valid food trail
    assert (a.col, a.row) != (10, 10)


def test_advance_does_not_mark_exploration_trail_when_not_walking():
    a_return = agent(0, 20, 20, state="return_nest")
    a_seek = agent(1, 30, 30, state="seek")
    w = world(a_return, a_seek)
    trail = Trail()
    advance(w, [a_return, a_seek], trail, 1.0, STEP_INTERVAL * 0.5, 0.0)
    assert trail.value(20, 20, 1.0) == 0.0
    assert trail.value(30, 30, 1.0) == 0.0


def test_advance_does_not_mark_food_trail_when_seeking_battery():
    a = agent(0, *MID, state="seek")
    a.has_food = True
    w = World(NAV, [a])
    trail = Trail()
    advance(w, [a], trail, 1.0, STEP_INTERVAL * 0.5, 0.0)
    assert w.food_trail.value(a.col, a.row, 1.0) == 0.0


def test_food_trail_accumulates_on_same_tile():
    ft = Trail(accumulate=True)
    ft.mark(5, 5, 0.0)
    val1 = ft.value(5, 5, 0.0)
    assert val1 == MARK_VALUE
    ft.mark(5, 5, 1.0)
    val2 = ft.value(5, 5, 1.0)
    assert val2 > MARK_VALUE
    assert val2 == pytest.approx(val1 * MARK_DECAY + MARK_VALUE)


def test_food_trail_accumulate_resets_when_decayed():
    ft = Trail(accumulate=True)
    ft.mark(5, 5, 0.0)
    ft.mark(5, 5, 200.0)
    assert ft.value(5, 5, 200.0) == MARK_VALUE


def test_exploration_trail_does_not_accumulate():
    t = Trail(accumulate=False)
    t.mark(5, 5, 0.0)
    t.mark(5, 5, 1.0)
    assert t.value(5, 5, 1.0) == MARK_VALUE


def test_advance_food_trail_does_not_multiply_on_same_tile_without_moving():
    a = agent(0, *MID, state="return_nest")
    a.has_food = True
    w = World(NAV, [a])
    trail = Trail()
    advance(w, [a], trail, 1.0, STEP_INTERVAL * 0.1, 0.0)
    val1 = w.food_trail.value(a.col, a.row, 1.0)
    assert val1 == MARK_VALUE
    advance(w, [a], trail, 1.05, STEP_INTERVAL * 0.1, 0.0)
    val2 = w.food_trail.value(a.col, a.row, 1.05)
    assert val2 <= val1


def test_walk_follows_less_pheromone_trail_at_split():
    a = agent(0, 50, 50, state="walk")
    food_trail = Trail(accumulate=True)
    # Branch 1 (North) is heavily traveled (400)
    food_trail.levels[49][50] = 400.0
    food_trail.stamp[49][50] = 0.0
    # Branch 2 (East) is newer/lighter (100)
    food_trail.levels[50][51] = 100.0
    food_trail.stamp[50][51] = 0.0
    w = World(NAV, [a], nest_nav=NEST_NAV, nest_set=set(NEST), food_trail=food_trail)
    a.step(w, Trail(), 0.0)
    # At the split, agent chooses the lesser pheromone branch (East) to balance traffic and prevent losing it
    assert (a.col, a.row) == (51, 50)


def test_walk_follows_single_continuous_food_trail():
    a = agent(0, 50, 50, state="walk")
    food_trail = Trail(accumulate=True)
    # Single continuous trail going North
    food_trail.levels[49][50] = 200.0
    food_trail.stamp[49][50] = 0.0
    w = World(NAV, [a], nest_nav=NEST_NAV, nest_set=set(NEST), food_trail=food_trail)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (50, 49)


def test_walk_approaching_trail_from_outside_steers_toward_scent():
    a = agent(0, 50, 50, state="walk")
    food_trail = Trail(accumulate=True)
    # Trail is 2 tiles away North (value on (50, 49) is 0, but cone senses (50, 48))
    food_trail.levels[48][50] = 200.0
    food_trail.stamp[48][50] = 0.0
    w = World(NAV, [a], nest_nav=NEST_NAV, nest_set=set(NEST), food_trail=food_trail)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (50, 49)


def test_make_static_obstacles_and_moving_barriers():
    obs = make_static_obstacles()
    barriers = make_moving_barriers()
    assert len(obs) > 0
    assert len(barriers) == 4
    stations = set(make_stations())
    nest = set(make_nest())
    assert not (obs & stations)
    assert not (obs & nest)
    for b in barriers:
        assert not (set(b.coords) & stations)
        assert not (set(b.coords) & nest)
        assert not (set(b.coords) & obs)


def test_build_nav_with_obstacles_and_unreachable():
    obs = make_static_obstacles()
    nav_obs = build_nav(STATIONS, obstacles=obs)
    assert nav_obs is not None
    # Enclosed tile (15, 15) surrounded by obstacles
    enc_obs = {(15, 14), (15, 16), (14, 15), (16, 15)}
    nav_enc = build_nav(STATIONS, obstacles=enc_obs)
    assert nav_enc.arrow[15][15] == (0, 0)


def test_world_obstacles_property_and_free():
    obs = {(10, 10)}
    b = MovingBarrier(coords=[(20, 20)], axis="y", direction=1, min_pos=15, max_pos=25)
    w = World(NAV, [], static_obstacles=obs, moving_barriers=[b])
    assert (10, 10) in w.obstacles
    assert (20, 20) in w.obstacles
    assert (20, 20) in w.barrier_coords
    assert not w.free((10, 10))
    assert not w.free((20, 20))
    assert w.free((30, 30))


def test_tokens_write_read_and_expiration():
    w = World(NAV, [])
    w.write_token((5, 5), "FOOD_DEPLETED", 0.0)
    assert w.read_token((5, 5), 10.0) == "FOOD_DEPLETED"
    assert w.read_token((5, 5), 50.0) is None
    assert w.read_token((9, 9), 0.0) is None


def test_active_tokens_cleanup():
    w = World(NAV, [])
    w.write_token((1, 1), "OLD", 0.0)
    w.write_token((2, 2), "FRESH", 50.0)
    active = w.active_tokens(now=55.0, ttl=30.0)
    assert (1, 1) not in active
    assert active.get((2, 2)) == "FRESH"


def test_timestamped_token_helpers_and_append():
    # 1. format_token_line
    assert format_token_line("[10.5s] Hello", now=20.0) == "[10.5s] Hello"
    assert format_token_line("Hello", now=20.0) == "[20.0s] Hello"

    # 2. parse_token_line
    ts, text = parse_token_line("[12.4s] Test message")
    assert ts == 12.4
    assert text == "Test message"
    ts_none, text_none = parse_token_line("Plain text")
    assert ts_none is None
    assert text_none == "Plain text"
    ts_inv, text_inv = parse_token_line("[..] invalid float")
    assert ts_inv is None

    # 3. prune_token_lines with blank lines, stale lines, and fresh lines
    multi = "\n[10.0s] Old line\n\n[25.0s] Fresh line\n"
    pruned = prune_token_lines(multi, now=30.0, max_age=10.0)
    assert "[10.0s] Old line" not in pruned
    assert "[25.0s] Fresh line" in pruned

    # 4. update_token_text: overrides obsolete guide lines when [DEPLETED] is posted
    old_text = "\n[5.0s] [GUIDE] Food ~10 steps East at (30, 20) [4u]\n\n[8.0s] [ALERT] Hazard at (10, 10)\n"
    new_line = "[DEPLETED] Food at (30, 20) exhausted"
    merged = update_token_text(old_text, new_line, now=12.0, max_age=20.0)
    assert "[GUIDE]" not in merged
    assert "[ALERT] Hazard at (10, 10)" in merged
    assert "[DEPLETED] Food at (30, 20) exhausted" in merged

    # 4b. update_token_text with depleted without coords, and duplicate line prevention
    dep_no_coord = update_token_text(merged, "[DEPLETED] all done", now=15.0)
    assert "[DEPLETED] all done" in dep_no_coord
    dup = update_token_text(dep_no_coord, "[DEPLETED] all done", now=15.0)
    assert dup == dep_no_coord

    # 5. World.append_token
    w = World(NAV, [])
    w.append_token((10, 10), "[GUIDE] Food ~5 steps at (15, 10)", now=0.0)
    assert "[0.0s]" in w.read_token((10, 10), now=5.0)
    # Append second line
    w.append_token((10, 10), "[ALERT] Moving barrier", now=10.0)
    read_multi = w.read_token((10, 10), now=12.0)
    assert "[0.0s]" in read_multi
    assert "[10.0s]" in read_multi
    # Advance time so first line expires (max_age=15)
    w.append_token(
        (10, 10), "[DEPLETED] Food at (15, 10) exhausted", now=20.0, max_age=15.0
    )
    read_pruned = w.read_token((10, 10), now=22.0)
    assert "[GUIDE]" not in read_pruned
    assert "[ALERT]" in read_pruned
    assert "[DEPLETED]" in read_pruned

    # 6. append_token cleans up if merged becomes empty
    w.tokens[(20, 20)] = ("", 0.0)
    w.append_token((20, 20), "", now=10.0)
    assert (20, 20) not in w.tokens
    # append_token on tile not in self.tokens with empty line
    w.append_token((99, 99), "", now=10.0)
    assert (99, 99) not in w.tokens


def test_dynamic_hazards_spawn_and_expiry():
    w = World(NAV, [])
    cluster = w.spawn_hazard(0.0, candidate=(12, 12), lifetime=25.0)
    assert cluster == [(12, 12)]
    assert (12, 12) in w.obstacles
    assert (12, 12) in w.dynamic_hazards
    # Multi-tile shape spawn with default random lifetime
    shape_cluster = w.spawn_hazard(
        0.0, candidate=(20, 20), shape=((0, 0), (1, 0), (0, 1))
    )
    assert shape_cluster == [(20, 20), (21, 20), (20, 21)]
    assert (21, 20) in w.dynamic_hazards
    # Tick past expiry
    w.tick_obstacles(now=40.0, dt=1.0)
    assert (12, 12) not in w.dynamic_hazards
    assert (20, 20) not in w.dynamic_hazards


def test_spawn_hazard_random_and_fallback():
    w = World(NAV, [])
    cluster = w.spawn_hazard(0.0)
    assert cluster is not None and len(cluster) > 0
    # Blocked candidate returns None
    w.static_obstacles = {(10, 10)}
    assert w.spawn_hazard(0.0, candidate=(10, 10)) is None
    # When all grid candidates are blocked
    w.static_obstacles = {
        (c, r) for r in range(10, GRID_SIZE - 10) for c in range(10, GRID_SIZE - 10)
    }
    assert w.spawn_hazard(0.0) is None


def test_moving_barrier_step_and_reversal():
    # Boundary reversal (y axis)
    b = MovingBarrier(coords=[(10, 50)], axis="y", direction=1, min_pos=45, max_pos=50)
    w = World(NAV, [])
    w._step_barrier(b)
    assert b.direction == -1
    assert b.coords == [(10, 50)]

    # Obstacle collision reversal
    b2 = MovingBarrier(coords=[(10, 48)], axis="y", direction=1, min_pos=40, max_pos=60)
    w2 = World(NAV, [], static_obstacles={(10, 49)})
    w2._step_barrier(b2)
    assert b2.direction == -1

    # Collision with another moving barrier
    b_other = MovingBarrier(
        coords=[(10, 49)], axis="x", direction=1, min_pos=5, max_pos=15
    )
    b_test = MovingBarrier(
        coords=[(10, 48)], axis="y", direction=1, min_pos=40, max_pos=60
    )
    w_barriers = World(NAV, [], moving_barriers=[b_test, b_other])
    w_barriers._step_barrier(b_test)
    assert b_test.direction == -1

    # Collision with dynamic hazard
    b_haz = MovingBarrier(
        coords=[(10, 48)], axis="y", direction=1, min_pos=40, max_pos=60
    )
    w_haz = World(NAV, [])
    w_haz.dynamic_hazards[(10, 49)] = 100.0
    w_haz._step_barrier(b_haz)
    assert b_haz.direction == -1

    # Normal step (x axis)
    b3 = MovingBarrier(coords=[(48, 10)], axis="x", direction=1, min_pos=40, max_pos=60)
    w3 = World(NAV, [])
    w3._step_barrier(b3)
    assert b3.coords == [(49, 10)]
    assert b3.direction == 1


def test_push_skips_obstacle_tile():
    blocker = agent(1, 10, 10, state="walk")
    obs = {(9, 10), (10, 9), (10, 11)}
    other = agent(2, 11, 10, state="charge")
    w = World(NAV, [blocker, other], static_obstacles=obs)
    claim = (3, 0, 0)
    assert not w.push(blocker, claim, (10, 10))


def test_make_food_clusters_with_obstacles():
    obs = {(c, r) for r in range(GRID_SIZE) for c in range(GRID_SIZE)}
    food = make_food_clusters(cluster_count=1, obstacles=obs)
    assert len(food) == 0


def test_seek_detours_around_obstacle():
    dc, dr = NAV.arrow[10][10]
    target = (10 + dc, 10 + dr)
    a = agent(0, 10, 10, state="seek", charge=10000)
    w = World(NAV, [a], static_obstacles={target})
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) != target
    assert (a.col, a.row) != (10, 10)
    assert a.moved


def test_return_nest_detours_around_obstacle():
    dc, dr = NEST_NAV.arrow[10][10]
    target = (10 + dc, 10 + dr)
    a = agent(0, 10, 10, state="return_nest")
    a.has_food = True
    w = World(NAV, [a], nest_nav=NEST_NAV, nest_set=set(NEST), static_obstacles={target})
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) != target
    assert (a.col, a.row) != (10, 10)
    assert a.moved


def test_advance_ticks_obstacles_and_barriers():
    b = MovingBarrier(
        coords=[(10, 48)],
        axis="y",
        direction=1,
        min_pos=40,
        max_pos=60,
        interval=0.1,
    )
    w = World(NAV, [], moving_barriers=[b])
    w.next_hazard_time = 0.05
    advance(w, [], Trail(), 1.0, 0.2, 0.0)
    assert len(w.dynamic_hazards) >= 1
    assert b.coords != [(10, 48)]


def test_seek_obstacle_boxed_in_no_detour():
    obs_around = {(11, 10), (9, 10), (10, 11), (10, 9)}
    a = agent(0, 10, 10, state="seek", charge=10000)
    w = World(NAV, [a], static_obstacles=obs_around)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (10, 10)


def test_return_nest_obstacle_boxed_in_no_detour():
    obs_around = {(11, 10), (9, 10), (10, 11), (10, 9)}
    a = agent(0, 10, 10, state="return_nest")
    a.has_food = True
    w = World(NAV, [a], nest_nav=NEST_NAV, nest_set=set(NEST), static_obstacles=obs_around)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (10, 10)


def test_moving_barrier_reversal_min_pos_x():
    b = MovingBarrier(coords=[(40, 10)], axis="x", direction=-1, min_pos=40, max_pos=50)
    w = World(NAV, [])
    w._step_barrier(b)
    assert b.direction == 1
    assert b.coords == [(40, 10)]


def test_dynamic_hazards_capped_at_twenty_and_timer_interval():
    b = MovingBarrier(
        coords=[(10, 48)], axis="y", direction=1, min_pos=40, max_pos=60, interval=5.0
    )
    w = World(NAV, [], moving_barriers=[b])
    w.dynamic_hazards = {(i, 10): 100.0 for i in range(20)}
    w.next_hazard_time = 0.0
    w.tick_obstacles(now=10.0, dt=0.01)
    assert len(w.dynamic_hazards) == 20
    assert b.timer == 0.01


def test_spawn_moving_barrier_and_temporary_expiry():
    w = World(NAV, [])
    # Valid spawn at candidate
    b = w.spawn_moving_barrier(now=0.0, candidate=(15, 15), axis="x", lifetime=20.0)
    assert b is not None
    assert b.expiry == 20.0
    assert len(b.coords) == 3
    assert (15, 15) in w.barrier_coords
    assert (15, 15) in w.tokens

    # Default random spawn
    b_rand = w.spawn_moving_barrier(now=5.0)
    assert b_rand is not None
    assert len(w.moving_barriers) == 2


def test_spawn_moving_barrier_fallbacks():
    w = World(NAV, [])
    # Out of bounds
    assert (
        w.spawn_moving_barrier(
            now=0.0, candidate=(GRID_SIZE - 1, GRID_SIZE - 1), axis="x"
        )
        is None
    )
    # Blocked by obstacle
    w.static_obstacles = {(10, 10)}
    assert w.spawn_moving_barrier(now=0.0, candidate=(10, 10)) is None
    # All candidates blocked
    w.static_obstacles = {
        (c, r) for r in range(12, GRID_SIZE - 12) for c in range(12, GRID_SIZE - 12)
    }
    assert w.spawn_moving_barrier(now=0.0) is None


def test_tick_obstacles_spawns_and_expires_moving_barriers():
    w = World(NAV, [])
    b = MovingBarrier(
        coords=[(10, 48)], axis="y", direction=1, min_pos=40, max_pos=60, expiry=10.0
    )
    w.moving_barriers = [b]
    w.next_barrier_time = 5.0
    # Before expiry
    w.tick_obstacles(now=8.0, dt=0.01)
    assert b in w.moving_barriers

    # Past expiry: b is removed and a new temporary barrier is spawned since now >= next_barrier_time
    w.tick_obstacles(now=12.0, dt=0.01)
    assert b not in w.moving_barriers
    assert len(w.moving_barriers) >= 1

    # When moving barriers are already capped at MAX_MOVING_BARRIERS
    w.moving_barriers = [
        MovingBarrier(
            coords=[(i, 40)], axis="x", direction=1, min_pos=0, max_pos=50
        )
        for i in range(MAX_MOVING_BARRIERS)
    ]
    w.next_barrier_time = 0.0
    w.tick_obstacles(now=15.0, dt=0.01)
    assert len(w.moving_barriers) == MAX_MOVING_BARRIERS


def test_agent_waypoint_movement():
    # 1. Agent steps when at waypoint -> clears waypoint and continues
    a = agent(0, 10, 10, state="walk", charge=50000)
    a.waypoint = (10, 10)
    w = World(NAV, [a])
    a.step(w, Trail(), 0.0)
    assert a.waypoint is None

    # 2. Agent steps towards remote waypoint
    a2 = agent(1, 10, 10, state="walk", charge=50000)
    a2.waypoint = (15, 10)
    w2 = World(NAV, [a2])
    a2.step(w2, Trail(), 0.0)
    assert (a2.col, a2.row) == (11, 10)
    assert a2.waypoint == (15, 10)


def test_charging_detour_suppresses_food_trail_from_station_to_nest():
    # 1. Agent starts with food, food_trail_valid is True
    a = agent(0, 50, 50, state="return_nest", charge=LOW_CHARGE - 1)
    a.has_food = True
    assert a.food_trail_valid is True

    w = World(NAV, [a])
    trail = Trail()

    # 2. Agent steps with low battery -> enters "seek" and marks food_trail_valid = False
    a.step(w, trail, 0.0)
    assert a.state == "seek"
    assert a.food_trail_valid is False

    # 3. Simulate agent completing charging at station
    a.state = "charge"
    a.charge = MAX_CHARGE - 1
    a.charge_up(dt=10.0)
    assert a.charge == MAX_CHARGE
    assert a.state == "return_nest"
    assert a.has_food is True
    assert a.food_trail_valid is False

    # 4. Agent advances while returning from charger -> no food trail marked on this leg
    advance(w, [a], trail, 1.0, STEP_INTERVAL * 0.5, 0.0)
    assert w.food_trail.value(a.col, a.row, 1.0) == 0.0

    # 5. Agent delivers food at nest
    a.moved = False
    w.is_nest = lambda t: True
    a.step(w, trail, 2.0)
    assert not a.has_food
    assert a.food_trail_valid is True
    assert w.food_collected == 1


def test_seek_agent_waypoint_movement():
    # 1. Agent steps when at waypoint in seek state -> clears waypoint and continues
    a = agent(0, 10, 10, state="seek", charge=10000)
    a.waypoint = (10, 10)
    w = World(NAV, [a])
    a.step(w, Trail(), 0.0)
    assert a.waypoint is None

    # 2. Agent steps towards remote waypoint in seek state
    a2 = agent(1, 10, 10, state="seek", charge=10000)
    a2.waypoint = (15, 10)
    w2 = World(NAV, [a2])
    a2.step(w2, Trail(), 0.0)
    assert (a2.col, a2.row) == (11, 10)
    assert a2.waypoint == (15, 10)

    # 3. Agent has waypoint but all surrounding tiles are blocked -> falls through
    obs_around = {(11, 10), (9, 10), (10, 11), (10, 9)}
    a3 = agent(2, 10, 10, state="seek", charge=10000)
    a3.waypoint = (15, 10)
    w3 = World(NAV, [a3], static_obstacles=obs_around)
    a3.step(w3, Trail(), 0.0)
    assert (a3.col, a3.row) == (10, 10)


def test_forager_prioritizes_food_pheromone_trail_over_waypoint():
    # 1. Forager on/near active food pheromone trail prioritizes the trail over straight-line waypoint
    a = agent(0, 50, 48, state="walk", charge=50000)
    a.role = "forager"
    a.waypoint = (55, 48)  # Straight line would go East to (51, 48)
    w = World(NAV, [a], nest_nav=NEST_NAV, nest_set=set(NEST))
    # Mark active food pheromone trail outward to the North at (50, 47)
    w.food_trail.mark(50, 47, 0.0)
    a.step(w, Trail(), 0.0)
    # The forager takes the pheromone trail outward at (50, 47) instead of blindly walking East
    assert (a.col, a.row) == (50, 47)
    assert a.waypoint == (55, 48)

    # 2. Scout with non-food waypoint ignores food pheromone trail and continues scouting mission
    b = agent(1, 50, 48, state="walk", charge=50000)
    b.role = "scout"
    b.waypoint = (55, 48)
    w2 = World(NAV, [b], nest_nav=NEST_NAV, nest_set=set(NEST))
    w2.food_trail.mark(50, 47, 0.0)
    b.step(w2, Trail(), 0.0)
    # Scout continues toward its scouting waypoint to the East
    assert (b.col, b.row) == (51, 48)

    # 3. Agent with waypoint in world.food also prioritizes the food pheromone trail
    c = agent(2, 50, 48, state="walk", charge=50000)
    c.waypoint = (50, 40)
    w3 = World(NAV, [c], nest_nav=NEST_NAV, nest_set=set(NEST), food={(50, 40): 10})
    w3.food_trail.mark(50, 47, 0.0)
    c.step(w3, Trail(), 0.0)
    assert (c.col, c.row) == (50, 47)

    # 4. Picking up food clears agent waypoint
    d = agent(3, 20, 20, state="walk", charge=50000)
    d.waypoint = (20, 20)
    w4 = World(NAV, [d], food={(20, 20): 5})
    d.step(w4, Trail(), 0.0)
    assert d.has_food is True
    assert d.state == "return_nest"
    assert d.waypoint is None


def test_is_dead_property_and_unpushable():
    a = agent(0, 10, 10, charge=0, state="walk")
    assert a.is_dead is True
    assert a.pushable(NAV, (10, 0, 99)) is False

    charging_a = agent(1, 10, 10, charge=0, state="charge")
    assert charging_a.is_dead is False

    live_a = agent(2, 10, 10, charge=MOVE_COST, state="walk")
    assert live_a.is_dead is False


def test_seek_waypoint_queue_waiting_zero_drain():
    # Agent at distance 1 to occupied station waits quietly without pacing
    a = agent(0, 50, 49, state="seek", charge=1000)
    a.waypoint = (50, 50)
    occ = agent(1, 50, 50, state="charge", charge=50000)
    w = World(NAV, [a, occ], nest_nav=NEST_NAV, nest_set=set(NEST))

    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (50, 49)
    assert a.charge == 1000  # Zero battery drain while waiting stationary in queue

    # Waypoint becomes free -> steps onto station
    del w.occ[(50, 50)]
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (50, 50)
    assert a.charge == 1000 - MOVE_COST


def test_seek_dead_agent_detour():
    a = agent(0, 10, 10, state="seek", charge=5000)
    dc, dr = NAV.arrow[10][10]
    target = (10 + dc, 10 + dr)
    dead_blocker = agent(1, *target, state="walk", charge=0)
    w = World(NAV, [a, dead_blocker], nest_nav=NEST_NAV, nest_set=set(NEST))

    a.step(w, Trail(), 0.0)
    # Detours around dead agent
    assert (a.col, a.row) != (10, 10)
    assert (a.col, a.row) != target
    assert a.charge == 5000 - MOVE_COST

    # Trapped seeker with no detours returns without moving
    trapped = agent(2, 20, 20, state="seek", charge=5000)
    tdc, tdr = NAV.arrow[20][20]
    dead_target = (20 + tdc, 20 + tdr)
    dead_b = agent(3, *dead_target, state="walk", charge=0)
    # Block all other 3 directions with static obstacles
    obs = {
        (20 + ddc, 20 + ddr)
        for ddc, ddr in DIRECTIONS
        if (20 + ddc, 20 + ddr) != dead_target
    }
    w_trapped = World(NAV, [trapped, dead_b], static_obstacles=obs, nest_nav=NEST_NAV, nest_set=set(NEST))
    trapped.step(w_trapped, Trail(), 0.0)
    assert (trapped.col, trapped.row) == (20, 20)


def test_return_nest_dead_agent_detour():
    a = agent(0, 48, 48, state="return_nest", charge=50000)
    a.has_food = True
    dc, dr = NEST_NAV.arrow[48][48]
    target = (48 + dc, 48 + dr)
    dead_blocker = agent(1, *target, state="walk", charge=0)
    w = World(NAV, [a, dead_blocker], nest_nav=NEST_NAV, nest_set=set(NEST))

    a.step(w, Trail(), 0.0)
    # Detours around dead agent toward nest
    assert (a.col, a.row) != (48, 48)
    assert (a.col, a.row) != target
    assert a.charge == 50000 - MOVE_COST

    # Trapped forager with no detours returns without moving
    trapped = agent(2, 30, 30, state="return_nest", charge=50000)
    trapped.has_food = True
    tdc, tdr = NEST_NAV.arrow[30][30]
    dead_target = (30 + tdc, 30 + tdr)
    dead_b = agent(3, *dead_target, state="walk", charge=0)
    obs = {
        (30 + ddc, 30 + ddr)
        for ddc, ddr in DIRECTIONS
        if (30 + ddc, 30 + ddr) != dead_target
    }
    w_trapped = World(NAV, [trapped, dead_b], static_obstacles=obs, nest_nav=NEST_NAV, nest_set=set(NEST))
    trapped.step(w_trapped, Trail(), 0.0)
    assert (trapped.col, trapped.row) == (30, 30)


def test_seek_non_station_waypoint_obstacle_detour():
    a = agent(0, 20, 20, state="seek", charge=50000)
    a.waypoint = (20, 18)  # 2 tiles North
    obs = {(20, 19)}
    w = World(NAV, [a], static_obstacles=obs)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) in [(19, 20), (21, 20), (20, 21)]


def test_seek_waypoint_queue_waiting_distance_two():
    st = (50, 50)
    a = agent(0, 50, 48, state="seek", charge=1000)
    a.waypoint = st
    queue_leader = agent(1, 50, 49, state="seek", charge=1000)
    w = World(NAV, [a, queue_leader], nest_nav=NEST_NAV, nest_set=set(NEST), stations=[st])
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (50, 48)
    assert a.charge == 1000


def test_seek_on_nest_deposits_food_and_continues_seeking():
    nx, ny = NEST[0]
    a = agent(0, nx, ny, state="seek", charge=10000)
    a.has_food = True
    w = World(NAV, [a], nest_nav=NEST_NAV, nest_set=set(NEST))
    a.step(w, Trail(), 0.0)
    assert a.has_food is False
    assert a.state == "seek"
    assert w.food_collected == 1
    assert a.waypoint is None


def test_seek_moves_onto_nest_deposits_food_and_continues_seeking():
    nx, ny = NEST[0]
    a = agent(0, nx - 1, ny, state="seek", charge=10000)
    a.has_food = True
    w = World(NAV, [a], nest_nav=NEST_NAV, nest_set=set(NEST))
    w.move(a, (nx, ny))
    assert a.has_food is False
    assert a.state == "seek"
    assert w.food_collected == 1
    assert a.waypoint is None


def test_trail_clear_tile_and_clear_area():
    t = Trail(accumulate=True)
    t.mark(10, 10, 1.0)
    t.mark(11, 10, 1.0)
    assert t.value(10, 10, 1.0) >= 0.5
    t.clear_tile(10, 10)
    assert t.value(10, 10, 1.0) == 0.0
    t.clear_area(11, 10, radius=2)
    assert t.value(11, 10, 1.0) == 0.0
    # Boundary check
    t.clear_tile(-1, -1)


def test_food_cluster_depletion_clears_trail_and_announces_depleted():
    here = (20, 20)
    a = agent(0, *here, state="walk")
    w = World(NAV, [a], food={here: 1})
    w.food_trail.mark(20, 20, 0.0)
    a.step(w, Trail(), 0.0)
    assert here not in w.food
    assert a.has_food is True
    assert a.food_trail_valid is False
    assert w.food_trail.value(20, 20, 0.0) == 100.0
    assert not any("[DEPLETED]" in text for pos, text in w.active_tokens(0.0).items())


def test_world_move_depletes_cluster():
    target = (30, 30)
    a = agent(0, 29, 30, state="walk")
    w = World(NAV, [a], food={target: 1})
    w.food_trail.mark(30, 30, 0.0)
    w.move(a, target)
    assert target not in w.food
    assert a.has_food is True
    assert a.food_trail_valid is False
    assert w.food_trail.value(30, 30, 0.0) == 100.0
    assert not any("[DEPLETED]" in text for pos, text in w.active_tokens(0.0).items())


def test_agent_arrives_at_depleted_last_food_tile():
    a = agent(0, 26, 25, state="walk")
    a.last_food_tile = (26, 25)
    w = World(NAV, [a], food={})
    w.food_trail.mark(26, 25, 0.0)
    w.food_trail.mark(27, 25, 0.0)
    a.step(w, Trail(), 0.0)
    assert a.last_food_tile is None
    assert w.food_trail.value(26, 25, 0.0) == 100.0
    assert w.food_trail.value(27, 25, 0.0) > 0.0


def test_agent_sees_depleted_last_food_tile_from_nearby():
    a = agent(0, 25, 25, state="walk")
    a.last_food_tile = (26, 25)
    w = World(NAV, [a], food={(28, 25): 5})
    w.food_trail.mark(25, 25, 0.0)
    w.food_trail.mark(26, 25, 0.0)
    a.step(w, Trail(), 0.0)
    assert a.last_food_tile is None
    # Neither tile is cleared because agent is not standing on last_food_tile and food is nearby
    assert w.food_trail.value(25, 25, 0.0) > 0.0
    assert w.food_trail.value(26, 25, 0.0) > 0.0


def test_food_tile_depletion_with_cluster_remaining_step():
    here = (20, 20)
    neighbor = (21, 20)
    a = agent(0, *here, state="walk")
    w = World(NAV, [a], food={here: 1, neighbor: 5})
    w.food_trail.mark(20, 20, 0.0)
    a.step(w, Trail(), 0.0)
    assert here not in w.food
    assert neighbor in w.food
    assert w.food_trail.value(20, 20, 0.0) > 0.0


def test_food_tile_depletion_with_cluster_remaining_move():
    target = (30, 30)
    neighbor = (31, 30)
    a = agent(0, 29, 30, state="walk")
    w = World(NAV, [a], food={target: 1, neighbor: 5})
    w.food_trail.mark(30, 30, 0.0)
    w.move(a, target)
    assert target not in w.food
    assert neighbor in w.food
    assert w.food_trail.value(30, 30, 0.0) > 0.0


def test_seeker_with_station_waypoint_detours_around_return_nest_blocker():
    seeker = agent(0, 3, 15, state="seek", charge=50000)
    seeker.waypoint = (0, 15)
    blocker = agent(1, 2, 15, state="return_nest")
    blocker.has_food = True
    w = World(NAV, [seeker, blocker], stations=[(0, 15)])
    seeker.step(w, Trail(), 0.0)
    assert (seeker.col, seeker.row) in ((4, 15), (3, 14), (3, 16))


def test_seeker_with_station_waypoint_waits_behind_fellow_seeker():
    seeker = agent(0, 3, 15, state="seek", charge=50000)
    seeker.waypoint = (0, 15)
    fellow_seeker = agent(1, 2, 15, state="seek")
    w = World(NAV, [seeker, fellow_seeker], stations=[(0, 15)])
    seeker.step(w, Trail(), 0.0)
    assert (seeker.col, seeker.row) == (3, 15)



def test_return_nest_detours_around_oncoming_seek_blocker():
    a = agent(0, 10, 10, state="return_nest")
    a.has_food = True
    dc, dr = NEST_NAV.arrow[10][10]
    target = (10 + dc, 10 + dr)
    blocker = agent(1, *target, state="seek")
    blocker.has_food = True
    w = World(NAV, [a, blocker], nest_nav=NEST_NAV, nest_set=set(NEST))
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) != (10, 10)
    assert (a.col, a.row) != target


def test_seek_detours_around_oncoming_return_nest_blocker(monkeypatch):
    a = agent(0, *MID, state="seek", charge=50000)
    target = arrow_target(*MID)
    blocker = agent(1, *target, state="return_nest")
    w = world(a, blocker)
    monkeypatch.setattr(w, "push", lambda *args: False)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) != MID
    assert (a.col, a.row) != target


def test_clear_trail_branch_clears_branch_towards_nest_and_stops_near_active_food():
    trail = Trail(accumulate=True)
    coords = [(20, 20), (21, 20), (21, 21), (22, 21), (22, 22), (23, 22), (23, 23), (24, 23), (24, 24), (25, 24)]
    for c, r in coords:
        trail.mark(c, r, 0.0)
    food = {(10, 10): 0, (20, 21): 5, (25, 24): 10, (49, 49): 20}
    trail.clear_trail_branch(20, 20, NEST_NAV, food)
    assert trail.value(20, 20, 0.0) == 0.0
    assert trail.value(21, 20, 0.0) == 0.0
    assert trail.value(25, 24, 0.0) > 0.0

    # If called on a tile already in protected_tiles, it immediately stops and does not clear
    trail.clear_trail_branch(25, 24, NEST_NAV, food)
    assert trail.value(25, 24, 0.0) > 0.0


def test_clear_trail_branch_stops_near_nest():
    trail = Trail(accumulate=True)
    trail.mark(48, 49, 0.0)
    trail.clear_trail_branch(48, 49, NEST_NAV, food={})
    assert trail.value(48, 49, 0.0) == 0.0


def test_forager_at_trail_end_with_no_food_clears_branch_and_switches_to_scout():
    a = agent(0, 20, 20, state="walk")
    a.role = "forager"
    w = World(NAV, [a], nest_nav=NEST_NAV, food={})
    w.food_trail.mark(21, 20, 0.0)
    w.food_trail.mark(20, 20, 0.0)
    a.step(w, Trail(), 0.0)
    assert a.role == "scout"
    assert a.waypoint is None
    assert w.food_trail.value(20, 20, 0.0) == 100.0
    assert w.food_trail.value(21, 20, 0.0) > 0.0
    assert not any("[DEPLETED]" in text for pos, text in w.active_tokens(0.0).items())


def test_forager_clears_tile_when_in_depleted_zone_with_outward_trail():
    a = agent(0, 20, 20, state="walk")
    a.role = "forager"
    w = World(NAV, [a], nest_nav=NEST_NAV, food={})
    w.food_trail.mark(20, 20, 0.0)
    w.food_trail.mark(19, 20, 0.0)  # (19, 20) is outward (farther from nest)
    w.tokens[(20, 20)] = ("[DEPLETED] Food exhausted", 0.0)
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (19, 20)
    assert w.food_trail.value(20, 20, 0.0) == 100.0


def test_exploring_scout_discovers_active_food_trail():
    a = agent(0, 20, 20, state="walk")
    a.role = "scout"
    a.waypoint = (10, 10)  # arbitrary sweep waypoint
    w = World(NAV, [a], nest_nav=NEST_NAV, food={(15, 20): 10})
    # Active food trail at (20, 20) and outward at (19, 20)
    w.food_trail.mark(20, 20, 0.0)
    w.food_trail.mark(19, 20, 0.0)
    a.step(w, Trail(), 0.0)
    # The scout on the food trail follows the food trail to (19, 20) instead of leaving it for (10, 10)
def test_walk_to_seek_clears_non_station_waypoint_and_preserves_station_waypoint():
    # 1. Non-station waypoint gets cleared when charge drops below LOW_CHARGE
    a = agent(0, 20, 20, state="walk", charge=LOW_CHARGE - 1)
    a.waypoint = (30, 30)  # non-station
    w = World(NAV, [a], nest_nav=NEST_NAV)
    a.step(w, Trail(), 0.0)
    assert a.state == "seek"
    assert a.waypoint is None

    # 2. Station waypoint is preserved when charge drops below LOW_CHARGE
    st = STATIONS[0]
    a2 = agent(1, 20, 20, state="walk", charge=LOW_CHARGE - 1)
    a2.waypoint = st
    w2 = World(NAV, [a2], nest_nav=NEST_NAV)
    a2.step(w2, Trail(), 0.0)
    assert a2.state == "seek"
    assert a2.waypoint == st


def test_erasing_trail_moves_inward_and_clears_tile_by_tile():
    a = agent(0, 30, 20, state="walk")
    a.erasing_trail = True
    w = World(NAV, [a], nest_nav=NEST_NAV, food={})
    w.food_trail.mark(30, 20, 0.0)
    w.food_trail.mark(31, 20, 0.0)  # (31, 20) is closer to nest (50, 50)

    a.step(w, Trail(), 0.0)
    # Agent stepped onto (31, 20) and cleared both standing tiles
    assert (a.col, a.row) == (31, 20)
    assert w.food_trail.value(30, 20, 0.0) == 0.0
    assert w.food_trail.value(31, 20, 0.0) == 0.0

    # Next step: no more active inward trail -> stops erasing and becomes scout
    a.moved = False
    a.step(w, Trail(), 0.1)
    assert a.erasing_trail is False
    assert a.role == "scout"


def test_agent_does_not_rejoin_depleted_trail_outside_food():
    # Agent at (20, 21) off the trail at (20, 20)
    a = agent(0, 20, 21, state="walk")
    a.role = "scout"
    w = World(NAV, [a], nest_nav=NEST_NAV, food={})
    w.food_trail.mark(20, 20, 0.0)
    w.tokens[(20, 20)] = ("[DEPLETED] Food exhausted", 0.0)

    a.step(w, Trail(), 0.0)
    # The agent does NOT steer into the depleted trail tile at (20, 20)
    assert (a.col, a.row) != (20, 20)


def test_agent_rejoins_active_food_trail_after_collision_detour():
    # Agent at (20, 21) off the active food trail at (20, 20)
    a = agent(0, 20, 21, state="walk")
    a.role = "forager"
    w = World(NAV, [a], nest_nav=NEST_NAV, food={(15, 20): 10})
    w.food_trail.mark(20, 20, 0.0)

    a.step(w, Trail(), 0.0)
    # The agent rejoins the active food trail at (20, 20)
    assert (a.col, a.row) == (20, 20)


def test_llm_mode_agent_at_trail_end_clears_tile_and_moves_inward():
    a = agent(0, 30, 20, state="walk")
    a.llm_mode = "global_contextual"
    w = World(NAV, [a], nest_nav=NEST_NAV, food={})
    w.food_trail.mark(30, 20, 0.0)
    w.food_trail.mark(31, 20, 0.0)  # inward closer to nest (50, 50)

    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (31, 20)
    assert w.food_trail.value(30, 20, 0.0) == 0.0
    assert w.food_trail.value(31, 20, 0.0) == 0.0
    assert a.erasing_trail is True


def test_llm_mode_agent_at_trail_end_with_no_inward_trail_clears_tile():
    a = agent(0, 30, 20, state="walk")
    a.llm_mode = "global_contextual"
    # Only standing tile has trail, no other neighbors
    w = World(NAV, [a], nest_nav=NEST_NAV, food={})
    w.food_trail.mark(30, 20, 0.0)

    a.step(w, Trail(), 0.0)
    assert w.food_trail.value(30, 20, 0.0) == 0.0
    assert a.erasing_trail is True
    assert a.role == "scout"


def test_approaching_trail_when_all_candidates_depleted_skips_steering():
    # Agent at (20, 27). Distance to (20, 20) is 7 (> 6 so in_depleted_zone is False)
    a = agent(0, 20, 27, state="walk")
    a.waypoint = None
    # Block other directions so only candidate is (20, 26)
    obstacles = {(19, 27), (21, 27), (20, 28)}
    w = World(NAV, [a], nest_nav=NEST_NAV, food={}, static_obstacles=obstacles)
    w.tokens[(20, 20)] = ("[DEPLETED] Food exhausted", 0.0)
    # Candidate (20, 26) is within 6 of (20, 20), so non_depleted_candidates is empty
    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) == (20, 26)


def test_agent_on_isolated_trail_tile_near_nest_falls_through_to_scenting():
    # Agent at (45, 50) where dist to nest is 5 (< 15).
    # Only (45, 50) is marked with food_trail; no neighbor tiles are marked.
    a = agent(0, 45, 50, state="walk")
    w = World(NAV, [a], nest_nav=NEST_NAV, food={})
    w.food_trail.mark(45, 50, 0.0)

    a.step(w, Trail(), 0.0)
    assert (a.col, a.row) != (45, 50)


def test_has_active_branch_outward_coverage():
    # 1. Empty food dict
    w = World(NAV, [], nest_nav=NEST_NAV, food={})
    assert sim.has_active_branch_outward(w, (50, 50), 0.0) is False

    # 2. Food exists but all 0 amount
    w.food = {(40, 40): 0}
    assert sim.has_active_branch_outward(w, (50, 50), 0.0) is False

    # 3. Tile is directly adjacent to active food (<= 1)
    w.food = {(40, 40): 10}
    assert sim.has_active_branch_outward(w, (40, 41), 0.0) is True

    # 4. Outward food trail leads to active food
    # Nest at (50, 50). Trail from (45, 50) to (40, 50), food at (39, 50)
    w = World(NAV, [], nest_nav=NEST_NAV, food={(39, 50): 5})
    for c in range(40, 46):
        w.food_trail.mark(c, 50, 0.0)
    assert sim.has_active_branch_outward(w, (45, 50), 0.0) is True

    # 5. Outward food trail leads only to depleted food
    w.food = {(39, 50): 0}
    assert sim.has_active_branch_outward(w, (45, 50), 0.0) is False

    # 6. Edge of grid (out-of-bounds neighbor) and visited neighbors
    w_edge = World(NAV, [], nest_nav=NEST_NAV, food={(0, 0): 5})
    w_edge.food_trail.mark(0, 1, 0.0)
    w_edge.food_trail.mark(0, 2, 0.0)
    assert sim.has_active_branch_outward(w_edge, (0, 2), 0.0) is True

    # 7. Neighbor with trail has nxt_dist < curr_dist (inward neighbor, not enqueued)
    nest = [(49, 49), (50, 49), (49, 50), (50, 50)]
    nst_nav = build_nav(nest)
    w_inward = World(nst_nav, [], nest_nav=nst_nav, food={(40, 20): 5})
    # Mark an inward trail tile at (40, 31) where dist to nest is smaller than (40, 30)
    w_inward.food_trail.mark(40, 31, 0.0)
    assert sim.has_active_branch_outward(w_inward, (40, 30), 0.0) is False


def test_long_sim_junction_and_trunk_preservation():
    nest = [(49, 49), (50, 49), (49, 50), (50, 50)]
    nst_nav = build_nav(nest)
    # Nest at (49..50, 49..50). Common trunk (41..45, 50). Junction at (40, 50).
    # Dead branch: (30..39, 50). Active food at (40, 40) (distance 10 away).
    a = Agent(0, 30, 50)
    a.role = "forager"
    a.llm_mode = "global_contextual"
    w = World(nst_nav, [a], nest_nav=nst_nav, nest_set=set(nest), food={(40, 40): 10})
    for c in range(30, 46):
        w.food_trail.mark(c, 50, 0.0)
    for r in range(40, 51):
        w.food_trail.mark(40, r, 0.0)
    trail = Trail()

    for i in range(15):
        w.tick([a], trail, 0.1 * i)

    # Dead branch is erased:
    dead_branch = [c for c in range(30, 40) if w.food_trail.value(c, 50, 2.0) >= 0.5]
    assert len(dead_branch) == 0

    # Common trunk is preserved:
    trunk = [c for c in range(41, 46) if w.food_trail.value(c, 50, 2.0) >= 0.5]
    assert len(trunk) == 5

    # Active branch is preserved:
    active = [r for r in range(40, 50) if w.food_trail.value(40, r, 2.0) >= 0.5]
    assert len(active) == 10

    # Junction is preserved:
    assert w.food_trail.value(40, 50, 2.0) >= 0.5


def test_standing_on_junction_erasing_trail_stops_immediately():
    nest = [(49, 49), (50, 49), (49, 50), (50, 50)]
    nst_nav = build_nav(nest)
    a = Agent(0, 40, 50)
    a.role = "forager"
    a.erasing_trail = True
    w = World(nst_nav, [a], nest_nav=nst_nav, nest_set=set(nest), food={(40, 40): 10})
    for r in range(40, 51):
        w.food_trail.mark(40, r, 0.0)
    a.step(w, Trail(), 0.0)

    # Junction tile was NOT cleared
    assert w.food_trail.value(40, 50, 0.0) > 0.0
    assert a.erasing_trail is False
    assert a.role == "scout"


def test_3x3_grid_deadlock_resolution():
    # 3 agents in row: A1 at (0,0), A2 at (1,0), A3 at (2,0)
    # A1 wants East (1,0), A2 wants East (2,0), A3 wants West (1,0)
    # A2 cannot be pushed because A3 is at (2,0).
    # Space at (2,1) is free.
    a1 = Agent(1, 0, 0)
    a1.state = "return_nest"
    a2 = Agent(2, 1, 0)
    a2.state = "return_nest"
    a3 = Agent(3, 2, 0)
    a3.state = "return_nest"

    arrow = [[(0, 0) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    dist = [[50 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    arrow[0][0] = (1, 0)
    arrow[0][1] = (1, 0)
    arrow[0][2] = (-1, 0)

    nav = sim.NavMap(offset=arrow, arrow=arrow, dist=dist)
    obs = {(3, 0), (-1, 0), (0, -1), (1, -1), (2, -1), (1, 1), (0, 1)}
    w = World(nav, [a1, a2, a3], nest_nav=nav, static_obstacles=obs)
    trail = Trail()

    w.tick([a1, a2, a3], trail, 0.0)

    # A3 detours to (2, 1), freeing (2, 0).
    # A2 advances to (2, 0).
    # A1 advances to (1, 0).
    assert (a3.col, a3.row) == (2, 1)
    assert (a2.col, a2.row) == (2, 0)
    assert (a1.col, a1.row) == (1, 0)


def test_llm_mode_pushable_by_released_occupant():
    a = agent(1, 10, 10, state="seek", charge=10000)
    a.llm_mode = "global_contextual"
    claim = (5, 0, 99)
    assert a.pushable(NAV, claim) is True
    a.state = "charge"
    assert a.pushable(NAV, claim) is False


def test_llm_seeker_yields_doorway_to_released_station_occupant():
    st = next(iter(STATIONS))
    occupant = agent(1, *st, state="walk")
    occupant.released = True

    # Seeker is adjacent to station at st[0], st[1] + 1
    seeker = agent(2, st[0], st[1] + 1, state="seek", charge=10000)
    seeker.waypoint = st
    seeker.llm_mode = "global_contextual"

    w = World(NAV, [occupant, seeker], stations=STATIONS)
    trail = Trail()

    seeker.step(w, trail, 0.0)
    assert (seeker.col, seeker.row) != (st[0], st[1] + 1)
    assert (seeker.col, seeker.row) != st


def test_llm_seeker_waits_if_no_side_options_at_station_doorway():
    st = next(iter(STATIONS))
    occupant = agent(1, *st, state="walk")
    occupant.released = True

    seeker = agent(2, st[0], st[1] + 1, state="seek", charge=10000)
    seeker.waypoint = st
    seeker.llm_mode = "global_contextual"

    # Block all other sides of (st[0], st[1] + 1)
    obs = {(st[0] - 1, st[1] + 1), (st[0] + 1, st[1] + 1), (st[0], st[1] + 2)}
    w = World(NAV, [occupant, seeker], stations=STATIONS, static_obstacles=obs)
    trail = Trail()

    seeker.step(w, trail, 0.0)
    assert (seeker.col, seeker.row) == (st[0], st[1] + 1)


def test_llm_seeker_detours_at_target_dist_greater_than_1():
    st = next(iter(STATIONS))
    seeker = agent(1, st[0], st[1] + 2, state="seek", charge=10000)
    seeker.waypoint = st
    seeker.llm_mode = "global_contextual"

    # Obstacle ahead: seeker detours around it from free side
    obs = {(st[0], st[1] + 1)}
    w = World(NAV, [seeker], stations=STATIONS, static_obstacles=obs)
    trail = Trail()

    seeker.step(w, trail, 0.0)
    assert (seeker.col, seeker.row) in {(st[0] - 1, st[1] + 2), (st[0] + 1, st[1] + 2)}


def test_llm_mode_steers_to_nearby_active_food_during_trail_erasing():
    a = agent(1, 20, 20, state="walk")
    a.erasing_trail = True
    a.llm_mode = "global_contextual"

    food_trail = Trail()
    food_trail.mark(20, 20, 0.0)
    food_trail.mark(19, 20, 0.0)

    w = World(NAV, [a], food_trail=food_trail, food={(22, 20): 10}, nest_nav=NEST_NAV)
    trail = Trail()

    a.step(w, trail, 0.0)
    assert a.role == "forager"
    assert a.erasing_trail is False
    assert (a.col, a.row) == (21, 20)


def test_llm_mode_steers_to_nearby_active_food_at_end_of_trail():
    a = agent(1, 20, 20, state="walk")
    a.role = "forager"
    a.llm_mode = "global_contextual"

    food_trail = Trail()
    food_trail.mark(20, 20, 0.0)
    # (21, 20) is inward toward nest at (49, 49)
    food_trail.mark(21, 20, 0.0)

    # Active food at (18, 20) within 6x6 neighborhood
    w = World(NAV, [a], food_trail=food_trail, food={(18, 20): 10}, nest_nav=NEST_NAV)
    trail = Trail()

    a.step(w, trail, 0.0)
    assert a.role == "forager"
    # Steers toward food at (18, 20), stepping to (19, 20) instead of inward trail to (21, 20)
    assert (a.col, a.row) == (19, 20)


def test_llm_mode_no_nearby_active_food_at_end_of_trail_erases():
    a = agent(1, 20, 20, state="walk")
    a.role = "forager"
    a.llm_mode = "global_contextual"

    food_trail = Trail()
    food_trail.mark(20, 20, 0.0)
    # (21, 20) is inward toward nest at (49, 49)
    food_trail.mark(21, 20, 0.0)

    w = World(NAV, [a], food_trail=food_trail, food={}, nest_nav=NEST_NAV)
    trail = Trail()

    a.step(w, trail, 0.0)
    assert a.erasing_trail is True
    assert (a.col, a.row) == (21, 20)
    assert w.food_trail.value(20, 20, 0.0) == 0.0


def test_llm_mode_distributes_at_trail_fork():
    food_trail = Trail()
    food_trail.mark(30, 30, 0.0)
    for c in range(20, 31):
        food_trail.mark(c, 30, 0.0)
    for r in range(20, 30):
        food_trail.mark(30, r, 0.0)

    foods = {(19, 30): 10, (30, 19): 10}
    w0 = World(NAV, [], food_trail=food_trail, food=foods, nest_nav=NEST_NAV)
    a0 = agent(0, 30, 30, state="walk")
    a0.role = "forager"
    a0.llm_mode = "global_contextual"
    w0.occ[(30, 30)] = a0
    a0.step(w0, Trail(), 0.0)

    w1 = World(NAV, [], food_trail=food_trail, food=foods, nest_nav=NEST_NAV)
    a1 = agent(1, 30, 30, state="walk")
    a1.role = "forager"
    a1.llm_mode = "global_contextual"
    w1.occ[(30, 30)] = a1
    a1.step(w1, Trail(), 0.0)

    assert (a0.col, a0.row) in {(29, 30), (30, 29)}
    assert (a1.col, a1.row) in {(29, 30), (30, 29)}
    assert (a0.col, a0.row) != (a1.col, a1.row)


def test_llm_mode_distributes_when_approaching_trail_from_outside():
    food_trail = Trail()
    for c in range(20, 30):
        food_trail.mark(c, 30, 0.0)
    for r in range(20, 30):
        food_trail.mark(30, r, 0.0)

    foods = {(19, 30): 10, (30, 19): 10}
    w0 = World(NAV, [], food_trail=food_trail, food=foods, nest_nav=NEST_NAV)
    a0 = agent(0, 30, 30, state="walk")
    a0.role = "forager"
    a0.llm_mode = "global_contextual"
    w0.occ[(30, 30)] = a0
    a0.step(w0, Trail(), 0.0)

    w1 = World(NAV, [], food_trail=food_trail, food=foods, nest_nav=NEST_NAV)
    a1 = agent(1, 30, 30, state="walk")
    a1.role = "forager"
    a1.llm_mode = "global_contextual"
    w1.occ[(30, 30)] = a1
    a1.step(w1, Trail(), 0.0)

    assert (a0.col, a0.row) in {(29, 30), (30, 29)}
    assert (a1.col, a1.row) in {(29, 30), (30, 29)}
    assert (a0.col, a0.row) != (a1.col, a1.row)


def test_llm_mode_already_erasing_skips_becoming_second_eraser():
    a0 = agent(0, 30, 20, state="walk")
    a0.erasing_trail = True
    a1 = agent(1, 30, 21, state="walk")
    a1.llm_mode = "global_contextual"
    w = World(NAV, [a0, a1], nest_nav=NEST_NAV, food={})
    w.food_trail.mark(30, 21, 0.0)

    a1.step(w, Trail(), 0.0)
    assert a1.erasing_trail is False
    assert a1.role == "forager"
    assert a1.waypoint is None


def test_llm_mode_empty_outward_trail_near_nest_switches_to_forager():
    a = agent(0, 48, 48, state="walk")
    a.llm_mode = "global_contextual"
    # Near nest (here_dist < 15), on a trail tile with an empty neighbor
    w = World(NAV, [a], nest_nav=NEST_NAV, food={})
    w.food_trail.mark(48, 48, 0.0)
    w.food_trail.mark(47, 48, 0.0)

    a.step(w, Trail(), 0.0)
    assert a.role == "forager"
    assert a.waypoint is None


def test_llm_mode_erasing_trail_stops_at_active_node_and_takes_active_branch():
    a = agent(0, 30, 30, state="walk")
    a.llm_mode = "global_contextual"
    a.erasing_trail = True
    # Trail branch from (29, 30) leads to food at (19, 30) (distance 11 > 6)
    w = World(NAV, [a], nest_nav=NEST_NAV, food={(19, 30): 10})
    w.food_trail.mark(30, 30, 0.0)
    for c in range(20, 30):
        w.food_trail.mark(c, 30, 0.0)

    a.step(w, Trail(), 0.0)
    assert a.erasing_trail is False
    assert a.role == "forager"
    assert (a.col, a.row) == (29, 30)


def test_llm_mode_erasing_trail_stops_near_nest_without_active_branch():
    a = agent(0, 49, 49, state="walk")
    a.llm_mode = "global_contextual"
    a.erasing_trail = True
    # Close to nest (here_dist <= 6) with no active outward branches
    w = World(NAV, [a], nest_nav=NEST_NAV, food={})
    w.food_trail.mark(49, 49, 0.0)

    a.step(w, Trail(), 0.0)
    assert a.erasing_trail is False
    assert a.role == "forager"
    assert a.waypoint is None


def test_llm_seeker_waits_in_queue_when_queue_not_long():
    st = next(iter(STATIONS))
    seeker = agent(1, st[0], st[1] + 2, state="seek", charge=10000)
    seeker.waypoint = st
    seeker.llm_mode = "global_contextual"

    blocker = agent(2, st[0], st[1] + 1, state="seek", charge=10000)
    w = World(NAV, [seeker, blocker], stations=STATIONS)
    seeker.step(w, Trail(), 0.0)
    # Seeker waits in queue at 0 battery drain rather than oscillating sideways
    assert (seeker.col, seeker.row) == (st[0], st[1] + 2)


def test_llm_seeker_diverts_to_alternative_station_when_queue_long():
    st1 = (10, 0)
    st2 = (20, 0)
    stations = [st1, st2]
    nav = build_nav(stations)

    # Station 1 has occupant and an agent in front: queue load = 2
    occ = agent(1, *st1, state="charge")
    blocker = agent(2, st1[0], st1[1] + 1, state="seek", charge=10000)
    blocker.waypoint = st1

    # Seeker is behind blocker at dist 2, with waypoint = st1
    seeker = agent(3, st1[0], st1[1] + 2, state="seek", charge=50000)
    seeker.waypoint = st1
    seeker.llm_mode = "global_contextual"

    w = World(nav, [occ, blocker, seeker], stations=stations)
    seeker.step(w, Trail(), 0.0)
    assert seeker.waypoint == st2


def test_llm_seeker_waits_in_long_queue_when_no_alternative_reachable():
    st1 = (10, 0)
    st2 = (90, 90)
    stations = [st1, st2]
    nav = build_nav(stations)

    occ = agent(1, *st1, state="charge")
    blocker = agent(2, st1[0], st1[1] + 1, state="seek", charge=10000)
    blocker.waypoint = st1

    # Seeker has low charge, cannot afford to detour to distant st2
    seeker = agent(3, st1[0], st1[1] + 2, state="seek", charge=500)
    seeker.waypoint = st1
    seeker.llm_mode = "global_contextual"

    w = World(nav, [occ, blocker, seeker], stations=stations)
    seeker.step(w, Trail(), 0.0)
    assert (seeker.col, seeker.row) == (st1[0], st1[1] + 2)
    assert seeker.waypoint == st1


def test_llm_seeker_detours_around_passerby():
    st = next(iter(STATIONS))
    seeker = agent(1, st[0], st[1] + 2, state="seek", charge=10000)
    seeker.waypoint = st
    seeker.llm_mode = "global_contextual"

    passerby = agent(2, st[0], st[1] + 1, state="return_nest", charge=10000)
    w = World(NAV, [seeker, passerby], stations=STATIONS)
    seeker.step(w, Trail(), 0.0)
    assert (seeker.col, seeker.row) in {(st[0] - 1, st[1] + 2), (st[0] + 1, st[1] + 2)}


def test_seeker_waypoint_in_obstacles_cleared():
    seeker = agent(1, 20, 20, state="seek", charge=10000)
    seeker.waypoint = (20, 21)
    w = World(NAV, [seeker], static_obstacles={(20, 21)})
    seeker.step(w, Trail(), 0.0)
    assert seeker.waypoint is None


def test_llm_seeker_target_dist_2_waits_when_occupant_released():
    st = next(iter(STATIONS))
    occupant = agent(1, *st, state="walk")
    occupant.released = True

    # Seeker is at (st[0] + 1, st[1] + 1), dist 2 from st
    # Doorway tile (st[0], st[1] + 1) is free (dist 1)
    seeker = agent(2, st[0] + 1, st[1] + 1, state="seek", charge=10000)
    seeker.waypoint = st
    seeker.llm_mode = "global_contextual"

    w = World(NAV, [occupant, seeker], stations=STATIONS)
    seeker.step(w, Trail(), 0.0)
    # Does not jump into doorway to prevent 2-block ping-pong oscillation
    assert (seeker.col, seeker.row) == (st[0] + 1, st[1] + 1)


def test_classical_seeker_waits_behind_agent_in_queue():
    st = next(iter(STATIONS))
    seeker = agent(1, st[0], st[1] + 2, state="seek", charge=10000)
    seeker.waypoint = st
    seeker.llm_mode = "classical"

    blocker = agent(2, st[0], st[1] + 1, state="seek", charge=10000)
    w = World(NAV, [seeker, blocker], stations=STATIONS)
    seeker.step(w, Trail(), 0.0)
    assert (seeker.col, seeker.row) == (st[0], st[1] + 2)


def test_walk_options_fallback_in_llm_mode_when_all_pressured():
    a = agent(0, *MID)
    a.llm_mode = "global_contextual"
    w = world(a)
    for dc, dr in DIRECTIONS:
        w.pressure[50 + dr][50 + dc] = PRESSURE_LIMIT
    a.step(w, Trail(), 0.0)
    # LLM agent does not freeze in place; moves to an available free tile
    assert (a.col, a.row) != MID


def test_spawn_random_food_unseeded_generates_clusters():
    w = World(NAV, [])
    w.spawn_random_food(seed=None)
    assert len(w.food) > 0


def test_spawn_synchronized_food_unseeded():
    from apeks.llm import ParallelSwarmBench
    bench = ParallelSwarmBench()
    init_food = len(bench.instances[0].world.food)
    bench.spawn_synchronized_food(seed=None)
    assert len(bench.instances[0].world.food) > init_food


def test_llm_seeker_queue_long_alternative_station_also_crowded():
    st1 = (10, 0)
    st2 = (20, 0)
    stations = [st1, st2]
    nav = build_nav(stations)

    occ1 = agent(1, *st1, state="charge")
    blocker1 = agent(2, st1[0], st1[1] + 1, state="seek", charge=10000)
    blocker1.waypoint = st1

    occ2 = agent(4, *st2, state="charge")
    blocker2 = agent(5, st2[0], st2[1] + 1, state="seek", charge=10000)
    blocker2.waypoint = st2
    blocker3 = agent(6, st2[0], st2[1] + 2, state="seek", charge=10000)
    blocker3.waypoint = st2

    seeker = agent(3, st1[0], st1[1] + 2, state="seek", charge=50000)
    seeker.waypoint = st1
    seeker.llm_mode = "global_contextual"

    w = World(nav, [occ1, blocker1, seeker, occ2, blocker2, blocker3], stations=stations)
    seeker.step(w, Trail(), 0.0)
    assert seeker.waypoint == st1
    assert (seeker.col, seeker.row) == (st1[0], st1[1] + 2)


def test_classical_seeker_blocked_by_walking_agent_waits():
    st = next(iter(STATIONS))
    seeker = agent(1, st[0], st[1] + 2, state="seek", charge=10000)
    seeker.waypoint = st
    seeker.llm_mode = "classical"

    walker = agent(2, st[0], st[1] + 1, state="walk", charge=10000)
    w = World(NAV, [seeker, walker], stations=STATIONS)
    seeker.step(w, Trail(), 0.0)
    assert (seeker.col, seeker.row) == (st[0], st[1] + 2)






