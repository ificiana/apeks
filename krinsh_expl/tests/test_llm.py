from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from apeks.sim import (
    Agent,
    MovingBarrier,
    NavMap,
    Trail,
    World,
    build_nav,
)

from apeks.config import GRID_SIZE, MAX_CHARGE, NEST_TILES
from apeks.llm import (
    AgentObservation,
    LLMBackend,
    LLMDecision,
    MockLLMBackend,
    OllamaBackend,
    ParallelSwarmBench,
    SwarmOrchestrator,
    extract_observation,
    extract_station_signpost,
    extract_station_signposts,
    select_best_station,
    synthesize_token_intel,
    to_narrative_briefing,
    to_prompt,
)


def make_dummy_nav() -> NavMap:
    dist = [[10 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    arrow = [[(0, 0) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    offset = [[(0, 0) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    return NavMap(dist, arrow, offset)


def make_world(agents: list[Agent] | None = None) -> World:
    nav = make_dummy_nav()
    return World(nav=nav, agents=agents or [])


# ---------------------------------------------------------------------------
# Test AgentObservation, extract_observation, to_prompt
# ---------------------------------------------------------------------------


def test_extract_observation_normal_and_boundary():
    # Agent at (10, 10) with food, token, obstacle inside and outside radius
    a = Agent(ident=1, col=10, row=10, charge=30000.0)
    a.state = "walk"
    a.role = "scout"
    a.waypoint = (12, 12)

    ally = Agent(ident=5, col=11, row=11, charge=25000.0)
    ally.state = "walk"
    ally.role = "forager"
    ally.has_food = True

    w = make_world([a, ally])
    w.food = {(11, 10): 5, (50, 50): 10}  # (11, 10) inside, (50, 50) outside
    w.write_token((10, 12), "HAZARD", 0.0)
    w.write_token((50, 50), "DISTANT", 0.0)
    w.static_obstacles = {(10, 9), (45, 45)}  # (10, 9) inside, (45, 45) outside
    w.stations = [(11, 11), (50, 50)]  # (11, 11) inside, (50, 50) outside

    trail = Trail()
    trail.mark(10, 9, 0.0)

    obs = extract_observation(a, w, trail, now=1.0, radius=6)
    assert obs.agent_id == 1
    assert obs.pos == (10, 10)
    assert obs.role == "scout"
    assert obs.has_food is False
    assert ((11, 10), 5) in obs.nearby_food
    assert len(obs.nearby_food) == 1
    assert ((10, 12), "HAZARD") in obs.nearby_tokens
    assert len(obs.nearby_tokens) == 1
    assert (10, 9) in obs.nearby_obstacles
    assert len(obs.nearby_obstacles) == 1
    assert (11, 11) in obs.nearby_stations
    assert len(obs.nearby_stations) == 1
    assert obs.waypoint == (12, 12)
    assert "N" in obs.scent_directions
    assert len(obs.nearby_allies) == 1
    assert obs.nearby_allies[0] == (5, (11, 11), "walk", "forager", True)

    briefing_with_notes = to_narrative_briefing(obs, mode="global_contextual")
    assert "AGENT #001 TACTICAL BRIEFING" in briefing_with_notes
    assert "[HAZARDS]: HAZARD (read at (10, 12))" in briefing_with_notes
    assert "Stations: 1 in line of sight" in briefing_with_notes

    # Multi-line note with blank lines to test subline.strip()
    obs_blank = AgentObservation(
        agent_id=9,
        pos=(10, 10),
        charge=30000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((10, 12), "\n[1.0s] HAZARD\n\n")],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[],
    )
    briefing_blank = to_narrative_briefing(obs_blank, mode="global_contextual")
    assert "[HAZARDS]: HAZARD (read at (10, 12))" in briefing_blank

    # Agent at border (0, 0) -> tests deltas outside grid returning 100.0
    a_border = Agent(ident=2, col=0, row=0, charge=30000.0)
    a_border.state = "walk"
    obs_border = extract_observation(a_border, w, trail, now=1.0)
    assert obs_border.scent_directions["N"] == 100.0
    assert obs_border.scent_directions["W"] == 100.0
    assert obs_border.role == "scout"
    assert obs_border.waypoint is None
    briefing_no_notes = to_narrative_briefing(obs_border, mode="classical")
    assert "(No text notes discovered in local environment)" in briefing_no_notes


def test_to_prompt_json_formatting():
    obs = AgentObservation(
        agent_id=7,
        pos=(15, 20),
        charge=30000.0,
        state="walk",
        role="forager",
        has_food=True,
        nearby_food=[((16, 20), 3)],
        nearby_tokens=[((15, 22), "WARNING")],
        nearby_obstacles=[(14, 20)],
        scent_directions={"N": 1.2, "S": 3.4, "E": 0.5, "W": 100.0},
        waypoint=(16, 20),
        nearby_allies=[(3, (16, 21), "walk", "scout", False)],
    )
    prompt_str = to_prompt(obs, mode="coarse_fine")
    data = json.loads(prompt_str)
    assert data["mode"] == "coarse_fine"
    assert "briefing" in data
    assert data["agent"]["id"] == 7
    assert data["agent"]["pos"] == [15, 20]
    assert data["agent"]["role"] == "forager"
    assert data["agent"]["has_food"] is True
    assert data["environment"]["nearby_food"] == [[[16, 20], 3]]
    assert len(data["allies"]) == 1


def test_llm_backend_base_class():
    backend = LLMBackend()
    obs = AgentObservation(
        agent_id=1,
        pos=(0, 0),
        charge=1000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
    )
    with pytest.raises(NotImplementedError):
        backend.decide(obs, "classical")


# ---------------------------------------------------------------------------
# Test MockLLMBackend
# ---------------------------------------------------------------------------


def test_mock_backend_carrying_food():
    backend = MockLLMBackend()
    obs = AgentObservation(
        agent_id=3,
        pos=(25, 25),
        charge=20000.0,
        state="return_nest",
        role="scout",
        has_food=True,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
    )
    dec = backend.decide(obs, mode="classical")
    assert dec.action_type == "RETURN_NEST"
    assert dec.waypoint == NEST_TILES[0]
    assert dec.role == "forager"


def test_mock_backend_evacuate_tokens():
    backend = MockLLMBackend()

    # Case 1: dx >= 0, dy >= 0 (alert token to top-left of agent)
    obs1 = AgentObservation(
        agent_id=1,
        pos=(20, 20),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((15, 15), "ALERT: Hazard spotted")],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
    )
    dec1 = backend.decide(obs1, mode="global_contextual")
    assert dec1.action_type == "EVACUATE"
    assert dec1.waypoint == (28, 28)

    # Case 2: dx < 0, dy < 0 (hazard token to bottom-right of agent)
    obs2 = AgentObservation(
        agent_id=2,
        pos=(20, 20),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((25, 25), "HAZARD: Deep pit")],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
    )
    dec2 = backend.decide(obs2, mode="global_contextual")
    assert dec2.action_type == "EVACUATE"
    assert dec2.waypoint == (12, 12)

    # Case 3: Food broadcast note convergence
    obs3 = AgentObservation(
        agent_id=3,
        pos=(20, 20),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((22, 22), "[FOOD] Rich patch at (22, 22)")],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
    )
    dec3 = backend.decide(obs3, mode="global_contextual")
    assert dec3.action_type == "HARVEST_CALL"
    assert dec3.waypoint == (22, 22)
    assert dec3.role == "forager"

    # Case 4: Sentry ignores food broadcast note and continues duty
    obs4 = AgentObservation(
        agent_id=4,
        pos=(20, 20),
        charge=20000.0,
        state="walk",
        role="sentry",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((22, 22), "[FOOD] Rich patch at (22, 22)")],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
    )
    dec4 = backend.decide(obs4, mode="global_contextual")
    assert dec4.action_type == "ROLE_SWITCH"
    assert dec4.role == "scout"


def test_mock_backend_sentry_marks_barrier():
    backend = MockLLMBackend()
    obs = AgentObservation(
        agent_id=4,
        pos=(18, 18),
        charge=20000.0,
        state="walk",
        role="sentry",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[(19, 18)],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
    )
    dec = backend.decide(obs, mode="global_contextual")
    assert dec.action_type == "STIGMERGY_MARK"
    assert dec.token_to_write == ((18, 18), "[ALERT] Barrier sector at (19, 18)")
    assert dec.role == "sentry"


def test_mock_backend_coarse_fine_food_and_coldest_scent():
    backend = MockLLMBackend()

    # Food detected
    obs_food = AgentObservation(
        agent_id=5,
        pos=(30, 30),
        charge=50000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[((35, 32), 4)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
    )
    dec_food = backend.decide(obs_food, mode="coarse_fine")
    assert dec_food.action_type == "COARSE_WAYPOINT"
    assert dec_food.waypoint == (35, 32)
    assert dec_food.role == "forager"

    # Obstacle flanking macro-planning (obs.pos < ox, oy)
    obs_flank1 = AgentObservation(
        agent_id=7,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[(22, 22)],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
    )
    dec_flank1 = backend.decide(obs_flank1, mode="coarse_fine")
    assert dec_flank1.action_type == "COARSE_WAYPOINT"
    assert dec_flank1.waypoint == (10, 10)

    # Obstacle flanking macro-planning (obs.pos >= ox, oy)
    obs_flank2 = AgentObservation(
        agent_id=8,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[(18, 18)],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
    )
    dec_flank2 = backend.decide(obs_flank2, mode="coarse_fine")
    assert dec_flank2.action_type == "COARSE_WAYPOINT"
    assert dec_flank2.waypoint == (30, 30)

    # Coldest scent exploration
    obs_scent = AgentObservation(
        agent_id=6,
        pos=(30, 30),
        charge=50000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 10.0, "S": 2.0, "E": 5.0, "W": 8.0},
        waypoint=None,
    )
    dec_scent = backend.decide(obs_scent, mode="coarse_fine")
    assert dec_scent.action_type == "COARSE_WAYPOINT"
    assert dec_scent.waypoint == (30, 42)  # S delta is (0, 12)
    assert dec_scent.role == "scout"


def test_mock_backend_global_contextual_roles():
    backend = MockLLMBackend()

    # Scout discovers rich food cluster (amt >= 2) -> broadcasts directional guide
    # East
    obs_rich_e = AgentObservation(
        agent_id=1,
        pos=(10, 10),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[((12, 10), 3)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
    )
    dec_rich_e = backend.decide(obs_rich_e, mode="global_contextual")
    assert dec_rich_e.action_type == "STIGMERGY_MARK"
    assert dec_rich_e.token_to_write == (
        (10, 10),
        "[GUIDE] Food ~2 steps East at (12, 10) [3u]",
    )
    assert dec_rich_e.role == "forager"

    # West & North-West & South & Here directions
    obs_w = AgentObservation(
        agent_id=1,
        pos=(10, 10),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[((8, 10), 3)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
    )
    assert (
        backend.decide(obs_w, mode="global_contextual").token_to_write[1]
        == "[GUIDE] Food ~2 steps West at (8, 10) [3u]"
    )

    obs_s = AgentObservation(
        agent_id=1,
        pos=(10, 10),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[((10, 14), 3)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
    )
    assert (
        backend.decide(obs_s, mode="global_contextual").token_to_write[1]
        == "[GUIDE] Food ~4 steps South at (10, 14) [3u]"
    )

    obs_nw = AgentObservation(
        agent_id=1,
        pos=(10, 10),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[((8, 8), 3)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
    )
    assert (
        backend.decide(obs_nw, mode="global_contextual").token_to_write[1]
        == "[GUIDE] Food ~4 steps North-West at (8, 8) [3u]"
    )

    obs_here = AgentObservation(
        agent_id=1,
        pos=(10, 10),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[((10, 10), 3)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
    )
    assert (
        backend.decide(obs_here, mode="global_contextual").token_to_write[1]
        == "[GUIDE] Food ~0 steps Here at (10, 10) [3u]"
    )

    # Obstacles nearby -> sentry role switch
    obs_obs = AgentObservation(
        agent_id=2,
        pos=(10, 10),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[(11, 10)],
        scent_directions={},
        waypoint=None,
    )
    dec_obs = backend.decide(obs_obs, mode="global_contextual")
    assert dec_obs.action_type == "ROLE_SWITCH"
    assert dec_obs.role == "sentry"

    # Modest food nearby (amt < 2) -> forager role switch
    obs_food = AgentObservation(
        agent_id=3,
        pos=(10, 10),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[((12, 10), 1)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
    )
    dec_food = backend.decide(obs_food, mode="global_contextual")
    assert dec_food.action_type == "ROLE_SWITCH"
    assert dec_food.role == "forager"

    # Open field -> scout role switch
    obs_open = AgentObservation(
        agent_id=4,
        pos=(10, 10),
        charge=20000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
    )
    dec_open = backend.decide(obs_open, mode="global_contextual")
    assert dec_open.action_type == "ROLE_SWITCH"
    assert dec_open.role == "scout"


def test_mock_backend_global_contextual_depleted_and_abort():
    backend = MockLLMBackend()

    # 1. Reaching waypoint where food is depleted
    obs_dep = AgentObservation(
        agent_id=1,
        pos=(25, 25),
        charge=20000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=(25, 25),
    )
    dec_dep = backend.decide(obs_dep, mode="global_contextual")
    assert dec_dep.action_type == "DEPLETED_MARK"
    assert dec_dep.role == "scout"
    assert dec_dep.waypoint is None
    assert dec_dep.token_to_write == ((25, 25), "[DEPLETED] Food at (25, 25) exhausted")

    # 1b. Reaching waypoint where food is still available
    obs_present = AgentObservation(
        agent_id=1,
        pos=(25, 25),
        charge=20000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[((25, 25), 4)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=(25, 25),
    )
    dec_present = backend.decide(obs_present, mode="global_contextual")
    assert dec_present.action_type != "DEPLETED_MARK"

    # 2. Reading [DEPLETED] note for current waypoint -> abort route
    obs_abort = AgentObservation(
        agent_id=2,
        pos=(20, 20),
        charge=20000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((21, 20), "[DEPLETED] Food at (30, 30) exhausted")],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=(30, 30),
    )
    dec_abort = backend.decide(obs_abort, mode="global_contextual")
    assert dec_abort.action_type == "ABORT_DEPLETED"
    assert dec_abort.waypoint is None
    assert dec_abort.role == "scout"

    # 2b. Reading [DEPLETED] note for a DIFFERENT waypoint -> do not abort
    obs_diff = AgentObservation(
        agent_id=3,
        pos=(20, 20),
        charge=20000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((21, 20), "[DEPLETED] Food at (40, 40) exhausted")],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=(30, 30),
    )
    dec_diff = backend.decide(obs_diff, mode="global_contextual")
    assert dec_diff.action_type != "ABORT_DEPLETED"

    # 2c. Reading [DEPLETED] note without coordinates -> does not abort
    obs_no_coord = AgentObservation(
        agent_id=4,
        pos=(20, 20),
        charge=20000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((21, 20), "[DEPLETED] Sector exhausted")],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=(30, 30),
    )
    dec_no_coord = backend.decide(obs_no_coord, mode="global_contextual")
    assert dec_no_coord.action_type != "ABORT_DEPLETED"


def test_mock_backend_pure_llm():
    backend = MockLLMBackend()

    # 1. Low battery triage (< 35% MAX_CHARGE)
    # 1a. Visible station detected in local field
    obs_low1 = AgentObservation(
        agent_id=1,
        pos=(15, 15),
        charge=MAX_CHARGE * 0.20,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
        nearby_stations=[(12, 12)],
    )
    dec_low1 = backend.decide(obs_low1, mode="pure_llm")
    assert dec_low1.action_type == "TACTICAL_CHARGE"
    assert dec_low1.waypoint == (12, 12)

    # 1b. Station signpost token read in environment
    obs_low2 = AgentObservation(
        agent_id=2,
        pos=(20, 20),
        charge=MAX_CHARGE * 0.20,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((18, 18), "[STATION] Charging depot at (10, 10)")],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
        nearby_stations=[],
    )
    dec_low2 = backend.decide(obs_low2, mode="pure_llm")
    assert dec_low2.action_type == "TACTICAL_CHARGE"
    assert dec_low2.waypoint == (10, 10)

    # 1c. No station visible and no signpost -> Tactical search sweep
    # (no global cheating)
    obs_low3 = AgentObservation(
        agent_id=3,
        pos=(25, 25),
        charge=MAX_CHARGE * 0.20,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
        nearby_stations=[],
    )
    dec_low3 = backend.decide(obs_low3, mode="pure_llm")
    assert dec_low3.action_type == "TACTICAL_SEARCH"
    assert dec_low3.waypoint is None

    # 2. Food target lock
    obs_food = AgentObservation(
        agent_id=3,
        pos=(30, 30),
        charge=MAX_CHARGE * 0.9,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[((35, 32), 3)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
    )
    dec_food = backend.decide(obs_food, mode="pure_llm")
    assert dec_food.action_type == "TACTICAL_MOVE"
    assert dec_food.waypoint == (35, 32)
    assert dec_food.role == "forager"

    # 3. Obstacle evasion
    # 3a. pos < ox and pos < oy -> -2, -2
    obs_evade1 = AgentObservation(
        agent_id=4,
        pos=(20, 20),
        charge=MAX_CHARGE * 0.9,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[(21, 21)],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
    )
    dec_evade1 = backend.decide(obs_evade1, mode="pure_llm")
    assert dec_evade1.action_type == "TACTICAL_EVADE"
    assert dec_evade1.waypoint == (18, 18)

    # 3b. pos >= ox and pos >= oy -> +2, +2
    obs_evade2 = AgentObservation(
        agent_id=5,
        pos=(20, 20),
        charge=MAX_CHARGE * 0.9,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[(19, 19)],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
    )
    dec_evade2 = backend.decide(obs_evade2, mode="pure_llm")
    assert dec_evade2.action_type == "TACTICAL_EVADE"
    assert dec_evade2.waypoint == (22, 22)

    # 4. Search sweep in coldest scent direction
    obs_sweep = AgentObservation(
        agent_id=6,
        pos=(40, 40),
        charge=MAX_CHARGE * 0.9,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 10.0, "S": 1.0, "E": 5.0, "W": 8.0},
        waypoint=None,
    )
    dec_sweep = backend.decide(obs_sweep, mode="pure_llm")
    assert dec_sweep.action_type == "TACTICAL_SWEEP"
    assert dec_sweep.waypoint == (40, 48)  # S is (0, 8)


def test_mock_backend_default_maintain():
    backend = MockLLMBackend()
    obs = AgentObservation(
        agent_id=9,
        pos=(20, 20),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
    )
    dec = backend.decide(obs, mode="classical")
    assert dec.action_type == "MAINTAIN"
    assert dec.waypoint is None


# ---------------------------------------------------------------------------
# Test OllamaBackend
# ---------------------------------------------------------------------------


def test_ollama_backend_success_with_and_without_waypoint():
    fallback = MockLLMBackend()
    ollama = OllamaBackend(
        model="qwen2.5:7b",
        url="http://localhost:11434/",
        timeout=2.0,
        fallback=fallback,
    )
    assert ollama.url == "http://localhost:11434"
    assert ollama.fallback is fallback

    obs = AgentObservation(
        agent_id=1,
        pos=(10, 10),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
    )

    # Success with waypoint
    resp_json = json.dumps(
        {
            "response": json.dumps(
                {
                    "thought": "Directing to quadrant",
                    "waypoint": [25, 30],
                    "role": "scout",
                    "action_type": "COARSE_WAYPOINT",
                }
            )
        }
    ).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = resp_json
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        dec = ollama.decide(obs, "coarse_fine")
        assert dec.thought == "Directing to quadrant"
        assert dec.waypoint == (25, 30)
        assert dec.role == "scout"
        assert dec.action_type == "COARSE_WAYPOINT"

    # Success without waypoint
    resp_no_wp = json.dumps(
        {
            "response": json.dumps(
                {
                    "thought": "Continuing scout duty",
                    "role": "scout",
                }
            )
        }
    ).encode("utf-8")
    mock_resp2 = MagicMock()
    mock_resp2.read.return_value = resp_no_wp
    mock_resp2.__enter__.return_value = mock_resp2

    with patch("urllib.request.urlopen", return_value=mock_resp2):
        dec2 = ollama.decide(obs, "coarse_fine")
        assert dec2.waypoint is None
        assert dec2.action_type == "OLLAMA_DECISION"


def test_ollama_backend_fallbacks_on_errors():
    ollama = OllamaBackend()
    assert isinstance(ollama.fallback, MockLLMBackend)

    obs = AgentObservation(
        agent_id=1,
        pos=(10, 10),
        charge=20000.0,
        state="walk",
        role="scout",
        has_food=True,  # triggers RETURN_NEST in mock fallback
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
    )

    # 1. URLError fallback
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        dec1 = ollama.decide(obs, "coarse_fine")
        assert dec1.action_type == "RETURN_NEST"

    # 2. TimeoutError fallback
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Timed out")):
        dec2 = ollama.decide(obs, "coarse_fine")
        assert dec2.action_type == "RETURN_NEST"

    # 3. JSONDecodeError fallback
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"invalid json"
    mock_resp.__enter__.return_value = mock_resp
    with patch("urllib.request.urlopen", return_value=mock_resp):
        dec3 = ollama.decide(obs, "coarse_fine")
        assert dec3.action_type == "RETURN_NEST"

    # 4. KeyError fallback
    mock_resp_key = MagicMock()
    mock_resp_key.read.return_value = json.dumps({"response": "not a dict"}).encode(
        "utf-8"
    )
    mock_resp_key.__enter__.return_value = mock_resp_key
    with patch("urllib.request.urlopen", return_value=mock_resp_key):
        dec4 = ollama.decide(obs, "coarse_fine")
        assert dec4.action_type == "RETURN_NEST"


# ---------------------------------------------------------------------------
# Test SwarmOrchestrator
# ---------------------------------------------------------------------------


def test_swarm_orchestrator_init_and_mode():
    # Defaults and custom
    orch1 = SwarmOrchestrator()
    assert orch1.mode == "classical"
    assert isinstance(orch1.backend, MockLLMBackend)

    orch2 = SwarmOrchestrator(mode="coarse_fine", interval=2.0)
    assert orch2.mode == "coarse_fine"
    assert orch2.interval == 2.0

    # Invalid mode falls back to classical
    orch3 = SwarmOrchestrator(mode="non_existent_mode")
    assert orch3.mode == "classical"

    # set_mode valid and invalid
    orch3.set_mode("global_contextual")
    assert orch3.mode == "global_contextual"
    orch3.set_mode("invalid_mode")
    assert orch3.mode == "global_contextual"


def test_swarm_orchestrator_step_modes_and_timers():
    # 1. Classical mode returns early
    orch = SwarmOrchestrator(mode="classical")
    w = make_world()
    orch.step(w, [], Trail(), now=1.0, dt=1.0)
    assert orch.stats["decisions"] == 0

    # 2. Timer below interval returns early
    orch.set_mode("coarse_fine")
    orch.interval = 1.0
    orch.step(w, [], Trail(), now=1.0, dt=0.5)
    assert orch.timer == 0.5
    assert orch.stats["decisions"] == 0


def test_swarm_orchestrator_step_executions():
    w = make_world()
    trail = Trail()

    a1 = Agent(ident=1, col=10, row=10, charge=30000.0)
    a1.state = "walk"
    a1.role = "scout"
    a2 = Agent(ident=2, col=15, row=15, charge=30000.0)
    a2.state = "seek"
    a2.role = "forager"
    a3 = Agent(ident=3, col=20, row=20, charge=30000.0)
    a3.state = "charge"  # filtered out of active pool

    agents = [a1, a2, a3]

    # Custom backend returning different action responses
    class CustomBackend(LLMBackend):
        def decide(self, obs: AgentObservation, mode: str) -> LLMDecision:
            if obs.agent_id == 1:
                return LLMDecision(
                    thought="Assigning waypoint and switching to sentry",
                    waypoint=(12, 12),
                    role="sentry",
                    token_to_write=((10, 10), "TEST_TOKEN"),
                    action_type="CUSTOM",
                )
            else:
                # Same role, no waypoint, no token
                return LLMDecision(
                    thought="Maintaining role",
                    role="forager",
                    action_type="MAINTAIN",
                )

    orch = SwarmOrchestrator(backend=CustomBackend(), mode="coarse_fine", interval=1.0)
    orch.step(w, agents, trail, now=5.0, dt=1.5)

    assert orch.stats["decisions"] == 2
    assert orch.stats["waypoints_assigned"] == 1
    assert orch.stats["roles_switched"] == 1
    assert orch.stats["tokens_written"] == 1

    assert a1.waypoint == (12, 12)
    assert a1.role == "sentry"
    assert a1.llm_thought == "Assigning waypoint and switching to sentry"
    assert a1.last_llm_time == 5.0
    assert w.read_token((10, 10), 5.0) == "[5.0s] TEST_TOKEN"

    assert a2.waypoint is None
    assert a2.role == "forager"
    assert a2.llm_thought == "Maintaining role"
    assert a3.llm_thought == ""  # charge state agent was not sampled


# ---------------------------------------------------------------------------
# Test ParallelSwarmBench
# ---------------------------------------------------------------------------


def test_parallel_swarm_bench_defaults_and_controls():
    bench = ParallelSwarmBench(seed=42)
    assert len(bench.instances) == 4
    assert bench.active_idx == 0
    assert bench.active_instance.mode == "classical"

    # Selection
    bench.set_active_idx(1)
    assert bench.active_idx == 1
    assert bench.active_instance.mode == "coarse_fine"

    # Out of bounds selection is ignored
    bench.set_active_idx(99)
    assert bench.active_idx == 1
    bench.set_active_idx(-1)
    assert bench.active_idx == 1

    # Toggle quad view
    assert bench.quad_view is False
    is_quad = bench.toggle_quad_view()
    assert is_quad is True
    assert bench.quad_view is True

    # Switching active index exits quad view
    bench.set_active_idx(2)
    assert bench.quad_view is False
    assert bench.active_instance.mode == "global_contextual"

    # Step simulation across all instances
    bench.step(now=1.0, dt=0.5)

    # Synchronized food spawn
    init_food_count = len(bench.instances[0].world.food)
    bench.spawn_synchronized_food(seed=99)
    for inst in bench.instances:
        assert len(inst.world.food) >= init_food_count


def test_parallel_swarm_bench_custom_args():
    custom_backend = MockLLMBackend()
    obs = {(5, 5), (6, 5)}
    stations = [(10, 10)]
    nav = build_nav(stations, obstacles=obs)
    nest = [(20, 20)]
    nest_nav = build_nav(nest, obstacles=obs)
    initial_food = {(30, 30): 5}
    initial_barriers = [
        MovingBarrier(coords=[(15, 15)], axis="x", direction=1, min_pos=10, max_pos=20)
    ]
    starts = [(12, 12) for _ in range(25)]

    bench = ParallelSwarmBench(
        backend=custom_backend,
        static_obstacles=obs,
        stations=stations,
        nav=nav,
        nest_tiles=nest,
        nest_nav=nest_nav,
        initial_food=initial_food,
        initial_barriers=initial_barriers,
        starts=starts,
    )
    assert len(bench.instances) == 4
    for inst in bench.instances:
        assert (5, 5) in inst.world.static_obstacles
        assert (30, 30) in inst.world.food
        assert len(inst.world.moving_barriers) == 1
        assert inst.world.moving_barriers[0].coords == [(15, 15)]
        assert len(inst.agents) == 25
        assert inst.agents[0].col == 12


def test_synthesize_token_intel():
    # Empty
    intel_empty = synthesize_token_intel([])
    assert intel_empty["food_leads"] == []
    assert intel_empty["depleted_zones"] == []
    assert intel_empty["hazards"] == []
    assert intel_empty["stations"] == []
    assert intel_empty["general"] == []

    # Various categories, blank lines, timestamped and untimestamped
    tokens = [
        ((10, 10), "\n[1.5s] [GUIDE] Food ~3 steps East at (13, 10) [4u]\n\n"),
        ((12, 10), "[2.0s] [FOOD] Harvest at (12, 10)"),
        ((15, 15), "[3.0s] [DEPLETED] Food at (15, 15) exhausted"),
        ((18, 18), "ALERT: Moving barrier detected"),
        ((20, 20), "[4.0s] HAZARD sector"),
        ((22, 22), "[5.0s] [STATION] Charging depot at (10, 10)"),
        ((25, 25), "Custom note left by scout"),
    ]
    intel = synthesize_token_intel(tokens)
    assert len(intel["food_leads"]) == 2
    assert (
        "[GUIDE] Food ~3 steps East at (13, 10) [4u] (read at (10, 10))"
        in intel["food_leads"]
    )
    assert "[FOOD] Harvest at (12, 10) (read at (12, 10))" in intel["food_leads"]
    assert len(intel["depleted_zones"]) == 1
    assert (
        "[DEPLETED] Food at (15, 15) exhausted (read at (15, 15))"
        in intel["depleted_zones"]
    )
    assert len(intel["hazards"]) == 2
    assert len(intel["stations"]) == 1
    assert (
        "[STATION] Charging depot at (10, 10) (read at (22, 22))" in intel["stations"]
    )
    assert len(intel["general"]) == 1
    assert "Custom note left by scout (read at (25, 25))" in intel["general"]


def test_mock_backend_coarse_fine_charging_triage():
    backend = MockLLMBackend()

    # 1. Low charge with visible station
    obs_vis = AgentObservation(
        agent_id=1,
        pos=(15, 15),
        charge=MAX_CHARGE * 0.2,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
        nearby_stations=[(12, 12)],
    )
    dec_vis = backend.decide(obs_vis, mode="coarse_fine")
    assert dec_vis.action_type == "COARSE_WAYPOINT"
    assert dec_vis.waypoint == (12, 12)

    # 2. Low charge with station signpost token in environment
    obs_sign = AgentObservation(
        agent_id=2,
        pos=(25, 25),
        charge=MAX_CHARGE * 0.2,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((24, 25), "[STATION] Charging depot at (10, 10)")],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
        nearby_stations=[],
    )
    dec_sign = backend.decide(obs_sign, mode="coarse_fine")
    assert dec_sign.action_type == "COARSE_WAYPOINT"
    assert dec_sign.waypoint == (10, 10)


def test_mock_backend_global_contextual_charging_triage_and_scout_station():
    backend = MockLLMBackend()

    # 1. Low charge with visible station -> Approaching & marks blackboard if not
    # present
    obs_vis = AgentObservation(
        agent_id=1,
        pos=(15, 15),
        charge=MAX_CHARGE * 0.2,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(12, 12)],
    )
    dec_vis = backend.decide(obs_vis, mode="global_contextual")
    assert dec_vis.action_type == "STATION_APPROACH"
    assert dec_vis.waypoint == (12, 12)
    assert dec_vis.token_to_write == ((15, 15), "[STATION] Charging depot at (12, 12)")

    # 1b. Low charge with visible station -> Token already at pos ->
    # token_to_write is None
    obs_vis_marked = AgentObservation(
        agent_id=1,
        pos=(15, 15),
        charge=MAX_CHARGE * 0.2,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((15, 15), "[STATION] Charging depot at (12, 12)")],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(12, 12)],
    )
    dec_vis_marked = backend.decide(obs_vis_marked, mode="global_contextual")
    assert dec_vis_marked.action_type == "STATION_APPROACH"
    assert dec_vis_marked.waypoint == (12, 12)
    assert dec_vis_marked.token_to_write is None

    # 2. Low charge with station signpost in environment
    obs_sign = AgentObservation(
        agent_id=2,
        pos=(25, 25),
        charge=MAX_CHARGE * 0.2,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((24, 25), "[STATION] Charging depot at (10, 10)")],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[],
    )
    dec_sign = backend.decide(obs_sign, mode="global_contextual")
    assert dec_sign.action_type == "STATION_SEEK"
    assert dec_sign.waypoint == (10, 10)

    # 3. Scout discovering station while having full charge -> writes signpost
    obs_scout = AgentObservation(
        agent_id=3,
        pos=(14, 14),
        charge=MAX_CHARGE,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(12, 12)],
    )
    dec_scout = backend.decide(obs_scout, mode="global_contextual")
    assert dec_scout.action_type == "STIGMERGY_MARK"
    assert dec_scout.token_to_write == (
        (14, 14),
        "[STATION] Charging depot at (12, 12)",
    )

    # 4. Low charge with NO visible station and NO signpost -> falls through to
    # role arbitration
    obs_none = AgentObservation(
        agent_id=4,
        pos=(20, 20),
        charge=MAX_CHARGE * 0.2,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[],
    )
    dec_none = backend.decide(obs_none, mode="global_contextual")
    assert dec_none.action_type == "ROLE_SWITCH"
    assert dec_none.role == "scout"


def test_mock_backend_coarse_fine_charging_fallthrough():
    backend = MockLLMBackend()
    # Low charge with NO visible station and NO signpost -> falls through to food
    # macro-planning
    obs = AgentObservation(
        agent_id=1,
        pos=(15, 15),
        charge=MAX_CHARGE * 0.2,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[((20, 20), 5)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
        nearby_stations=[],
    )
    dec = backend.decide(obs, mode="coarse_fine")
    assert dec.action_type == "COARSE_WAYPOINT"
    assert dec.waypoint == (20, 20)
    assert dec.role == "forager"


def test_extract_station_signpost():
    # Empty
    assert extract_station_signpost([]) is None
    # No station token
    assert extract_station_signpost([((1, 1), "[ALERT] Obstacle ahead")]) is None
    # Station token without coordinates
    assert (
        extract_station_signpost([((1, 1), "[STATION] Charging zone nearby")]) is None
    )
    # Station token with coordinates
    assert extract_station_signpost(
        [((1, 1), "[STATION] Charging depot at (14, 18)")]
    ) == (14, 18)


def test_select_best_station():
    # 1. No candidates available
    obs_none = AgentObservation(
        agent_id=0,
        pos=(10, 10),
        charge=10000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[],
        nearby_allies=[],
    )
    assert select_best_station(obs_none) is None

    # 2. Critical low charge (< 20%) with multiple candidates:
    # Must wait at closest station regardless of queue (stationary wait costs 0 battery)
    obs_low = AgentObservation(
        agent_id=0,
        pos=(10, 10),
        charge=MAX_CHARGE * 0.15,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(12, 12), (16, 16)],
        nearby_allies=[(1, (12, 12), "charge", "scout", False)],
    )
    assert select_best_station(obs_low) == (12, 12)

    # 3. Surplus charge (> 20%) with closest queued:
    # Safely detours to less-busy alternative station
    obs_surplus = AgentObservation(
        agent_id=0,
        pos=(10, 10),
        charge=MAX_CHARGE * 0.50,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(12, 12), (14, 14)],
        nearby_allies=[(1, (12, 12), "charge", "scout", False)],
    )
    assert select_best_station(obs_surplus) == (14, 14)

    # 4. Surplus charge, closest queued, but alternative also queued:
    # Stays at closest
    obs_both_busy = AgentObservation(
        agent_id=0,
        pos=(10, 10),
        charge=MAX_CHARGE * 0.50,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(12, 12), (14, 14)],
        nearby_allies=[
            (1, (12, 12), "charge", "scout", False),
            (2, (14, 14), "charge", "scout", False),
        ],
    )
    assert select_best_station(obs_both_busy) == (12, 12)

    # 5. Surplus charge, closest queued, alternative free, but not enough charge for
    # detour travel
    obs_unaffordable = AgentObservation(
        agent_id=0,
        pos=(10, 10),
        charge=MAX_CHARGE * 0.21,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(11, 10), (200, 200)],
        nearby_allies=[(1, (11, 10), "charge", "scout", False)],
    )
    assert select_best_station(obs_unaffordable) == (11, 10)

    # 6. Surplus charge and closest is completely free:
    # Selects closest
    obs_free = AgentObservation(
        agent_id=0,
        pos=(10, 10),
        charge=MAX_CHARGE * 0.50,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(12, 12), (14, 14)],
        nearby_allies=[],
    )
    assert select_best_station(obs_free) == (12, 12)

    # 7. Candidate discovery from known_stations when not in nearby_stations
    obs_known = AgentObservation(
        agent_id=0,
        pos=(10, 10),
        charge=MAX_CHARGE * 0.50,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(12, 12)],
        known_stations=[(12, 12), (14, 14)],
        nearby_allies=[(1, (12, 12), "charge", "scout", False)],
    )
    assert select_best_station(obs_known) == (14, 14)

    # 8. Active detour commitment: agent maintaining route to alternative waypoint
    obs_detour_committed = AgentObservation(
        agent_id=0,
        pos=(13, 13),
        charge=MAX_CHARGE * 0.50,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=(14, 14),
        nearby_stations=[(12, 12), (14, 14)],
        nearby_allies=[(1, (12, 12), "charge", "scout", False)],
    )
    assert select_best_station(obs_detour_committed) == (14, 14)

    # 9. Active detour aborted when charge becomes insufficient for remaining distance
    obs_detour_unaffordable = AgentObservation(
        agent_id=0,
        pos=(10, 10),
        charge=100.0,  # too low for (14, 14) travel cost
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=(14, 14),
        nearby_stations=[(12, 12), (14, 14)],
        nearby_allies=[],
    )
    assert select_best_station(obs_detour_unaffordable) == (12, 12)

    # 10. Active detour aborted when destination station itself is crowded (queue >= 2)
    obs_detour_destination_crowded = AgentObservation(
        agent_id=0,
        pos=(13, 13),
        charge=MAX_CHARGE * 0.50,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=(14, 14),
        nearby_stations=[(12, 12), (14, 14)],
        nearby_allies=[
            (1, (14, 14), "charge", "scout", False),
            (2, (14, 13), "seek", "scout", False),
        ],
    )
    assert select_best_station(obs_detour_destination_crowded) == (12, 12)

    # 11. Queue detection includes allies at distance 2 and 3 in seek state
    obs_distant_queue = AgentObservation(
        agent_id=0,
        pos=(10, 10),
        charge=MAX_CHARGE * 0.50,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(12, 12), (14, 14)],
        nearby_allies=[
            (1, (12, 10), "seek", "scout", False),  # distance 2 from (12, 12)
            (2, (12, 9), "seek", "scout", False),  # distance 3 from (12, 12)
        ],
    )
    assert select_best_station(obs_distant_queue) == (14, 14)

    # 12. Adjacent station (dist 1) with long queue (>= 2):
    # Safely detours to less-busy alternative station
    obs_adjacent_queue_long = AgentObservation(
        agent_id=0,
        pos=(10, 10),
        charge=MAX_CHARGE * 0.50,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(11, 10), (14, 14)],
        nearby_allies=[
            (1, (11, 10), "charge", "scout", False),
            (2, (11, 11), "seek", "scout", False),
        ],
    )
    assert select_best_station(obs_adjacent_queue_long) == (14, 14)

    # 13. Adjacent station with long queue, but alternative station is also busy:
    # Stays at adjacent station
    obs_adjacent_queue_long_no_better_alt = AgentObservation(
        agent_id=0,
        pos=(10, 10),
        charge=MAX_CHARGE * 0.50,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(11, 10), (14, 14)],
        nearby_allies=[
            (1, (11, 10), "charge", "scout", False),
            (2, (11, 11), "seek", "scout", False),
            (3, (14, 14), "charge", "scout", False),
            (4, (14, 13), "seek", "scout", False),
        ],
    )
    assert select_best_station(obs_adjacent_queue_long_no_better_alt) == (11, 10)


def test_extract_station_signposts():
    # 1. No tokens
    assert extract_station_signposts([]) == []
    assert extract_station_signpost([]) is None

    # 2. Token with no [STATION] tag
    tokens_no_station = [((5, 5), "[FOOD] at (10, 10)")]
    assert extract_station_signposts(tokens_no_station) == []
    assert extract_station_signpost(tokens_no_station) is None

    # 3. Multiple [STATION] tokens including duplicates
    tokens_with_stations = [
        ((5, 5), "[STATION] Charging depot at (0, 20)"),
        ((6, 6), "[STATION] Charging depot at (0, 40)"),
        ((7, 7), "[STATION] Duplicate at (0, 20) and (0, 60)"),
    ]
    signposts = extract_station_signposts(tokens_with_stations)
    assert signposts == [(0, 20), (0, 40), (0, 60)]
    assert extract_station_signpost(tokens_with_stations) == (0, 20)


def test_extract_observation_dead_agent_detected_as_obstacle():
    nav = build_nav([(0, 0)])
    a = Agent(0, 10, 10, charge=50000)
    dead_ally = Agent(1, 12, 12, charge=0)
    w = World(nav, [a, dead_ally])
    trail = Trail()
    obs = extract_observation(a, w, trail, 0.0, radius=6)
    assert (12, 12) in obs.nearby_obstacles


def test_swarm_orchestrator_step_assigns_station_triggers_seek():
    st = [(0, 15)]
    nav = build_nav(st)
    a = Agent(0, 10, 10, charge=50000)
    a.state = "walk"
    w = World(nav, [a], stations=st)
    trail = Trail()
    backend = MagicMock()
    backend.decide.return_value = LLMDecision(
        thought="Navigating to charging station",
        waypoint=(0, 15),
        role="scout",
        action_type="COARSE_WAYPOINT",
    )
    orch = SwarmOrchestrator(backend=backend, mode="coarse_fine", interval=0.1)
    orch.step(w, [a], trail, 0.2, 0.2)
    assert a.waypoint == (0, 15)
    assert a.state == "seek"

    # Step again while already seeking -> stays seek without re-triggering
    orch.step(w, [a], trail, 0.4, 0.2)
    assert a.state == "seek"


def test_swarm_orchestrator_detours_agent_when_closest_station_is_queued():
    st = [(0, 20), (0, 40)]
    nav = build_nav(st)
    trail = Trail()

    # Agent 0 is at (0, 20) charging
    a0 = Agent(0, 0, 20, charge=50000)
    a0.state = "charge"

    # Agent 1 is at (1, 20) seeking (queued)
    a1 = Agent(1, 1, 20, charge=15000)
    a1.state = "seek"

    # Agent 2 is approaching at (3, 20) with surplus battery (30,000)
    a2 = Agent(2, 3, 20, charge=30000)
    a2.state = "seek"

    # Agent 3 is approaching at (3, 21) with critical battery (12,000 < 20% MAX_CHARGE)
    a3 = Agent(3, 3, 21, charge=12000)
    a3.state = "seek"

    w = World(nav, [a0, a1, a2, a3], stations=st)
    orch = SwarmOrchestrator(mode="coarse_fine", interval=0.1)

    orch.step(w, [a0, a1, a2, a3], trail, 0.2, 0.2)

    # Agent 2 with surplus battery detours to alternative station (0, 40)
    assert a2.waypoint == (0, 40)

    # Agent 3 with critical battery (< 20%) commits to closest station (0, 20) to
    # avoid dying in transit
    assert a3.waypoint == (0, 20)

    orch_ctx = SwarmOrchestrator(mode="global_contextual", interval=0.1)
    orch_ctx.step(w, [a0, a1, a2, a3], trail, 0.4, 0.2)
    assert a2.waypoint == (0, 40)
    assert a3.waypoint == (0, 20)

    orch_pure = SwarmOrchestrator(mode="pure_llm", interval=0.1)
    orch_pure.step(w, [a0, a1, a2, a3], trail, 0.6, 0.2)
    assert a2.waypoint == (0, 40)
    assert a3.waypoint == (0, 20)


def test_mock_backend_seeking_with_food_seeks_charger_not_nest():
    backend = MockLLMBackend()
    # Agent carrying food but in seek state (low battery) must seek charger instead
    # of returning to nest
    obs = AgentObservation(
        agent_id=1,
        pos=(20, 20),
        charge=10000.0,
        state="seek",
        role="forager",
        has_food=True,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[(22, 22)],
        nearby_allies=[],
    )
    dec = backend.decide(obs, mode="coarse_fine")
    assert dec.action_type == "COARSE_WAYPOINT"
    assert dec.waypoint == (22, 22)


def test_mock_backend_seek_gradient_when_no_station_in_range():
    backend = MockLLMBackend()
    obs = AgentObservation(
        agent_id=2,
        pos=(20, 20),
        charge=10000.0,
        state="seek",
        role="scout",
        has_food=False,
        nearby_food=[
            ((25, 25), 5)
        ],  # Even with food nearby, must not chase food while seeking
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        nearby_stations=[],
        nearby_allies=[],
    )
    # coarse_fine mode falls back to SEEK_GRADIENT instead of chasing food
    dec_cf = backend.decide(obs, mode="coarse_fine")
    assert dec_cf.action_type == "SEEK_GRADIENT"
    assert dec_cf.waypoint is None

    # global_contextual mode falls back to SEEK_GRADIENT
    dec_gc = backend.decide(obs, mode="global_contextual")
    assert dec_gc.action_type == "SEEK_GRADIENT"
    assert dec_gc.waypoint is None


def test_swarm_orchestrator_discards_non_station_waypoint_for_seeker():
    st = [(0, 15)]
    nav = build_nav(st)
    a = Agent(0, 10, 10, charge=10000)
    a.state = "seek"
    w = World(nav, [a], stations=st)
    trail = Trail()
    backend = MagicMock()
    # Hallucinated non-station waypoint (e.g., food or nest tile)
    backend.decide.return_value = LLMDecision(
        thought="Hallucinated food target",
        waypoint=(50, 50),
        role="forager",
        action_type="COARSE_WAYPOINT",
    )
    orch = SwarmOrchestrator(backend=backend, mode="coarse_fine", interval=0.1)
    orch.step(w, [a], trail, 0.2, 0.2)
    # Orchestrator clears invalid non-station waypoint for seeker
    assert a.waypoint is None

    # When decision waypoint is None and agent is seeking, clears stale waypoint
    a.waypoint = (40, 40)
    backend.decide.return_value = LLMDecision(
        thought="Seek gradient",
        waypoint=None,
        role="scout",
        action_type="SEEK_GRADIENT",
    )
    orch.step(w, [a], trail, 0.4, 0.2)
    assert a.waypoint is None

    # Normal walking agent with waypoint=None does not get its waypoint cleared
    walker = Agent(1, 10, 10, charge=50000)
    walker.state = "walk"
    walker.waypoint = (30, 30)
    backend.decide.return_value = LLMDecision(
        thought="Maintain",
        waypoint=None,
        action_type="MAINTAIN",
    )
    orch.step(w, [walker], trail, 0.6, 0.2)
    assert walker.waypoint == (30, 30)


def test_extract_observation_senses_food_trail():
    a = Agent(0, 10, 10, charge=50000)
    w = World(build_nav([(0, 0)]), [a])
    w.food_trail.mark(12, 10, 0.0)
    trail = Trail()
    obs = extract_observation(a, w, trail, 0.0, radius=6)
    assert obs.sensed_food_trail is True


def test_coarse_fine_depleted_waypoint_and_trail_forage():
    backend = MockLLMBackend()
    # Depleted waypoint
    obs_dep = AgentObservation(
        agent_id=1,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=(21, 20),
    )
    dec_dep = backend.decide(obs_dep, mode="coarse_fine")
    assert dec_dep.action_type == "ABORT_DEPLETED"
    assert dec_dep.waypoint is None
    assert dec_dep.role == "scout"
    assert dec_dep.token_to_write is None

    # Sensed food trail -> TRAIL_FORAGE
    obs_trail = AgentObservation(
        agent_id=2,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
        sensed_food_trail=True,
    )
    dec_trail = backend.decide(obs_trail, mode="coarse_fine")
    assert dec_trail.action_type == "TRAIL_FORAGE"
    assert dec_trail.role == "forager"

    # Sensed food trail but near depleted note without food -> does not TRAIL_FORAGE
    obs_trail_dep = AgentObservation(
        agent_id=3,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((20, 20), "[DEPLETED] Food at (21, 20) exhausted")],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
        sensed_food_trail=True,
    )
    dec_trail_dep = backend.decide(obs_trail_dep, mode="coarse_fine")
    assert dec_trail_dep.action_type != "TRAIL_FORAGE"


def test_pure_llm_depleted_waypoint_and_trail_forage():
    backend = MockLLMBackend()
    obs_dep = AgentObservation(
        agent_id=1,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=(21, 20),
    )
    dec_dep = backend.decide(obs_dep, mode="pure_llm")
    assert dec_dep.action_type == "DEPLETED_MARK"

    obs_trail = AgentObservation(
        agent_id=2,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
        sensed_food_trail=True,
    )
    dec_trail = backend.decide(obs_trail, mode="pure_llm")
    assert dec_trail.action_type == "TRAIL_FORAGE"


def test_swarm_orchestrator_clears_trail_on_depleted_mark():
    st = [(0, 15)]
    nav = build_nav(st)
    a = Agent(0, 10, 10, charge=50000)
    w = World(nav, [a], stations=st)
    w.food_trail.mark(25, 25, 0.0)
    w.food_trail.mark(10, 10, 0.0)
    w.food_trail.mark(25, 25, 0.0)
    trail = Trail()
    backend = MagicMock()
    backend.decide.return_value = LLMDecision(
        thought="Food depleted",
        token_to_write=((10, 10), "[DEPLETED] Food at (25, 25) exhausted"),
        action_type="DEPLETED_MARK",
    )
    orch = SwarmOrchestrator(backend=backend, mode="global_contextual", interval=0.1)
    orch.step(w, [a], trail, 0.2, 0.2)
    # The agent strictly updates ONLY the tile it is standing on (10, 10)
    assert w.food_trail.value(10, 10, 0.0) == 0.0
    # Distant tiles like (25, 25) are NOT cleared by an agent standing at (10, 10)
    assert w.food_trail.value(25, 25, 0.0) > 0.0


def test_coarse_fine_and_pure_llm_waypoint_with_food_present():
    backend = MockLLMBackend()
    obs = AgentObservation(
        agent_id=1,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[((21, 20), 5)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=(21, 20),
    )
    dec_cf = backend.decide(obs, mode="coarse_fine")
    assert dec_cf.action_type != "DEPLETED_MARK"
    assert dec_cf.role == "forager"

    dec_pure = backend.decide(obs, mode="pure_llm")
    assert dec_pure.action_type == "TACTICAL_MOVE"
    assert dec_pure.role == "forager"


def test_swarm_orchestrator_depleted_token_without_coordinates():
    st = [(0, 15)]
    nav = build_nav(st)
    a = Agent(0, 10, 10, charge=50000)
    w = World(nav, [a], stations=st)
    w.food_trail.mark(10, 10, 0.0)
    trail = Trail()
    backend = MagicMock()
    backend.decide.return_value = LLMDecision(
        thought="Depleted notice without coords",
        token_to_write=((10, 10), "[DEPLETED] Cluster finished"),
        action_type="DEPLETED_MARK",
    )
    orch = SwarmOrchestrator(backend=backend, mode="global_contextual", interval=0.1)
    orch.step(w, [a], trail, 0.2, 0.2)
    assert orch.stats["tokens_written"] == 1
    assert w.food_trail.value(10, 10, 0.0) == 0.0


def test_coarse_fine_does_not_clear_trail_on_depleted_token():
    st = [(0, 15)]
    nav = build_nav(st)
    a = Agent(0, 10, 10, charge=50000)
    w = World(nav, [a], stations=st)
    w.food_trail.mark(25, 25, 0.0)
    trail = Trail()
    backend = MagicMock()
    backend.decide.return_value = LLMDecision(
        thought="Food depleted",
        token_to_write=((10, 10), "[DEPLETED] Food at (25, 25) exhausted"),
        action_type="DEPLETED_MARK",
    )
    orch = SwarmOrchestrator(backend=backend, mode="coarse_fine", interval=0.1)
    orch.step(w, [a], trail, 0.2, 0.2)
    # coarse_fine does not clear trails!
    assert w.food_trail.value(25, 25, 0.0) > 0.0


def test_global_contextual_ignores_guide_if_target_depleted():
    backend = MockLLMBackend()
    obs = AgentObservation(
        agent_id=1,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="scout",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[
            ((20, 20), "[GUIDE] Food at (25, 25) [10u]"),
            ((20, 21), "[DEPLETED] Food at (25, 25) exhausted"),
        ],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=None,
    )
    dec = backend.decide(obs, mode="global_contextual")
    assert dec.action_type != "HARVEST_CALL"


def test_global_contextual_aborts_route_when_waypoint_depleted():
    backend = MockLLMBackend()
    obs = AgentObservation(
        agent_id=1,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[
            ((20, 21), "[DEPLETED] Food at (25, 25) exhausted"),
        ],
        nearby_obstacles=[],
        scent_directions={"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
        waypoint=(25, 25),
    )
    dec = backend.decide(obs, mode="global_contextual")
    assert dec.action_type == "ABORT_DEPLETED"
    assert dec.waypoint is None
    assert dec.role == "scout"


def test_depleted_clears_only_dead_sub_branch_preserving_common_and_active_branch():
    st = [(0, 15)]
    nav = build_nav(st)
    nest = [(49, 49), (50, 49), (49, 50), (50, 50)]
    nst_nav = build_nav(nest)
    a = Agent(0, 25, 25, charge=50000)

    # Active food source B at (40, 40)
    w = World(
        nav,
        [a],
        nest_nav=nst_nav,
        stations=st,
        food={(40, 40): 10},  # Food B is active!
    )

    # Common trail and branches:
    # Common trunk near nest: (48, 48)
    w.food_trail.mark(48, 48, 0.0)
    # Active Branch B: (40, 40)
    w.food_trail.mark(40, 40, 0.0)
    # Depleted Branch A: (25, 25)
    w.food_trail.mark(25, 25, 0.0)

    trail = Trail()
    backend = MagicMock()
    backend.decide.return_value = LLMDecision(
        thought="Food A depleted",
        token_to_write=((25, 25), "[DEPLETED] Food at (25, 25) exhausted"),
        action_type="DEPLETED_MARK",
    )
    orch = SwarmOrchestrator(backend=backend, mode="global_contextual", interval=0.1)
    orch.step(w, [a], trail, 0.2, 0.2)

    # Depleted food A branch is cleared
    assert w.food_trail.value(25, 25, 0.0) == 0.0
    # Active food B branch is completely PRESERVED!
    assert w.food_trail.value(40, 40, 0.0) > 0.0
    # Common trunk is completely PRESERVED!
    assert w.food_trail.value(48, 48, 0.0) > 0.0


def test_global_contextual_trail_end_depleted_and_already_marked():
    backend = MockLLMBackend()
    # 1. Trail end reached with no food, no outward trail, far from nest
    # (nest_dist >= 15)
    obs_trail_end = AgentObservation(
        agent_id=1,
        pos=(30, 30),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        sensed_food_trail=True,
        has_outward_trail=False,
        nest_dist=25,
    )
    dec = backend.decide(obs_trail_end, mode="global_contextual")
    assert dec.action_type == "DEPLETED_MARK"
    assert dec.role == "scout"
    assert dec.waypoint is None
    assert dec.token_to_write == ((30, 30), "[DEPLETED] Food at (30, 30) exhausted")

    # 2. Trail end already marked with [DEPLETED] token -> abort immediately to scout
    obs_already = AgentObservation(
        agent_id=2,
        pos=(30, 30),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((30, 30), "[DEPLETED] Food at (30, 30) exhausted")],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        sensed_food_trail=True,
        has_outward_trail=False,
        nest_dist=25,
    )
    dec_already = backend.decide(obs_already, mode="global_contextual")
    assert dec_already.action_type == "ABORT_DEPLETED"
    assert dec_already.role == "scout"
    assert dec_already.waypoint is None
    assert dec_already.token_to_write is None


def test_pure_llm_trail_end_depleted_and_already_marked():
    backend = MockLLMBackend()
    # 1. Trail end reached in pure_llm with no food and no outward trail
    obs_trail_end = AgentObservation(
        agent_id=1,
        pos=(30, 30),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        sensed_food_trail=True,
        has_outward_trail=False,
        nest_dist=25,
    )
    dec = backend.decide(obs_trail_end, mode="pure_llm")
    assert dec.action_type == "DEPLETED_MARK"
    assert dec.role == "scout"
    assert dec.waypoint is None
    assert dec.token_to_write == ((30, 30), "[DEPLETED] Food at (30, 30) exhausted")

    # 2. Trail end already marked in pure_llm -> abort immediately to scout
    obs_already = AgentObservation(
        agent_id=2,
        pos=(30, 30),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[],
        nearby_tokens=[((30, 30), "[DEPLETED] Food at (30, 30) exhausted")],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=None,
        sensed_food_trail=True,
        has_outward_trail=False,
        nest_dist=25,
    )
    dec_already = backend.decide(obs_already, mode="pure_llm")
    assert dec_already.action_type == "ABORT_DEPLETED"
    assert dec_already.role == "scout"
    assert dec_already.waypoint is None
    assert dec_already.token_to_write is None


def test_live_forager_at_trail_end_clears_branch_and_marks_depleted():
    st = [(0, 15)]
    nav = build_nav(st)
    nest = [(49, 49), (50, 49), (49, 50), (50, 50)]
    nst_nav = build_nav(nest)
    a = Agent(0, 30, 50, charge=50000)
    a.state = "walk"
    a.role = "forager"
    a.waypoint = None

    food_trail = Trail(accumulate=True)
    for c in range(30, 45):
        food_trail.mark(c, 50, 0.0)

    w = World(nav, [a], nest_nav=nst_nav, stations=st, food={}, food_trail=food_trail)
    orch = SwarmOrchestrator(MockLLMBackend(), mode="global_contextual", interval=0.1)
    trail = Trail()
    orch.step(w, [a], trail, now=0.2, dt=0.2)

    assert a.role == "scout"
    assert a.waypoint is None
    # Agent strictly updates ONLY the tile it is standing on: (30, 50)
    assert w.food_trail.value(30, 50, 0.2) == 0.0
    # Other tiles on the trail like (35, 50) are NOT cleared yet because agent is
    # not standing on them
    assert w.food_trail.value(35, 50, 0.2) > 0.0
    tokens = w.active_tokens(0.2)
    assert any("[DEPLETED]" in text for text in tokens.values())

    # When agent moves to (35, 50), on the next step it clears (35, 50)
    a.col, a.row = 35, 50
    orch.step(w, [a], trail, now=0.4, dt=0.2)
    assert w.food_trail.value(35, 50, 0.4) == 0.0


def test_orchestrator_depleted_on_junction_or_active_branch_does_not_clear():
    st = [(0, 15)]
    nav = build_nav(st)
    nest = [(49, 49), (50, 49), (49, 50), (50, 50)]
    nst_nav = build_nav(nest)
    # Agent standing at junction (40, 50) leading outward to active food at (40, 40)
    a = Agent(0, 40, 50, charge=50000)
    a.state = "walk"
    a.role = "forager"

    food_trail = Trail(accumulate=True)
    for r in range(40, 51):
        food_trail.mark(40, r, 0.0)

    w = World(
        nav,
        [a],
        nest_nav=nst_nav,
        stations=st,
        food={(40, 40): 10},
        food_trail=food_trail,
    )
    trail = Trail()
    backend = MagicMock()
    backend.decide.return_value = LLMDecision(
        thought="Spurious depleted announcement",
        token_to_write=((40, 50), "[DEPLETED] Food at (20, 20) exhausted"),
        action_type="DEPLETED_MARK",
    )
    orch = SwarmOrchestrator(backend=backend, mode="global_contextual", interval=0.1)
    orch.step(w, [a], trail, now=0.2, dt=0.2)

    # Junction/active branch is protected and NOT cleared!
    assert w.food_trail.value(40, 50, 0.2) > 0.0
    assert a.erasing_trail is False


def test_coarse_fine_depleted_waypoint_retargets_to_nearby_active_food():
    backend = MockLLMBackend()
    obs = AgentObservation(
        agent_id=1,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[
            ((23, 20), 8)
        ],  # Food at waypoint (20, 20) is exhausted, but (23, 20) is active
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=(20, 20),
    )
    dec = backend.decide(obs, mode="coarse_fine")
    assert dec.action_type == "COARSE_WAYPOINT"
    assert dec.waypoint == (23, 20)
    assert dec.role == "forager"


def test_global_contextual_depleted_waypoint_retargets_to_nearby_active_food():
    backend = MockLLMBackend()
    obs = AgentObservation(
        agent_id=2,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[((24, 20), 6)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=(20, 20),
    )
    dec = backend.decide(obs, mode="global_contextual")
    assert dec.action_type == "HARVEST_CALL"
    assert dec.waypoint == (24, 20)
    assert dec.role == "forager"


def test_pure_llm_depleted_waypoint_retargets_to_nearby_active_food():
    backend = MockLLMBackend()
    obs = AgentObservation(
        agent_id=3,
        pos=(20, 20),
        charge=50000.0,
        state="walk",
        role="forager",
        has_food=False,
        nearby_food=[((25, 20), 7)],
        nearby_tokens=[],
        nearby_obstacles=[],
        scent_directions={},
        waypoint=(20, 20),
    )
    dec = backend.decide(obs, mode="pure_llm")
    assert dec.action_type == "TACTICAL_MOVE"
    assert dec.waypoint == (25, 20)
    assert dec.role == "forager"


def test_swarm_orchestrator_skips_erasing_if_already_erasing():
    a0 = Agent(0, 20, 20)
    a0.erasing_trail = True
    a1 = Agent(1, 21, 20)
    st = [(0, 5)]
    nav = build_nav(st)
    nst_nav = build_nav([(50, 50)])
    w = World(nav, [a0, a1], nest_nav=nst_nav, stations=st, food={})
    w.food_trail.mark(21, 20, 0.0)
    trail = Trail()
    backend = MagicMock()
    backend.decide.return_value = LLMDecision(
        thought="Depleted announcement",
        token_to_write=((21, 20), "[DEPLETED] Food empty"),
        action_type="DEPLETED_MARK",
    )
    orch = SwarmOrchestrator(backend=backend, mode="global_contextual", interval=0.1)
    orch.step(w, [a1], trail, now=0.2, dt=0.2)

    # Since a0 is already erasing, a1 does not clear the tile and does not become an
    # eraser
    assert a1.erasing_trail is False
    assert a1.role == "forager"
    assert a1.waypoint is None
