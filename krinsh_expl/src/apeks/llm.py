from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from apeks.config import (
    AGENT_COUNT,
    GRID_SIZE,
    LOW_CHARGE,
    MAX_CHARGE,
    MOVE_COST,
    NEST_TILES,
)
from apeks.sim import (
    Agent,
    MovingBarrier,
    NavMap,
    Tile,
    Trail,
    World,
    advance,
    build_nav,
    has_active_branch_outward,
    make_food_clusters,
    make_moving_barriers,
    make_nest,
    make_static_obstacles,
    make_stations,
    parse_token_line,
)


@dataclass
class AgentObservation:
    agent_id: int
    pos: Tile
    charge: float
    state: str
    role: str
    has_food: bool
    nearby_food: list[tuple[Tile, int]]
    nearby_tokens: list[tuple[Tile, str]]
    nearby_obstacles: list[Tile]
    scent_directions: dict[str, float]
    waypoint: Tile | None
    nearby_allies: list[tuple[int, Tile, str, str, bool]] = field(default_factory=list)
    nearby_stations: list[Tile] = field(default_factory=list)
    sensed_food_trail: bool = False
    known_stations: list[Tile] = field(default_factory=list)
    has_outward_trail: bool = True
    nest_dist: int = 999


@dataclass
class LLMDecision:
    thought: str
    waypoint: Tile | None = None
    role: str | None = None
    token_to_write: tuple[Tile, str] | None = None
    action_type: str = "MAINTAIN"


def synthesize_token_intel(
    nearby_tokens: list[tuple[Tile, str]],
) -> dict[str, list[str]]:
    """Synthesizes structured linguistic intelligence from raw environmental
    blackboard tokens.

    Implements the Token Synthesizer described in the MAS implementation blueprint,
    parsing timestamped strings to extract active resource leads, depleted zones,
    hazard alerts, and charging station signposts.
    """
    intel: dict[str, list[str]] = {
        "food_leads": [],
        "depleted_zones": [],
        "hazards": [],
        "stations": [],
        "general": [],
    }
    for tile, raw_text in nearby_tokens:
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            _, content = parse_token_line(line)
            if "[FOOD]" in content or "[GUIDE]" in content:
                intel["food_leads"].append(f"{content} (read at {tile})")
            elif "[DEPLETED]" in content:
                intel["depleted_zones"].append(f"{content} (read at {tile})")
            elif "ALERT" in content or "HAZARD" in content:
                intel["hazards"].append(f"{content} (read at {tile})")
            elif "[STATION]" in content:
                intel["stations"].append(f"{content} (read at {tile})")
            else:
                intel["general"].append(f"{content} (read at {tile})")
    return intel


def extract_station_signposts(nearby_tokens: list[tuple[Tile, str]]) -> list[Tile]:
    """Extract destination coordinates from environmental [STATION] signpost tokens."""
    res: list[Tile] = []
    for _, note in nearby_tokens:
        if "[STATION]" in note:
            for m in re.finditer(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", note):
                tile = (int(m.group(1)), int(m.group(2)))
                if tile not in res:
                    res.append(tile)
    return res


def extract_station_signpost(nearby_tokens: list[tuple[Tile, str]]) -> Tile | None:
    """Extract destination coordinate from an environmental [STATION] signpost token."""
    signposts = extract_station_signposts(nearby_tokens)
    return signposts[0] if signposts else None


def select_best_station(obs: AgentObservation) -> Tile | None:
    """Select the best charging station using decentralized sensing, queue load,
    and remaining charge.

    Zero energy is consumed while stationary. If charge is critically low (< 20%),
    the agent
    must not risk traveling to a farther station and must commit to the closest station,
    waiting patiently in queue at 0 battery drain to avoid dying in the field.
    Only if the agent has sufficient surplus charge can it safely bypass a busy
    station and
    divert to an alternative nearby station with a smaller queue.
    """
    candidates: list[Tile] = list(obs.nearby_stations)
    for st in obs.known_stations:
        if st not in candidates:
            candidates.append(st)
    for signpost in extract_station_signposts(obs.nearby_tokens):
        if signpost not in candidates:
            candidates.append(signpost)

    if not candidates:
        return None

    def station_queue(st: Tile) -> int:
        return sum(
            1
            for ally in obs.nearby_allies
            if ally[1] == st
            or (
                abs(ally[1][0] - st[0]) + abs(ally[1][1] - st[1]) <= 3
                and ally[2] in ("charge", "seek")
            )
        )

    candidates_by_dist = sorted(
        candidates,
        key=lambda st: abs(obs.pos[0] - st[0]) + abs(obs.pos[1] - st[1]),
    )
    closest = candidates_by_dist[0]

    # If already at or adjacent to a candidate station (front of queue), stay committed
    # unless queue is long (>= 2) and agent has safe surplus charge to detour to a
    # less-busy station.
    near_stations = [
        st
        for st in candidates
        if abs(obs.pos[0] - st[0]) + abs(obs.pos[1] - st[1]) <= 1
    ]
    if near_stations:
        st = near_stations[0]
        st_q = station_queue(st)
        if st_q < 2 or obs.charge < MAX_CHARGE * 0.20:
            return st
        for alt in candidates_by_dist:
            if alt != st:
                alt_dist = abs(obs.pos[0] - alt[0]) + abs(obs.pos[1] - alt[1])
                travel_cost = (alt_dist + 5) * MOVE_COST
                if obs.charge >= travel_cost and station_queue(alt) < st_q:
                    return alt
        return st

    # Critical low charge (< 20%): cannot afford travel to farther station, wait at
    # closest
    if obs.charge < MAX_CHARGE * 0.20 or len(candidates_by_dist) == 1:
        return closest

    # If currently en route to an alternative station with sufficient charge and no
    # heavy queue, maintain commitment
    if (
        obs.waypoint is not None
        and obs.waypoint in candidates
        and obs.waypoint != closest
    ):
        wp_dist = abs(obs.pos[0] - obs.waypoint[0]) + abs(obs.pos[1] - obs.waypoint[1])
        wp_travel_cost = (wp_dist + 5) * MOVE_COST
        if obs.charge >= wp_travel_cost and station_queue(obs.waypoint) < 2:
            return obs.waypoint

    closest_q = station_queue(closest)
    if closest_q == 0:
        return closest

    # Surplus charge: detour to an alternative station only if it has a shorter queue
    # and agent has enough battery to reach it safely with margin.
    for alt in candidates_by_dist[1:]:
        alt_dist = abs(obs.pos[0] - alt[0]) + abs(obs.pos[1] - alt[1])
        travel_cost = (alt_dist + 5) * MOVE_COST
        if obs.charge >= travel_cost and station_queue(alt) < closest_q:
            return alt

    return closest


def extract_observation(
    agent: Agent,
    world: World,
    trail: Trail,
    now: float,
    radius: int = 6,
) -> AgentObservation:
    """Extract local egocentric observation around an agent."""
    c, r = agent.col, agent.row

    nearby_food: list[tuple[Tile, int]] = []
    for tile, amt in world.food.items():
        if abs(tile[0] - c) + abs(tile[1] - r) <= radius:
            nearby_food.append((tile, amt))

    tokens = world.active_tokens(now)
    nearby_tokens: list[tuple[Tile, str]] = []
    for tile, text in tokens.items():
        if abs(tile[0] - c) + abs(tile[1] - r) <= radius:
            nearby_tokens.append((tile, text))

    nearby_obstacles: list[Tile] = []
    for tile in world.obstacles:
        if abs(tile[0] - c) + abs(tile[1] - r) <= radius:
            nearby_obstacles.append(tile)
    for other in world.occ.values():
        if (
            other.id != agent.id
            and getattr(other, "is_dead", False)
            and abs(other.col - c) + abs(other.row - r) <= radius
        ):
            nearby_obstacles.append((other.col, other.row))

    nearby_stations: list[Tile] = []
    for st in getattr(world, "stations", []):
        if abs(st[0] - c) + abs(st[1] - r) <= radius:
            nearby_stations.append(st)

    nearby_allies: list[tuple[int, Tile, str, str, bool]] = []
    for other in world.occ.values():
        if other.id != agent.id and abs(other.col - c) + abs(other.row - r) <= radius:
            nearby_allies.append(
                (
                    other.id,
                    (other.col, other.row),
                    other.state,
                    getattr(other, "role", "scout"),
                    other.has_food,
                )
            )

    scents: dict[str, float] = {}
    deltas = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
    for name, (dc, dr) in deltas.items():
        tc, tr = c + dc, r + dr
        if 0 <= tc < GRID_SIZE and 0 <= tr < GRID_SIZE:
            scents[name] = round(trail.value(tc, tr, now), 1)
        else:
            scents[name] = 100.0

    here_dist = world.nest_nav.dist[r][c]
    sensed_food_trail = False
    has_outward_trail = False
    for dx in range(-radius, radius + 1):
        for dy in range(-(radius - abs(dx)), radius - abs(dx) + 1):
            tc, tr = c + dx, r + dy
            if 0 <= tc < GRID_SIZE and 0 <= tr < GRID_SIZE:
                if world.food_trail.value(tc, tr, now) >= 0.5:
                    sensed_food_trail = True
                    if world.nest_nav.dist[tr][tc] > here_dist:
                        has_outward_trail = True
                        break
        if sensed_food_trail and has_outward_trail:
            break

    return AgentObservation(
        agent_id=agent.id,
        pos=(c, r),
        charge=agent.charge,
        state=agent.state,
        role=getattr(agent, "role", "scout"),
        has_food=agent.has_food,
        nearby_food=nearby_food,
        nearby_tokens=nearby_tokens,
        nearby_obstacles=nearby_obstacles,
        scent_directions=scents,
        waypoint=getattr(agent, "waypoint", None),
        nearby_allies=nearby_allies,
        nearby_stations=nearby_stations,
        sensed_food_trail=sensed_food_trail,
        known_stations=list(getattr(world, "stations", [])),
        has_outward_trail=has_outward_trail,
        nest_dist=here_dist,
    )


def to_narrative_briefing(obs: AgentObservation, mode: str) -> str:
    """Format an observation into a structured, highly legible tactical text
    briefing."""
    charge_pct = round(obs.charge / MAX_CHARGE * 100, 1)
    lines = [
        f"=== AGENT #{obs.agent_id:03d} TACTICAL BRIEFING ({mode.upper()}) ===",
        f"Role: {obs.role} | State: {obs.state} | Energy: {charge_pct}% "
        f"| Pos: {obs.pos} | Has Food: {obs.has_food}",
        "Sensory Field (Radius 6):",
        f"  - Food Patches: {len(obs.nearby_food)} visible -> {obs.nearby_food}",
        f"  - Stations: {len(obs.nearby_stations)} in line of sight "
        f"-> {obs.nearby_stations}",
        f"  - Threats/Obstacles: {len(obs.nearby_obstacles)} detected "
        f"-> {obs.nearby_obstacles[:3]}",
        f"  - Nearby Swarm Allies: {len(obs.nearby_allies)} in range",
        f"  - Scent Gradients: {obs.scent_directions}",
        "Environmental Text Intelligence (Token Synthesizer):",
    ]
    intel = synthesize_token_intel(obs.nearby_tokens)
    has_intel = False
    for category, items in intel.items():
        if items:
            has_intel = True
            lines.append(f"  - [{category.upper()}]: {', '.join(items)}")
    if not has_intel:
        lines.append("  - (No text notes discovered in local environment)")
    return "\n".join(lines)


def to_prompt(obs: AgentObservation, mode: str) -> str:
    """Format an observation into a structured Chain-of-Thought JSON prompt with
    embedded narrative briefing."""
    briefing = to_narrative_briefing(obs, mode)
    intel = synthesize_token_intel(obs.nearby_tokens)
    payload = {
        "mode": mode,
        "briefing": briefing,
        "agent": {
            "id": obs.agent_id,
            "pos": obs.pos,
            "charge_pct": round(obs.charge / MAX_CHARGE * 100, 1),
            "state": obs.state,
            "role": obs.role,
            "has_food": obs.has_food,
            "waypoint": obs.waypoint,
        },
        "allies": obs.nearby_allies,
        "environment": {
            "nearby_food": obs.nearby_food,
            "nearby_stations": obs.nearby_stations,
            "nearby_stigmergy_tokens": obs.nearby_tokens,
            "synthesized_intel": intel,
            "nearby_obstacles": obs.nearby_obstacles,
            "scents": obs.scent_directions,
        },
    }
    return json.dumps(payload, indent=2)


class LLMBackend:
    """Base class for multi-model LLM backends."""

    def decide(self, obs: AgentObservation, mode: str) -> LLMDecision:
        raise NotImplementedError


class MockLLMBackend(LLMBackend):
    """Deterministic, fast offline reasoner with specialized logic per paradigm."""

    def decide(self, obs: AgentObservation, mode: str) -> LLMDecision:
        # Priority 1 across all active modes: If carrying food, steer directly to
        # central nest depot
        # Charging takes absolute priority if the agent is actively seeking charge.
        if obs.has_food and obs.state != "seek":
            nest_target = NEST_TILES[0]
            return LLMDecision(
                thought=(
                    f"Agent #{obs.agent_id:03d} carrying harvest. "
                    "Navigating to nest depot."
                ),
                waypoint=nest_target,
                role="forager",
                action_type="RETURN_NEST",
            )

        # -------------------------------------------------------------
        # Mode 2: COARSE-FINE HYBRID (Strategic Cartographer)
        # -------------------------------------------------------------
        if mode == "coarse_fine":
            # Low charge macro-triage (strictly decentralized)
            if obs.charge < MAX_CHARGE * 0.35 or obs.state == "seek":
                best_station = select_best_station(obs)
                if best_station is not None:
                    return LLMDecision(
                        thought=(
                            "Strategic Cartographer: Station triage selected "
                            f"{best_station} based on charge "
                            f"({round(obs.charge / MAX_CHARGE * 100)}%) and queue load."
                        ),
                        waypoint=best_station,
                        role="scout",
                        action_type="COARSE_WAYPOINT",
                    )
                if obs.state == "seek":
                    return LLMDecision(
                        thought=(
                            f"Strategic Cartographer: Agent #{obs.agent_id:03d} "
                            "seeking charge with no station in sensing range. "
                            "Following local gradient to charger."
                        ),
                        waypoint=None,
                        role="scout",
                        action_type="SEEK_GRADIENT",
                    )

            # Priority A0: Food waypoint reached and found depleted
            if (
                obs.waypoint is not None
                and abs(obs.pos[0] - obs.waypoint[0])
                + abs(obs.pos[1] - obs.waypoint[1])
                <= 6
                and not obs.has_food
            ):
                food_at_pos = any(
                    t == obs.waypoint and amt > 0 for t, amt in obs.nearby_food
                )
                if not food_at_pos:
                    if obs.nearby_food:
                        food_pos = obs.nearby_food[obs.agent_id % len(obs.nearby_food)][
                            0
                        ]
                        return LLMDecision(
                            thought=(
                                f"Strategic Cartographer: Waypoint {obs.waypoint} "
                                f"depleted, but active food at {food_pos} visible. "
                                "Retargeting."
                            ),
                            waypoint=food_pos,
                            role="forager",
                            action_type="COARSE_WAYPOINT",
                        )
                    return LLMDecision(
                        thought=(
                            f"Strategic Cartographer: Target source at {obs.waypoint} "
                            "is depleted. Resuming frontier scout exploration."
                        ),
                        waypoint=None,
                        role="scout",
                        action_type="ABORT_DEPLETED",
                    )

            # Priority A: Food cluster macro-planning
            if obs.nearby_food:
                food_pos = obs.nearby_food[obs.agent_id % len(obs.nearby_food)][0]
                return LLMDecision(
                    thought=(
                        f"Strategic Cartographer: Food cluster detected at {food_pos}. "
                        "Setting coarse macro-waypoint."
                    ),
                    waypoint=food_pos,
                    role="forager",
                    action_type="COARSE_WAYPOINT",
                )

            # Priority A2: Active food trail detected (unless near a known depleted
            # marker with no food)
            has_depleted_note = any("[DEPLETED]" in t[1] for t in obs.nearby_tokens)
            if obs.sensed_food_trail and not (
                has_depleted_note and not obs.nearby_food
            ):
                return LLMDecision(
                    thought=(
                        "Strategic Cartographer: Active food pheromone trail "
                        "detected. Forager following trail to harvest source."
                    ),
                    waypoint=None,
                    role="forager",
                    action_type="TRAIL_FORAGE",
                )
            # Priority B: Obstacle flanking macro-planning
            if obs.nearby_obstacles:
                ox, oy = obs.nearby_obstacles[0]
                dx = 10 if obs.pos[0] >= ox else -10
                dy = 10 if obs.pos[1] >= oy else -10
                flank_target = (
                    max(3, min(GRID_SIZE - 4, obs.pos[0] + dx)),
                    max(3, min(GRID_SIZE - 4, obs.pos[1] + dy)),
                )
                return LLMDecision(
                    thought=(
                        f"Strategic Cartographer: Barrier detected at ({ox}, {oy}). "
                        f"Setting macro-flanking waypoint {flank_target}."
                    ),
                    waypoint=flank_target,
                    role="scout",
                    action_type="COARSE_WAYPOINT",
                )
            # Priority C: Explore quadrant with coldest scent
            best_dir = min(obs.scent_directions, key=obs.scent_directions.get)  # type: ignore[arg-type]
            dir_map = {"N": (0, -12), "S": (0, 12), "E": (12, 0), "W": (-12, 0)}
            dc, dr = dir_map[best_dir]
            target = (
                max(4, min(GRID_SIZE - 5, obs.pos[0] + dc)),
                max(4, min(GRID_SIZE - 5, obs.pos[1] + dr)),
            )
            return LLMDecision(
                thought=(
                    "Strategic Cartographer: Macro-planning frontier sweep toward "
                    f"coldest scent {best_dir} at sector {target}."
                ),
                waypoint=target,
                role="scout",
                action_type="COARSE_WAYPOINT",
            )

        # -------------------------------------------------------------
        # Mode 3: GLOBAL-CONTEXTUAL (Semantic Blackboard & Dynamic Roles)
        # -------------------------------------------------------------
        if mode == "global_contextual":
            # Priority 0: Low charge triage (strictly decentralized)
            if obs.charge < MAX_CHARGE * 0.35 or obs.state == "seek":
                best_station = select_best_station(obs)
                if best_station is not None:
                    is_visible = best_station in obs.nearby_stations
                    action_type = "STATION_APPROACH" if is_visible else "STATION_SEEK"
                    thought = (
                        f"Contextual Agent #{obs.agent_id:03d}: Charging depot "
                        f"selected at {best_station} "
                        f"(charge {round(obs.charge / MAX_CHARGE * 100)}%). "
                        "Approaching station."
                        if is_visible
                        else "Blackboard Note: Read station signpost. "
                        f"Heading to charging depot at {best_station}."
                    )
                    token_to_write = (
                        (obs.pos, f"[STATION] Charging depot at {best_station}")
                        if (
                            is_visible
                            and not any(
                                "[STATION]" in t[1]
                                for t in obs.nearby_tokens
                                if t[0] == obs.pos
                            )
                        )
                        else None
                    )
                    return LLMDecision(
                        thought=thought,
                        waypoint=best_station,
                        role="scout",
                        token_to_write=token_to_write,
                        action_type=action_type,
                    )
                if obs.state == "seek":
                    return LLMDecision(
                        thought=(
                            f"Contextual Agent #{obs.agent_id:03d}: Seeking charge "
                            "with no station signpost in range. "
                            "Following local gradient to station."
                        ),
                        waypoint=None,
                        role="scout",
                        action_type="SEEK_GRADIENT",
                    )

            # Priority A: Check if current waypoint reached or dead trail end was found
            # depleted
            is_depleted_waypoint = (
                obs.waypoint is not None
                and abs(obs.pos[0] - obs.waypoint[0])
                + abs(obs.pos[1] - obs.waypoint[1])
                <= 6
                and not obs.has_food
                and not any(t == obs.waypoint and amt > 0 for t, amt in obs.nearby_food)
            )
            is_depleted_trail_end = (
                not obs.has_food
                and obs.sensed_food_trail
                and not obs.has_outward_trail
                and not any(amt > 0 for _, amt in obs.nearby_food)
                and obs.nest_dist >= 15
            )

            if is_depleted_waypoint or is_depleted_trail_end:
                if obs.nearby_food:
                    food_pos = obs.nearby_food[obs.agent_id % len(obs.nearby_food)][0]
                    return LLMDecision(
                        thought=(
                            f"Forager at {obs.pos}: Target source depleted, but "
                            f"nearby active food at {food_pos} visible. Retargeting."
                        ),
                        waypoint=food_pos,
                        role="forager",
                        action_type="HARVEST_CALL",
                    )
                target = obs.waypoint if is_depleted_waypoint else obs.pos
                already_marked = any(
                    "[DEPLETED]" in text
                    and abs(pos[0] - obs.pos[0]) + abs(pos[1] - obs.pos[1]) <= 6
                    for pos, text in obs.nearby_tokens
                )
                if already_marked:
                    return LLMDecision(
                        thought=(
                            f"Forager at {obs.pos}: Sector already marked "
                            "[DEPLETED]. Dispersing as scout."
                        ),
                        waypoint=None,
                        role="scout",
                        action_type="ABORT_DEPLETED",
                    )
                depleted_msg = f"[DEPLETED] Food at {target} exhausted"
                return LLMDecision(
                    thought=(
                        f"Forager at {obs.pos}: Target source at {target} is "
                        "depleted. Announcing [DEPLETED]."
                    ),
                    waypoint=None,
                    role="scout",
                    token_to_write=(obs.pos, depleted_msg),
                    action_type="DEPLETED_MARK",
                )

            # Priority B: Read spatial blackboard notes
            if obs.nearby_tokens:
                depleted_coords: set[Tile] = set()
                for _, text in obs.nearby_tokens:
                    if "[DEPLETED]" in text:
                        m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
                        if m:
                            depleted_coords.add((int(m.group(1)), int(m.group(2))))

                # B1: Waypoint marked depleted -> Abort route immediately
                if obs.waypoint is not None and obs.waypoint in depleted_coords:
                    return LLMDecision(
                        thought=(
                            f"Blackboard Note: Waypoint {obs.waypoint} marked "
                            "depleted. Aborting route to scout."
                        ),
                        waypoint=None,
                        role="scout",
                        action_type="ABORT_DEPLETED",
                    )

                note_pos, note_text = obs.nearby_tokens[0]
                # B2: Hazard warning note -> Evacuate
                if "ALERT" in note_text or "HAZARD" in note_text:
                    dx = obs.pos[0] - note_pos[0]
                    dy = obs.pos[1] - note_pos[1]
                    detour_x = max(
                        2, min(GRID_SIZE - 3, obs.pos[0] + (1 if dx >= 0 else -1) * 8)
                    )
                    detour_y = max(
                        2, min(GRID_SIZE - 3, obs.pos[1] + (1 if dy >= 0 else -1) * 8)
                    )
                    first_line = note_text.splitlines()[0]
                    return LLMDecision(
                        thought=(
                            f"Blackboard Note: Read alert '{first_line}'. "
                            f"Evacuating sector to ({detour_x}, {detour_y})."
                        ),
                        waypoint=(detour_x, detour_y),
                        role=obs.role,
                        action_type="EVACUATE",
                    )
                # B3: Food or Guide note -> Converge if not sentry and not depleted
                food_leads = []
                for _, text in obs.nearby_tokens:
                    if "[FOOD]" in text or "[GUIDE]" in text:
                        coord_m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
                        tgt = (
                            (int(coord_m.group(1)), int(coord_m.group(2)))
                            if coord_m
                            else note_pos
                        )
                        if tgt not in depleted_coords and tgt not in food_leads:
                            food_leads.append(tgt)
                if food_leads and obs.role != "sentry":
                    target = food_leads[obs.agent_id % len(food_leads)]
                    last_line = note_text.splitlines()[-1]
                    return LLMDecision(
                        thought=(
                            f"Blackboard Note: Read announcement '{last_line}'. "
                            f"Converging on food source at {target}."
                        ),
                        waypoint=target,
                        role="forager",
                        action_type="HARVEST_CALL",
                    )

            # Priority C: Write to spatial blackboard
            # C1: Sentry spots barrier -> Broadcast warning
            if obs.role == "sentry" and obs.nearby_obstacles:
                warning = f"[ALERT] Barrier sector at {obs.nearby_obstacles[0]}"
                return LLMDecision(
                    thought=(
                        f"Sentry at {obs.pos}: Broadcasting barrier warning "
                        "onto spatial blackboard."
                    ),
                    role="sentry",
                    token_to_write=(obs.pos, warning),
                    action_type="STIGMERGY_MARK",
                )
            # C2: Scout spots station in visual range -> Broadcast station signpost if
            # not already marked
            if (
                obs.role == "scout"
                and obs.nearby_stations
                and not any(
                    "[STATION]" in t[1] for t in obs.nearby_tokens if t[0] == obs.pos
                )
            ):
                st_tile = obs.nearby_stations[0]
                return LLMDecision(
                    thought=(
                        f"Scout at {obs.pos}: Spotted charging depot at {st_tile}. "
                        "Broadcasting signpost."
                    ),
                    role="scout",
                    token_to_write=(obs.pos, f"[STATION] Charging depot at {st_tile}"),
                    action_type="STIGMERGY_MARK",
                )
            # C3: Scout discovers rich food cluster -> Broadcast directional guide
            # signpost
            if obs.role == "scout" and obs.nearby_food and obs.nearby_food[0][1] >= 2:
                food_tile, amt = obs.nearby_food[obs.agent_id % len(obs.nearby_food)]
                d = abs(food_tile[0] - obs.pos[0]) + abs(food_tile[1] - obs.pos[1])
                dx = food_tile[0] - obs.pos[0]
                dy = food_tile[1] - obs.pos[1]
                ew = "East" if dx > 0 else "West" if dx < 0 else ""
                ns = "South" if dy > 0 else "North" if dy < 0 else ""
                dir_str = f"{ns}-{ew}" if (ns and ew) else (ns or ew or "Here")
                announcement = (
                    f"[GUIDE] Food ~{d} steps {dir_str} at {food_tile} [{amt}u]"
                )
                return LLMDecision(
                    thought=(
                        f"Scout at {obs.pos}: Discovered harvest of {amt}u at "
                        f"{food_tile} (~{d} steps {dir_str}). Leaving signpost."
                    ),
                    waypoint=food_tile,
                    role="forager",
                    token_to_write=(obs.pos, announcement),
                    action_type="STIGMERGY_MARK",
                )

            # Priority D: Dynamic role arbitration based on sector density
            has_depleted_note = any("[DEPLETED]" in t[1] for t in obs.nearby_tokens)
            is_dead_trail = (
                obs.sensed_food_trail
                and not obs.has_outward_trail
                and not obs.nearby_food
                and obs.nest_dist >= 15
            )
            if obs.nearby_obstacles:
                new_role = "sentry"
            elif obs.nearby_food or (
                obs.sensed_food_trail
                and not is_dead_trail
                and not (has_depleted_note and not obs.nearby_food)
            ):
                new_role = "forager"
            else:
                new_role = "scout"

            return LLMDecision(
                thought=(
                    "Role Arbitration: Sector density analysis switching agent "
                    f"role to {new_role}."
                ),
                waypoint=None,
                role=new_role,
                action_type="TRAIL_FORAGE"
                if (new_role == "forager" and not obs.nearby_food)
                else "ROLE_SWITCH",
            )

        # -------------------------------------------------------------
        # Mode 4: PURE-LLM (Direct Tactical Commander)
        # -------------------------------------------------------------
        if mode == "pure_llm":
            # Tactical Priority A: Low battery alert (strictly decentralized)
            if obs.charge < MAX_CHARGE * 0.35 or obs.state == "seek":
                best_station = select_best_station(obs)
                if best_station is not None:
                    return LLMDecision(
                        thought=(
                            f"Tactical Commander #{obs.agent_id:03d}: Low charge "
                            f"alert ({round(obs.charge / MAX_CHARGE * 100)}%). "
                            f"Navigating to charging zone at {best_station}."
                        ),
                        waypoint=best_station,
                        role="scout",
                        action_type="TACTICAL_CHARGE",
                    )
                return LLMDecision(
                    thought=(
                        f"Tactical Commander #{obs.agent_id:03d}: Low charge "
                        f"alert ({round(obs.charge / MAX_CHARGE * 100)}%), but no "
                        "charging zone visible or signpost in sector. "
                        "Executing tactical search."
                    ),
                    waypoint=None,
                    role="scout",
                    action_type="TACTICAL_SEARCH",
                )
            # Tactical Priority B0: Depleted waypoint or dead trail end check
            is_depleted_waypoint = (
                obs.waypoint is not None
                and abs(obs.pos[0] - obs.waypoint[0])
                + abs(obs.pos[1] - obs.waypoint[1])
                <= 6
                and not obs.has_food
                and not any(t == obs.waypoint and amt > 0 for t, amt in obs.nearby_food)
            )
            is_depleted_trail_end = (
                not obs.has_food
                and obs.sensed_food_trail
                and not obs.has_outward_trail
                and not any(amt > 0 for _, amt in obs.nearby_food)
                and obs.nest_dist >= 15
            )
            if is_depleted_waypoint or is_depleted_trail_end:
                if obs.nearby_food:
                    target_food = obs.nearby_food[obs.agent_id % len(obs.nearby_food)][
                        0
                    ]
                    return LLMDecision(
                        thought=(
                            f"Tactical Commander #{obs.agent_id:03d}: Target food "
                            f"exhausted, but active food locked at {target_food}. "
                            "Retargeting approach."
                        ),
                        waypoint=target_food,
                        role="forager",
                        action_type="TACTICAL_MOVE",
                    )
                target = obs.waypoint if is_depleted_waypoint else obs.pos
                already_marked = any(
                    "[DEPLETED]" in text
                    and abs(pos[0] - obs.pos[0]) + abs(pos[1] - obs.pos[1]) <= 6
                    for pos, text in obs.nearby_tokens
                )
                if already_marked:
                    return LLMDecision(
                        thought=(
                            f"Tactical Commander #{obs.agent_id:03d}: Food depleted "
                            f"note detected at {obs.pos}. Switching to scout sweep."
                        ),
                        waypoint=None,
                        role="scout",
                        action_type="ABORT_DEPLETED",
                    )
                return LLMDecision(
                    thought=(
                        f"Tactical Commander #{obs.agent_id:03d}: Target food at "
                        f"{target} exhausted. Announcing [DEPLETED]."
                    ),
                    waypoint=None,
                    role="scout",
                    token_to_write=(obs.pos, f"[DEPLETED] Food at {target} exhausted"),
                    action_type="DEPLETED_MARK",
                )
            # Tactical Priority B: Direct harvest targeting
            if obs.nearby_food:
                target_food = obs.nearby_food[obs.agent_id % len(obs.nearby_food)][0]
                return LLMDecision(
                    thought=(
                        f"Tactical Commander #{obs.agent_id:03d}: Target food "
                        f"cluster locked at {target_food}. "
                        "Initiating harvest approach."
                    ),
                    waypoint=target_food,
                    role="forager",
                    action_type="TACTICAL_MOVE",
                )
            # Tactical Priority B2: Follow food trail if sensed
            has_depleted_note = any("[DEPLETED]" in t[1] for t in obs.nearby_tokens)
            is_dead_trail = (
                obs.sensed_food_trail
                and not obs.has_outward_trail
                and not obs.nearby_food
                and obs.nest_dist >= 15
            )
            if (
                obs.sensed_food_trail
                and not is_dead_trail
                and not (has_depleted_note and not obs.nearby_food)
            ):
                return LLMDecision(
                    thought=(
                        f"Tactical Commander #{obs.agent_id:03d}: Active food "
                        "pheromone trail detected. Forager advancing along trail."
                    ),
                    waypoint=None,
                    role="forager",
                    action_type="TRAIL_FORAGE",
                )
            # Tactical Priority C: Obstacle evasion
            if obs.nearby_obstacles:
                ox, oy = obs.nearby_obstacles[0]
                evade_x = max(
                    2, min(GRID_SIZE - 3, obs.pos[0] + (2 if obs.pos[0] >= ox else -2))
                )
                evade_y = max(
                    2, min(GRID_SIZE - 3, obs.pos[1] + (2 if obs.pos[1] >= oy else -2))
                )
                return LLMDecision(
                    thought=(
                        f"Tactical Commander #{obs.agent_id:03d}: Tactical "
                        f"obstacle avoidance maneuver away from ({ox}, {oy}) to "
                        f"({evade_x}, {evade_y})."
                    ),
                    waypoint=(evade_x, evade_y),
                    action_type="TACTICAL_EVADE",
                )
            # Tactical Priority D: Autonomous search sweep
            sweep_dir = min(obs.scent_directions, key=obs.scent_directions.get)  # type: ignore[arg-type]
            dir_map = {"N": (0, -8), "S": (0, 8), "E": (8, 0), "W": (-8, 0)}
            dc, dr = dir_map[sweep_dir]
            sweep_target = (
                max(3, min(GRID_SIZE - 4, obs.pos[0] + dc)),
                max(3, min(GRID_SIZE - 4, obs.pos[1] + dr)),
            )
            return LLMDecision(
                thought=(
                    f"Tactical Commander #{obs.agent_id:03d}: Autonomous search "
                    f"sweep in direction {sweep_dir} toward {sweep_target}."
                ),
                waypoint=sweep_target,
                action_type="TACTICAL_SWEEP",
            )

        # Default fallback
        return LLMDecision(
            thought=f"Maintaining default autonomous operation at {obs.pos}.",
            action_type="MAINTAIN",
        )


class OllamaBackend(LLMBackend):
    """Local Ollama bridge (Qwen 2.5 / Llama 3) with robust fallback."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        url: str = "http://localhost:11434",
        timeout: float = 1.0,
        fallback: LLMBackend | None = None,
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.fallback = fallback if fallback is not None else MockLLMBackend()

    def decide(self, obs: AgentObservation, mode: str) -> LLMDecision:
        prompt = to_prompt(obs, mode)
        req_data = json.dumps(
            {
                "model": self.model,
                "prompt": (
                    "You are an AI swarm agent coordinator. Return JSON only with "
                    "fields thought, waypoint, role, action_type.\n"
                    f"{prompt}"
                ),
                "stream": False,
                "format": "json",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response_text = data.get("response", "{}")
                parsed = json.loads(response_text)
                waypoint = (
                    tuple(parsed["waypoint"])
                    if parsed.get("waypoint") is not None
                    else None
                )
                return LLMDecision(
                    thought=parsed.get("thought", "Ollama LLM decision."),
                    waypoint=waypoint,  # type: ignore[arg-type]
                    role=parsed.get("role"),
                    action_type=parsed.get("action_type", "OLLAMA_DECISION"),
                )
        except urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError:
            return self.fallback.decide(obs, mode)


class SwarmOrchestrator:
    """Manages high-level hybrid swarm paradigms across the simulation."""

    MODES = ("classical", "coarse_fine", "global_contextual", "pure_llm")

    def __init__(
        self,
        backend: LLMBackend | None = None,
        mode: str = "classical",
        interval: float = 1.0,
    ) -> None:
        self.backend: LLMBackend = backend if backend is not None else MockLLMBackend()
        self.mode = mode if mode in self.MODES else "classical"
        self.interval = interval
        self.timer: float = 0.0
        self.latest_thought: str = ""
        self.stats = {
            "decisions": 0,
            "waypoints_assigned": 0,
            "roles_switched": 0,
            "tokens_written": 0,
        }

    def set_mode(self, new_mode: str) -> None:
        if new_mode in self.MODES:
            self.mode = new_mode

    def step(
        self,
        world: World,
        agents: list[Agent],
        trail: Trail,
        now: float,
        dt: float,
        batch_size: int = 15,
    ) -> None:
        """Run one step of hybrid orchestration."""
        if self.mode == "classical":
            return

        self.timer += dt
        if self.timer < self.interval:
            return
        self.timer -= self.interval

        for agent in agents:
            agent.llm_mode = self.mode

        # Pick agents due for macro-planning (least recently planned first)
        active_pool = [a for a in agents if a.state in ("walk", "seek")]
        sample = sorted(active_pool, key=lambda a: a.last_llm_time)[:batch_size]

        for agent in sample:
            obs = extract_observation(agent, world, trail, now)
            decision = self.backend.decide(obs, self.mode)
            self.stats["decisions"] += 1

            agent.llm_thought = decision.thought
            agent.last_llm_time = now
            self.latest_thought = (
                f"#{agent.id:03d} [{decision.action_type}]: {decision.thought}"
            )

            if decision.waypoint is not None:
                if (
                    agent.state == "seek" or agent.charge < LOW_CHARGE
                ) and not world.is_station(decision.waypoint):
                    agent.waypoint = None
                else:
                    agent.waypoint = decision.waypoint
                    if world.is_station(decision.waypoint) and agent.state != "seek":
                        agent.state = "seek"
                        agent.food_trail_valid = False
                        agent.age = 0
                        world.press((agent.col, agent.row), 1)
                    self.stats["waypoints_assigned"] += 1
            elif (
                agent.state == "seek"
                or agent.charge < LOW_CHARGE
                or decision.action_type
                in (
                    "SEEK_GRADIENT",
                    "TACTICAL_SEARCH",
                    "ABORT_DEPLETED",
                    "DEPLETED_MARK",
                    "TRAIL_FORAGE",
                )
            ):
                agent.waypoint = None

            if decision.role is not None and decision.role != getattr(
                agent, "role", None
            ):
                agent.role = decision.role
                self.stats["roles_switched"] += 1

            if decision.token_to_write is not None:
                world.append_token(
                    decision.token_to_write[0],
                    decision.token_to_write[1],
                    now,
                )
                if (
                    self.mode in ("global_contextual", "pure_llm")
                    and "[DEPLETED]" in decision.token_to_write[1]
                ):
                    if not has_active_branch_outward(
                        world, (agent.col, agent.row), now
                    ):
                        already_erasing = any(
                            other is not agent
                            and getattr(other, "erasing_trail", False)
                            and abs(other.col - agent.col) + abs(other.row - agent.row)
                            <= 25
                            for other in world.occ.values()
                        )
                        if not already_erasing:
                            world.food_trail.clear_tile(agent.col, agent.row)
                self.stats["tokens_written"] += 1

            if self.mode in ("global_contextual", "pure_llm"):
                has_depleted_note = any(
                    "[DEPLETED]" in text
                    and abs(pos[0] - agent.col) + abs(pos[1] - agent.row) <= 6
                    for pos, text in world.active_tokens(now).items()
                )
                nearby_has_food = any(
                    amt > 0
                    for t, amt in world.food.items()
                    if abs(t[0] - agent.col) + abs(t[1] - agent.row) <= 6
                )
                if (
                    decision.action_type in ("DEPLETED_MARK", "ABORT_DEPLETED")
                    or (has_depleted_note and not nearby_has_food)
                ) and not agent.has_food:
                    agent.avoided_collision_from_trail = False
                    if not has_active_branch_outward(
                        world, (agent.col, agent.row), now
                    ):
                        already_erasing = any(
                            other is not agent
                            and getattr(other, "erasing_trail", False)
                            and abs(other.col - agent.col) + abs(other.row - agent.row)
                            <= 25
                            for other in world.occ.values()
                        )
                        if not already_erasing:
                            world.food_trail.clear_tile(agent.col, agent.row)
                            agent.erasing_trail = True
                        else:
                            agent.erasing_trail = False
                            agent.role = "forager"
                            agent.waypoint = None


@dataclass
class SwarmInstance:
    mode: str
    name: str
    color: tuple[int, int, int]
    world: World
    agents: list[Agent]
    trail: Trail
    food_trail: Trail
    orchestrator: SwarmOrchestrator
    step_timer: float = 0.0


class ParallelSwarmBench:
    """Synchronously executes 4 parallel swarm simulation instances with identical
    initial conditions."""

    MODES: tuple[tuple[str, str, tuple[int, int, int]], ...] = (
        ("classical", "CLASSICAL", (100, 255, 200)),
        ("coarse_fine", "COARSE-FINE", (255, 220, 80)),
        ("global_contextual", "GLOBAL-CONTEXTUAL", (240, 140, 255)),
        ("pure_llm", "PURE-LLM", (180, 160, 255)),
    )

    def __init__(
        self,
        backend: LLMBackend | None = None,
        seed: int = 42,
        static_obstacles: set[Tile] | None = None,
        stations: list[Tile] | None = None,
        nav: NavMap | None = None,
        nest_tiles: list[Tile] | None = None,
        nest_nav: NavMap | None = None,
        initial_food: dict[Tile, int] | None = None,
        initial_barriers: list[MovingBarrier] | None = None,
        starts: list[Tile] | None = None,
    ) -> None:
        self.active_idx: int = 0
        self.quad_view: bool = False
        self.backend = backend if backend is not None else MockLLMBackend()
        self.instances: list[SwarmInstance] = []

        rng = random.Random(seed)
        obs = (
            static_obstacles
            if static_obstacles is not None
            else make_static_obstacles()
        )
        st_tiles = stations if stations is not None else make_stations()
        n_nav = nav if nav is not None else build_nav(st_tiles, obstacles=obs)
        n_tiles = nest_tiles if nest_tiles is not None else make_nest()
        nst_nav = (
            nest_nav if nest_nav is not None else build_nav(n_tiles, obstacles=obs)
        )
        f_init = (
            initial_food
            if initial_food is not None
            else make_food_clusters(seed=seed, obstacles=obs)
        )
        b_init = (
            initial_barriers if initial_barriers is not None else make_moving_barriers()
        )

        b_tiles = {c for b in b_init for c in b.coords}
        nest_set = set(n_tiles)
        if starts is None:
            free_starts = [
                (t % GRID_SIZE, t // GRID_SIZE)
                for t in range(GRID_SIZE * GRID_SIZE)
                if (t % GRID_SIZE, t // GRID_SIZE) not in obs
                and (t % GRID_SIZE, t // GRID_SIZE) not in nest_set
                and (t % GRID_SIZE, t // GRID_SIZE) not in b_tiles
            ]
            starts_chosen = rng.sample(free_starts, AGENT_COUNT)
        else:
            starts_chosen = starts

        # Build 4 parallel swarm instances with identical topology and starting state
        for mode, name, color in self.MODES:
            agents = [Agent(i, col, row) for i, (col, row) in enumerate(starts_chosen)]
            for a in agents:
                a.llm_mode = mode
            barriers = [
                MovingBarrier(
                    coords=list(b.coords),
                    axis=b.axis,
                    direction=b.direction,
                    min_pos=b.min_pos,
                    max_pos=b.max_pos,
                    interval=b.interval,
                    expiry=b.expiry,
                )
                for b in b_init
            ]
            trail = Trail()
            food_trail = Trail(accumulate=True)
            world = World(
                n_nav,
                agents,
                nest_nav=nst_nav,
                nest_set=nest_set,
                food=dict(f_init),
                food_trail=food_trail,
                static_obstacles=obs,
                moving_barriers=barriers,
                stations=st_tiles,
            )
            orch = SwarmOrchestrator(backend=self.backend, mode=mode)
            self.instances.append(
                SwarmInstance(
                    mode=mode,
                    name=name,
                    color=color,
                    world=world,
                    agents=agents,
                    trail=trail,
                    food_trail=food_trail,
                    orchestrator=orch,
                )
            )

    @property
    def active_instance(self) -> SwarmInstance:
        return self.instances[self.active_idx]

    def set_active_idx(self, idx: int) -> None:
        if 0 <= idx < len(self.instances):
            self.active_idx = idx
            self.quad_view = False

    def toggle_quad_view(self) -> bool:
        self.quad_view = not self.quad_view
        return self.quad_view

    def step(self, now: float, dt: float) -> None:
        """Step all parallel instances synchronously."""
        for inst in self.instances:
            inst.orchestrator.step(inst.world, inst.agents, inst.trail, now, dt)
            inst.step_timer = advance(
                inst.world, inst.agents, inst.trail, now, dt, inst.step_timer
            )

    def spawn_synchronized_food(self, seed: int | None = None) -> None:
        """Spawn the exact same food in all 4 parallel worlds to preserve fair
        benchmark conditions."""
        new_food = make_food_clusters(
            cluster_count=1 if seed is not None else random.randint(1, 2),
            seed=seed,
            obstacles=self.instances[0].world.static_obstacles,
        )
        for inst in self.instances:
            for tile, amt in new_food.items():
                inst.world.food[tile] = inst.world.food.get(tile, 0) + amt
