"""Simulation constants. Distances are map units, time is ticks."""

DT = 1.0

# Vessels
MAX_SPEED = 2.0
MAX_ACCEL = 0.2
CRUISE_FRACTION = 0.9  # share of max speed used for route time estimates
ARRIVE_RADIUS = 30.0  # convoy centre within this of a node counts as arrived
NEAR_MISS = 4.0  # vessels closer than this count as a near miss
HOLD_WIDTH = 100.0  # lane width assumed while holding at a node or port

# Ports
SERVICE_TICKS = 40  # time a convoy spends loading or unloading at a berth

# Radio
RADIO_RANGE = 260.0
RADIO_LOSS = 0.05
RADIO_BANDWIDTH = 96  # transmissions that get through per tick
RADIO_NOISE = 0.03  # position-estimate error per unit of distance
NOTICE_EVERY = 20  # ticks between port stations repeating navigation notices

# LLM policy authors
LLM_LATENCY = 20.0  # sim ticks before a requested policy takes effect
LLM_BUDGET = 20  # maximum policy requests per author per run
