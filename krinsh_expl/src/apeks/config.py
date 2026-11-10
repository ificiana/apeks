GRID_SIZE = 100  # tiles per side
TILE_SIZE = 50  # px per tile
WORLD_SIZE = GRID_SIZE * TILE_SIZE
CAM_SPEED = 1000  # px/s

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
AGENT_COUNT = 100
SPEED = 8.0  # tiles/s, constant
MASS = 1.0  # kg
STEP_INTERVAL = 1 / SPEED  # s per one-tile move
MOVE_COST = round(MASS * SPEED**2)  # U per tile (m*v^2)
MAX_CHARGE = 100000  # U
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
AGENT_COLOR = (220, 60, 60)
CHARGE_RATE = 10000  # U/s while on a station
LOW_CHARGE = MAX_CHARGE * 0.25  # below this the agent heads for a station
MARK_VALUE = 100  # value an agent writes to the tile it stands on
MARK_DECAY = 0.93  # per-second multiplier on written values
WALK_NOISE = 0.5  # random jitter (in value units) breaking near-ties
MARK_TINT = (185, 215, 255)  # cool ice-blue tile tint for exploration trail
PUSH_REACH = 5  # sensing radius in tiles
PUSH_FLOOR_SEEK = 40  # a seeker needs this much slack (moves) to be pushed
PUSH_FLOOR_WALK = 3
URGENT_SLACK = 20  # seekers below this slack jump the queue
PRESSURE_LIMIT = 4  # walkers avoid tiles with this many seekers in reach
SEEK_RING = (255, 200, 0)
CHARGE_RING = (60, 220, 90)

STATION_COUNT = 20
STATION_COLOR = (190, 235, 200)
STATION_DOT = (120, 255, 160)
ARROW_COLOR = (30, 80, 160)
TEXT_COLOR = (40, 40, 40)
VALUE_COLOR = (40, 90, 160)

BP_BG = (20, 60, 120)  # blueprint blue
BP_LINE = (200, 225, 255)  # pale chalk-blue lines
MINIMAP_SIZE = 100  # px

NEST_SIZE = 2
NEST_TILES = tuple((49 + c, 49 + r) for r in range(NEST_SIZE) for c in range(NEST_SIZE))
FOOD_CLUSTER_COUNT = 6
FOOD_PER_CLUSTER = 35
FOOD_AMOUNT_MIN = 20
FOOD_AMOUNT_MAX = 50
NEST_COLOR = (70, 130, 190)
NEST_DOT = (100, 180, 255)
FOOD_COLOR = (245, 180, 30)
FOOD_RING = (255, 215, 0)
FOOD_TINT = (255, 175, 0)  # vibrant golden amber tile tint for food trail
FOOD_DOT = (255, 200, 50)

OBSTACLE_COLOR = (70, 80, 95)
OBSTACLE_DOT = (80, 95, 115)
BARRIER_COLOR = (225, 125, 45)
BARRIER_DOT = (255, 150, 40)
HAZARD_COLOR = (185, 75, 75)
HAZARD_DOT = (220, 85, 85)
TOKEN_BADGE = (165, 95, 245)
TOKEN_TTL = 30.0
OBSTACLE_DRIFT_INTERVAL = 0.8
BARRIER_LIFETIME_MIN = 20.0
BARRIER_LIFETIME_MAX = 45.0
BARRIER_SPAWN_INTERVAL_MIN = 8.0
BARRIER_SPAWN_INTERVAL_MAX = 16.0
MAX_MOVING_BARRIERS = 4
HAZARD_SPAWN_INTERVAL_MIN = 5.0
HAZARD_SPAWN_INTERVAL_MAX = 10.0
HAZARD_LIFETIME = 20.0
HAZARD_LIFETIME_MIN = 15.0
HAZARD_LIFETIME_MAX = 35.0

HAZARD_SHAPES = (
    ((0, 0),),  # 1-tile dot
    ((0, 0), (1, 0)),  # 2-tile horizontal bar
    ((0, 0), (0, 1)),  # 2-tile vertical bar
    ((0, 0), (1, 0), (2, 0)),  # 3-tile bar H
    ((0, 0), (0, 1), (0, 2)),  # 3-tile bar V
    ((0, 0), (1, 0), (0, 1)),  # 3-tile corner
    ((0, 0), (1, 0), (0, 1), (1, 1)),  # 2x2 box
    ((0, 0), (0, 1), (0, 2), (1, 2)),  # L-shape
    ((0, 0), (1, 0), (2, 0), (1, 1)),  # T-shape
    ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)),  # Cross
)
