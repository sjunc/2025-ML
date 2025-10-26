# agents/planner/submission.py
"""
Planner agent using fast A* with bucketed priority queue (Dial-like / multi-bucket).
This is a practical implementation that brings the "sorting-barrier-breaking" idea
to a grid-based POMDP environment by removing heavy heap sorting via bucketed queues.
Returns an action index compatible with evaluation_local.py's actions_map logic.
"""

import math
import numpy as np
from collections import defaultdict

# Bucket scale: tuning parameter. Smaller -> more fine-grained buckets (slower bucket iteration),
# larger -> coarser grouping. Tune per environment.
BUCKET_SCALE = 1.0

# Default discrete actions_map fallback (if one not passed)
_default_actions_map = {
    0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30],
    6: [-40, -30], 7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30],
    12: [20, -30], 13: [20, -18], 14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30],
    18: [80, -30], 19: [80, -18], 20: [80, -6], 21: [80, 6], 22: [80, 18], 23: [80, 30],
    24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6], 28: [140, 18], 29: [140, 30],
    30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18], 35: [200, 30]
}

def nearest_action_index(force, angle, actions_map=None):
    if actions_map is None:
        actions_map = _default_actions_map
    best_i = 0
    best_dist = float('inf')
    for i, fa in actions_map.items():
        df = (fa[0] - force)
        da = (fa[1] - angle)
        d = df*df + da*da
        if d < best_dist:
            best_dist = d
            best_i = i
    return best_i

class BucketedPQ:
    def __init__(self, scale=BUCKET_SCALE):
        self.scale = scale
        self.buckets = defaultdict(list)
        self.min_bucket = None
        self.size = 0

    def _bidx(self, key_float):
        return int(math.floor(key_float / self.scale))

    def push(self, key_float, item):
        bi = self._bidx(key_float)
        self.buckets[bi].append((key_float, item))
        if self.min_bucket is None or bi < self.min_bucket:
            self.min_bucket = bi
        self.size += 1

    def pop(self):
        if self.size == 0:
            raise IndexError("pop from empty BucketedPQ")
        bi = self.min_bucket
        # find first non-empty bucket
        while bi not in self.buckets or not self.buckets[bi]:
            bi += 1
            self.min_bucket = bi
        bucket = self.buckets[bi]
        # pick minimal by linear scan inside bucket (buckets are small)
        best_idx = 0
        best_key = bucket[0][0]
        for idx in range(1, len(bucket)):
            if bucket[idx][0] < best_key:
                best_key = bucket[idx][0]
                best_idx = idx
        key, item = bucket.pop(best_idx)
        self.size -= 1
        return key, item

    def empty(self):
        return self.size == 0

def improved_astar(start_xy, goal_xy, obstacle_set, grid_size=1.0, max_nodes=5000, bucket_scale=BUCKET_SCALE):
    """
    start_xy, goal_xy: continuous/binned coordinates in same basis as obstacle_set (here we use grid coords)
    obstacle_set: set of integer grid coordinates (x,y) representing blocked cells (in same basis)
    grid_size: mapping scale (we assume obs grid is unit)
    returns: list of continuous coordinates for path or [] if fail
    """
    sx, sy = int(round(start_xy[0])), int(round(start_xy[1]))
    gx, gy = int(round(goal_xy[0])), int(round(goal_xy[1]))

    def heuristic(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    start = (sx, sy)
    goal = (gx, gy)

    openp = BucketedPQ(scale=bucket_scale)
    g_score = {start: 0.0}
    openp.push(heuristic(start, goal), start)
    came_from = {}
    visited = set()
    directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    nodes = 0

    while not openp.empty() and nodes < max_nodes:
        nodes += 1
        fcur, current = openp.pop()
        if current in visited:
            continue
        visited.add(current)

        # goal check: within 0.9 cell
        if heuristic(current, goal) <= 0.9:
            # reconstruct path
            path_nodes = [current]
            while path_nodes[-1] in came_from:
                path_nodes.append(came_from[path_nodes[-1]])
            path_nodes.reverse()
            # convert to continuous coords (center)
            return [((x * grid_size + 0.5 * grid_size), (y * grid_size + 0.5 * grid_size)) for (x,y) in path_nodes]

        for dx, dy in directions:
            nb = (current[0] + dx, current[1] + dy)
            if nb in obstacle_set:
                continue
            tentative_g = g_score[current] + math.hypot(dx, dy)
            if nb not in g_score or tentative_g < g_score[nb]:
                g_score[nb] = tentative_g
                fnb = tentative_g + heuristic(nb, goal)
                openp.push(fnb, nb)
                came_from[nb] = current

    return []

def parse_obs_to_local_grid(obs):
    """
    Parse the 25x25 local observation and return:
      - start (continuous coords) assumed as (0,0)
      - list of goal cells in same grid coords (integers)
      - obstacle_set: set of ints coords (x,y) representing walls
    We treat center cell (12,12) as agent's position (0,0).
    """
    arr = np.array(obs)
    if arr.size == 25*25:
        arr = arr.reshape((25,25))
    else:
        # try reshape or raise
        try:
            arr = arr.reshape((25,25))
        except:
            raise ValueError("Unexpected obs shape")

    center = (12, 12)
    obstacle_set = set()
    goal_cells = []
    for i in range(25):
        for j in range(25):
            val = int(arr[i, j])
            gx = j - center[1]
            gy = i - center[0]
            # Using integer grid coordinates offset by center
            if val == 6:
                obstacle_set.add((gx, gy))
            elif val == 7:
                goal_cells.append((gx, gy))
    # start is (0,0)
    start = (0.0, 0.0)
    return start, goal_cells, obstacle_set

def choose_action(obs_flat, actions_map=None):
    """
    Public interface used by evaluation_local.get_join_actions.
    obs_flat: flattened (25*25) observation
    actions_map: dict mapping action indices -> [force, angle]
    returns: action_index
    """
    if actions_map is None:
        actions_map = _default_actions_map

    try:
        start, goal_cells, obstacles = parse_obs_to_local_grid(obs_flat)
    except Exception as e:
        # if parse fails, return random
        return np.random.choice(list(actions_map.keys()))

    # If no visible goal, fallback to heuristic front move
    if not goal_cells:
        # try to move forward (angle 0) with medium force
        force = 150.0
        angle = 0.0
        return nearest_action_index(force, angle, actions_map)

    # choose nearest goal cell
    goal_cell = min(goal_cells, key=lambda c: math.hypot(c[0], c[1]))
    goal_xy = (goal_cell[0], goal_cell[1])

    # Run improved_astar in the local grid coord basis
    path = improved_astar(start_xy=start, goal_xy=goal_xy, obstacle_set=obstacles, grid_size=1.0, max_nodes=8000, bucket_scale=BUCKET_SCALE)

    if not path:
        # fallback greedy towards goal
        dx = goal_xy[0] - start[0]
        dy = goal_xy[1] - start[1]
        angle = math.degrees(math.atan2(dy, dx))
        force = 150.0
        return nearest_action_index(force, angle, actions_map)

    # pick next point (after start)
    if len(path) >= 2:
        nextpt = path[1]
    else:
        nextpt = path[0]
    dx = nextpt[0] - start[0]
    dy = nextpt[1] - start[1]
    angle = math.degrees(math.atan2(dy, dx))
    dist = math.hypot(dx, dy)
    if dist < 0.5:
        force = 50.0
    elif dist < 1.5:
        force = 120.0
    else:
        force = 200.0
    return nearest_action_index(force, angle, actions_map)
