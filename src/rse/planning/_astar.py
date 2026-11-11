import numpy as np
import heapq


def inflate_grid(occ_grid, inflation_radius_m, ds):
    """
    Returns a copy of occ_grid where every free cell (==0.0) whose Euclidean
    distance to ANY obstacle cell (==1.0) OR to the grid boundary is
    <= inflation_radius_m is marked as 0.5. Original 1.0 cells stay 1.0.

    Treating the grid boundary as obstacle is physically correct: the robot's
    workspace ends at the grid edge, so cells within r of the edge must carry
    the same safety penalty as cells within r of an interior wall.
    """
    H, W = occ_grid.shape
    radius_cells = int(np.ceil(inflation_radius_m / ds))

    # Pad with obstacle cells so the boundary acts as a wall during inflation.
    padded = np.pad(occ_grid, radius_cells, mode='constant', constant_values=1.0)

    # Build disk-shaped offset list (excluding center).
    offsets = []
    for di in range(-radius_cells, radius_cells + 1):
        for dj in range(-radius_cells, radius_cells + 1):
            if di == 0 and dj == 0:
                continue
            if np.hypot(di, dj) * ds <= inflation_radius_m:
                offsets.append((di, dj))

    # Two-buffer pass: read from `padded`, write 0.5 into `inflated`.
    inflated = padded.copy()
    Hp, Wp = padded.shape
    for i in range(Hp):
        for j in range(Wp):
            if padded[i, j] != 1.0:
                continue
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < Hp and 0 <= nj < Wp and inflated[ni, nj] == 0.0:
                    inflated[ni, nj] = 0.5

    # Crop the padding back off.
    return inflated[radius_cells:radius_cells + H, radius_cells:radius_cells + W]


class RoboticPlanner:
    def __init__(self, resolution=0.1, radius=0.2, grid_dims=(30, 40), obstacle_penalty=8.0):
        self.ds = resolution
        self.r = radius
        self.c_obs = obstacle_penalty
        self.grid_w, self.grid_h = grid_dims

        # 8-connected motion set: (dx, dy, cost)
        self.motions = [
            (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2))
        ]

    def wtg(self, x):
        """World-to-Grid: map continuous world coordinate to discrete grid index."""
        ix = int(np.floor(x[0] / self.ds + 0.5)) + (self.grid_w // 2)
        iy = int(np.floor(x[1] / self.ds + 0.5)) + (self.grid_h // 2)
        return np.array([ix, iy])

    def gtw(self, i):
        """Grid-to-World: return the world-coordinate center of the cell."""
        wx = (i[0] - (self.grid_w // 2)) * self.ds 
        wy = (i[1] - (self.grid_h // 2)) * self.ds
        return np.array([wx, wy])

    def a_star(self, start_pos, goal_pos, grid, use_inflation=False):
        start_idx = tuple(self.wtg(start_pos))
        goal_idx = tuple(self.wtg(goal_pos))

        open_set = []
        heapq.heappush(open_set, (0.0, start_idx))

        parents = {start_idx: None}
        reached_costs = {start_idx: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_idx:
                return self.reconstruct(parents, goal_idx)

            for di, dj, move_cost in self.motions:
                neighbor = (current[0] + di, current[1] + dj)

                if not (0 <= neighbor[0] < self.grid_w and 0 <= neighbor[1] < self.grid_h):
                    continue

                if grid[neighbor] == 1.0:
                    continue

                penalty = self.c_obs if (use_inflation and grid[neighbor] == 0.5) else 0.0
                new_cost = reached_costs[current] + (move_cost * self.ds) + penalty

                if neighbor not in reached_costs or new_cost < reached_costs[neighbor]:
                    reached_costs[neighbor] = new_cost
                    h = np.linalg.norm(np.array(neighbor) - np.array(goal_idx)) * self.ds
                    heapq.heappush(open_set, (new_cost + h, neighbor))
                    parents[neighbor] = current

        return None

    def reconstruct(self, parents, current):
        path = []
        while current is not None:
            path.append(self.gtw(current))
            current = parents[current]
        return path[::-1]