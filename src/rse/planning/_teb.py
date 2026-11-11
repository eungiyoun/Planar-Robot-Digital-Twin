import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.spatial import cKDTree


@dataclass
class TEBResult:
    waypoints: np.ndarray        # shape (N, 2)
    dt: np.ndarray               # shape (N-1,)
    t: np.ndarray                # shape (N,)
    timed_waypoints: list        # [(t0, x0), (t1, x1), ...]
    segment_speed: np.ndarray    # shape (N-1,)
    success: bool
    message: str
    cost: float


class TEBPlanner:
    """
    Timed Elastic Band optimizer.

    Input:
        p_inf : list or ndarray - World-coordinate path returned by A*

    Output:
        timed waypoint sequence 
    """

    def __init__(
        self,
        obstacle_points: np.ndarray | None,
        vmax: float,
        dmin: float,
        n_waypoints: int,
        gamma_time: float,
        gamma_smooth: float,
        gamma_obs: float,
        gamma_vel: float,
        gamma_bc: float,
        dt_min: float,
        dt_max: float,
    ) -> None:
        self.vmax = float(vmax)
        self.dmin = float(dmin)
        self.n_waypoints = int(n_waypoints)

        self.gamma_time = float(gamma_time)
        self.gamma_smooth = float(gamma_smooth)
        self.gamma_obs = float(gamma_obs)
        self.gamma_vel = float(gamma_vel)
        self.gamma_bc = float(gamma_bc)

        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

        if obstacle_points is None:
            self.obstacle_points = np.zeros((0, 2), dtype=float)
        else:
            self.obstacle_points = np.asarray(obstacle_points, dtype=float)

        self.kdtree = cKDTree(self.obstacle_points) if len(self.obstacle_points) > 0 else None
            

    @staticmethod
    def grid_to_obstacle_points(grid: np.ndarray, planner, include_soft: bool = False) -> np.ndarray:
        points = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if include_soft:
                    if grid[i, j] >= 0.5:
                        points.append(planner.gtw((i, j)))
                else:
                    if grid[i, j] >= 1.0:
                        points.append(planner.gtw((i, j)))
        if len(points) == 0:
            return np.zeros((0, 2), dtype=float)
        return np.asarray(points, dtype=float)

    def _to_numpy_path(self, p_inf) -> np.ndarray:
        """
        Convert A* output p_inf to ndarray of shape (M, 2).
        The A* planner returns a Python list of 2D world-coordinate points.
        """
        path = np.asarray(p_inf, dtype=float)
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError("Input A* path p_inf must have shape (M, 2)")
        if len(path) < 2:
            raise ValueError("A* path must contain at least 2 points")
        return path

    def _subsample_path(self, path_world: np.ndarray) -> np.ndarray:
        """
        Uniformly subsample the A* path to reduce optimization dimension.
        Keep start and goal fixed.
        """
        if len(path_world) <= self.n_waypoints:
            return path_world.copy()

        idx = np.linspace(0, len(path_world) - 1, self.n_waypoints).astype(int)
        idx = np.unique(idx)

        if idx[0] != 0:
            idx[0] = 0
        if idx[-1] != len(path_world) - 1:
            idx[-1] = len(path_world) - 1

        return path_world[idx]

    def _pack(self, waypoints: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """
        Pack interior waypoint coordinates and time intervals into one vector.
        """
        interior = waypoints[1:-1].reshape(-1)
        return np.concatenate([interior, dt])

    def _unpack(self, z: np.ndarray, start: np.ndarray, goal: np.ndarray, n: int):
        """
        Unpack optimization vector back to waypoints and dt.
        """
        n_interior = n - 2
        xy_flat = z[: 2 * n_interior]
        dt = z[2 * n_interior:]

        waypoints = np.zeros((n, 2), dtype=float)
        waypoints[0] = start
        waypoints[-1] = goal

        if n_interior > 0:
            waypoints[1:-1] = xy_flat.reshape(n_interior, 2)

        return waypoints, dt

    def _segment_speed(self, waypoints: np.ndarray, dt: np.ndarray) -> np.ndarray:
        seg = waypoints[1:] - waypoints[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        return seg_len / dt

    def _nearest_obstacle_distance(self, points: np.ndarray) -> np.ndarray:
        if self.kdtree is None or len(points) == 0:
            return np.full(len(points), np.inf)
        d, _ = self.kdtree.query(points)
        return d

    def _cost(self, z: np.ndarray, start: np.ndarray, goal: np.ndarray, n: int) -> float:
        waypoints, dt = self._unpack(z, start, goal, n)

        if np.any(dt <= 0):
            return 1e12

        seg = waypoints[1:] - waypoints[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        speed = seg_len / dt

        # 1. total traversal time
        j_time = np.sum(dt)

        # 2. smoothness surrogate
        j_smooth = np.sum(seg_len**2)

        # 3. obstacle clearance penalty on interior waypoints
        interior = waypoints[1:-1]
        d_obs = self._nearest_obstacle_distance(interior)
        obs_violation = np.maximum(self.dmin - d_obs, 0.0)
        j_obs = np.sum(obs_violation**2)

        # 4. max speed penalty
        vel_violation = np.maximum(speed - self.vmax, 0.0)
        j_vel = np.sum(vel_violation**2)

        # 5. boundary-condition penalty: near rest at start and end
        j_bc = speed[0] ** 2 + speed[-1] ** 2

        return (
            self.gamma_time * j_time
            + self.gamma_smooth * j_smooth
            + self.gamma_obs * j_obs
            + self.gamma_vel * j_vel
            + self.gamma_bc * j_bc
        )

    def solve(self, p_inf) -> TEBResult:
        """
        Main TEB entry point.

        Parameters
        ----------
        p_inf : list or ndarray
            Direct output of RoboticPlanner.a_star(...)

        Returns
        -------
        TEBResult
            Includes timed waypoint sequence {(t_k, xbar_k^*)}
        """
        path_world = self._to_numpy_path(p_inf)
        path0 = self._subsample_path(path_world)

        n = len(path0)
        start = path0[0].copy()
        goal = path0[-1].copy()

        # Initial time allocation from segment length / vmax
        seg0 = path0[1:] - path0[:-1]
        seg_len0 = np.linalg.norm(seg0, axis=1)
        dt0 = np.maximum(seg_len0 / self.vmax, self.dt_min)

        z0 = self._pack(path0, dt0)

        bounds = []
        n_interior = n - 2

        # Interior waypoint positions are free
        for _ in range(2 * n_interior):
            bounds.append((None, None))

        # Time intervals are bounded and positive
        for _ in range(n - 1):
            bounds.append((self.dt_min, self.dt_max))

        res = minimize(
            fun=self._cost,
            x0=z0,
            args=(start, goal, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000,
                     "maxfun": 100000,}
        )

        waypoints_opt, dt_opt = self._unpack(res.x, start, goal, n)
        speed_opt = self._segment_speed(waypoints_opt, dt_opt)
        t_opt = np.concatenate([[0.0], np.cumsum(dt_opt)])

        timed_waypoints = [
            (float(ti), waypoints_opt[i].copy())
            for i, ti in enumerate(t_opt)
        ]

        return TEBResult(
            waypoints=waypoints_opt,
            dt=dt_opt,
            t=t_opt,
            timed_waypoints=timed_waypoints,
            segment_speed=speed_opt,
            success=bool(res.success),
            message=str(res.message),
            cost=float(res.fun),
        )