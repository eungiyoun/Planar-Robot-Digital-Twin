import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.spatial import cKDTree


@dataclass
class TEBDynResult:
    waypoints: np.ndarray
    dt: np.ndarray
    t: np.ndarray
    timed_waypoints: list
    segment_speed: np.ndarray
    success: bool
    message: str
    cost: float


class TEBDynPlanner:
    def __init__(
        self,
        obstacle_points: np.ndarray | None,
        vmax: float,
        dmin_static: float,
        dmin_dynamic: float,
        n_waypoints: int,
        gamma_time: float,
        gamma_smooth: float,
        gamma_obs: float,
        gamma_vel: float,
        gamma_bc: float,
        dt_min: float,
        dt_max: float,
        dynamic_obstacle_fn=None,
        dynamic_obstacle_radius: float = 0.0,
        robot_radius: float = 0.0,
        horizon_time: float = 1.0,
        gamma_horizon: float = 10.0,
    ) -> None:
        self.vmax = float(vmax)
        self.dmin_static = float(dmin_static)
        self.dmin_dynamic = float(dmin_dynamic)
        self.n_waypoints = int(n_waypoints)

        self.gamma_time = float(gamma_time)
        self.gamma_smooth = float(gamma_smooth)
        self.gamma_obs = float(gamma_obs)
        self.gamma_vel = float(gamma_vel)
        self.gamma_bc = float(gamma_bc)

        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

        self.dynamic_obstacle_fn = dynamic_obstacle_fn
        self.dynamic_obstacle_radius = float(dynamic_obstacle_radius)
        self.robot_radius = float(robot_radius)

        self.horizon_time = float(horizon_time)
        self.gamma_horizon = float(gamma_horizon)

        if obstacle_points is None:
            self.obstacle_points = np.zeros((0, 2), dtype=float)
        else:
            self.obstacle_points = np.asarray(obstacle_points, dtype=float)

        self.kdtree = cKDTree(self.obstacle_points) if len(self.obstacle_points) > 0 else None

        # warm start memory
        self.prev_solution = None

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
        path = np.asarray(p_inf, dtype=float)
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError("Input A* path p_inf must have shape (M, 2)")
        if len(path) < 2:
            raise ValueError("A* path must contain at least 2 points")
        return path

    def _subsample_path(self, path_world: np.ndarray) -> np.ndarray:
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
        interior = waypoints[1:-1].reshape(-1)
        return np.concatenate([interior, dt])

    def _unpack(self, z: np.ndarray, start: np.ndarray, goal: np.ndarray, n: int):
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
        """
        Static obstacle net clearance:
            distance(point, nearest static obstacle point) - robot_radius
        """
        if self.kdtree is None or len(points) == 0:
            return np.full(len(points), np.inf)

        d_center, _ = self.kdtree.query(points)
        d_net = d_center - self.robot_radius
        return d_net

    def _dynamic_obstacle_distance(self, points: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Dynamic obstacle net clearance:
            ||p - p_obs(t)|| - dynamic_obstacle_radius - robot_radius
        """
        if self.dynamic_obstacle_fn is None or len(points) == 0:
            return np.full(len(points), np.inf)

        d = np.zeros(len(points))
        for k, (p, tk) in enumerate(zip(points, times)):
            obs_pos = np.asarray(self.dynamic_obstacle_fn(float(tk)), dtype=float)
            d[k] = (
                np.linalg.norm(p - obs_pos)
                - self.dynamic_obstacle_radius
                - self.robot_radius
            )
        return d

    def _cost(self, z: np.ndarray, start: np.ndarray, goal: np.ndarray, n: int, t0: float = 0.0) -> float:
        waypoints, dt = self._unpack(z, start, goal, n)

        if np.any(dt <= 0):
            return 1e12

        seg = waypoints[1:] - waypoints[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        speed = seg_len / dt

        # cumulative global time for each waypoint
        t_local = np.concatenate([[0.0], np.cumsum(dt)])
        t_global = t0 + t_local

        # 1. total traversal time
        j_time = np.sum(dt)

        # 2. smoothness surrogate
        j_smooth = np.sum(seg_len**2)

        # 3. obstacle clearance penalty on interior waypoints
        interior = waypoints[1:-1]
        interior_times = t_global[1:-1]

        # both are now NET clearance
        d_static = self._nearest_obstacle_distance(interior)
        d_dynamic = self._dynamic_obstacle_distance(interior, interior_times)

        static_violation = np.maximum(self.dmin_static - d_static, 0.0)
        dynamic_violation = np.maximum(self.dmin_dynamic - d_dynamic, 0.0)

        j_obs_static = np.sum(static_violation**2)
        j_obs_dynamic = np.sum(dynamic_violation**2)
        j_obs = j_obs_static + j_obs_dynamic

        # 4. max speed penalty
        vel_violation = np.maximum(speed - self.vmax, 0.0)
        j_vel = np.sum(vel_violation**2)

        # 5. boundary-condition penalty
        if t0 <= 1e-9:
            j_bc = speed[0] ** 2 + speed[-1] ** 2
        else:
            j_bc = speed[-1] ** 2

        # 6. keep the planned time horizon close to horizon_time
        j_horizon = (np.sum(dt) - self.horizon_time) ** 2

        return (
            self.gamma_time * j_time
            + self.gamma_smooth * j_smooth
            + self.gamma_obs * j_obs
            + self.gamma_vel * j_vel
            + self.gamma_bc * j_bc
            + self.gamma_horizon * j_horizon
        )
    def _build_cold_start(self, path0: np.ndarray):
        seg0 = path0[1:] - path0[:-1]
        seg_len0 = np.linalg.norm(seg0, axis=1)
        dt0 = np.maximum(seg_len0 / self.vmax, self.dt_min)
        dt0 = np.clip(dt0, self.dt_min, self.dt_max)
        return path0.copy(), dt0

    def _resample_waypoints(self, pts: np.ndarray, n_target: int) -> np.ndarray:
    
        pts = np.asarray(pts, dtype=float)
        if len(pts) == n_target:
            return pts.copy()
        if len(pts) < 2:
            raise ValueError("Need at least 2 points to resample.")

        seg = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = s[-1]

        if total_len < 1e-9:
            return np.repeat(pts[:1], n_target, axis=0)

        s_new = np.linspace(0.0, total_len, n_target)
        out = np.zeros((n_target, 2), dtype=float)

        for i, si in enumerate(s_new):
            k = np.searchsorted(s, si, side="right") - 1
            k = min(max(k, 0), len(seg_len) - 1)

            if seg_len[k] < 1e-12:
                out[i] = pts[k]
            else:
                alpha = (si - s[k]) / seg_len[k]
                out[i] = (1 - alpha) * pts[k] + alpha * pts[k + 1]

        return out

    def _build_warm_start(self, start: np.ndarray, goal: np.ndarray, path0: np.ndarray):
 
        # Use previous optimized solution as initial guess.

        if self.prev_solution is None:
            return self._build_cold_start(path0)

        prev_wp = self.prev_solution["waypoints"]
        prev_dt = self.prev_solution["dt"]

        if prev_wp is None or len(prev_wp) < 2 or prev_dt is None:
            return self._build_cold_start(path0)

       
        d = np.linalg.norm(prev_wp - start[None, :], axis=1)
        idx_start = int(np.argmin(d))

        truncated_wp = prev_wp[idx_start:].copy()
        if len(truncated_wp) < 2:
            return self._build_cold_start(path0)

        truncated_wp[0] = start
        truncated_wp[-1] = goal

        warm_wp = self._resample_waypoints(truncated_wp, len(path0))
        warm_wp[0] = start
        warm_wp[-1] = goal

        prev_dt_tail = prev_dt[idx_start:] if idx_start < len(prev_dt) else prev_dt[-1:]
        if len(prev_dt_tail) < 1:
            return self._build_cold_start(path0)

        if len(prev_dt_tail) == len(path0) - 1:
            warm_dt = prev_dt_tail.copy()
        else:
            # 简单线性插值到新的段数
            old_x = np.linspace(0.0, 1.0, len(prev_dt_tail))
            new_x = np.linspace(0.0, 1.0, len(path0) - 1)
            warm_dt = np.interp(new_x, old_x, prev_dt_tail)

        warm_dt = np.clip(warm_dt, self.dt_min, self.dt_max)

        return warm_wp, warm_dt

    def solve(self, p_inf, t0: float = 0.0, use_warm_start: bool = True) -> TEBDynResult:
        path_world = self._to_numpy_path(p_inf)
        path0 = self._subsample_path(path_world)

        n = len(path0)
        start = path0[0].copy()
        goal = path0[-1].copy()

        if use_warm_start:
            wp0, dt0 = self._build_warm_start(start, goal, path0)
        else:
            wp0, dt0 = self._build_cold_start(path0)

        z0 = self._pack(wp0, dt0)

        bounds = []
        n_interior = n - 2

        for _ in range(2 * n_interior):
            bounds.append((None, None))

        for _ in range(n - 1):
            bounds.append((self.dt_min, self.dt_max))

        res = minimize(
            fun=self._cost,
            x0=z0,
            args=(start, goal, n, t0),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500}
        )

        waypoints_opt, dt_opt = self._unpack(res.x, start, goal, n)
        speed_opt = self._segment_speed(waypoints_opt, dt_opt)
        t_opt = t0 + np.concatenate([[0.0], np.cumsum(dt_opt)])

        timed_waypoints = [
            (float(ti), waypoints_opt[i].copy())
            for i, ti in enumerate(t_opt)
        ]

        # 保存本轮结果，供下一轮 warm start
        self.prev_solution = {
            "waypoints": waypoints_opt.copy(),
            "dt": dt_opt.copy(),
            "t0": float(t0),
        }

        return TEBDynResult(
            waypoints=waypoints_opt,
            dt=dt_opt,
            t=t_opt,
            timed_waypoints=timed_waypoints,
            segment_speed=speed_opt,
            success=bool(res.success),
            message=str(res.message),
            cost=float(res.fun),
        )

    def solve_receding_horizon(self, p_inf, t0: float = 0.0, exec_steps: int = 1):
        teb_res = self.solve(p_inf, t0=t0, use_warm_start=True)

        exec_steps = min(exec_steps, len(teb_res.waypoints) - 1)

        executed_waypoints = teb_res.waypoints[:exec_steps + 1].copy()
        executed_dt = teb_res.dt[:exec_steps].copy()

        return teb_res, executed_waypoints, executed_dt

    def reset_warm_start(self):
        self.prev_solution = None