import numpy as np

class ClampedCubicSpline:
    def __init__(self, times: np.ndarray, waypoints: np.ndarray):
        """
        Fits a C2-smooth piecewise cubic polynomial to the provided waypoints and times
        using clamped boundary conditions (zero velocity at start and end).
        
        Args:
            times: 1D numpy array of execution times.
            waypoints: 2D numpy array of (x, y) spatial waypoints.
        """
        times = np.asarray(times)
        waypoints = np.asarray(waypoints)

        # Require strictly increasing time knots
        _, unique_indices = np.unique(times, return_index=True)
        self.t_knots = times[unique_indices]
        self.pos_knots = waypoints[unique_indices]

        if len(self.t_knots) < 2:
            raise ValueError("At least 2 unique time knots are required to fit a spline.")

        self.max_t = self.t_knots[-1]
        self.N = len(self.t_knots) - 1  # Number of polynomial segments

        # Solve for exact cubic coefficients [a, b, c, d] for X and Y components independently
        self.ax, self.bx, self.cx, self.dx = self._fit_1d_spline(self.t_knots, self.pos_knots[:, 0])
        self.ay, self.by, self.cy, self.dy = self._fit_1d_spline(self.t_knots, self.pos_knots[:, 1])

    def _fit_1d_spline(self, t: np.ndarray, y: np.ndarray):
        """
        Solves the tridiagonal system for the second derivatives (M) at the knots,
        then calculates the piecewise polynomial coefficients a, b, c, d.
        """
        N = self.N
        h = np.diff(t)  # Segment durations
        
        # We solve the linear system A * M = B to find the second derivatives at the knots
        A = np.zeros((N + 1, N + 1))
        B = np.zeros(N + 1)
        
        # 1. Clamped Boundary Condition at Start (v_start = 0)
        A[0, 0] = 2 * h[0]
        A[0, 1] = h[0]
        B[0] = 6 * (y[1] - y[0]) / h[0]
        
        # 2. Clamped Boundary Condition at End (v_end = 0)
        A[N, N - 1] = h[N - 1]
        A[N, N] = 2 * h[N - 1]
        B[N] = -6 * (y[N] - y[N - 1]) / h[N - 1]
        
        # 3. Interior Knots (C1 & C2 Continuity)
        for k in range(1, N):
            A[k, k - 1] = h[k - 1]
            A[k, k] = 2 * (h[k - 1] + h[k])
            A[k, k + 1] = h[k]
            B[k] = 6 * ((y[k + 1] - y[k]) / h[k] - (y[k] - y[k - 1]) / h[k - 1])
            
        # Solve for the second derivatives (Moments)
        M = np.linalg.solve(A, B)
        
        # 4. Map moments to polynomial coefficients for each interval k
        a = (M[1:] - M[:-1]) / (6 * h)
        b = M[:-1] / 2
        c = (y[1:] - y[:-1]) / h - h * (2 * M[:-1] + M[1:]) / 6
        d = y[:-1]
        
        return a, b, c, d
        
    def _evaluate_1d(self, t_query, a, b, c, d):
        """
        Evaluates the piecewise polynomial and its analytical derivative at time t.
        """
        t_query = np.clip(t_query, self.t_knots[0], self.max_t)
        
        # Find which segment 't' belongs to
        idx = np.searchsorted(self.t_knots, t_query, side='right') - 1
        idx = np.clip(idx, 0, self.N - 1)
        
        dt = t_query - self.t_knots[idx]
        
        # Position: x(t) = a*dt^3 + b*dt^2 + c*dt + d
        pos = a[idx] * dt**3 + b[idx] * dt**2 + c[idx] * dt + d[idx]
        
        # Velocity: x'(t) = 3*a*dt^2 + 2*b*dt + c
        vel = 3 * a[idx] * dt**2 + 2 * b[idx] * dt + c[idx]
        
        return pos, vel

    def get_pose(self, t: float) -> np.ndarray:
        """Evaluates the spline to return just the 2D position."""
        x_pos, _ = self._evaluate_1d(t, self.ax, self.bx, self.cx, self.dx)
        y_pos, _ = self._evaluate_1d(t, self.ay, self.by, self.cy, self.dy)
        return np.array([x_pos, y_pos])

    def get_reference(self, t: float):
        """Returns reference position and velocity."""
        x_pos, x_vel = self._evaluate_1d(t, self.ax, self.bx, self.cx, self.dx)
        y_pos, y_vel = self._evaluate_1d(t, self.ay, self.by, self.cy, self.dy)
        return np.array([x_pos, y_pos]), np.array([x_vel, y_vel])

    def __call__(self, t: float) -> np.ndarray:
        """Allows the object to be called directly like a function."""
        return self.get_pose(t)