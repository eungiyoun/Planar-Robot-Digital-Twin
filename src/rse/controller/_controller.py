from typing import Protocol
import numpy as np
from numpy.typing import NDArray
from rse.estimation import Estimator


class ControllerProtocol(Protocol):
    def calculate_body_error(self, destination: NDArray) -> NDArray: ...
    def control(self, destination: float) -> NDArray: ...


class Controller(ControllerProtocol):
    def __init__(
        self,
        estimator: Estimator,
        Kp: float = 1.0,
        Ki: float = 1.0,
        decay_factor: float = 1.0,
        dt: float = 0.001,
    ):
        self.estimator = estimator

        self.Kp = Kp
        self.Ki = Ki
        self.decay_factor = decay_factor
        self.dt = dt

        self.prev_error = 0.0
        self.curr_error = 0.0

        self.prev_integral = 0.0
        self.curr_integral = 0.0

    def update_error(self, error: float) -> None:
        """Calculates the positional error and transforms it to the robot's body frame."""
        # Note: estimate_pose() should return a 2x2 rotation matrix (SO(2)).
        self.prev_error = self.curr_error
        self.curr_error = error

    def control(self, error: float, u_max: float = 1.0) -> NDArray:
        """
        Main control loop.
        Calculates the required control input (forces) to reach the destination.
        """
        # --- [1] Update Error State ---
        # We do this ONLY ONCE per step to prevent advancing the state multiple times
        self.update_error(error)

        # --- [2] Proportional (P) Control ---
        p_term = self.Kp * self.curr_error

        # --- [3] Integral (I) Control (Leaky Integrator) ---
        self.prev_integral = self.curr_integral

        # Calculate leaky integrator coefficients
        decay = np.exp(-self.decay_factor * self.dt)
        if self.decay_factor > 0.0001:
            coef = (1.0 - decay) / self.decay_factor
        else:
            coef = self.dt  # Fallback to standard integration if decay is 0

        # I_{k} = e^(-w*dt) * I_{k-1} + coef * e_{k-1}
        tentative_integral = decay * self.prev_integral + coef * self.prev_error
        i_term = self.Ki * tentative_integral

        # --- Anti-Windup Mechanism  ---
        tentative_control = p_term + i_term

        # If we exceed the physical actuator limits, clamp the output
        # AND freeze the integrator so it doesn't wind up!
        if abs(tentative_control) > u_max:
            control_effort = np.clip(tentative_control, -u_max, u_max)

            # Back-calculate the integral to exactly match the saturated limit
            if self.Ki != 0:
                self.curr_integral = (control_effort - p_term) / self.Ki
            else:
                self.curr_integral = tentative_integral
        else:
            self.curr_integral = tentative_integral
            control_effort = tentative_control

        return control_effort

    def reset(self) -> None:
        """Reset integral state to zero (call at start of each segment)."""
        self.prev_error = 0.0
        self.curr_error = 0.0
        self.prev_integral = 0.0
        self.curr_integral = 0.0
