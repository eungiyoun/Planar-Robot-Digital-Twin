from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from rse.estimation._protocol import SensorReadingProtocol


class Estimator:
    """Low-pass + complementary filter for robot state estimation.

    Fuses GPS, compass, and gyroscope readings to produce filtered
    estimates of position, heading, and orientation matrix.

    Position filter (low-pass, HW2 Problem 3, Definition 3.1.1)::

        x_hat_k = (1 - alpha_x) * x_hat_{k-1} + alpha_x * y_gps_k

    Orientation filter (complementary, HW2 Problem 4, Definition 3.2.1)::

        theta_hat_k = alpha_theta * (theta_hat_{k-1} + y_gyro_k * dt)
                    + (1 - alpha_theta) * y_compass_k

    Parameters
    ----------
    sensor : SensorReadingProtocol
        Object exposing ``y_gps``, ``y_compass``, ``y_gyro``.
    alpha_x : float
        Low-pass gain for position, in (0, 1).
    alpha_theta : float
        Complementary filter weight for gyro integration, in (0, 1).
    dt : float
        Sampling period in seconds.
    """

    def __init__(
        self,
        sensor: SensorReadingProtocol,
        alpha_x: float = 0.01,
        alpha_theta: float = 0.98,
        dt: float = 0.001,
    ) -> None:
        self._sensor = sensor
        self.alpha_x = alpha_x  # triggers validation via setter
        self.alpha_theta = alpha_theta  # triggers validation via setter

        if dt <= 0:
            raise ValueError("dt must be positive.")
        self._dt = dt

        self._x_hat: NDArray | None = None
        self._theta_hat: float | None = None
        self._initialized: bool = False

    # --- Gain properties with validation ---

    @property
    def alpha_x(self) -> float:
        return self._alpha_x

    @alpha_x.setter
    def alpha_x(self, value: float) -> None:
        if not (0 < value < 1):
            raise ValueError(f"alpha_x must be in (0, 1), got {value}")
        self._alpha_x = value

    @property
    def alpha_theta(self) -> float:
        return self._alpha_theta

    @alpha_theta.setter
    def alpha_theta(self, value: float) -> None:
        if not (0 < value < 1):
            raise ValueError(f"alpha_theta must be in (0, 1), got {value}")
        self._alpha_theta = value

    def update(self) -> None:
        """Incorporate the latest sensor reading and advance the filter."""
        y_gps = np.asarray(self._sensor.y_gps, dtype=float)
        y_compass = float(self._sensor.y_compass)
        y_gyro = float(self._sensor.y_gyro)

        if not self._initialized:
            # Seed from first reading
            self._x_hat = y_gps.copy()
            self._theta_hat = y_compass
            self._initialized = True
            return

        # 1. Low-pass filter for position (Works perfectly)
        self._x_hat = (1.0 - self._alpha_x) * self._x_hat + self._alpha_x * y_gps

        # 2. PI Complementary filter for heading

        # Predict where we are based on the gyro
        theta_pred = self._theta_hat + y_gyro * self._dt

        # Find the SHORTEST angular error between prediction and compass
        def wrap_to_pi(angle: float) -> float:
            return (angle + np.pi) % (2 * np.pi) - np.pi

        angle_error = wrap_to_pi(y_compass - theta_pred)

        # Update the heading estimate (Wrapping the final result!)
        self._theta_hat = wrap_to_pi(
            theta_pred + (1.0 - self._alpha_theta) * angle_error
        )

    # --- Read-only state estimates ---

    @property
    def x_hat(self) -> NDArray:
        """Filtered position estimate, shape (2,)."""
        if self._x_hat is None:
            raise RuntimeError("Estimator not initialized — call update() first.")
        return self._x_hat

    @property
    def theta_hat(self) -> float:
        """Filtered heading estimate in radians."""
        if self._theta_hat is None:
            raise RuntimeError("Estimator not initialized — call update() first.")
        return self._theta_hat

    @property
    def Q_hat(self) -> NDArray:
        """Filtered orientation matrix (2x2 SO(2))."""
        th = self.theta_hat  # will raise if not initialized
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s], [s, c]])

    @property
    def pose(self) -> Tuple[NDArray, NDArray]:
        """Convenience accessor returning ``(x_hat, Q_hat)``."""
        return self.x_hat, self.Q_hat

    # --- Static utilities ---

    @staticmethod
    def gain_from_cutoff(omega_c: float, dt: float) -> float:
        """Convert cut-off frequency to low-pass gain.

        alpha = 1 - exp(-omega_c * dt)
        """
        if omega_c <= 0:
            raise ValueError("omega_c must be positive.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        return 1.0 - np.exp(-omega_c * dt)

    @staticmethod
    def crossover_frequency(alpha_theta: float, dt: float) -> float:
        """Compute the crossover frequency of the complementary filter.

        omega_cross = -ln(alpha_theta) / dt

        This is the frequency at which the gyro and compass contributions
        have equal magnitude.
        """
        if not (0 < alpha_theta < 1):
            raise ValueError("alpha_theta must be in (0, 1).")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        # return -np.log(alpha_theta) / dt
        return (1.0 - alpha_theta) / (alpha_theta * dt)
