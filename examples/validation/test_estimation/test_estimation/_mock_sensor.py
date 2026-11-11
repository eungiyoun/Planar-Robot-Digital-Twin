import numpy as np
from numpy.typing import NDArray
from rse.system._robot import Robot


class MockSensor:
    """Test double satisfying SensorReadingProtocol.

    Attributes can be set directly to control what the estimator sees.
    """

    def __init__(
        self,
        y_gps: NDArray = np.zeros(2),
        y_compass: float = 0.0,
        y_gyro: float = 0.0,
    ) -> None:
        self.y_gps = np.asarray(y_gps, dtype=float)
        self.y_compass = float(y_compass)
        self.y_gyro = float(y_gyro)


class NoisySensorSim:
    """Simulated noisy sensor wrapping a Robot for integration tests.

    Generates GPS, compass, and gyroscope measurements from the robot's
    true state with additive Gaussian noise.  Compass bias follows a
    Brownian-motion model (random walk).

    Default noise parameters from Table 2:
        sigma_gps^2   = 0.01   =>  sigma_gps   = 0.1
        sigma_compass^2 = 0.25 =>  sigma_compass = 0.5
        sigma_gyro^2  = 0.04   =>  sigma_gyro  = 0.2
        beta_0        = 0.1    (initial compass bias)
        sigma_beta    = 0.01   (bias random walk std per step)
    """

    def __init__(
        self,
        robot: Robot,
        sigma_gps: float = 0.1,
        sigma_compass: float = 0.5,
        sigma_gyro: float = 0.2,
        beta_0: float = 0.1,
        sigma_beta: float = 0.01,
        dt: float = 0.001,
        rng_seed: int | None = None,
    ) -> None:
        self._robot = robot
        self._sigma_gps = sigma_gps
        self._sigma_compass = sigma_compass
        self._sigma_gyro = sigma_gyro
        self._sigma_beta = sigma_beta
        self._dt = dt
        self._rng = np.random.default_rng(rng_seed)

        # Compass bias (Brownian motion)
        self._beta = beta_0

        # Latest readings (updated by measure())
        self._y_gps: NDArray = np.zeros(2)
        self._y_compass: float = 0.0
        self._y_gyro: float = 0.0

    def measure(self) -> None:
        """Sample noisy measurements from the robot's current true state."""
        # True heading from orientation matrix
        Q = self._robot.orientation
        theta_true = np.arctan2(Q[1, 0], Q[0, 0])

        # True angular velocity
        omega_true = self._robot.angular_momentum / self._robot.moment_of_inertia

        # GPS: position + Gaussian noise
        self._y_gps = self._robot.position + self._rng.normal(
            0.0, self._sigma_gps, size=2
        )

        # Compass bias random walk
        self._beta += self._rng.normal(0.0, self._sigma_beta)

        # Compass: heading + bias + Gaussian noise
        self._y_compass = (
            theta_true + self._beta + self._rng.normal(0.0, self._sigma_compass)
        )

        # Gyro: angular velocity + Gaussian noise
        self._y_gyro = omega_true + self._rng.normal(0.0, self._sigma_gyro)

    @property
    def y_gps(self) -> NDArray:
        return self._y_gps

    @property
    def y_compass(self) -> float:
        return self._y_compass

    @property
    def y_gyro(self) -> float:
        return self._y_gyro
