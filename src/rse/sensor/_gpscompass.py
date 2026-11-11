import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from rse.system import RobotProtocol


@dataclass
class GPSCompassMeasurement:
    gps: NDArray
    compass: float


class GPSCompass:
    def __init__(
        self,
        gps_variance: float = 0.01,
        compass_variance: float = 0.25,
        seed: int | None = None,
    ) -> None:
        """
        GPS + compass sensor model.

        Args:
            gps_variance (float): variance of 2D Gaussian position noise
            compass_variance (float): variance of heading noise
            seed (int | None): random seed for reproducibility
        """
        assert gps_variance >= 0.0, "gps_variance must be non-negative."
        assert compass_variance >= 0.0, "compass_variance must be non-negative."

        self.gps_variance = gps_variance
        self.compass_variance = compass_variance
        self.rng = np.random.default_rng(seed)

    def measure(self, robot: RobotProtocol) -> GPSCompassMeasurement:
        """
        Returns a noisy GPS + compass measurement from the robot true state.
        """
        x_true = np.asarray(robot.position, dtype=float).copy()

        # Robot orientation is stored as a 2x2 rotation matrix
        Q = np.asarray(robot.orientation, dtype=float)
        theta_true = np.arctan2(Q[1, 0], Q[0, 0])

        gps_noise = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(self.gps_variance),
            size=2,
        )
        compass_noise = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(self.compass_variance),
        )

        y_gps = x_true + gps_noise
        y_compass = self._wrap_angle(theta_true + compass_noise)

        return GPSCompassMeasurement(gps=y_gps, compass=y_compass)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
