from typing import Protocol
from numpy.typing import NDArray


class SensorReadingProtocol(Protocol):
    """Minimal interface the estimator expects from ANY sensor.

    Any sensor implementation (real or mock) must expose these three
    measurement properties so the estimator can fuse them.
    """

    @property
    def y_gps(self) -> NDArray:
        """GPS position measurement, shape (2,)."""
        ...

    @property
    def y_compass(self) -> float:
        """Compass heading measurement in radians."""
        ...

    @property
    def y_gyro(self) -> float:
        """Gyroscope angular velocity measurement in rad/s."""
        ...


class EstimatorProtocol(Protocol):
    """Interface the controller will expect from ANY estimator.

    After each call to ``update()``, the filtered state estimates
    are available through the read-only properties below.
    """

    def update(self) -> None:
        """Incorporate the latest sensor reading and advance the filter."""
        ...

    @property
    def x_hat(self) -> NDArray:
        """Filtered position estimate, shape (2,)."""
        ...

    @property
    def theta_hat(self) -> float:
        """Filtered heading estimate in radians."""
        ...

    @property
    def Q_hat(self) -> NDArray:
        """Filtered orientation matrix (2x2 SO(2))."""
        ...
