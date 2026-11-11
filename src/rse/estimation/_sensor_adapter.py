from numpy.typing import NDArray
from rse.sensor._gpscompass import GPSCompass, GPSCompassMeasurement
from rse.sensor._gyro import Gyro, GyroMeasurement
from rse.system._system import RobotProtocol


class SensorAdapter:
    """Bridges teammate's sensor classes to the EstimatorProtocol's expected interface.

    Wraps ``GPSCompass`` and ``Gyro`` into a single object that:
    - Calls both sensors' ``measure()`` methods
    - Stores the latest readings
    - Exposes ``y_gps``, ``y_compass``, ``y_gyro`` as properties

    This satisfies ``SensorReadingProtocol`` without modifying either
    the sensor or estimator implementations.

    Parameters
    ----------
    gps_compass : GPSCompass
        GPS + compass sensor instance.
    gyro : Gyro
        Gyroscope sensor instance.
    dt : float
        Sampling period passed to ``gyro.measure(robot, dt)``.
    """

    def __init__(
        self,
        gps_compass: GPSCompass,
        gyro: Gyro,
        dt: float = 0.001,
    ) -> None:
        self._gps_compass = gps_compass
        self._gyro = gyro
        self._dt = dt

        self._gps_compass_reading: GPSCompassMeasurement | None = None
        self._gyro_reading: GyroMeasurement | None = None

    def measure(self, robot: RobotProtocol) -> None:
        """Read both sensors and store the latest measurements."""
        self._gps_compass_reading = self._gps_compass.measure(robot)
        self._gyro_reading = self._gyro.measure(robot, self._dt)

    @property
    def y_gps(self) -> NDArray:
        """GPS position measurement, shape (2,)."""
        if self._gps_compass_reading is None:
            raise RuntimeError("No measurement available — call measure(robot) first.")
        return self._gps_compass_reading.gps

    @property
    def y_compass(self) -> float:
        """Compass heading measurement in radians."""
        if self._gps_compass_reading is None:
            raise RuntimeError("No measurement available — call measure(robot) first.")
        return self._gps_compass_reading.compass

    @property
    def y_gyro(self) -> float:
        """Gyroscope angular velocity measurement in rad/s."""
        if self._gyro_reading is None:
            raise RuntimeError("No measurement available — call measure(robot) first.")
        return self._gyro_reading.omega
