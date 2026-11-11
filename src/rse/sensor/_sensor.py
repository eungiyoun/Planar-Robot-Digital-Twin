from typing import Protocol
from dataclasses import dataclass
from rse.system import RobotProtocol
from rse.sensor._gpscompass import GPSCompass, GPSCompassMeasurement
from rse.sensor._gyro import Gyro, GyroMeasurement
import numpy as np


class SensorProtocol(Protocol):
    def measure(self, robot: RobotProtocol, time_step: float): ...


@dataclass
class SensorMeasurement:
    gps_compass: GPSCompassMeasurement
    gyro: GyroMeasurement


class Sensor(SensorProtocol):
    def __init__(
        self,
        gps_variance: float = 0.01,
        compass_variance: float = 0.25,
        gyro_variance: float = 0.04,
        bias_initial: float = 0.1,
        bias_noise_intensity: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.gps = GPSCompass(
            gps_variance=gps_variance, compass_variance=compass_variance, seed=seed
        )
        self.gyro = Gyro(
            gyro_variance=gyro_variance,
            bias_initial=bias_initial,
            bias_noise_intensity=bias_noise_intensity,
            seed=seed,
        )
        self._measurement: SensorMeasurement | None = None

    def measure(self, robot: RobotProtocol, time_step: float) -> None:
        gps_measurement = self.gps.measure(robot)
        gyro_measurement = self.gyro.measure(robot, time_step)

        measurement = SensorMeasurement(gps_measurement, gyro_measurement)
        self._measurement = measurement

    @property
    def y_gps(self) -> np.ndarray:
        if self._measurement is None:
            return np.zeros(2)
        return self._measurement.gps_compass.gps

    @property
    def y_compass(self) -> float:
        if self._measurement is None:
            return 0.0
        return self._measurement.gps_compass.compass

    @property
    def y_gyro(self) -> float:
        if self._measurement is None:
            return 0.0
        return self._measurement.gyro.omega
