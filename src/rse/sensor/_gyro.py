import numpy as np
from dataclasses import dataclass
from rse.system import RobotProtocol


@dataclass
class GyroMeasurement:
    omega: float
    bias: float


class Gyro:
    def __init__(
        self,
        gyro_variance: float = 0.04,
        bias_initial: float = 0.1,
        bias_noise_intensity: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """
        Gyroscope sensor model.

        Args:
            gyro_variance (float): variance of white measurement noise
            bias_initial (float): initial gyroscope bias beta_0
            bias_noise_intensity (float): sigma_beta in the bias random walk
            seed (int | None): random seed for reproducibility
        """
        assert gyro_variance >= 0.0, "gyro_variance must be non-negative."
        assert bias_noise_intensity >= 0.0, "bias_noise_intensity must be non-negative."

        self.gyro_variance = gyro_variance
        self.bias = float(bias_initial)
        self.bias_noise_intensity = bias_noise_intensity
        self.rng = np.random.default_rng(seed)

    def measure(self, robot: RobotProtocol, time_step: float) -> GyroMeasurement:
        """
        Returns a noisy gyroscope measurement of angular velocity.

        Args:
            robot (RobotProtocol): robot containing the true state
            time_step (float): simulation time step Delta t

        Returns:
            GyroMeasurement:
                omega = noisy angular velocity measurement
                bias = current internal bias after update
        """
        assert time_step > 0.0, "time_step must be positive."

        # True angular velocity from robot momentum state
        omega_true = robot.angular_momentum / robot.moment_of_inertia

        # Bias random walk:
        # beta_{k+1} = beta_k + sigma_beta * epsilon_k, epsilon_k ~ N(0, Delta t)
        bias_increment = self.bias_noise_intensity * self.rng.normal(
            loc=0,
            scale=np.sqrt(time_step),
        )
        self.bias += bias_increment

        # White measurement noise:
        # epsilon_tilde ~ N(0, sigma_gyro^2)
        white_noise = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(self.gyro_variance),
        )

        # Gyro reading:
        # y_gyro = omega + beta + epsilon_tilde
        y_gyro = omega_true + self.bias + white_noise

        return GyroMeasurement(omega=y_gyro, bias=self.bias)
