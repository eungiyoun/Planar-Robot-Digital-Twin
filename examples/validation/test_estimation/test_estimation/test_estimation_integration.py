"""Integration test: run full physics simulation with noisy sensor and estimator.

5-stage input sequence (10 s total, dt = 0.001):
    Stage 1  [0, 2)   — stationary (no input)
    Stage 2  [2, 3)   — forward  u_l = u_r = 1
    Stage 3  [3, 6)   — coast    (no input)
    Stage 4  [6, 7)   — rotate   u_l = -1, u_r = 1
    Stage 5  [7, 10)  — coast    (no input)
"""

import numpy as np
import pytest

from rse.system import Robot
from rse.environment import Environment
from rse.integrator import PositionVerletIntegrator
from rse.simulator import Simulator
from rse.action._action import ActionProtocol
from rse.estimation import Estimator
from tests.test_estimation._mock_sensor import NoisySensorSim


class FiveStageAction(ActionProtocol):
    """Piecewise control for the integration test."""

    def __init__(self, track_width: float) -> None:
        self.w = track_width

    def apply(self, robot, current_time: float) -> None:
        t = current_time
        if 2.0 <= t < 3.0:
            ul, ur = 1.0, 1.0  # forward
        elif 6.0 <= t < 7.0:
            ul, ur = -1.0, 1.0  # rotate
        else:
            ul, ur = 0.0, 0.0  # stationary / coast

        force_x = ul + ur
        torque = (ur - ul) * (self.w / 2.0)
        robot.apply_force_body(np.array([force_x, 0.0]))
        robot.apply_torque(torque)


def _run_estimation_simulation():
    """Run the full 10-second simulation and return error arrays."""
    dt = 0.001
    robot = Robot(mass=2.0, moment_of_inertia=0.05, radius=0.2, track_width=0.15)
    robot.position = np.array([-1.2, -1.3])

    environment = Environment()
    integrator = PositionVerletIntegrator()
    action = FiveStageAction(track_width=robot.track_width)
    simulator = Simulator(robot, environment, integrator, dt)

    sensor_sim = NoisySensorSim(robot, dt=dt, rng_seed=42)
    estimator = Estimator(sensor_sim, alpha_x=0.01, alpha_theta=0.98, dt=dt)

    num_steps = int(10.0 / dt)

    gps_pos_errors = []
    filtered_pos_errors = []
    gyro_heading_errors = []
    filtered_heading_errors = []

    # Track integrated-gyro heading for comparison
    gyro_integrated_heading = 0.0

    for i in range(num_steps):
        # Ground truth
        Q_true = robot.orientation
        theta_true = np.arctan2(Q_true[1, 0], Q_true[0, 0])
        pos_true = robot.position.copy()

        # Physics step
        simulator.step(action)

        # Sensor measurement
        sensor_sim.measure()

        # Estimator update
        estimator.update()

        # Track pure gyro integration (for comparison)
        if i == 0:
            gyro_integrated_heading = sensor_sim.y_compass
        else:
            gyro_integrated_heading += sensor_sim.y_gyro * dt

        # Compute errors
        gps_pos_errors.append(np.linalg.norm(sensor_sim.y_gps - pos_true))
        filtered_pos_errors.append(np.linalg.norm(estimator.x_hat - pos_true))
        gyro_heading_errors.append(abs(gyro_integrated_heading - theta_true))
        filtered_heading_errors.append(abs(estimator.theta_hat - theta_true))

    return (
        np.array(gps_pos_errors),
        np.array(filtered_pos_errors),
        np.array(gyro_heading_errors),
        np.array(filtered_heading_errors),
    )


class TestEstimationIntegration:
    """End-to-end estimation tests using the full physics engine."""

    @pytest.fixture(scope="class")
    def errors(self):
        return _run_estimation_simulation()

    def test_filtered_position_rmse_less_than_gps(self, errors):
        gps_err, filt_err, _, _ = errors
        # Skip first 500 steps (transient)
        rmse_gps = np.sqrt(np.mean(gps_err[500:] ** 2))
        rmse_filt = np.sqrt(np.mean(filt_err[500:] ** 2))
        assert rmse_filt < rmse_gps, (
            f"Filtered RMSE ({rmse_filt:.4f}) should be less than GPS RMSE ({rmse_gps:.4f})"
        )

    def test_filtered_heading_rmse_less_than_gyro_only(self, errors):
        _, _, gyro_err, filt_err = errors
        # Skip first 500 steps (transient)
        rmse_gyro = np.sqrt(np.mean(gyro_err[500:] ** 2))
        rmse_filt = np.sqrt(np.mean(filt_err[500:] ** 2))
        assert rmse_filt < rmse_gyro, (
            f"Filtered heading RMSE ({rmse_filt:.4f}) should be less than "
            f"gyro-only RMSE ({rmse_gyro:.4f})"
        )
