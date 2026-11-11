import pytest
import numpy as np
from rse.system import Robot


@pytest.fixture
def robot() -> Robot:
    """Fixture to provide a clean robot instance for each test."""
    # Using parameters from Table 1 of the Midterm Project
    return Robot(mass=2.0, moment_of_inertia=0.05, radius=0.2, track_width=0.15)


def test_initialization(robot: Robot) -> None:
    """Verify that the robot initializes with correct default states."""
    assert robot.mass == 2.0
    np.testing.assert_allclose(robot.position, [0.0, 0.0])
    np.testing.assert_allclose(robot.orientation, [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(robot.linear_momentum, [0.0, 0.0])


def test_kinematics_straight_line(robot: Robot) -> None:
    """Test if linear momentum correctly translates the robot."""
    # p = m * v. If we want v = 1.0 m/s forward, p = 2.0 * 1.0 = 2.0
    robot.linear_momentum = np.array([2.0, 0.0])

    # Step kinematics by 1 second
    robot.compute_kinematics(time_current=0.0, time_step=1.0)

    # It should have moved 1 meter along the positive x-axis
    np.testing.assert_allclose(robot.position, [1.0, 0.0])


def test_dynamics_force_acceleration(robot: Robot) -> None:
    """Test if applied force correctly increases linear momentum."""
    # Apply 10 N of force forward
    robot.applied_force = np.array([10.0, 0.0])

    # Step dynamics by 0.1 seconds
    robot.compute_dynamics(time_current=0.0, time_step=0.1)

    # Momentum should increase by Force * dt (10.0 * 0.1 = 1.0)
    np.testing.assert_allclose(robot.linear_momentum, [1.0, 0.0])
