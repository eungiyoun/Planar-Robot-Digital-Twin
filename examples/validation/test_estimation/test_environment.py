import numpy as np

from rse.environment._environment import Environment


# ---------------------------------------------------------------------------
# Mock Robot
# A minimal stand-in for the real Robot class. Does NOT import robot.py.
# ---------------------------------------------------------------------------


class MockRobot:
    def __init__(self) -> None:
        self._position: np.ndarray = np.zeros(2)
        self._orientation: np.ndarray = np.eye(2)
        self._linear_momentum: np.ndarray = np.zeros(2)
        self._angular_momentum: float = 0.0
        self.applied_force: np.ndarray = np.zeros(2)
        self.applied_torque: float = 0.0
        self.radius: float = 0.2
        self.mass: float = 2.0
        self.moment_of_inertia: float = 0.05
        self.track_width: float = 0.15

    @property
    def position(self) -> np.ndarray:
        return self._position

    @property
    def orientation(self) -> np.ndarray:
        return self._orientation

    @property
    def linear_momentum(self) -> np.ndarray:
        return self._linear_momentum

    @property
    def angular_momentum(self) -> float:
        return self._angular_momentum

    def apply_force_body(self, force: np.ndarray) -> None:
        self.applied_force = self.applied_force + np.asarray(force, dtype=float)

    def apply_torque(self, torque: float) -> None:
        self.applied_torque = self.applied_torque + float(torque)


# ---------------------------------------------------------------------------
# TestGetFrictionCoefficient
# ---------------------------------------------------------------------------


class TestGetFrictionCoefficient:
    def setup_method(self) -> None:
        self.env = Environment()

    def test_no_friction_zone(self) -> None:
        # 4(-1) + 3(-1) = -7 < 0 → no friction
        result = self.env._get_friction_coefficient(np.array([-1.0, -1.0]))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_friction_zone(self) -> None:
        # 4(1) + 3(1) = 7 > 0 → friction active
        result = self.env._get_friction_coefficient(np.array([1.0, 1.0]))
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_on_boundary(self) -> None:
        # 4(0) + 3(0) = 0 → boundary counts as no friction (<= 0)
        result = self.env._get_friction_coefficient(np.array([0.0, 0.0]))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_just_inside_friction(self) -> None:
        # 4(0.1) + 3(0.1) = 0.7 > 0 → friction active
        result = self.env._get_friction_coefficient(np.array([0.1, 0.1]))
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_robot_start_position(self) -> None:
        # 4(-1.2) + 3(-1.3) = -8.7 < 0 → robot starts in no-friction zone
        result = self.env._get_friction_coefficient(np.array([-1.2, -1.3]))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TestTransformToBodyFrame
# ---------------------------------------------------------------------------


class TestTransformToBodyFrame:
    def setup_method(self) -> None:
        self.env = Environment()

    def test_identity_orientation(self) -> None:
        # Q = I → Q^T @ f = f
        Q = np.eye(2)
        f = np.array([3.0, 4.0])
        result = self.env._transform_to_body_frame(f, Q)
        np.testing.assert_allclose(result, [3.0, 4.0], atol=1e-10)

    def test_90_degree_rotation(self) -> None:
        # Q = 90° CCW rotation: [[0,-1],[1,0]]
        # Q^T = [[0,1],[-1,0]]
        # Q^T @ [1,0] = [0,-1]
        Q = np.array([[0.0, -1.0], [1.0, 0.0]])
        f = np.array([1.0, 0.0])
        result = self.env._transform_to_body_frame(f, Q)
        np.testing.assert_allclose(result, [0.0, -1.0], atol=1e-10)

    def test_zero_force(self) -> None:
        # Zero force → zero result regardless of orientation
        Q = np.array([[0.0, -1.0], [1.0, 0.0]])
        f = np.array([0.0, 0.0])
        result = self.env._transform_to_body_frame(f, Q)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# TestPointToSegmentDistance
# ---------------------------------------------------------------------------


class TestPointToSegmentDistance:
    def setup_method(self) -> None:
        self.env = Environment()

    def test_perpendicular_hit(self) -> None:
        # Segment (0,0)→(2,0), point (1,1): closest=(1,0), dist=1, normal=[0,1]
        dist, normal = self.env._point_to_segment_distance(
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
        )
        np.testing.assert_allclose(dist, 1.0, atol=1e-10)
        np.testing.assert_allclose(normal, [0.0, 1.0], atol=1e-10)

    def test_past_start_endpoint(self) -> None:
        # Segment (0,0)→(2,0), point (-1,0): t clamps to 0, closest=(0,0)
        # diff = (-1,0), dist=1, normal=(-1,0)
        dist, normal = self.env._point_to_segment_distance(
            np.array([-1.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
        )
        np.testing.assert_allclose(dist, 1.0, atol=1e-10)
        np.testing.assert_allclose(normal, [-1.0, 0.0], atol=1e-10)

    def test_past_end_endpoint(self) -> None:
        # Segment (0,0)→(2,0), point (3,0): t clamps to 1, closest=(2,0)
        # diff = (1,0), dist=1, normal=(1,0)
        dist, normal = self.env._point_to_segment_distance(
            np.array([3.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
        )
        np.testing.assert_allclose(dist, 1.0, atol=1e-10)
        np.testing.assert_allclose(normal, [1.0, 0.0], atol=1e-10)

    def test_point_on_segment(self) -> None:
        # Segment (0,0)→(2,0), point (1,0): dist=0, fallback normal=(0,1)
        dist, normal = self.env._point_to_segment_distance(
            np.array([1.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
        )
        np.testing.assert_allclose(dist, 0.0, atol=1e-10)
        np.testing.assert_allclose(normal, [0.0, 1.0], atol=1e-10)

    def test_vertical_segment(self) -> None:
        # Segment (1,0)→(1,3), point (0,1.5): closest=(1,1.5), dist=1, normal=(-1,0)
        dist, normal = self.env._point_to_segment_distance(
            np.array([0.0, 1.5]),
            np.array([1.0, 0.0]),
            np.array([1.0, 3.0]),
        )
        np.testing.assert_allclose(dist, 1.0, atol=1e-10)
        np.testing.assert_allclose(normal, [-1.0, 0.0], atol=1e-10)

    def test_diagonal_segment(self) -> None:
        # Segment (0,0)→(1,1), point (0,1)
        # t = dot((0,1),(1,1)) / dot((1,1),(1,1)) = 1/2 = 0.5
        # closest = (0.5, 0.5), diff = (-0.5, 0.5)
        # dist = sqrt(0.5), normal = [-1/sqrt(2), 1/sqrt(2)]
        dist, normal = self.env._point_to_segment_distance(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
        )
        np.testing.assert_allclose(dist, np.sqrt(0.5), atol=1e-10)
        np.testing.assert_allclose(
            normal, [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], atol=1e-10
        )


# ---------------------------------------------------------------------------
# TestDetectCollision
# ---------------------------------------------------------------------------


class TestDetectCollision:
    def setup_method(self) -> None:
        self.env = Environment()
        self.robot = MockRobot()

    def test_no_collision(self) -> None:
        # Center of stem — nearest walls (inner stem sides) are 0.9 away, radius=0.2
        self.robot._position = np.array([0.0, 0.0])
        self.env.detect_collision(self.robot)
        assert len(self.env._collisions) == 0

    def test_collision_with_one_wall(self) -> None:
        # Right inner stem wall at x=0.9; robot at x=0.8 → distance=0.1 < radius=0.2
        self.robot._position = np.array([0.8, 0.0])
        self.env.detect_collision(self.robot)
        assert len(self.env._collisions) == 1
        depth, normal = self.env._collisions[0]
        np.testing.assert_allclose(depth, 0.1, atol=1e-10)
        np.testing.assert_allclose(normal, [-1.0, 0.0], atol=1e-10)

    def test_collision_in_corner(self) -> None:
        # Near inner top-right corner: close to both the stem wall and the step wall
        self.robot._position = np.array([0.85, 0.95])
        self.env.detect_collision(self.robot)
        assert len(self.env._collisions) >= 1


# ---------------------------------------------------------------------------
# TestComputeLoads
# ---------------------------------------------------------------------------


class TestComputeLoads:
    def setup_method(self) -> None:
        self.env = Environment()
        self.robot = MockRobot()

    def test_no_friction_no_collision(self) -> None:
        # Robot at (-1.2, -1.3): 4(-1.2)+3(-1.3)=-8.7 < 0 → both wheels in no-friction zone
        # All walls > 0.3 m away → no collision → load stays zero
        self.robot._position = np.array([-1.2, -1.3])
        self.robot._linear_momentum = np.array([1.0, 0.5])
        self.robot._angular_momentum = 0.1
        self.env.detect_collision(self.robot)
        self.env.compute_loads(self.robot, time_step=0.01)
        np.testing.assert_allclose(self.robot.applied_force, [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(self.robot.applied_torque, 0.0, atol=1e-10)

    def test_friction_only(self) -> None:
        # Robot at (0, 1.5): both wheels in friction zone (mu_f=1.0), no wall collisions
        # Per-wheel model: force = [-2.0, -1.0] (same as old model)
        # Torque differs from old model due to per-wheel differential friction:
        #   omega=10, tau_right=-0.13125, tau_left=0.01875 → total=-0.1125
        self.robot._position = np.array([0.0, 1.5])
        self.robot._linear_momentum = np.array([2.0, 1.0])
        self.robot._angular_momentum = 0.5
        self.env.detect_collision(self.robot)
        assert len(self.env._collisions) == 0  # verify position is safe
        self.env.compute_loads(self.robot, time_step=0.01)
        np.testing.assert_allclose(self.robot.applied_force, [-2.0, -1.0], atol=1e-10)
        np.testing.assert_allclose(self.robot.applied_torque, -0.1125, atol=1e-10)

    def test_contact_only(self) -> None:
        # Robot at (-0.82, 0.0): 4(-0.82)+3(0)=-3.28<0 → no friction
        # Left inner stem wall at x=-0.9, distance=0.08 < radius=0.2 → single collision
        # Normal = (1,0) [wall→robot, pointing right], penetration = 0.12
        # Contact force global: 1000 * 0.12 * (1,0) = (120, 0)
        # With identity orientation: body_load = [120, 0, 0], torque = 0
        self.robot._position = np.array([-0.82, 0.0])
        self.robot._linear_momentum = np.zeros(2)
        self.robot._angular_momentum = 0.0
        self.env.detect_collision(self.robot)
        assert len(self.env._collisions) == 1  # confirm single wall contact
        self.env.compute_loads(self.robot, time_step=0.01)
        np.testing.assert_allclose(self.robot.applied_force[0], 120.0, atol=1e-10)
        np.testing.assert_allclose(self.robot.applied_force[1], 0.0, atol=1e-10)
        np.testing.assert_allclose(self.robot.applied_torque, 0.0, atol=1e-10)

    def test_friction_and_contact_together(self) -> None:
        # Robot at (0.0, 1.85): 4(0)+3(1.85)=5.55>0 → friction zone
        # Top wall at y=2, distance=0.15 < radius=0.2 → collision
        # Both friction and contact contribute → external_load is nonzero
        self.robot._position = np.array([0.0, 1.85])
        self.robot._linear_momentum = np.array([1.0, 0.5])
        self.robot._angular_momentum = 0.2
        self.env.detect_collision(self.robot)
        assert len(self.env._collisions) >= 1  # confirm wall contact
        self.env.compute_loads(self.robot, time_step=0.01)
        # Total load should be nonzero (friction + contact both contribute)
        assert np.linalg.norm(self.robot.applied_force) > 0.0
        # Per-wheel friction torque: omega=4, tau_right=-0.06, tau_left=0.015 → total=-0.045
        np.testing.assert_allclose(self.robot.applied_torque, -0.045, atol=1e-10)
