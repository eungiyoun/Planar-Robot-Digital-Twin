from typing import List, Tuple, Protocol
import numpy as np
from numpy.typing import NDArray
from rse.system import RobotProtocol


class EnvironmentProtocol(Protocol):
    """
    Interface for environmental interactions including
    collision detection and external load computation (friction, contact forces).
    """

    def detect_collision(self, robot: RobotProtocol) -> None: ...

    def compute_loads(self, robot: RobotProtocol, time_step: float) -> None: ...


class Environment(EnvironmentProtocol):
    """
    Concrete implementation of EnvironmentProtocol for an I-shaped room.
    Handles collision detection against wall segments and computes
    contact and friction loads on the robot.
    """

    def __init__(self) -> None:
        # 12 segments of the I-shaped room boundary.
        # Vertices listed clockwise from bottom-left corner.
        vertices: NDArray = np.array(
            [
                [-1.5, -2.0],
                [1.5, -2.0],
                [1.5, -1.0],
                [0.9, -1.0],
                [0.9, 1.0],
                [1.5, 1.0],
                [1.5, 2.0],
                [-1.5, 2.0],
                [-1.5, 1.0],
                [-0.9, 1.0],
                [-0.9, -1.0],
                [-1.5, -1.0],
            ]
        )  # shape: (12, 2)

        # Build segments as consecutive vertex pairs (wraps to close the loop)
        self._walls: NDArray = np.array(
            [
                [vertices[i], vertices[(i + 1) % len(vertices)]]
                for i in range(len(vertices))
            ]
        )  # shape: (12, 2, 2)

        self._mu_c: float = 1000.0  # contact stiffness (N/m), penalty spring
        self._mu_f: float = 1.0  # friction damping coefficient (s^-1), viscous

        # Friction boundary: 4x + 3y > 0  =>  friction active
        self._friction_boundary_coeffs: Tuple[float, float] = (4.0, 3.0)

        # Collision state: populated by detect_collision(), consumed by compute_loads()
        # Each entry: (penetration_depth: float, contact_normal: NDArray shape (2,))
        self._collisions: List[Tuple[float, NDArray]] = []

        # Sanity checks on friction boundary
        a, b = self._friction_boundary_coeffs
        assert a * (-1.0) + b * (-1.0) < 0, "(-1,-1) should be in no-friction zone"
        assert a * (1.0) + b * (1.0) > 0, "(1,1) should be in friction zone"
        assert a * (0.0) + b * (0.0) == 0, "(0,0) should be on the boundary"

    def detect_collision(self, robot: RobotProtocol) -> None:
        """
        Detects collisions between the robot and all wall segments.

        For each wall segment, computes the shortest distance from the robot center
        to the segment. If that distance is less than robot.radius, a collision is
        recorded as (penetration_depth, contact_normal), where:
            penetration_depth = robot.radius - distance  (> 0 when penetrating)
            contact_normal    = unit vector pointing from wall toward robot center
        """
        self._collisions.clear()
        for wall in self._walls:
            distance, contact_normal = self._point_to_segment_distance(
                robot.position, wall[0], wall[1]
            )
            if distance < robot.radius:
                penetration_depth = robot.radius - distance
                self._collisions.append((penetration_depth, contact_normal))

    def compute_loads(self, robot: RobotProtocol, time_step: float) -> None:
        """
        Computes and applies external loads to the robot.

        Two contributions:

        1. Contact force (penalty spring model):
               f_contact = mu_c * penetration_depth * contact_normal
           Summed over all active collisions in the global frame, then transformed
           to the body frame. A circular robot produces no contact torque.

        2. Friction (differential per-wheel viscous damping):
           Each wheel position is tested independently against the friction boundary
           (4x + 3y > 0). Active wheels apply f = -mu_f * eta_wheel, where
           eta_wheel = 0.5 * m * v_wheel. The differential drag produces both
           a net force and a torque on the robot.
        """
        # Contact forces: sum in global frame, then transform to body frame
        global_contact_force: NDArray = np.zeros(2)
        for penetration_depth, contact_normal in self._collisions:
            global_contact_force += self._mu_c * penetration_depth * contact_normal

        body_contact_force = self._transform_to_body_frame(
            global_contact_force, robot.orientation
        )
        robot.apply_force_body(body_contact_force)  # circular robot: no contact torque

        # Friction: differential per-wheel model
        r_right = np.array([0.0, -0.5 * robot.track_width])
        r_left = np.array([0.0, 0.5 * robot.track_width])

        pos_right_global = robot.position + (robot.orientation @ r_right)
        pos_left_global = robot.position + (robot.orientation @ r_left)

        mu_r = self._get_friction_coefficient(pos_right_global)
        mu_l = self._get_friction_coefficient(pos_left_global)

        if mu_r > 0 or mu_l > 0:
            v_com = robot.linear_momentum / robot.mass
            omega = robot.angular_momentum / robot.moment_of_inertia

            # Wheel velocity: v_wheel = v_com + omega x r  (2D cross product)
            v_right = v_com + omega * np.array([-r_right[1], r_right[0]])
            v_left = v_com + omega * np.array([-r_left[1], r_left[0]])

            # Each wheel carries half the total mass
            f_right = -mu_r * (0.5 * robot.mass * v_right)
            f_left = -mu_l * (0.5 * robot.mass * v_left)

            # 2D torque: tau = r x F = r_x * F_y - r_y * F_x
            tau_right = r_right[0] * f_right[1] - r_right[1] * f_right[0]
            tau_left = r_left[0] * f_left[1] - r_left[1] * f_left[0]

            robot.apply_force_body(f_right + f_left)
            robot.apply_torque(tau_right + tau_left)

    # --- Private Helpers ---

    def _point_to_segment_distance(
        self,
        point: NDArray,
        seg_start: NDArray,
        seg_end: NDArray,
    ) -> Tuple[float, NDArray]:
        """
        Shortest distance from a 2D point to a line segment, and the contact normal
        (unit vector from the closest point on the segment toward the given point).
        """
        segment = seg_end - seg_start
        seg_length = np.linalg.norm(segment)
        if seg_length == 0:
            diff = point - seg_start
            return float(np.linalg.norm(diff)), diff / np.linalg.norm(diff)

        projection = np.clip(np.dot(point - seg_start, segment) / seg_length**2, 0, 1)
        closest_point = seg_start + projection * segment
        diff = point - closest_point
        diff_norm = np.linalg.norm(diff)
        contact_normal = np.array([0.0, 1.0]) if diff_norm == 0 else diff / diff_norm
        return float(diff_norm), contact_normal

    def _get_friction_coefficient(self, position: NDArray) -> float:
        """
        Returns mu_f where friction is active (4x + 3y > 0), else 0.
        """
        a, b = self._friction_boundary_coeffs
        return self._mu_f if a * position[0] + b * position[1] > 0 else 0.0

    def _transform_to_body_frame(
        self,
        force_global: NDArray,
        orientation: NDArray,
    ) -> NDArray:
        """
        Transforms a force from the global frame to the robot body frame (Q^T @ f).
        """
        return orientation.T @ force_global
