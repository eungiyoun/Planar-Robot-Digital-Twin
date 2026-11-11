import numpy as np
from numpy.typing import NDArray
from rse.system._system import RobotProtocol


class Robot(RobotProtocol):
    def __init__(
        self,
        mass: float = 2.0,
        moment_of_inertia: float = 0.05,
        radius: float = 0.2,
        track_width: float = 0.15,
    ) -> None:
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.radius = radius
        self.track_width = track_width

        # Pose q = [Q, x; 0, 1]: position x in R^2, orientation Q in SO(2)
        self._position: NDArray = np.zeros(2)
        self._orientation: NDArray = np.eye(2)

        # Momentum p = [eta; l]: linear eta in R^2, angular l in R
        self._linear_momentum: NDArray = np.zeros(2)
        self._angular_momentum: float = 0.0

        # Applied force and torque (reset by simulator at the start of each step)
        self._applied_force: NDArray = np.zeros(2)
        self._applied_torque: float = 0.0

    # --- Physical Properties ---

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        assert value > 0, "Mass must be positive."
        self._mass = value

    @property
    def moment_of_inertia(self) -> float:
        return self._moment_of_inertia

    @moment_of_inertia.setter
    def moment_of_inertia(self, value: float) -> None:
        assert value > 0, "Moment of inertia must be positive."
        self._moment_of_inertia = value

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        assert value > 0, "Radius must be positive."
        self._radius = value

    @property
    def track_width(self) -> float:
        return self._track_width

    @track_width.setter
    def track_width(self, value: float) -> None:
        assert value > 0, "Track width must be positive."
        self._track_width = value

    # --- Pose Properties ---

    @property
    def position(self) -> NDArray:
        return self._position

    @position.setter
    def position(self, value: NDArray) -> None:
        val_array = np.asarray(value, dtype=float)
        assert val_array.shape == (2,), "Position must be a 2D vector."
        self._position = val_array

    @property
    def orientation(self) -> NDArray:
        return self._orientation

    @orientation.setter
    def orientation(self, value: NDArray) -> None:
        val = np.asarray(value)
        assert val.shape == (2, 2)
        self._orientation = val

    # --- Momentum Properties ---

    @property
    def linear_momentum(self) -> NDArray:
        return self._linear_momentum

    @linear_momentum.setter
    def linear_momentum(self, value: NDArray) -> None:
        self._linear_momentum = np.asarray(value, dtype=float)

    @property
    def angular_momentum(self) -> float:
        return self._angular_momentum

    @angular_momentum.setter
    def angular_momentum(self, value: float) -> None:
        self._angular_momentum = float(value)

    # --- Applied Load Properties ---

    @property
    def applied_force(self) -> NDArray:
        return self._applied_force

    @applied_force.setter
    def applied_force(self, value: NDArray) -> None:
        val_array = np.asarray(value, dtype=float)
        assert val_array.shape == (2,), "Applied force must be a 2D vector."
        self._applied_force = val_array

    @property
    def applied_torque(self) -> float:
        return self._applied_torque

    @applied_torque.setter
    def applied_torque(self, value: float) -> None:
        self._applied_torque = float(value)

    def apply_force_body(self, force: NDArray) -> None:
        """Accumulates a body-frame force onto the robot."""
        force_array = np.asarray(force, dtype=float)
        assert force_array.shape == (2,), "Force must be a 2D vector."
        self.applied_force = self.applied_force + force_array

    def apply_torque(self, torque: float) -> None:
        """Accumulates a torque about the robot's center of mass."""
        self.applied_torque = self.applied_torque + float(torque)

    # --- Kinematics and Dynamics ---

    def compute_kinematics(self, time_current: float, time_step: float) -> None:
        """
        Updates position and orientation using the exact SE(2) Left Jacobian.

        For a rotation angle dtheta = omega * dt over the step, the Left Jacobian
        maps body-frame velocity to world-frame displacement, correctly accounting
        for curved motion and preventing first-order Euler drift.
        """
        v_body = self.linear_momentum / self.mass
        omega = self.angular_momentum / self.moment_of_inertia
        dtheta = omega * time_step

        if abs(dtheta) < 1e-9:
            # Degenerate case: straight-line motion
            v_mat = time_step * np.eye(2)
        else:
            a = np.sin(dtheta) / dtheta
            b = (1.0 - np.cos(dtheta)) / dtheta
            v_mat = time_step * np.array([[a, -b], [b, a]])

        self.position = self.position + self.orientation @ (v_mat @ v_body)

        c, s = np.cos(dtheta), np.sin(dtheta)
        self.orientation = self.orientation @ np.array([[c, -s], [s, c]])

    def compute_dynamics(self, time_current: float, time_step: float) -> None:
        """
        Updates momentum using exact co-adjoint rotation to prevent energy drift.

        Linear momentum is rotated by the body-frame angular velocity before
        accumulating the applied impulse, removing the Euler-method drift that
        would otherwise accumulate in the body-frame momentum coordinates.
        """
        omega = self.angular_momentum / self.moment_of_inertia

        c, s = np.cos(-omega * time_step), np.sin(-omega * time_step)
        rot = np.array([[c, -s], [s, c]])

        self.linear_momentum = (rot @ self.linear_momentum) + (
            self.applied_force * time_step
        )
        self.angular_momentum = self.angular_momentum + self.applied_torque * time_step
