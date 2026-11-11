from ._integrator import IntegratorProtocol
from rse.system import RobotProtocol


class PositionVerletIntegrator(IntegratorProtocol):
    def step(
        self,
        time_current: float,
        time_step: float,
        robot: RobotProtocol,
        force_callback=None,
    ) -> None:
        """
        Perform a single integration step using the Position Verlet method.

        Args:
            time_current (float): Current simulation time (seconds)
            time_step (float): Time step for integration (seconds)
            robot (RobotProtocol): An instance of RobotProtocol to integrate
            force_callback (Optional[Callable[[], None]]): Optional callback to update forces before integration
        """

        half_time_step = 0.5 * time_step
        # 1. Half-step of position (using current velocity)
        robot.compute_kinematics(time_current, half_time_step)

        # 2. Full-step of momentum (using current accumulated forces)
        if force_callback is not None:
            force_callback()

        robot.compute_dynamics(time_current + half_time_step, time_step)

        # 3. Final half-step of position (using newly updated velocity)
        robot.compute_kinematics(time_current + half_time_step, half_time_step)
