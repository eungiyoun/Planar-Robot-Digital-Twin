from typing import Optional, Callable
from ._integrator import IntegratorProtocol
from rse.system import RobotProtocol


class EulerIntegrator(IntegratorProtocol):
    def step(
        self,
        time_current: float,
        time_step: float,
        robot: RobotProtocol,
        force_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Perform a single integration step using the Euler method.

        Args:
            time_current (float): Current simulation time (seconds)
            time_step (float): Time step for integration (seconds)
            robot (RobotProtocol): An instance of RobotProtocol to integrate
            force_callback (Optional[Callable[[], None]]): Optional callback to update forces before integration
        """

        robot.compute_kinematics(time_current, time_step)
        robot.compute_dynamics(time_current, time_step)
