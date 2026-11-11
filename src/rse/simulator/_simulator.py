from typing import Optional
from rse.system._robot import RobotProtocol
from rse.environment._environment import EnvironmentProtocol
from rse.integrator._integrator import IntegratorProtocol
from rse.action._action import ActionProtocol
import numpy as np


class Simulator:
    def __init__(
        self,
        robot: RobotProtocol,
        environment: EnvironmentProtocol,
        integrator: IntegratorProtocol,
        time_step: float = 0.001,
    ) -> None:
        """Initializes the simulation environment."""
        self.robot = robot
        self.environment = environment
        self.integrator = integrator

        assert time_step > 0, "Time step must be positive."
        self.time_step = time_step
        self.time_current = 0.0

    def _evaluate_forces(
        self, action: Optional[ActionProtocol], current_time: float
    ) -> None:
        """Helper method to clear and recalculate all forces at a specific time."""
        self.robot.applied_force = np.zeros(2)
        self.robot.applied_torque = 0.0

        # Apply fresh forces based on the correct evaluation time
        if action is not None:
            action.apply(self.robot, current_time)
        self.environment.detect_collision(self.robot)
        self.environment.compute_loads(self.robot, self.time_step)

    def step(self, action: Optional[ActionProtocol] = None) -> None:
        """Executes one single simulation step."""
        # 1. Initial force calculation at t (used by Euler/Symplectic)
        self._evaluate_forces(action, self.time_current)

        # 2. Step the integrator. For Position Verlet, the callback evaluates at the mid-point.
        mid_time = self.time_current + 0.5 * self.time_step
        self.integrator.step(
            self.time_current,
            self.time_step,
            self.robot,
            force_callback=lambda: self._evaluate_forces(action, mid_time),
        )

        # 3. Advance Simulation Clock
        self.time_current += self.time_step

    def run(self, time_horizon: float, action: Optional[ActionProtocol] = None) -> None:
        """Runs the simulation continuously until the time horizon is reached."""
        assert time_horizon > 0, "Time horizon must be positive."

        # Calculate how many discrete steps we need to take
        num_steps = int(time_horizon / self.time_step)

        for _ in range(num_steps):
            self.step(action)
