from typing import Protocol, Optional, Callable
from rse.system._robot import RobotProtocol


class IntegratorProtocol(Protocol):
    """
    Interface for all numerical integrators.
    """

    def step(
        self,
        time_current: float,
        time_step: float,
        robot: RobotProtocol,
        force_callback: Optional[Callable[[], None]] = None,
    ) -> None: ...
