from typing import Protocol
from rse.system import RobotProtocol


class ActionProtocol(Protocol):
    """
    Interface for controller policies or external actions.
    """

    def apply(self, robot: RobotProtocol, current_time: float) -> None: ...
