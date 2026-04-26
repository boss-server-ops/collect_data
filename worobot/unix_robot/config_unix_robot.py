from dataclasses import dataclass, field

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("unix_robot")
@dataclass
class UnixRobotConfig(RobotConfig):

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    #teleop mode
    teleop: bool =  False
