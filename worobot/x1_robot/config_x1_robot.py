from dataclasses import dataclass, field

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("x1_robot")
@dataclass
class X1RobotConfig(RobotConfig):

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    #teleop mode
    teleop: bool =  False
