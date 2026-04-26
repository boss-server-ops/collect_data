import logging
from pprint import pformat

from .robot import Robot
from .config import RobotConfig
# x1_robot and unix_robot pull in rclpy (ROS2). For PiperRobot users without
# ROS2 installed, only piper_robot needs to be importable at module load time.
try:
    from .x1_robot import X1Robot  # noqa: F401
except ImportError:
    X1Robot = None
try:
    from .unix_robot import UnixRobot  # noqa: F401
except ImportError:
    UnixRobot = None
from .piper_robot import PiperRobot  # noqa: F401

def make_robot_from_config(config: RobotConfig) -> Robot:
    if config.type == "unix_robot":
        from .unix_robot import UnixRobot

        return UnixRobot(config)
    elif config.type == "x1_robot":
        from .x1_robot import X1Robot

        return X1Robot(config)
    elif config.type == "piper_robot":
        from .piper_robot import PiperRobot

        return PiperRobot(config)
    else:
        raise ValueError(config.type)
    
    # return X1Robot(config)