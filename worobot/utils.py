import logging
from pprint import pformat

from .robot import Robot
from .config import RobotConfig
from .x1_robot import X1Robot
from .unix_robot import UnixRobot
from .piper_robot import PiperRobot

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