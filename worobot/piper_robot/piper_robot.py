#!/usr/bin/env python
"""PiperRobot — ROS1 (rospy) version.

Subscribes to the kai0-style topics that are already running on the
recording rig:
  /master/joint_left, /master/joint_right  -> action source (teleop)
  /puppet/joint_left, /puppet/joint_right  -> state source (slave actual)

For inference / replay (config.teleop=False) it publishes commands back
on /master/joint_left and /master/joint_right.

go_home() calls the kai0 ROS1 services /can_{left,right}/go_zero_master_slave
(std_srvs/Trigger), waits for arms to settle, then restores master-slave
coupling via /can_{left,right}/restore_ms_mode.
"""

import logging
import time
from functools import cached_property
from threading import Lock
from typing import Any

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_piper_robot import PiperRobotConfig

logger = logging.getLogger(__name__)


class AdaptiveKalmanFilter:
    """Inline adaptive Kalman filter (matches the original piper_robot.py
    signature: ``dim, process_variance, measurement_variance, threshold,
    scale_factor``). The standalone ``kalman_filter.py`` in this package
    is a separate research demo with a different API; do not import from it
    here, otherwise its top-level matplotlib code blocks the program.
    """
    def __init__(self, dim, process_variance, measurement_variance,
                 threshold=5.0, scale_factor=10.0):
        self.dim = dim
        self.x = np.zeros(dim)
        self.P = np.ones(dim)
        self.Q = process_variance * np.eye(dim)
        self.R = measurement_variance * np.eye(1)
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.prev_measurement = np.zeros(1)
        self.delta_t = 0.03333
        self.A = np.array([[1, self.delta_t], [0, 1]])

    def initialize_state_from_measurement(self, initial_measurement):
        self.x[0] = initial_measurement[0]
        self.x[1] = 0.0
        self.P = np.array([0.1, 0.1])
        self.prev_measurement[0] = initial_measurement[0]

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measurement):
        delta = np.abs(measurement - self.prev_measurement)
        if np.any(delta > self.threshold):
            self.R = self.scale_factor * np.eye(1)
        else:
            self.R = 7.5 * np.eye(1)

        H = np.array([[1, 0]])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = measurement - H @ self.x
        self.x += K @ y
        self.P = (np.eye(self.dim) - K @ H) @ self.P
        self.prev_measurement = measurement
        return self.x


# Topic / service names.
#
# IMPORTANT — physical-vs-label correction.
# On this rig, the kai0 ROS topics named `/master/joint_left` and
# `/puppet/joint_left` are physically wired to the operator/robot's RIGHT arm
# (and vice versa). Recording datasets without correcting this produces
# left/right-flipped data that requires a deployment-time swap.
#
# Fix at the source: cross-wire the topic names below so that LEFT_MOTORS
# (which become action[0:7] / state[0:7] in the dataset) actually carry the
# physical LEFT arm's joints. Result: datasets match the standard fold
# convention out of the box and need no swap at deploy time.
#
# If you ever fix the underlying ROS bringup so /master/joint_left really is
# the physical left arm, just swap the two MASTER and the two PUPPET strings
# back to the literal names.
MASTER_LEFT_TOPIC = "/master/joint_right"   # physically left arm
MASTER_RIGHT_TOPIC = "/master/joint_left"   # physically right arm
PUPPET_LEFT_TOPIC = "/puppet/joint_right"   # physically left arm
PUPPET_RIGHT_TOPIC = "/puppet/joint_left"   # physically right arm
GO_ZERO_SERVICES = ("/can_left/go_zero_master_slave", "/can_right/go_zero_master_slave")
RESTORE_MS_SERVICES = ("/can_left/restore_ms_mode", "/can_right/restore_ms_mode")


class PiperRobot(Robot):
    config_class = PiperRobotConfig
    name = "piper_robot"

    LEFT_MOTORS = [
        "left_joint_1", "left_joint_2", "left_joint_3",
        "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
    ]
    RIGHT_MOTORS = [
        "right_joint_1", "right_joint_2", "right_joint_3",
        "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper",
    ]

    def __init__(self, config: PiperRobotConfig):
        Robot.__init__(self, config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        self.motors = self.LEFT_MOTORS + self.RIGHT_MOTORS

        self._is_connected = False
        self._lock = Lock()
        self._master_left = None
        self._master_right = None
        self._puppet_left = None
        self._puppet_right = None

        self._sub_master_left = None
        self._sub_master_right = None
        self._sub_puppet_left = None
        self._sub_puppet_right = None

        # Action publishers used only in non-teleop (replay/inference) mode.
        self._pub_master_left = None
        self._pub_master_right = None

        # Kalman smoothing on outgoing actions (only used when we publish).
        self._kalman_initialized = False
        self.kalman_filters = {
            motor: AdaptiveKalmanFilter(
                dim=2, process_variance=0.5, measurement_variance=5,
                threshold=0.075, scale_factor=15.0,
            )
            for motor in self.motors
        }

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _master_left_cb(self, msg: JointState):
        with self._lock:
            self._master_left = list(msg.position)

    def _master_right_cb(self, msg: JointState):
        with self._lock:
            self._master_right = list(msg.position)

    def _puppet_left_cb(self, msg: JointState):
        with self._lock:
            self._puppet_left = list(msg.position)

    def _puppet_right_cb(self, msg: JointState):
        with self._lock:
            self._puppet_right = list(msg.position)

    # ------------------------------------------------------------------
    # Feature schemas (must match fold_clothes_data167 dataset format)
    # ------------------------------------------------------------------

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Lazily init the ROS node so multiple Robot instances can coexist.
        if not rospy.core.is_initialized():
            rospy.init_node("woan_piper_robot", anonymous=True, disable_signals=True)

        self._sub_master_left = rospy.Subscriber(
            MASTER_LEFT_TOPIC, JointState, self._master_left_cb, queue_size=100, tcp_nodelay=True,
        )
        self._sub_master_right = rospy.Subscriber(
            MASTER_RIGHT_TOPIC, JointState, self._master_right_cb, queue_size=100, tcp_nodelay=True,
        )
        self._sub_puppet_left = rospy.Subscriber(
            PUPPET_LEFT_TOPIC, JointState, self._puppet_left_cb, queue_size=100, tcp_nodelay=True,
        )
        self._sub_puppet_right = rospy.Subscriber(
            PUPPET_RIGHT_TOPIC, JointState, self._puppet_right_cb, queue_size=100, tcp_nodelay=True,
        )

        if not self.config.teleop:
            self._pub_master_left = rospy.Publisher(MASTER_LEFT_TOPIC, JointState, queue_size=10)
            self._pub_master_right = rospy.Publisher(MASTER_RIGHT_TOPIC, JointState, queue_size=10)

        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        logger.info(f"{self} connected (teleop={self.config.teleop})")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for sub in (self._sub_master_left, self._sub_master_right,
                    self._sub_puppet_left, self._sub_puppet_right):
            if sub is not None:
                sub.unregister()

        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------
    # Read APIs
    # ------------------------------------------------------------------

    def _combined(self, left: list, right: list) -> dict[str, float]:
        """Combine left[7] + right[7] into a 14-dim {motor.pos: value} dict."""
        out = {}
        for i, motor in enumerate(self.LEFT_MOTORS):
            out[f"{motor}.pos"] = float(left[i])
        for i, motor in enumerate(self.RIGHT_MOTORS):
            out[f"{motor}.pos"] = float(right[i])
        return out

    def get_observation(self) -> dict[str, Any]:
        """state = puppet (slave actual) + camera frames."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._lock:
            pl, pr = self._puppet_left, self._puppet_right

        if pl is None or pr is None:
            return {}
        if len(pl) < len(self.LEFT_MOTORS) or len(pr) < len(self.RIGHT_MOTORS):
            logger.warning(
                f"puppet joints too short: left={len(pl)} right={len(pr)}, expected >= 7 each"
            )
            return {}

        obs_dict = self._combined(pl, pr)
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()
        return obs_dict

    def get_action(self) -> dict[str, Any]:
        """action = master (teleop command) — what the human operator just commanded."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._lock:
            ml, mr = self._master_left, self._master_right

        if ml is None or mr is None:
            return {}
        if len(ml) < len(self.LEFT_MOTORS) or len(mr) < len(self.RIGHT_MOTORS):
            logger.warning(
                f"master joints too short: left={len(ml)} right={len(mr)}, expected >= 7 each"
            )
            return {}

        return self._combined(ml, mr)

    # ------------------------------------------------------------------
    # Write APIs (replay / inference; not used during teleop recording)
    # ------------------------------------------------------------------

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._pub_master_left is None or self._pub_master_right is None:
            raise RuntimeError(
                "send_action called but action publishers are not initialised "
                "(robot was constructed in teleop mode)."
            )

        # Adaptive Kalman smoothing on each motor channel.
        if not self._kalman_initialized:
            for key, value in action.items():
                if key.endswith(".pos"):
                    motor = key.removesuffix(".pos")
                    if motor in self.kalman_filters:
                        self.kalman_filters[motor].initialize_state_from_measurement(np.array([value]))
            self._kalman_initialized = True

        filtered = {}
        for key, value in action.items():
            if not key.endswith(".pos"):
                continue
            motor = key.removesuffix(".pos")
            if motor in self.kalman_filters:
                kf = self.kalman_filters[motor]
                kf.predict()
                filtered[key] = float(kf.update(np.array([value]))[0])
            else:
                filtered[key] = value

        # Split into left / right and publish on the master command channels.
        left_msg = JointState()
        left_msg.name = list(self.LEFT_MOTORS)
        left_msg.position = [float(filtered[f"{m}.pos"]) for m in self.LEFT_MOTORS]
        left_msg.header.stamp = rospy.Time.now()

        right_msg = JointState()
        right_msg.name = list(self.RIGHT_MOTORS)
        right_msg.position = [float(filtered[f"{m}.pos"]) for m in self.RIGHT_MOTORS]
        right_msg.header.stamp = rospy.Time.now()

        self._pub_master_left.publish(left_msg)
        self._pub_master_right.publish(right_msg)

        return {f"{m}.pos": filtered[f"{m}.pos"] for m in self.motors}

    # ------------------------------------------------------------------
    # Homing
    # ------------------------------------------------------------------

    def go_home(self, settle_time: float = 3.0, srv_timeout: float = 2.0) -> None:
        """Move both master and slave arms to zero pose using kai0 ROS1 services."""
        if not self._is_connected:
            logger.warning("go_home called while disconnected; skipping")
            return

        used_service = True
        for srv_name in GO_ZERO_SERVICES:
            try:
                rospy.wait_for_service(srv_name, timeout=srv_timeout)
                rospy.ServiceProxy(srv_name, Trigger)()
            except (rospy.ROSException, rospy.ServiceException) as e:
                logger.warning(f"go_home: {srv_name} unavailable ({e})")
                used_service = False

        if used_service:
            time.sleep(settle_time)
            for srv_name in RESTORE_MS_SERVICES:
                try:
                    rospy.wait_for_service(srv_name, timeout=srv_timeout)
                    rospy.ServiceProxy(srv_name, Trigger)()
                except (rospy.ROSException, rospy.ServiceException) as e:
                    logger.warning(f"go_home: {srv_name} unavailable ({e})")
            logger.info("go_home: arms homed via service, master-slave mode restored")
            return

        # Fallback: publish zero JointState on /master/joint_* (slave-only).
        if self._pub_master_left is None or self._pub_master_right is None:
            logger.warning(
                "go_home fallback unavailable: action publishers are None "
                "(robot constructed in teleop mode). Master-slave service is required."
            )
            return
        for pub, motors in (
            (self._pub_master_left, self.LEFT_MOTORS),
            (self._pub_master_right, self.RIGHT_MOTORS),
        ):
            msg = JointState()
            msg.name = list(motors)
            msg.position = [0.0] * len(motors)
            msg.header.stamp = rospy.Time.now()
            pub.publish(msg)
        time.sleep(settle_time)
        logger.info("go_home: published zeros on /master/joint_* (fallback)")
