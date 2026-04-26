#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_piper_robot import PiperRobotConfig

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
import numpy as np

logger = logging.getLogger(__name__)

class AdaptiveKalmanFilter:
    def __init__(self, dim, process_variance, measurement_variance, threshold=5.0, scale_factor=10.0):
        """
        自适应卡尔曼过滤器。
        Args:
            dim (int): 状态变量的维度
            process_variance (float): 初始过程噪声的协方差
            measurement_variance (float): 初始测量噪声的协方差
            threshold (float): 判断变化幅度是否较大的阈值
            scale_factor (float): 自适应变化的缩放因子
        """
        self.dim = dim
        self.x = np.zeros(dim)  # 状态向量初始化
        self.P = np.ones(dim)   # 协方差矩阵初始化
        self.Q = process_variance * np.eye(dim)  # 系统过程噪声
        self.R = measurement_variance * np.eye(1)  # 测量噪声
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.prev_measurement = np.zeros(1)
        self.delta_t = 0.03333

        # 状态转移矩阵（角度和速度的预测公式）
        self.A = np.array([[1, self.delta_t],   # position = position + velocity * delta_t
                           [0, 1]])             # velocity = velocity (保持速度)

    def initialize_state_from_measurement(self, initial_measurement):
        """
        根据初始观测初始化状态。
        Args:
            initial_measurement (np.ndarray): 初始测量值
        """
        self.x[0] = initial_measurement[0]
        self.x[1] = 0.0  # 假设初始速度为零
        self.P = np.array([0.1, 0.1])  # 初始化较小协方差
        self.prev_measurement[0] = initial_measurement[0]

    def predict(self):
        """
        使用运动模型预测当前状态。
        """
        # 预测下一状态（基于状态转移矩阵）
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q  # 更新协方差矩阵

    def update(self, measurement):
        """
        更新状态估计。
        Args:
            measurement (np.ndarray): 测量值（如角度）
        Returns:
            np.ndarray: 更新后的状态 [位置, 速度]
        """
        # 动态调整测量噪声 R
        # 动态调整测量噪声 R
        delta = np.abs(measurement - self.prev_measurement)
        # delta_rate = delta / self.delta_t  # 引入速率变化判断
        if np.any(delta > self.threshold):  # 如果变化速率较大，增大测量噪声
            self.R = self.scale_factor * np.eye(1)
        else:  # 如果变化速率较小，恢复测量噪声
            self.R = 7.5 * np.eye(1)
 
        # 卡尔曼增益计算
        H = np.array([[1, 0]])  # 观测矩阵，仅观测位置
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
 
        # 更新状态
        y = measurement - H @ self.x  # 残差
        self.x += K @ y
        self.P = (np.eye(self.dim) - K @ H) @ self.P
 
        # 保存当前测量值
        self.prev_measurement = measurement
 
        return self.x

class PiperRobot(Robot, Node):
    config_class = PiperRobotConfig
    name = "piper_robot"

    def __init__(self, config: PiperRobotConfig):
        Robot.__init__(self, config)
        Node.__init__(self, "piper_robot_node")
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        self._is_connected = False

        self.joint_states = None
        self.joint_actions = None

        self.motors = [
            "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
            "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper"
        ]

        self.action_publisher = None

        # kalman滤波初始化标志
        self._kalman_initialized = False

        # 为每个关节创建自适应卡尔曼滤波器实例
        self.kalman_filters = {
            motor: AdaptiveKalmanFilter(dim=2, process_variance=0.5, measurement_variance=5, 
                                         threshold=0.075, scale_factor=15.0)
            for motor in self.motors
        }

        self.create_subscription(JointState, "/piper/recorded_joint_actions_states", self._joint_states_callback, 10)


        if not config.teleop:
            self.action_publisher = self.create_publisher(JointState, "/piper/sent_actions", 10)
    
    def _joint_states_callback(self, msg: JointState):
        if not self._is_connected:
            return
        # logger.info(f"Received joint states: {msg.name} with positions {msg.position}")
        # logger.info(f"Received joint velocities: {msg.name} with velocities {msg.velocity}")
        
        self.joint_actions = {f"{name}.pos": position for name, position in zip(msg.name, msg.position)}
        self.joint_states = {f"{name}.pos": position for name, position in zip(msg.name, msg.velocity)}

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

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        logger.info(f"{self} connected.")
        logger.info(self._is_connected)

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        

        if self.joint_states is None:
            return {}

        obs_dict = self.joint_states.copy()

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    # def get_observation(self) -> dict[str, Any]:
    #     if not self._is_connected:
    #         raise DeviceNotConnectedError(f"{self} is not connected.")
        
    #     logger.info(f"[DEBUG] get_observation called, joint_states: {self.joint_states}")


    #     obs_dict = {}

    #     # === 修复：保证 joint state 出现在 observation.state 下 ===
    #     state_dict = {}
    #     if self.joint_states is not None and len(self.joint_states) > 0:
    #         state_dict.update(self.joint_states)
    #     elif self.joint_actions is not None and len(self.joint_actions) > 0:
    #         state_dict.update(self.joint_actions)

    #     if len(state_dict) > 0:
    #         obs_dict["observation.state"] = state_dict

    #     # === 相机图像 ===
    #     for cam_key, cam in self.cameras.items():
    #         obs_dict[f"observation.images.{cam_key}"] = cam.async_read()

    #     return obs_dict



    def get_action(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.joint_actions is None:
            return {}

        action_dict = self.joint_actions.copy()

        return action_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # 如果还未初始化，则用第一个动作来初始化卡尔曼滤波器的状态
        if not self._kalman_initialized:
            for key, value in action.items():
                if key.endswith(".pos"):
                    motor = key.removesuffix(".pos")
                    if motor in self.kalman_filters:
                        self.kalman_filters[motor].initialize_state_from_measurement(np.array([value]))
            self._kalman_initialized = True
        
        # 对每个动作进行自适应卡尔曼滤波
        filtered_action = {}
        for key, value in action.items():
            if key.endswith(".pos"):
                motor = key.removesuffix(".pos")
                if motor in self.kalman_filters:
                    # 应用自适应卡尔曼滤波器
                    kf = self.kalman_filters[motor]
                    kf.predict()  # 预测状态
                    filtered_value = float(kf.update(np.array([value]))[0])  # 使用卡尔曼滤波器更新状态
                    filtered_action[key] = filtered_value
                else:
                    # 如果没有对应的滤波器，直接传递原始值
                    filtered_action[key] = value

        # logger.info(f"Original action: {action}")
        # logger.info(f"Filtered action: {filtered_action}")

        # goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        goal_pos = {key.removesuffix(".pos"): val for key, val in filtered_action.items() if key.endswith(".pos")}

        # 发布 JointState（除 gripper 外的关节动作）
        joint_state_msg = JointState()
        joint_state_msg.name = list(goal_pos.keys())
        # joint_state_msg.position = list(goal_pos.values())
        joint_state_msg.position = [float(val) for val in goal_pos.values()]
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()

        if self.action_publisher is not None:
            self.action_publisher.publish(joint_state_msg)


        # 返回合并动作
        result = {f"{motor}.pos": val for motor, val in goal_pos.items()}

        return result

    def go_home(self, settle_time: float = 3.0, srv_timeout: float = 2.0) -> None:
        """Move both master and slave arms to zero pose.

        Tries ROS2 services /can_{left,right}/go_zero_master_slave first
        (kai0-style — homes both master and slave simultaneously). If those
        services are unavailable, falls back to publishing a zero JointState
        on /piper/sent_actions (slave-only, master stays put).

        After service success, restores master-slave coupling mode via
        /can_{left,right}/restore_ms_mode so teleop continues to work.
        """
        if not self._is_connected:
            logger.warning("go_home called while disconnected; skipping")
            return

        used_service = True
        for srv_name in ("/can_left/go_zero_master_slave", "/can_right/go_zero_master_slave"):
            client = self.create_client(Trigger, srv_name)
            if not client.wait_for_service(timeout_sec=srv_timeout):
                logger.warning(f"go_home: service {srv_name} not available")
                used_service = False
                self.destroy_client(client)
                continue
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=settle_time)
            if future.done():
                resp = future.result()
                if resp is not None and not resp.success:
                    logger.warning(f"go_home: {srv_name} returned success=False ({resp.message})")
            else:
                logger.warning(f"go_home: {srv_name} timed out")
            self.destroy_client(client)

        if used_service:
            time.sleep(settle_time)
            for srv_name in ("/can_left/restore_ms_mode", "/can_right/restore_ms_mode"):
                client = self.create_client(Trigger, srv_name)
                if client.wait_for_service(timeout_sec=srv_timeout):
                    future = client.call_async(Trigger.Request())
                    rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                self.destroy_client(client)
            logger.info("go_home: arms homed via service, master-slave mode restored")
            return

        # Fallback: publish zero position on the action channel (slave only)
        if self.action_publisher is None:
            logger.warning(
                "go_home fallback unavailable: action_publisher is None "
                "(robot constructed in teleop mode). Master-slave service is required."
            )
            return
        msg = JointState()
        msg.name = list(self.motors)
        msg.position = [0.0] * len(self.motors)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.action_publisher.publish(msg)
        time.sleep(settle_time)
        logger.info("go_home: published zeros to /piper/sent_actions (fallback, slave-only)")
    
