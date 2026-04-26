#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt8
from std_msgs.msg import Int32
import time
import argparse
import sys
import threading


class FullCommandRecorder(Node):
    def __init__(self, output_file):
        super().__init__('full_command_recorder')
        self._lock = threading.Lock()

        self.last_log_time = 0
        self.last_action_time = -1
        self.last_state_time = -1
        self.last_left_gripper_time = -1
        self.last_right_gripper_time = -1

        self.joint_action = None
        self.joint_action_left = None
        self.joint_action_right = None

        self.joint_state = None
        self.joint_state_left = None
        self.joint_state_right = None

        self.joint_state_pub = self.create_publisher(JointState, '/piper/recorded_joint_actions_states', 10)

        # self.create_subscription(
        #     JointState,
        #     '/joint_states_ctrl_left',
        #     self.joint_action_left_callback,
        #     10
        # )
        # self.create_subscription(
        #     JointState,
        #     '/joint_states_ctrl_right',
        #     self.joint_action_right_callback,
        #     10
        # )

        self.create_subscription(
            JointState,
            '/joint_states_left',
            self.joint_state_left_callback,
            10
        )
        self.create_subscription(
            JointState,
            '/joint_states_right',
            self.joint_state_right_callback,
            10
        )

        self.create_timer(1.0 / 100.0, self.timer_callback)

    # def joint_action_left_callback(self, msg):
    #     with self._lock:
    #         self.joint_action_left = list(msg.position)

    # def joint_action_right_callback(self, msg):
    #     with self._lock:
    #         self.joint_action_right = list(msg.position)

    def joint_state_left_callback(self, msg):
        with self._lock:
            self.joint_state_left = list(msg.position)

    def joint_state_right_callback(self, msg):
        with self._lock:
            self.joint_state_right = list(msg.position)


    def timer_callback(self):
        # if self.joint_state_left is None or self.joint_state_right is None or self.joint_action_left is None or self.joint_action_right is None:
        #     self.get_logger().warn("Incomplete joint state data, skipping this cycle.")
        #     return
        # with self._lock:
        #     self.joint_action = self.joint_action_left + self.joint_action_right
        #     self.joint_state = self.joint_state_left + self.joint_state_right

        if self.joint_state_left is None or self.joint_state_right is None:
            self.get_logger().warn("Incomplete joint state data, skipping this cycle.")
            return
        with self._lock:
            self.joint_state = self.joint_state_left + self.joint_state_right

        timestamp_ms = int(time.time() * 1000)
        current_time = time.time()  # 当前时间（单位：秒）

        # 创建 JointState 消息
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = [
            "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
            "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper"
        ]

        # 确保所有元素都是 float 类型
        # joint_state_msg.position = list(map(float, self.joint_action)) 
        joint_state_msg.position = list(map(float, self.joint_state)) 
        joint_state_msg.velocity = list(map(float, self.joint_state))
        joint_state_msg.effort = [0.0] * 14  # 假设没有力矩信息，这里填充为0

        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            self.get_logger().info(f"Timestamp: {timestamp_ms} ms")
            self.get_logger().info(f"JointState name: {joint_state_msg.name}")
            self.get_logger().info(f"JointState position (action): {joint_state_msg.position}")
            self.get_logger().info(f"JointState velocity (state): {joint_state_msg.velocity}")

        # 发布 JointState 消息
        self.joint_state_pub.publish(joint_state_msg)

    def destroy_node(self):
        super().destroy_node()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='full_joint_gripper_commands.txt',
                        help='Output file name')
    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)  # 把 ROS 参数和 argparse 参数分开
    node = FullCommandRecorder(args.output)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
