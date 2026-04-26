#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState
import time
import argparse
import sys
from rclpy.qos import QoSProfile, ReliabilityPolicy

class ActionPlayer(Node):
    def __init__(self):
        super().__init__('action_player')

        self.last_log_time = 0
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(JointState, '/piper/sent_actions', self.action_callback, qos_profile)

        # self.joint_commands_pub = self.create_publisher(Float64MultiArray, '/record_data', qos_profile)
        # self.gripper_command_pub = self.create_publisher(Int32, '/joystick_info', qos_profile)
        self.left_pub = self.create_publisher(JointState, "/joint_ctrl_cmd_left", 10)
        self.right_pub = self.create_publisher(JointState, "/joint_ctrl_cmd_right", 10)


    def action_callback(self, msg: JointState):
        current_time = time.time() 
        joint_command = [0.0] * 14
        left_gripper_pos = 0.0
        right_gripper_pos = 0.0

        joint_name_to_index = {
            "left_joint_1": 0, "left_joint_2": 1, "left_joint_3": 2, "left_joint_4": 3, "left_joint_5": 4, "left_joint_6": 5, "left_gripper": 6,
            "right_joint_1": 7, "right_joint_2": 8, "right_joint_3": 9, "right_joint_4": 10, "right_joint_5": 11, "right_joint_6": 12, "right_gripper": 13
        }

          # 定义左臂和右臂的关节名称
        left_joint_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"
        ]
        right_joint_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"
        ]

        # 初始化左臂和右臂的 JointState 消息
        left_msg = JointState()
        right_msg = JointState()

        # 设置时间戳
        left_msg.header.stamp = self.get_clock().now().to_msg()
        right_msg.header.stamp = self.get_clock().now().to_msg()

        # 遍历接收到的关节数据，根据名称拆分到左右臂
        for name, position in zip(msg.name, msg.position):
            if name.startswith("left_"):
                # 去掉前缀并移除下划线，映射到左臂关节名称
                stripped_name = name.replace("left_", "").replace("_", "")
                if stripped_name in left_joint_names:
                    left_msg.name.append(stripped_name)
                    left_msg.position.append(position)
                    # left_msg.velocity.append(velocity)
                    # left_msg.effort.append(effort)
            elif name.startswith("right_"):
                # 去掉前缀并移除下划线，映射到右臂关节名称
                stripped_name = name.replace("right_", "").replace("_", "")
                if stripped_name in right_joint_names:
                    right_msg.name.append(stripped_name)
                    right_msg.position.append(position)
                    # right_msg.velocity.append(velocity)
                    # right_msg.effort.append(effort)

        print("left_msg:",left_msg)
        print("right_msg:",right_msg)
        # 发布左右臂的 JointState 消息
        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

    def destroy_node(self):
        # self.file.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = ActionPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
