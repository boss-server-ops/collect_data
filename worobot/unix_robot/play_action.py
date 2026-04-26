#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from unix_msgs.msg import GripperCommand
from sensor_msgs.msg import JointState
import time
import argparse
import sys

class ActionPlayer(Node):
    def __init__(self):
        super().__init__('action_player')

        self.last_log_time = 0

        self.create_subscription(JointState, 'unix/sent_actions', self.action_callback, 10)

        self.joint_commands_pub = self.create_publisher(Float64MultiArray, '/dual_arm_forward_position_controller/commands', 10)

        self.left_gripper_command_pub = self.create_publisher(GripperCommand, '/left_unix_gripper2_controller2/gripper_command', 10)

        self.right_gripper_command_pub = self.create_publisher(GripperCommand, '/right_unix_gripper2_controller2/gripper_command', 10)

    def action_callback(self, msg: JointState):
        current_time = time.time() 
        joint_command = [0.0] * 16
        left_gripper_pos = 0.0
        right_gripper_pos = 0.0

        joint_name_to_index = {
            "la0": 0, "la1": 1, "la2": 2, "la3": 3, "la4": 4, "la5": 5, "la6": 6, "la7": 7,
            "ra0": 8, "ra1": 9, "ra2": 10, "ra3": 11, "ra4": 12, "ra5": 13, "ra6": 14, "ra7": 15
        }

        for name, position in zip(msg.name, msg.position):
            if name in joint_name_to_index:
                joint_command[joint_name_to_index[name]] = position
            elif name == "left_gripper":
                left_gripper_pos = position
            elif name == "right_gripper":
                right_gripper_pos = position

        if left_gripper_pos < 0.5:
            left_gripper_pos = 0.0
        else:
            left_gripper_pos = 1.0
        
        if right_gripper_pos < 0.5:
            right_gripper_pos = 0.0
        else:
            right_gripper_pos = 1.0

        # 发布关节控制命令
        joint_msg = Float64MultiArray()
        joint_msg.data = joint_command
        self.joint_commands_pub.publish(joint_msg)

        # 发布左右夹爪控制命令
        left_cmd = GripperCommand()
        left_cmd.command.position = [left_gripper_pos]
        self.left_gripper_command_pub.publish(left_cmd)

        right_cmd = GripperCommand()
        right_cmd.command.position = [right_gripper_pos]
        self.right_gripper_command_pub.publish(right_cmd)

        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            self.get_logger().info(f"publish joint commands:  {joint_command}")
            self.get_logger().info(f"publish left gripper command:  {left_gripper_pos}")
            self.get_logger().info(f"publish right gripper command:  {right_gripper_pos}")

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
