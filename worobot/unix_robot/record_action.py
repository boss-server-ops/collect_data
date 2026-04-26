#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from unix_msgs.msg import GripperCommand
from sensor_msgs.msg import JointState
import time
import argparse
import sys

class FullCommandRecorder(Node):
    def __init__(self, output_file):
        super().__init__('full_command_recorder')

        self.last_log_time = 0

        # self.file = open(output_file, 'w')
        # self.get_logger().info(f"Recording to {output_file}")

        self.joint_action = None
        self.left_gripper_action = -1
        self.right_gripper_action = -1

        self.joint_state = None
        self.left_gripper_state = -1
        self.right_gripper_state = -1

        self.joint_state_pub = self.create_publisher(JointState, 'unix/recorded_joint_states', 10)


        self.create_subscription(
            Float64MultiArray,
            '/dual_arm_forward_position_controller/commands',
            self.joint_action_callback,
            10
        )
        self.create_subscription(
            GripperCommand,
            '/left_unix_gripper2_controller2/gripper_command',
            self.left_gripper_action_callback,
            10
        )
        self.create_subscription(
            GripperCommand,
            '/right_unix_gripper2_controller2/gripper_command',
            self.right_gripper_action_callback,
            10
        )
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.create_subscription(JointState,
            '/left_unix_gripper2_controller2/joint_states',
            self.left_gripper_state_callback,
            10
        )
        self.create_subscription(JointState,
            '/right_unix_gripper2_controller2/joint_states',
            self.right_gripper_state_callback,
            10
        )

        self.create_timer(1.0 / 100.0, self.timer_callback)

    def joint_action_callback(self, msg):
        self.joint_action = msg.data

    def left_gripper_action_callback(self, msg):
        if msg.command.position:
            self.left_gripper_action = msg.command.position[0]

    def right_gripper_action_callback(self, msg):
        if msg.command.position:
            self.right_gripper_action = msg.command.position[0]

    def joint_state_callback(self, msg):
        self.joint_state = [0.0] * 16

        # 定义关节名称与索引的映射
        joint_name_to_index = {
            "joint_la0": 0, "joint_la1": 1, "joint_la2": 2, "joint_la3": 3, "joint_la4": 4, "joint_la5": 5, "joint_la6": 6, "joint_la7": 7,
            "joint_ra0": 8, "joint_ra1": 9, "joint_ra2": 10, "joint_ra3": 11, "joint_ra4": 12, "joint_ra5": 13, "joint_ra6": 14, "joint_ra7": 15
        }

        # 遍历接收到的关节名称和位置
        for name, position in zip(msg.name, msg.position):
            if name in joint_name_to_index:
                index = joint_name_to_index[name]
                self.joint_state[index] = position

    def left_gripper_state_callback(self, msg):
        if not msg.position:
            return
        state = 1.0 - msg.position[0]
        if state > 0.5:
            self.left_gripper_state = 1.0
        else :
            self.left_gripper_state = 0.0

    def right_gripper_state_callback(self, msg):
        if not msg.position:
            return
        state = 1.0 - msg.position[0]
        if state > 0.5:
            self.right_gripper_state = 1.0
        else :
            self.right_gripper_state = 0.0

    def timer_callback(self):
        if self.joint_action is None or len(self.joint_action) < 16 or self.left_gripper_action < 0 or self.right_gripper_action < 0:
            self.get_logger().warn("Incomplete data, skipping this cycle.")
            return

        timestamp_ms = int(time.time() * 1000)
        current_time = time.time()  # 当前时间（单位：秒）

        left_arm_action = self.joint_action[0:8]
        right_arm_action = self.joint_action[8:16]

        # 创建 JointState 消息
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = [
            "la0", "la1", "la2", "la3", "la4", "la5", "la6", "la7", "left_gripper",
            "ra0", "ra1", "ra2", "ra3", "ra4", "ra5", "ra6", "ra7", "right_gripper"
        ]

        # 确保所有元素都是 float 类型
        joint_state_msg.position = list(map(float, left_arm_action)) + [float(self.left_gripper_action)] + list(map(float, right_arm_action)) + [float(self.right_gripper_action)]
        joint_state_msg.velocity = list(map(float, self.joint_state[0:8])) + [float(self.left_gripper_state)] + list(map(float, self.joint_state[8:16])) + [float(self.right_gripper_state)]
        joint_state_msg.effort = [0.0] * 18  # 假设没有力矩信息，这里填充为0

        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            self.get_logger().info(f"Timestamp: {timestamp_ms} ms")
            self.get_logger().info(f"JointState name: {joint_state_msg.name}")
            self.get_logger().info(f"JointState position (action): {joint_state_msg.position}")
            self.get_logger().info(f"JointState velocity (state): {joint_state_msg.velocity}")

        # 发布 JointState 消息
        self.joint_state_pub.publish(joint_state_msg)
        

        # todo: # 记录到文件
        # left_arm_str = ' '.join([f'{x:.6f}' for x in left_arm_action])
        # right_arm_str = ' '.join([f'{x:.6f}' for x in right_arm_action])

        # line = f'{timestamp_ms} {left_arm_str} {self.left_gripper_action:.6f} {right_arm_str} {self.right_gripper_action:.6f}'
        # # print(line, file=sys.stderr)
        # self.get_logger().info(f'Recorded: {line}')
        # self.file.write(line + '\n')
        # self.file.flush()

    def destroy_node(self):
        # self.file.close()
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
