#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from airsim_interfaces.msg import StringArray


class EchoNode(Node):
    def __init__(self):
        super().__init__('echo_node')
        self.sub = self.create_subscription(
            StringArray,
            '/hercules_node/Drone1/lidar/labels/LidarSensor1',
            self.callback,
            10
        )

    def callback(self, msg: StringArray):
        self.get_logger().info(f"Received message with {len(msg.data)} elements.")
        for i, label in enumerate(msg.data):
            print(f"{i}: {label}")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = EchoNode()
    rclpy.spin(node)   # <- use node reference directly in Humble
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
