#!/usr/bin/env python3
"""
full_topic_echo.py – print one StringArray message, with optional filtering.

Toggles
-------
FILTER_EXCLUDED  : if True, skip any label that appears in EXCLUDED_LABELS
UNIQUE_ONLY      : if True, print each kept label only once (deduplicated)
"""

import rclpy
from rclpy.node import Node
from airsim_interfaces.msg import StringArray

# ─── USER-ADJUSTABLE SWITCHES ──────────────────────────────────────────────────
FILTER_EXCLUDED = True          # set False to print every element
UNIQUE_ONLY     = True          # set False to keep duplicates
# ------------------------------------------------------------------------------
EXCLUDED_LABELS = {
    "out_of_range",
    "LandscapeStreamingProxy_D7T4VF4LBP34PHV9XOEVM5TIG_1_4_4_0",
}

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
        # Decide which labels to keep
        kept = []
        seen = set()

        for label in msg.data:
            if FILTER_EXCLUDED and label in EXCLUDED_LABELS:
                continue                      # skip unwanted labels

            if UNIQUE_ONLY:
                if label in seen:
                    continue                  # skip duplicates
                seen.add(label)

            kept.append(label)

        self.get_logger().info(
            f"Received {len(msg.data)} elements, printing {len(kept)}."
        )
        for i, label in enumerate(kept):
            print(f"{i}: {label}")

        rclpy.shutdown()


def main():
    rclpy.init()
    node = EchoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
