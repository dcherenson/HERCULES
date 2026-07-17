#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

class MarkerPublisher(Node):
    def __init__(self):
        super().__init__('marker_publisher')
        self.publisher_ = self.create_publisher(Marker, 'visualization_marker', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.locations = [
            {'x': 1.0, 'y': 2.0, 'z': 0.0}, # Red marker
            {'x': 2.0, 'y': 0.0, 'z': 0.0},   # Green marker
            {'x': -1.0, 'y': -2.0, 'z': 0.0},  # Blue marker
        ]
        # Define unique colors for each marker: red, green, blue.
        self.colors = [
            {'r': 1.0, 'g': 0.0, 'b': 0.0},  # Red
            {'r': 0.0, 'g': 1.0, 'b': 0.0},  # Green
            {'r': 0.0, 'g': 0.0, 'b': 1.0},  # Blue
        ]
        self.get_logger().info("Marker publisher started.")

    def timer_callback(self):
        for idx, loc in enumerate(self.locations):
            marker = Marker()
            marker.header.frame_id = "map"  # Change to your desired reference frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "my_markers"
            marker.id = idx           # Unique ID for each marker
            marker.type = Marker.SPHERE  # Marker type (can be changed to CUBE, ARROW, etc.)
            marker.action = Marker.ADD
            # Set the marker's pose (position and orientation)
            marker.pose.position.x = loc['x']
            marker.pose.position.y = loc['y']
            marker.pose.position.z = loc['z']
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            # Make markers larger: scale increased to 0.5
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            # Set unique color for each marker
            marker.color.a = 1.0  # Fully opaque
            marker.color.r = self.colors[idx]['r']
            marker.color.g = self.colors[idx]['g']
            marker.color.b = self.colors[idx]['b']
            # Publish the marker
            self.publisher_.publish(marker)
            self.get_logger().info(
                f"Published marker {idx} at (x: {loc['x']}, y: {loc['y']}, z: {loc['z']})"
            )

def main(args=None):
    rclpy.init(args=args)
    node = MarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Allow graceful shutdown on Ctrl+C
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
