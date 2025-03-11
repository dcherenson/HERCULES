#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion
import yaml
import cv2
import os
import math
import numpy as np

class MapPublisher(Node):
    def __init__(self):
        super().__init__('custom_map_server')
        # Declare a parameter for the YAML file path
        self.declare_parameter('yaml_file', '/home/sgarimella34/multi-robot-coordination/trajectory_data/occupancy_grid_maps/ogm_test1.yaml')
        yaml_file = self.get_parameter('yaml_file').value

        # Load the YAML file
        try:
            with open(yaml_file, 'r') as f:
                map_yaml = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load YAML file: {yaml_file} ({e})")
            return

        # Extract parameters from the YAML
        image_file = map_yaml.get('image')
        resolution = map_yaml.get('resolution')
        origin = map_yaml.get('origin')  # Expecting [x, y, theta]
        negate = map_yaml.get('negate')
        occupied_thresh = map_yaml.get('occupied_thresh')
        free_thresh = map_yaml.get('free_thresh')

        if image_file is None or resolution is None or origin is None:
            self.get_logger().error("YAML file is missing required parameters.")
            return

        # Build the full image path (assuming image is in the same directory as the YAML)
        yaml_dir = os.path.dirname(os.path.abspath(yaml_file))
        image_path = os.path.join(yaml_dir, image_file)

        # Load the map image as grayscale
        map_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if map_image is None:
            self.get_logger().error(f"Failed to load map image from {image_path}")
            return

        # Create and fill the OccupancyGrid message
        self.occupancy_grid = OccupancyGrid()
        self.occupancy_grid.header.frame_id = 'map'
        self.occupancy_grid.info.resolution = resolution
        self.occupancy_grid.info.width = map_image.shape[1]
        self.occupancy_grid.info.height = map_image.shape[0]
        self.occupancy_grid.info.origin.position.x = float(origin[0])
        self.occupancy_grid.info.origin.position.y = float(origin[1])
        self.occupancy_grid.info.origin.position.z = 0.0

        # Convert yaw (origin[2]) to a quaternion for the map's orientation
        yaw = float(origin[2])
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        self.occupancy_grid.info.origin.orientation = q

        # Convert the image to occupancy data.
        # Each pixel is normalized to [0,1] and, if needed, inverted using the negate flag.
        # Then we apply thresholds:
        #   - If occ > occupied_thresh:
        #         if occ >= 0.9: value = 100 (fully occupied)
        #         else:          value = 50  (partially occupied)
        #   - If occ < free_thresh: value = 0 (free)
        #   - Else: unknown (-1)
        rows, cols = map_image.shape
        data = np.empty((rows, cols), dtype=np.int8)

        for y in range(rows):
            for x in range(cols):
                occ = map_image[y, x] / 255.0  # normalize to [0,1]
                if negate:
                    occ = 1.0 - occ

                if occ > occupied_thresh:
                    if occ >= 0.9:
                        value = 100
                    else:
                        value = 50
                elif occ < free_thresh:
                    value = 0
                else:
                    value = -1
                data[y, x] = value

        # ROS occupancy grids are defined with the origin at the bottom-left.
        # Flip the data vertically.
        data = np.flipud(data)
        self.occupancy_grid.data = data.flatten().tolist()

        # Create a publisher on the desired topic "sliced_projected_map"
        self.publisher = self.create_publisher(OccupancyGrid, 'sliced_projected_map', 10)
        # Publish the map periodically (for example, every second)
        self.timer = self.create_timer(1.0, self.publish_map)

    def publish_map(self):
        self.occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.occupancy_grid)
        self.get_logger().info("Published occupancy grid map on 'sliced_projected_map'")

def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
