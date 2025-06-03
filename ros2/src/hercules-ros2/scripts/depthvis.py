#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import argparse
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

class DepthToPointCloudNode(Node):
    def __init__(self, robot_name):
        super().__init__('depth_to_pointcloud_node')
        self.bridge = CvBridge()
        self.robot_name = robot_name
        self.camera_info = None

        self.depth_topic = f'/hercules_node/{robot_name}/front_center_DepthPerspective/image'
        self.info_topic = f'/hercules_node/{robot_name}/front_center_DepthPerspective/camera_info'

        self.get_logger().info(f'Subscribing to:\n  {self.depth_topic}\n  {self.info_topic}')

        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(CameraInfo, self.info_topic, self.info_callback, latched_qos)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)

        self.pc_pub = self.create_publisher(PointCloud2, f'/depth_pointcloud/{robot_name}', 10)

    def info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info('Received and cached camera info.')

    def depth_callback(self, depth_msg):
        if self.camera_info is None:
            self.get_logger().warn('Waiting for camera info...')
            return

        # convert ROS Image → NumPy float32 array (ray‐lengths in meters)
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Could not convert depth image: {e}')
            return

        height, width = depth_image.shape
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # trim unstable border pixels if you like
        margin = 5
        u_all, v_all = np.meshgrid(
            np.arange(margin, width - margin),
            np.arange(margin, height - margin)
        )

        # r = DepthPerspective ray‐length at each pixel
        r_all = depth_image[v_all, u_all]

        # compute normalized ray‐direction components
        xdir = (u_all - cx) / fx
        ydir = (v_all - cy) / fy

        # ray norm = sqrt(xdir^2 + ydir^2 + 1)
        ray_norm = np.sqrt(xdir**2 + ydir**2 + 1.0)

        # recover true forward depth Zc = r / ||d||
        zc_all = r_all / ray_norm

        # mask out invalids
        valid = np.logical_and(zc_all > 0.1, zc_all < 500.0)
        if not np.any(valid):
            self.get_logger().warn('No valid points in depth image.')
            return

        # gather only valid points
        zc = zc_all[valid]
        x = xdir[valid] * zc
        y = ydir[valid] * zc

        # assemble Nx3 array
        points = np.vstack((x, y, zc)).T

        # publish as PointCloud2
        cloud_msg = pc2.create_cloud_xyz32(depth_msg.header, points)
        self.pc_pub.publish(cloud_msg)
        self.get_logger().info(f'Published point cloud with {points.shape[0]} points.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', required=True, choices=['Drone1', 'Drone2', 'Husky1', 'Husky2'],
                        help='Robot name')
    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = DepthToPointCloudNode(args.robot)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
