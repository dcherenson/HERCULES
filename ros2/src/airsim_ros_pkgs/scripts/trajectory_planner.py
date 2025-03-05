#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import random
import math
import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header

# Octomap message type
from octomap_msgs.msg import Octomap

# Try to import a hypothetical Python binding for octomap.
# You must have a module 'octomap' that provides OcTree functionality.
try:
    import octomap
except ImportError:
    octomap = None


class TrajectoryPlanner(Node):
    def __init__(self):
        super().__init__('trajectory_planner')
        
        # Declare and read parameters
        self.declare_parameter('z_height', 0.25)
        self.declare_parameter('trajectory_length', 100.0)  # meters
        self.declare_parameter('square_size', 100.0)          # planning area side (meters)
        
        self.z_height = self.get_parameter('z_height').value
        self.trajectory_length = self.get_parameter('trajectory_length').value
        self.square_size = self.get_parameter('square_size').value
        
        # Set up a 2D occupancy grid covering the planning square.
        # The grid is defined with an origin (lower left) and a resolution.
        self.grid_origin = (-self.square_size/2, -self.square_size/2)
        self.grid_resolution = 0.25   # meters per cell
        self.grid_width = int(self.square_size / self.grid_resolution)
        self.grid_height = int(self.square_size / self.grid_resolution)
        # Initialize the grid as completely free (0 = free, 1 = occupied).
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        
        # Subscribe to the octomap topic (assumed to be published as a binary octomap).
        self.octomap_sub = self.create_subscription(
            Octomap,
            '/octomap_binary',
            self.octomap_callback,
            10)
        
        # Publishers for the planned path and for visualization in RViz2.
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.marker_pub = self.create_publisher(Marker, '/trajectory_marker', 10)
        
        # Timer to plan and publish a new trajectory every second.
        self.create_timer(1.0, self.timer_callback)
        
        self.get_logger().info("Trajectory Planner Node Initialized")
    
    def octomap_callback(self, msg: Octomap):
        self.get_logger().info("Received Octomap message, updating occupancy grid")
        if octomap is None:
            self.get_logger().warn("octomap python module not installed; occupancy grid not updated")
            return
        
        # Create an OcTree from the binary data.
        try:
            # Construct the OcTree using the resolution from the message.
            tree = octomap.OcTree(msg.resolution)
            # Convert the msg.data (a list of uint8) to a bytearray and read it.
            tree.readBinaryFromBytes(bytearray(msg.data))
        except Exception as e:
            self.get_logger().error(f"Failed to decode octomap: {e}")
            return
        
        # Create a new occupancy grid (2D array).
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        
        # Iterate over all leaf nodes in the tree.
        # (Assumes the octomap.OcTree object is iterable over its leaf nodes.)
        try:
            for node in tree:
                # Check if the node is marked as occupied.
                if node.occupied:
                    # Get the center coordinates of this node (x, y, z).
                    x, y, z = node.center()  # Assumes this returns a tuple.
                    # Project the node onto the planning plane if its z is close to our fixed height.
                    if abs(z - self.z_height) < (self.grid_resolution / 2):
                        ix = int((x - self.grid_origin[0]) / self.grid_resolution)
                        iy = int((y - self.grid_origin[1]) / self.grid_resolution)
                        if 0 <= ix < self.grid_width and 0 <= iy < self.grid_height:
                            grid[iy, ix] = 1
        except Exception as e:
            self.get_logger().error(f"Error iterating over octomap nodes: {e}")
            return
        
        self.occupancy_grid = grid
        self.get_logger().info("Occupancy grid updated")
    
    def is_free(self, x, y, z):
        # Check if a point (x, y, z) is free using the occupancy grid.
        ix = int((x - self.grid_origin[0]) / self.grid_resolution)
        iy = int((y - self.grid_origin[1]) / self.grid_resolution)
        if ix < 0 or ix >= self.grid_width or iy < 0 or iy >= self.grid_height:
            return False  # Out of bounds.
        return self.occupancy_grid[iy, ix] == 0
    
    def check_line_free(self, p1, p2):
        # Check if the line segment between points p1 and p2 is free by sampling intermediate points.
        distance = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        steps = int(distance / (self.grid_resolution / 2))
        for i in range(steps+1):
            t = i / steps
            x = p1[0] + t * (p2[0]-p1[0])
            y = p1[1] + t * (p2[1]-p1[1])
            z = self.z_height
            if not self.is_free(x, y, z):
                return False
        return True
    
    def plan_trajectory(self):
        """
        Plan a collision-free trajectory by randomly sampling points in free space (within a square)
        at the fixed z-height. Points are connected if the line between them is collision-free.
        The process stops when the cumulative path length reaches the desired trajectory_length.
        """
        trajectory = []
        
        # Find a random free starting point.
        while True:
            start_x = random.uniform(self.grid_origin[0], self.grid_origin[0] + self.square_size)
            start_y = random.uniform(self.grid_origin[1], self.grid_origin[1] + self.square_size)
            if self.is_free(start_x, start_y, self.z_height):
                start_point = (start_x, start_y, self.z_height)
                break
        
        trajectory.append(start_point)
        current_point = start_point
        total_length = 0.0
        max_attempts = 1000
        
        for _ in range(max_attempts):
            if total_length >= self.trajectory_length:
                break
            # Randomly sample a candidate point.
            next_x = random.uniform(self.grid_origin[0], self.grid_origin[0] + self.square_size)
            next_y = random.uniform(self.grid_origin[1], self.grid_origin[1] + self.square_size)
            next_point = (next_x, next_y, self.z_height)
            # Check candidate: must be free and the connecting line collision-free.
            if self.is_free(next_x, next_y, self.z_height) and self.check_line_free(current_point, next_point):
                seg_length = math.sqrt((next_point[0]-current_point[0])**2 +
                                       (next_point[1]-current_point[1])**2)
                if total_length + seg_length > self.trajectory_length:
                    # Trim the final segment to exactly reach the desired trajectory length.
                    remaining = self.trajectory_length - total_length
                    dx = next_point[0] - current_point[0]
                    dy = next_point[1] - current_point[1]
                    scale = remaining / math.sqrt(dx*dx + dy*dy)
                    next_point = (current_point[0] + dx*scale,
                                  current_point[1] + dy*scale,
                                  self.z_height)
                    if not self.check_line_free(current_point, next_point):
                        continue
                    trajectory.append(next_point)
                    total_length += remaining
                    break
                trajectory.append(next_point)
                total_length += seg_length
                current_point = next_point
        
        self.get_logger().info(f"Planned trajectory length: {total_length:.2f} m with {len(trajectory)} waypoints")
        return trajectory
    
    def publish_trajectory(self, trajectory):
        # Publish the trajectory as a nav_msgs/Path message.
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        for pt in trajectory:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = pt[2]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        
        # Also publish a Marker for visualization (as a red line strip).
        marker = Marker()
        marker.header = path_msg.header
        marker.ns = "trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # Line width.
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        for pt in trajectory:
            p = Point()
            p.x = pt[0]
            p.y = pt[1]
            p.z = pt[2]
            marker.points.append(p)
        self.marker_pub.publish(marker)
    
    def timer_callback(self):
        # Plan and publish a new trajectory at every timer callback.
        trajectory = self.plan_trajectory()
        self.publish_trajectory(trajectory)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
