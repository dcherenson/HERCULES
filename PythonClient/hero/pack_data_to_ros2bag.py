#!/usr/bin/env python3
"""
pack_to_ros2bag.py

Reads raw_data_dir/{imu.txt, odom.txt, rgb/,depth/,seg/,lidar/}
and writes a ROS2 Humble bag at bag_out, deleting any existing bag_out first.

You can filter which streams go into the bag via:
  --include-topics imu,odom,rgb,depth,seg,lidar,tf   (default: all)
  --exclude-topics imu,tf                           (default: none)
"""

import os
import shutil
import glob
import argparse
import numpy as np
import cv2

from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Imu, Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from rclpy.serialization import serialize_message

def parse_txt(fname):
    """Return dict mapping timestamp → list of float fields."""
    d = {}
    with open(fname) as f:
        for line in f:
            parts = line.strip().split()
            t = float(parts[0])
            vals = list(map(float, parts[1:]))
            d[t] = vals
    return d

def make_pointcloud2(points, stamp, frame_id="base_link"):
    """Convert Nx3 ndarray → sensor_msgs/PointCloud2."""
    msg = PointCloud2()
    msg.header.stamp = Time(sec=int(stamp), nanosec=int((stamp % 1) * 1e9))
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points.shape[0]
    msg.is_dense = True
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = points.astype(np.float32).tobytes()
    return msg

def main(raw_data_dir, bag_out, include, exclude):
    bridge = CvBridge()

    # parse raw data
    imu_data  = parse_txt(os.path.join(raw_data_dir, "imu.txt"))
    odom_data = parse_txt(os.path.join(raw_data_dir, "odom.txt"))

    img_topics = {
        "rgb":   "camera/color/image_raw",
        "depth": "camera/depth/image_raw",
        "seg":   "camera/seg/image_raw",
    }
    img_map = {
        sub: {
            float(os.path.basename(f)[:-4]): f
            for f in glob.glob(f"{raw_data_dir}/{sub}/*.png")
        }
        for sub in img_topics
    }

    lidar_map = {
        float(os.path.basename(f)[:-4]): f
        for f in glob.glob(f"{raw_data_dir}/lidar/*.npy")
    }

    # collect all timestamps
    all_ts = sorted(
        set(imu_data)
        | set(odom_data)
        | set(img_map["rgb"])
        | set(lidar_map)
    )

    # clear existing bag directory
    if os.path.exists(bag_out):
        shutil.rmtree(bag_out)

    # open rosbag2 writer
    writer = SequentialWriter()
    storage_opts = StorageOptions(uri=bag_out, storage_id='sqlite3')
    conv_opts = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    writer.open(storage_opts, conv_opts)

    # helper to decide inclusion
    def want(topic):
        return ("all" in include or topic in include) and topic not in exclude

    # register topics
    if want("imu"):
        writer.create_topic(TopicMetadata(
            name="/imu",
            type="sensor_msgs/msg/Imu",
            serialization_format="cdr"
        ))
    if want("odom"):
        writer.create_topic(TopicMetadata(
            name="/odom",
            type="nav_msgs/msg/Odometry",
            serialization_format="cdr"
        ))
    for sub, ros_topic in img_topics.items():
        if want(sub):
            writer.create_topic(TopicMetadata(
                name=f"/{ros_topic}",
                type="sensor_msgs/msg/Image",
                serialization_format="cdr"
            ))
    if want("lidar"):
        writer.create_topic(TopicMetadata(
            name="/lidar",
            type="sensor_msgs/msg/PointCloud2",
            serialization_format="cdr"
        ))
    if want("tf"):
        writer.create_topic(TopicMetadata(
            name="/tf",
            type="tf2_msgs/msg/TFMessage",
            serialization_format="cdr"
        ))

    # write messages in order
    for t in all_ts:
        ts_nsec = int(t * 1e9)
        stamp = Time(sec=ts_nsec // 1_000_000_000,
                     nanosec=ts_nsec % 1_000_000_000)

        # IMU
        if want("imu") and t in imu_data:
            ax, ay, az, gx, gy, gz = imu_data[t]
            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = "base_link"
            imu.linear_acceleration.x = ax
            imu.linear_acceleration.y = ay
            imu.linear_acceleration.z = az
            imu.angular_velocity.x    = gx
            imu.angular_velocity.y    = gy
            imu.angular_velocity.z    = gz
            writer.write("/imu", serialize_message(imu), ts_nsec)

        # Odometry
        if want("odom") and t in odom_data:
            px, py, pz, qw, qx, qy, qz = odom_data[t]
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = "odom"
            odom.child_frame_id  = "base_link"
            odom.pose.pose.position.x    = px
            odom.pose.pose.position.y    = py
            odom.pose.pose.position.z    = pz
            odom.pose.pose.orientation.w = qw
            odom.pose.pose.orientation.x = qx
            odom.pose.pose.orientation.y = qy
            odom.pose.pose.orientation.z = qz
            writer.write("/odom", serialize_message(odom), ts_nsec)

        # Images
        for sub, ros_topic in img_topics.items():
            if want(sub) and t in img_map[sub]:
                if sub == "rgb":
                    cv_img = cv2.imread(img_map[sub][t], cv2.IMREAD_COLOR)
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    encoding = "rgb8"
                else:
                    cv_img = cv2.imread(img_map[sub][t], cv2.IMREAD_GRAYSCALE)
                    encoding = "mono8"
                img_msg = bridge.cv2_to_imgmsg(cv_img, encoding=encoding)
                img_msg.header.stamp = stamp
                img_msg.header.frame_id = "camera_link"
                writer.write(f"/{ros_topic}", serialize_message(img_msg), ts_nsec)

        # LiDAR
        if want("lidar") and t in lidar_map:
            pts = np.load(lidar_map[t])
            pc2 = make_pointcloud2(pts, t)
            writer.write("/lidar", serialize_message(pc2), ts_nsec)

        # TF
        if want("tf"):
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = "odom"
            tf.child_frame_id  = "base_link"
            tf.transform.rotation.w = 1.0
            tf_msg = TFMessage(transforms=[tf])
            writer.write("/tf", serialize_message(tf_msg), ts_nsec)

    print(f"Written bag to {bag_out}")

if __name__=="__main__":
    p = argparse.ArgumentParser(
        description="Pack raw Cosys-AirSim data into a ROS2 bag"
    )
    p.add_argument("raw_data_dir",
                   help="Folder with imu.txt, odom.txt, rgb/, depth/, seg/, lidar/")
    p.add_argument("bag_out",
                   help="Empty (or non-existent) folder to create the bag in")
    p.add_argument("--include-topics", default="all",
                   help="Comma-separated: imu,odom,rgb,depth,seg,lidar,tf or 'all'")
    p.add_argument("--exclude-topics", default="",
                   help="Comma-separated keys to skip")
    args = p.parse_args()

    include = set(args.include_topics.split(",")) if args.include_topics else {"all"}
    exclude = set(args.exclude_topics.split(",")) if args.exclude_topics else set()

    main(args.raw_data_dir, args.bag_out, include, exclude)
