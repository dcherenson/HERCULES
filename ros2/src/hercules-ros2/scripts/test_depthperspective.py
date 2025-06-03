#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import cv2
from cv_bridge import CvBridge

class DepthDebugger(Node):
    def __init__(self):
        super().__init__('depth_debugger')
        self.bridge = CvBridge()
        self.latest_depth = None

        # Choose 'jet' for color map or 'gray' for plain grayscale
        self.display_mode = 'jet'

        # Topics
        depth_topic = '/hercules_node/Husky1/front_center_DepthPerspective/image'
        info_topic  = '/hercules_node/Husky1/front_center_DepthPerspective/camera_info'

        # Subscribe to camera_info (latched) so we can get fx,fy,cx,cy
        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.create_subscription(
            CameraInfo,
            info_topic,
            self._info_cb,
            latched_qos
        )

        # Subscribe to depth perspective images
        self.create_subscription(
            Image,
            depth_topic,
            self._image_cb,
            10
        )

        # Windows
        cv2.namedWindow('Depth Perspective', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Clicked Depth', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Clicked Depth', 300, 100)
        cv2.setMouseCallback('Depth Perspective', DepthDebugger._on_mouse, self)

        # max display depth (m)
        self.max_depth = 1000.0

    def _info_cb(self, msg: CameraInfo):
        if not hasattr(self, 'camera_info'):
            self.camera_info = msg
            self.get_logger().info('Cached camera info.')

    @staticmethod
    def _on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self = param
        if self.latest_depth is None:
            return

        h, w = self.latest_depth.shape
        if not (0 <= y < h and 0 <= x < w):
            return

        d = float(self.latest_depth[y, x])
        disp = np.zeros((100, 300, 3), dtype=np.uint8)
        text = f"{d:.2f} m"
        cv2.putText(
            disp, text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0,
            (255, 255, 255), thickness=3, lineType=cv2.LINE_AA
        )
        cv2.imshow('Clicked Depth', disp)
        cv2.waitKey(1)

    def _image_cb(self, msg: Image):
        # 1) convert slant-range (ray-length) image
        try:
            r_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
            return

        # Wait for intrinsics
        if not hasattr(self, 'camera_info'):
            self.get_logger().warn('Waiting for camera info...')
            return

        # unpack intrinsics
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # image shape and pixel grid
        h, w = r_img.shape
        u, v = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32)
        )

        # compute ray directions and norms
        xdir = (u - cx) / fx
        ydir = (v - cy) / fy
        ray_norm = np.sqrt(xdir*xdir + ydir*ydir + 1.0)

        # true forward depth Zc
        zc = r_img / ray_norm
        self.latest_depth = zc  # for click callback

        # log center pixel + neighborhood
        cyi, cxi = h // 2, w // 2
        center = float(zc[cyi, cxi])
        self.get_logger().info(f'Depth at center ({cyi},{cxi}): {center:.2f} m')
        for dy in (-1, 0, 1):
            row = zc[cyi+dy, cxi-2:cxi+3]
            vals = ', '.join(f"{v:.2f}" for v in row)
            self.get_logger().info(f'Row {cyi+dy}: {vals}')

        # prepare for display
        clipped = np.clip(zc, 0.0, self.max_depth)
        vis_uint8 = (clipped / self.max_depth * 255.0).astype(np.uint8)

        if self.display_mode == 'jet':
            vis_color = cv2.applyColorMap(vis_uint8, cv2.COLORMAP_JET)
            cv2.imshow('Depth Perspective', vis_color)
        else:
            cv2.imshow('Depth Perspective', vis_uint8)

        cv2.waitKey(1)

    def destroy_node(self):
        super().destroy_node()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = DepthDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
