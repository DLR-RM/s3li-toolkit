#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from s3li_ros.calde_parser import parse_calibration_file, parse_matrix
import numpy as np


def create_camera_info(cam_dict):
    """Creates a sensor_msgs/msg/CameraInfo from parsed camera data."""
    cam_info = CameraInfo()

    # Intrinsics matrix
    A = parse_matrix(cam_dict['A'])
    cam_info.k = A

    # Distortion coefficients (only k1, k2, p1, p2 provided)
    k1 = float(cam_dict.get('k1', 0.0))
    k2 = float(cam_dict.get('k2', 0.0))
    p1 = float(cam_dict.get('p1', 0.0))
    p2 = float(cam_dict.get('p2', 0.0))
    cam_info.d = [k1, k2, p1, p2]
    cam_info.distortion_model = "plumb_bob"

    # Rectification and projection matrices
    T = parse_matrix(cam_dict['T'])
    cam_info.r = parse_matrix(cam_dict['R'])

    cam_info.p = [A[0], A[1], A[2], 0.001 * T[0],
                  A[3], A[4], A[5], 0.001 * T[1],
                  A[6], A[7], A[8], 0.001 * T[2]]

    cam_info.width = int(cam_dict.get('width', 0))
    cam_info.height = int(cam_dict.get('height', 0))

    return cam_info


class StereoCameraInfoPublisher(Node):
    def __init__(self):
        super().__init__('stereo_camera_info_publisher')

        # Declare and get parameters
        self.declare_parameter('calibration_file')
        self.declare_parameter('left_namespace', 'left')
        self.declare_parameter('right_namespace', 'right')

        calib_file = self.get_parameter('calibration_file').get_parameter_value().string_value
        left_ns = self.get_parameter('left_namespace').get_parameter_value().string_value
        right_ns = self.get_parameter('right_namespace').get_parameter_value().string_value

        cameras = parse_calibration_file(calib_file)

        if 0 not in cameras or 1 not in cameras:
            self.get_logger().error("Both camera.0 and camera.1 must be present in calibration file.")
            return

        self.left_info = create_camera_info(cameras[0])
        self.right_info = create_camera_info(cameras[1])

        if self.left_info.width == 0.0 or self.left_info.height == 0.0:
            self.left_info.width = self.right_info.width
            self.left_info.height = self.right_info.height
            self.get_logger().warn(f"Right camera info has null width and height. "
                                   f"Using left's: {self.left_info.width} x {self.left_info.height}")

        # Publishers
        self.left_pub = self.create_publisher(CameraInfo, f"{left_ns}/camera_info", 10)
        self.right_pub = self.create_publisher(CameraInfo, f"{right_ns}/camera_info", 10)

        self.get_logger().info(
            f"StereoCameraInfoPublisher started.\n"
            f"Opened camera calibration file: {calib_file}\n"
            f"Left topic: {left_ns}/camera_info, Right topic: {right_ns}/camera_info"
        )

        # Timer at 100 Hz (0.01 sec)
        self.timer = self.create_timer(0.01, self.publish_info)


    def publish_info(self):
        now = self.get_clock().now().to_msg()

        self.left_info.header = Header(stamp=now, frame_id='left_camera_frame')
        self.right_info.header = Header(stamp=now, frame_id='right_camera_frame')

        self.left_pub.publish(self.left_info)
        self.right_pub.publish(self.right_info)


if __name__ == "__main__":
    rclpy.init()
    node = StereoCameraInfoPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

