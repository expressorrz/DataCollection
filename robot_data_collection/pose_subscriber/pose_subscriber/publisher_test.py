#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ros2_aruco_interfaces.msg import ArucoMarkers
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Time

class ArucoPublisher(Node):
    def __init__(self):
        super().__init__('aruco_publisher')

        self.publisher_ = self.create_publisher(ArucoMarkers, '/aruco_markers_eyeinhand', 10)
        timer_period = 1.0  # 每1秒发布一次
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info("Aruco publisher started.")

    def timer_callback(self):
        msg = ArucoMarkers()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.marker_ids = [0]

        pose = Pose()
        pose.position.x = 0.1
        pose.position.y = 0.2
        pose.position.z = 0.3
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        msg.poses.append(pose)

        self.publisher_.publish(msg)
        self.get_logger().info("Published test ArucoMarkers message.")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()