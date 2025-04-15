"""
This node locates Aruco AR markers in images and publishes their ids and poses.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf_transformations
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseArray, Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType


class ArucoNode(Node):
    def __init__(self):
        super().__init__("aruco_node")

        # Declare and read parameters
        self.marker_size = self.declare_and_get_param("marker_size", 0.0625, ParameterType.PARAMETER_DOUBLE, "Size of the markers in meters.")
        dictionary_id_name = self.declare_and_get_param("aruco_dictionary_id", "DICT_5X5_250", ParameterType.PARAMETER_STRING, "Dictionary that was used to generate markers.")
        image_topic = self.declare_and_get_param("image_topic", "", ParameterType.PARAMETER_STRING, "Image topic to subscribe to.")
        info_topic = self.declare_and_get_param("camera_info_topic", "", ParameterType.PARAMETER_STRING, "Camera info topic to subscribe to.")
        self.camera_frame = self.declare_and_get_param("camera_frame", "", ParameterType.PARAMETER_STRING, "Camera optical frame to use.")

        self.get_logger().info(f"Marker size: {self.marker_size}")
        self.get_logger().info(f"Marker type: {dictionary_id_name}")
        self.get_logger().info(f"Image topic: {image_topic}")
        self.get_logger().info(f"Image info topic: {info_topic}")

        # Validate dictionary id
        dictionary_id = self.validate_dictionary_id(dictionary_id_name)

        # Set up subscriptions
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_callback, qos_profile_sensor_data)
        self.create_subscription(Image, image_topic, self.image_callback, qos_profile_sensor_data)

        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, "aruco_poses", 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, "aruco_markers", 10)

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.bridge = CvBridge()

    def declare_and_get_param(self, name, value, param_type, description):
        self.declare_parameter(name, value, ParameterDescriptor(type=param_type, description=description))
        return self.get_parameter(name).get_parameter_value().double_value if param_type == ParameterType.PARAMETER_DOUBLE else self.get_parameter(name).get_parameter_value().string_value

    def validate_dictionary_id(self, dictionary_id_name):
        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if not isinstance(dictionary_id, int):
                raise AttributeError
            return dictionary_id
        except AttributeError:
            self.get_logger().error(f"bad aruco_dictionary_id: {dictionary_id_name}")
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error(f"valid options: {options}")
            raise

    def info_callback(self, info_msg):
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        self.destroy_subscription(self.info_sub)

    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        markers = ArucoMarkers()
        pose_array = PoseArray()
        frame_id = self.camera_frame or self.info_msg.header.frame_id

        markers.header.frame_id = frame_id
        pose_array.header.frame_id = frame_id
        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        corners, marker_ids, _ = cv2.aruco.detectMarkers(cv_image, self.aruco_dictionary, parameters=self.aruco_parameters)
        if marker_ids is not None:
            rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.intrinsic_mat, self.distortion)
            for i, marker_id in enumerate(marker_ids):
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = tvecs[i][0]
                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                quat = tf_transformations.quaternion_from_matrix(rot_matrix)
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat

                pose_array.poses.append(pose)
                markers.poses.append(pose)
                markers.marker_ids.append(marker_id[0])

            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
