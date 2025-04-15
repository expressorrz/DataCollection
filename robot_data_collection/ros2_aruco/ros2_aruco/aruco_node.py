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
        self.enable_viewer = self.declare_and_get_param("enable_viewer", True, ParameterType.PARAMETER_BOOL, "Enable OpenCV viewer for pose visualization.")
        self.marker_size = self.declare_and_get_param("marker_size", 0.0625, ParameterType.PARAMETER_DOUBLE, "Size of the markers in meters.")
        dictionary_id_name = self.declare_and_get_param("aruco_dictionary_id", "DICT_5X5_250", ParameterType.PARAMETER_STRING, "Dictionary that was used to generate markers.")
        image_topic_eyeinhand = self.declare_and_get_param("image_topic_eyeinhand", "", ParameterType.PARAMETER_STRING, "Image topic to subscribe to.")
        info_topic_eyeinhand = self.declare_and_get_param("camera_info_topic_eyeinhand", "", ParameterType.PARAMETER_STRING, "Camera info topic to subscribe to.")
        self.camera_frame_eyeinhand = self.declare_and_get_param("camera_frame_eyeinhand", "", ParameterType.PARAMETER_STRING, "Camera optical frame to use.")

        image_topic_eyetohand = self.declare_and_get_param("image_topic_eyetohand", "", ParameterType.PARAMETER_STRING, "Image topic to subscribe to.")
        info_topic_eyetohand = self.declare_and_get_param("camera_info_topic_eyetohand", "", ParameterType.PARAMETER_STRING, "Camera info topic to subscribe to.")
        self.camera_frame_eyetohand = self.declare_and_get_param("camera_frame_eyetohand", "", ParameterType.PARAMETER_STRING, "Camera optical frame to use.")
        
        self.get_logger().info(f"Marker size: {self.marker_size}")
        self.get_logger().info(f"Marker type: {dictionary_id_name}")
        self.get_logger().info(f"Image topic arm_in_hand: {image_topic_eyeinhand}")
        self.get_logger().info(f"Image info topic arm_in_hand: {info_topic_eyeinhand}")
        self.get_logger().info(f"Image topic arm_to_hand: {image_topic_eyetohand}")
        self.get_logger().info(f"Image info topic arm_to_hand: {info_topic_eyetohand}")

        # Validate dictionary id
        dictionary_id = self.validate_dictionary_id(dictionary_id_name)

        # Set up arm_in_hand subscriptions
        self.info_sub_eyeinhand = self.create_subscription(CameraInfo, info_topic_eyeinhand, self.eyeinhand_info_callback, qos_profile_sensor_data)
        self.create_subscription(Image, image_topic_eyeinhand, self.eyeinhand_image_callback, qos_profile_sensor_data)

        # Set up arm_to_hand subscriptions
        self.info_sub_eyetohand = self.create_subscription(CameraInfo, info_topic_eyetohand, self.eyetohand_info_callback, qos_profile_sensor_data)
        self.create_subscription(Image, image_topic_eyetohand, self.eyetohand_image_callback, qos_profile_sensor_data)

        # Set up arm_in_hand publisher
        self.poses_pub_eyeinhand = self.create_publisher(PoseArray, "aruco_poses_eyeinhand", 10)
        self.markers_pub_eyeinhand = self.create_publisher(ArucoMarkers, "aruco_markers_eyeinhand", 10)

        # Set up arm_to_hand publisher
        self.poses_pub_eyetohand = self.create_publisher(PoseArray, "aruco_poses_eyetohand", 10)
        self.markers_pub_eyetohand = self.create_publisher(ArucoMarkers, "aruco_markers_eyetohand", 10)

        # Set up fields for arm_in_hand camera parameters
        self.info_msg_eyeinhand = None
        self.intrinsic_mat_eyeinhand = None
        self.distortion_eyeinhand = None

        # Set up fields for arm_to_hand camera parameters
        self.info_msg_eyetohand = None
        self.intrinsic_mat_eyetohand = None
        self.distortion_eyetohand = None

        # Set up aruco dictionary and parameters
        self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.bridge = CvBridge()

    def declare_and_get_param(self, name, value, param_type, description):
        self.declare_parameter(name, value, ParameterDescriptor(type=param_type, description=description))
        if param_type == ParameterType.PARAMETER_DOUBLE:
            return self.get_parameter(name).get_parameter_value().double_value
        elif param_type == ParameterType.PARAMETER_STRING:
            return self.get_parameter(name).get_parameter_value().string_value
        elif param_type == ParameterType.PARAMETER_BOOL:
            return self.get_parameter(name).get_parameter_value().bool_value

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

    def eyeinhand_info_callback(self, info_msg_eyeinhand):
        self.info_msg_eyeinhand = info_msg_eyeinhand
        self.intrinsic_mat_eyeinhand = np.reshape(np.array(self.info_msg_eyeinhand.k), (3, 3))
        self.distortion_eyeinhand = np.array(self.info_msg_eyeinhand.d)
        self.destroy_subscription(self.info_sub_eyeinhand)

    def eyeinhand_image_callback(self, img_msg):
        if self.info_msg_eyeinhand is None:
            self.get_logger().warn("No camera info has been received!")
            return

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        markers = ArucoMarkers()
        pose_array = PoseArray()
        if self.camera_frame_eyeinhand == "":
            markers.header.frame_id = self.info_msg_eyeinhand.header.frame_id
            pose_array.header.frame_id = self.info_msg_eyeinhand.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame_eyeinhand
            pose_array.header.frame_id = self.camera_frame_eyeinhand


        corners, marker_ids, rejected = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dictionary, parameters=self.aruco_parameters
        )
        if marker_ids is not None:
            if cv2.__version__ > "4.0.0":
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.intrinsic_mat_eyeinhand, self.distortion_eyeinhand
                )
            else:
                rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.intrinsic_mat_eyeinhand, self.distortion_eyeinhand
                )
            for i, marker_id in enumerate(marker_ids):
                pose = Pose()
                pose.position.x = tvecs[i][0][0]
                pose.position.y = tvecs[i][0][1]
                pose.position.z = tvecs[i][0][2]

                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                quat = tf_transformations.quaternion_from_matrix(rot_matrix)

                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                markers.header.stamp = img_msg.header.stamp
                pose_array.header.stamp = img_msg.header.stamp
                pose_array.poses.append(pose)
                markers.poses.append(pose)
                markers.marker_ids.append(marker_id[0])

                # ✅ Draw axis on the image
                if self.enable_viewer:
                    cv2.aruco.drawAxis(
                        cv_image, self.intrinsic_mat_eyeinhand, self.distortion_eyeinhand,
                        rvecs[i], tvecs[i], self.marker_size * 0.5
                    )

            # ✅ Show the image with overlays
            if self.enable_viewer:
                cv2.imshow("Aruco Viewer - Eye-in-hand", cv_image)
                cv2.waitKey(1)

            self.poses_pub_eyeinhand.publish(pose_array)
            self.markers_pub_eyeinhand.publish(markers)

    def eyetohand_info_callback(self, info_msg_eyetohand):
        self.info_msg_eyetohand = info_msg_eyetohand
        self.intrinsic_mat_eyetohand = np.reshape(np.array(self.info_msg_eyetohand.k), (3, 3))
        self.distortion_eyetohand = np.array(self.info_msg_eyetohand.d)
        self.destroy_subscription(self.info_sub_eyetohand)

    def eyetohand_image_callback(self, img_msg):
        if self.info_msg_eyetohand is None:
            self.get_logger().warn("No camera info has been received!")
            return

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        markers = ArucoMarkers()
        pose_array = PoseArray()
        if self.camera_frame_eyetohand == "":
            markers.header.frame_id = self.info_msg_eyetohand.header.frame_id
            pose_array.header.frame_id = self.info_msg_eyetohand.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame_eyetohand
            pose_array.header.frame_id = self.camera_frame_eyetohand

        corners, marker_ids, rejected = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dictionary, parameters=self.aruco_parameters
        )
        if marker_ids is not None:
            if cv2.__version__ > "4.0.0":
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.intrinsic_mat_eyetohand, self.distortion_eyetohand
                )
            else:
                rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.intrinsic_mat_eyetohand, self.distortion_eyetohand
                )
            for i, marker_id in enumerate(marker_ids):
                pose = Pose()
                pose.position.x = tvecs[i][0][0]
                pose.position.y = tvecs[i][0][1]
                pose.position.z = tvecs[i][0][2]

                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                quat = tf_transformations.quaternion_from_matrix(rot_matrix)

                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                markers.header.stamp = img_msg.header.stamp
                pose_array.header.stamp = img_msg.header.stamp
                pose_array.poses.append(pose)
                markers.poses.append(pose)
                markers.marker_ids.append(marker_id[0])

                # ✅ Draw axis on the image
                if self.enable_viewer:
                    cv2.aruco.drawAxis(
                        cv_image, self.intrinsic_mat_eyetohand, self.distortion_eyetohand,
                        rvecs[i], tvecs[i], self.marker_size * 0.5
                    )
                    
            

            # 在图像上绘制你提供的固定位姿
            # 固定位姿信息
            # fixed_position = [0.06700228154659271, 0.1537950485944748, 0.7934461832046509]
            # fixed_orientation = [0.9665464162826538, 0.04399968311190605, -0.06192246451973915, -0.2449849396944046]

            # fixed_position = [0.21840135753154755, 0.16549208760261536, 0.7465377449989319]
            # fixed_orientation = [0.9411773085594177, -0.013762395828962326, -0.07128103822469711, -0.33002424240112305]

            fixed_position = [0.21352414786815643, 0.1611885279417038, 0.705443263053894]
            fixed_orientation = [0.937416672706604, -0.005212195683270693, -0.044262614101171494, -0.34534505009651184]
            
            # fixed_position = [0.23210585043629645, 0.17277722226966277, 0.7683665140857224]
            # fixed_orientation = [-0.9581805226183138, -0.03302465247718234, 0.08064683062211771, 0.2725720952597017]

            # 将四元数转换为 4x4 变换矩阵，再提取旋转部分
            fixed_rot_matrix = tf_transformations.quaternion_matrix(fixed_orientation)
            R_fixed = fixed_rot_matrix[0:3, 0:3]
            rvec_fixed, _ = cv2.Rodrigues(R_fixed)
            tvec_fixed = np.array(fixed_position).reshape((3, 1))

            # 绘制固定位姿的坐标轴
            if self.enable_viewer:
                cv2.aruco.drawAxis(
                    cv_image, self.intrinsic_mat_eyetohand, self.distortion_eyetohand,
                    rvec_fixed, tvec_fixed, self.marker_size * 1.5
                )

            # ✅ Show the image with overlays
            if self.enable_viewer:
                cv2.imshow("Aruco Viewer - Eye-to-hand", cv_image)
                cv2.waitKey(1)

            self.poses_pub_eyetohand.publish(pose_array)
            self.markers_pub_eyetohand.publish(markers)
    


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)

    if node.enable_viewer:
        cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
