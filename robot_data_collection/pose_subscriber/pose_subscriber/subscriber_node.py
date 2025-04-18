#!/usr/bin/env python3
# filepath: /home/ipu/Documents/robot_learning/ICG_ROS2/src/pose_subscriber.py

import numpy as np
import rclpy
from rclpy.node import Node
from ros2_aruco_interfaces.msg import ArucoMarkers
import os
import zarr

def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = str(len(dirs) - 1)

    return log_dir_index

class Msg_Subscriber(Node):
    def __init__(self, save_path):
        super().__init__('twist_subscriber')

        self.save_path = save_path

        self.marker_list = [0]
        topic_name = '/aruco_markers_eyeinhand'
        
        self.pose_subscription = self.create_subscription(
            ArucoMarkers,
            topic_name,
            self.pose_callback,
            10  # QoS
        )

        if os.path.exists(os.path.join(save_path, 'pose_buffer.zarr')):
            pose_buffer_path = os.path.join(save_path, 'pose_buffer.zarr')
            if os.path.isdir(pose_buffer_path):
                import shutil
                shutil.rmtree(pose_buffer_path)
            elif os.path.exists(pose_buffer_path):
                os.remove(pose_buffer_path)

        root = zarr.open(os.path.join(save_path, 'pose_buffer.zarr'), mode='a')
        self.zarr_group = root.require_group('pose_data')
        for marker_id in self.marker_list:
            self.zarr_group.require_dataset(
                f'marker_{marker_id}',
                shape=(0, 9),
                maxshape=(None, 9),
                chunks=(1, 9),
                dtype='f8'
            )

        self.get_logger().info(f'Subscribed to {topic_name}')
    
    def pose_callback(self, msg):
        for marker_id in self.marker_list:
            if marker_id not in msg.marker_ids:
                continue
            else:
                idx = msg.marker_ids.index(marker_id)
                self.get_logger().info(f'idx: {idx}, marker_ids: {msg.marker_ids}, marker_id: {marker_id}')

                if msg.poses[idx].position.x != 0.0:
                    timestamp_sec = msg.header.stamp.sec
                    timestamp_nanosec = msg.header.stamp.nanosec
                    position = msg.poses[idx].position
                    orientation = msg.poses[idx].orientation

                    x = position.x
                    y = position.y
                    z = position.z
                    qx = orientation.x
                    qy = orientation.y
                    qz = orientation.z
                    qw = orientation.w

                    self.get_logger().info(
                        f'received object pose:'
                        f'position: x={x:.3f}, y={y:.3f}, z={z:.3f}'
                        f'orientation: x={qx:.3f}, y={qy:.3f}, z={qz:.3f}, w={qw:.3f}'
                    )

                    pose_data = [timestamp_sec, timestamp_nanosec, x, y, z, qx, qy, qz, qw]
                else:
                    pose_data = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                self.zarr_group[f'marker_{marker_id}'].append(np.array([pose_data], dtype=np.float64))
                

def main(args=None):
    import sys
    rclpy.init(args=args)

    # --------------------------------------------------------

    save_dir = '/home/ipu/codes/DP4HRC/data_collection/src/data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    idx = get_log_dir_index(save_dir)
    save_path = '%s/%s' % (save_dir, idx)
    os.makedirs(save_path, exist_ok=True)

    print(f'save_path: {save_path}')
    # --------------------------------------------------------

    msg_subscriber = Msg_Subscriber(save_path)

    try:
        rclpy.spin(msg_subscriber)
    except KeyboardInterrupt:
        print('User interrupted with Ctrl-C')
    finally:
        msg_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()