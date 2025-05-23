import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    pose_subscriber_node = Node(
        package='pose_subscriber',
        executable='subscriber_node',
        name='pose_subscriber',
        output='screen'
    )

    return LaunchDescription([
        pose_subscriber_node
    ])
