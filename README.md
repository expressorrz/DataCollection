# Data Collection

This project is designed for collecting data using Intel RealSense cameras. It provides tools and scripts to capture, process, and store depth and RGB data for various applications such as computer vision, robotics, and machine learning.

## Features

- Support for Intel RealSense cameras.
- Capture synchronized RGB and depth data.
- Save data in an organized format for easy access.
- Configurable parameters for data collection.

## Requirements

- Python 3.7 or higher.
- Intel RealSense SDK.


## Usage

Run the main script to start data collection:

**Single camera**: 
```bash
python main_single_realsense.py
```

**Multiple Cameras**:
```bash
python main_multi_realsense.py
```

**Collecting Poses**
```
ros2 launch realsense2_camera rs_launch.py camera_namespace:=robot1 camera_name:=D455_1 serial_no:=_247122071632 

source ../install/setup.bash

ros2 run pose_subscriber subscriber_node
ros2 run pose_subscriber test_publisher

# ros2 launch pose_subscriber pose_subscriber.launch.py
```


You can customize parameters such as resolution, frame rate, and output directory using command-line arguments.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.