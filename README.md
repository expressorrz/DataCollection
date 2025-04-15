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

You can customize parameters such as resolution, frame rate, and output directory using command-line arguments.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.