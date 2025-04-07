import cv2
import numpy as np
import pyrealsense2 as rs
import time

def get_center_depth(depth):
    # Get the depth frame's dimensions
    width = depth.get_width()
    height = depth.get_height()

    center_x = int(width / 2)
    center_y = int(height / 2)

    print(width, " ", height)
    dis_center = round(depth.get_distance(center_x, center_y)*100, 2)
    print("The camera is facing an object ", dis_center, " cm away.")
    return dis_center, (center_x, center_y)

if __name__ == '__main__':
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Configure and start the pipeline
    pipeline.start(config)

    while True:
        start = time.time()
        # Block program until frames arrive
        frames = pipeline.wait_for_frames()

        # RGB image
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        # depth image
        depth = frames.get_depth_frame()
        dis_center, center_coordinate = get_center_depth(depth)

        print("color_image:", color_image.shape)
        cv2.circle(color_image, center_coordinate, 5, (0,0,255), 0)
        cv2.imshow("color_image", color_image)
        
        print("FPS:", 1/(time.time()-start), "/s")

        # press Esc to stop
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()