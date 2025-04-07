#!/usr/bin/env python
# coding=utf-8
import time
import h5py         # 深度图格式所在库
import pyrealsense2 as rs
import numpy as np
import cv2
import os
 
class Camera(object):
    '''
    Class for RealSense camera
    '''
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
        self.pipeline.start(self.config)
 
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        colorizer = rs.colorizer()
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)
 
        # Align the depth frame to color frame
        align_to = rs.stream.color
        align = rs.align(align_to)
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is the depth frame aligned to color
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())  # raw depth image

        return color_image, depth_image, depth_colormap
 
    def release(self):
        self.pipeline.stop()
 
 
if __name__ == '__main__':
    video_path = './media/targetvideo_rgb.mp4'
    video_depthc_path = './media/targetvideo_depthcolor.mp4'
    video_depth16_path = './media/targetvideo_depth.h5'
 
    # Initialize video parameters
    fps, width, high = 30, 640, 480 
    mp4 = cv2.VideoWriter_fourcc(*'mp4v')  # video format setting
 
    # Initialize camera
    realsense_camera = Camera(width, high, fps)
    flag_V = 0
    idx = 0
    id = 0
    print('Press "s" to start recording, "q" to quit.')
 
    while True:
        # Read RGB, depth, depth color frames from camera
        color_image, depth_image, depth_colormap = realsense_camera.get_frame()
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)
 
        if key & 0xFF == ord('s'):
            flag_V = 1
            # construct video writer for RGB , Depth and disparity
            color_wr = cv2.VideoWriter(video_path, mp4, fps, (width, high), isColor=True)
            colordepth_wr = cv2.VideoWriter(video_depthc_path, mp4, fps, (width, high), isColor=True)
            idepth_wr = h5py.File(video_depth16_path, 'w')
            print('>> Recording the video...')
            
        if flag_V == 1:
            # Save the RGB and depth frames
            color_wr.write(color_image)
            colordepth_wr.write(depth_colormap)

            depth16_image = cv2.imencode('.png', depth_image)[1]
            depth_map_name = str(id).zfill(5) + '_depth.png'
            idepth_wr[depth_map_name] = depth16_image
            idx += 1
            id += 1
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            print('>> Recording finished.')
            break
    # Release resources
    color_wr.release()
    colordepth_wr.release()

    idepth_wr.close()
    realsense_camera.release()
    print(f'Video saved to {video_path}, {video_depthc_path}, {video_depth16_path}')
