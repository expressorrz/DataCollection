#!/usr/bin/env python
# coding=utf-8
import time
import h5py
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
from utils import pointcloud_xyz_rgb, depth2xyz, depth2xyzrgb, save_2_ply, save_2_pcd
 
class Camera(object):
    '''
    Class for RealSense camera
    '''
    def __init__(self, width=640, height=480, fps=30):
        # Configure depth and color streams
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)

        pip_profile = self.pipeline.start(self.config)

        # Get device and depth scale
        _device = pip_profile.get_device()
        self.depth_sensor = _device.first_depth_sensor()
        self.g_depth_scale = self.depth_sensor.get_depth_scale()  # 0.00100...
        
        self.depth_sensor = pip_profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        self.clipping_distance_in_meters = 6  # 6 meter
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        self.initialize_filter()

        # Create pointcloud object
        self.pc = rs.pointcloud()
        self.g_colorizer = rs.colorizer(0)
 
    def get_frame(self, store_pc=False):
        frames = self.pipeline.wait_for_frames()
        colorizer = rs.colorizer()
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)

        frame_num = frames.get_frame_number()
        timestamp = frames.get_timestamp()
        
 
        # Align the depth frame to color frame
        align_to = rs.stream.color
        align = rs.align(align_to)
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()  # depth_frame is the depth frame aligned to color
        color_frame = aligned_frames.get_color_frame()

        # process depth frame with filters
        depth_frame_filter = depth_frame
        for filter in self.g_rs_depth_postprocess_list:
            depth_frame_filter = filter.process(depth_frame_filter)
        depth_frame = depth_frame_filter

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = np.asanyarray(self.g_colorizer.colorize(depth_frame).get_data())

        # get the pointcloud data
        if store_pc:
            pointcloud_data_ply, pointcloud_data_pcd = self.get_pointcloud(frames, color_frame, depth_frame, color_image, depth_image, depth_colormap, self.pc)
        else:
            pointcloud_data_ply, pointcloud_data_pcd = None, None
        

        return timestamp, frame_num, color_image, depth_image, depth_colormap, pointcloud_data_ply, pointcloud_data_pcd
    
    def get_pointcloud(self, frames, color_frame, depth_frame, color_image, depth_image, depth_colormap, pc, PointCloud_Process='method1'):

        depth_intrin = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        g_depth_intrinsics_matrix = np.array([
            [depth_intrin.fx, 0., depth_intrin.ppx],
            [0., depth_intrin.fy, depth_intrin.ppy],
            [0, 0, 1.]
        ])


        # 6. Get pointcloud data
        if PointCloud_Process == 'method1':
            pointcloud_data_ply, pointcloud_data_pcd = pointcloud_xyz_rgb(color_frame, depth_frame, color_image, depth_colormap, pc)
        elif PointCloud_Process == 'method2':
            pointcloud_data_ply = depth2xyz(depth_image, g_depth_intrinsics_matrix, self.g_depth_scale)
            pointcloud_data_pcd = depth2xyzrgb(color_image, depth_image, g_depth_intrinsics_matrix, self.g_depth_scale)

        return pointcloud_data_ply, pointcloud_data_pcd
    
    def initialize_filter(self):
        g_rs_downsample_filter = rs.decimation_filter(magnitude=2 ** 1,)   # downsample rate
        g_rs_thres_filter = rs.threshold_filter(min_dist=0.1, max_dist=5.0)
        g_rs_spatical_filter = rs.spatial_filter(
            magnitude=2,
            smooth_alpha=0.5,
            smooth_delta=20,
            hole_fill=0,
        )
        g_rs_templ_filter = rs.temporal_filter(
            smooth_alpha=0.1,
            smooth_delta=40.,
            persistence_control=3
        )
        g_rs_depth2disparity_trans = rs.disparity_transform(True)
        g_rs_disparity2depth_trans = rs.disparity_transform(False)
        self.g_rs_depth_postprocess_list = [
            g_rs_downsample_filter,
            g_rs_thres_filter,
            g_rs_depth2disparity_trans,
            g_rs_spatical_filter,
            g_rs_templ_filter,
            g_rs_disparity2depth_trans
        ]
 
    def release(self):
        self.pipeline.stop()
 
 
if __name__ == '__main__':
    video_path = './media/targetvideo_rgb.mp4'
    video_depthc_path = './media/targetvideo_depthcolor.mp4'
    video_depth16_path = './media/targetvideo_depth.h5'

    save_path = './media'
    if not os.path.exists(f'{save_path}/color') or not os.path.exists(f'{save_path}/depth') or not os.path.exists(f'{save_path}/pointcloud_3d'):
        print('save_path not exists, create it ...')
        os.makedirs(f'{save_path}/data')
        os.makedirs(f'{save_path}/color')
        os.makedirs(f'{save_path}/depth')
        os.makedirs(f'{save_path}/pointcloud_3d')
 
    # Initialize video parameters
    fps, width, high = 30, 640, 480 
    mp4 = cv2.VideoWriter_fourcc(*'mp4v')  # video format setting
    store_pc = True
 
   # Initialize camera
    realsense_camera = Camera(width, high, fps)
    idx = 0
    is_recording = False  # 是否开始录制
    print('Press "s" to start/stop recording, "q" to quit.')

    while True:
        # Read RGB, depth, depth color frames from camera
        data_list = realsense_camera.get_frame(store_pc)
        timestamp, frame_num, color_image, depth_image, depth_colormap, pointcloud_data_ply, pointcloud_data_pcd = data_list

        readable_timestamp = datetime.fromtimestamp(timestamp / 1000.0)
        print(f'Frame {idx} at {readable_timestamp} ms, frame_num: {frame_num}')
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('s'):
            is_recording = not is_recording  # 切换录制状态
            if is_recording:
                print('>> Start recording...')
            else:
                print('>> Pause recording...')

        if is_recording:
            img_bgr = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{save_path}/color/{frame_num}.png', img_bgr)
            cv2.imwrite(f'{save_path}/depth/{frame_num}.png', depth_colormap)

            save_2_ply(pointcloud_data_ply, f'{save_path}/pointcloud_3d/{frame_num}.ply')
            save_2_pcd(pointcloud_data_pcd, f'{save_path}/pointcloud_3d/{frame_num}.pcd')

            data_dict = {
                'frame_num': frame_num,
                'timestamp': timestamp,
                'color_img': color_image,
                'depth_img': depth_image,
            }
            np.savez(f'{save_path}/data/{frame_num}.npz', **data_dict)

        if key & 0xFF == ord('q') or key == 27:
            print('>> Recording finished.')
            break

        idx += 1

    realsense_camera.release()
    cv2.destroyAllWindows()
    print(f'Video saved to {video_path}, {video_depthc_path}, {video_depth16_path}')

    # while True:
    #     # Read RGB, depth, depth color frames from camera
    #     data_list = realsense_camera.get_frame(store_pc)
    #     timestamp, frame_num, color_image, depth_image, depth_colormap, pointcloud_data_ply, pointcloud_data_pcd = data_list

    #     readable_timestamp = datetime.fromtimestamp(timestamp / 1000.0)
    #     print(f'Frame {idx} at {readable_timestamp} ms, frame_num: {frame_num}')
    #     cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    #     cv2.imshow('RealSense', color_image)
    #     key = cv2.waitKey(1)
 
    #     if key & 0xFF == ord('s'):
    #         flag_V = 1
    #         # construct video writer for RGB , Depth and disparity
    #         color_wr = cv2.VideoWriter(video_path, mp4, fps, (width, high), isColor=True)
    #         colordepth_wr = cv2.VideoWriter(video_depthc_path, mp4, fps, (width, high), isColor=True)
    #         idepth_wr = h5py.File(video_depth16_path, 'w')
    #         print('>> Recording the video...')
            
    #     if flag_V == 1:
    #         # Save the RGB and depth frames
    #         color_wr.write(color_image)
    #         colordepth_wr.write(depth_colormap)

    #         depth16_image = cv2.imencode('.png', depth_image)[1]
    #         depth_map_name = str(id).zfill(5) + '_depth.png'
    #         idepth_wr[depth_map_name] = depth16_image
    #         idx += 1
    #         id += 1
    #     if key & 0xFF == ord('q') or key == 27:
    #         cv2.destroyAllWindows()
    #         print('>> Recording finished.')
    #         break
    # # Release resources
    # color_wr.release()
    # colordepth_wr.release()

    # idepth_wr.close()
    # realsense_camera.release()
    # print(f'Video saved to {video_path}, {video_depthc_path}, {video_depth16_path}')
