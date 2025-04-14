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
    def __init__(self, device_id, width=640, height=480, fps=30, enbale_filter=True):
        # Configure depth and color streams
        self.enbale_filter = enbale_filter

        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # self.config.enable_device(device_id)
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
        if self.enbale_filter:
            depth_frame_filter = depth_frame
            for filter in self.g_rs_depth_postprocess_list:
                depth_frame_filter = filter.process(depth_frame_filter)
            depth_frame = depth_frame_filter

        color_image = np.asanyarray(color_frame.get_data())
        print(color_image)
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


def main(device_id, width, high, fps, store_pc, task_id, operator_id, run_id):
    save_path = './data'
    save_color_path = f'{save_path}/{device_id}/task_{task_id}/op_{operator_id}/id_{run_id}/color'
    save_depth_path = f'{save_path}/{device_id}/task_{task_id}/op_{operator_id}/id_{run_id}/depth'
    save_pc_path = f'{save_path}/{device_id}/task_{task_id}/op_{operator_id}/id_{run_id}/pointcloud_3d'
    save_npz_path = f'{save_path}/{device_id}/task_{task_id}/op_{operator_id}/id_{run_id}/data'

    for path in [save_color_path, save_depth_path, save_pc_path, save_npz_path]:
        os.makedirs(path, exist_ok=True)

    # Initialize camera
    realsense_camera = Camera(device_id, width, high, fps)

    print("Press 'r' to start recording, 'q' to quit.")

    recording = False
    cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)


    while True:
        data_list = realsense_camera.get_frame(store_pc)
        timestamp, frame_num, color_image, depth_image, depth_colormap, pointcloud_data_ply, pointcloud_data_pcd = data_list
        print(f'color_image.shape: {color_image.shape}')
        exit()

        cv2.imshow("Color Stream", color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if not recording:
                print(">>> Start recording ...")
            recording = True

        elif key == ord('q'):
            cv2.destroyAllWindows()
            realsense_camera.release()
            print(">>> Recording ended, exiting program.")
            print(f'Video saved to {save_color_path}, {save_depth_path}, {save_pc_path}, {save_npz_path}')
            break

        if recording:
            readable_timestamp = datetime.fromtimestamp(timestamp / 1000.0)
            print(f'Frame {frame_num} at {readable_timestamp} ms')


            # cv2.imwrite(f'{save_color_path}/{frame_num}.png', color_image)
            # cv2.imwrite(f'{save_depth_path}/{frame_num}.png', depth_colormap)

            if store_pc:
                save_2_ply(pointcloud_data_ply, f'{save_pc_path}/{frame_num}.ply')
                save_2_pcd(pointcloud_data_pcd, f'{save_pc_path}/{frame_num}.pcd')

            data_dict = {
                'frame_num': frame_num,
                'timestamp': timestamp,
                'color_img': color_image,
                'depth_img': depth_image,
            }
            np.savez(f'{save_npz_path}/{frame_num}.npz', **data_dict)
    
 
if __name__ == '__main__':
    task_id = input("Task No.").strip()
    operator_id = input("Human No.").strip()
    run_id = input("ID No.").strip()
    device_id = input("Device No.").strip()

    # Initialize video parameters
    device_ids = ['327122075831', '239722072823', '247122071632']
    device_id = device_ids[int(device_id)]
    fps, width, high = 30, 640, 480 
    store_pc = False

    main(device_id, width, high, fps, store_pc, task_id, operator_id, run_id)
 
   
