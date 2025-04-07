#!/home/ipu/software/miniconda3/envs/pytorch
"""
@desc:  读取realsense流, 保存数据(包括图像数据  深度数据  点云数据)
        连接realsense相机进行录像, 按照帧序号保存 彩色图 深度图 点云图(ply & pcd)
        经过测试, 保存的点云图都可以使用pcl_viewer 打开和查看, pcd文件的纹理色彩也是没问题的
"""

import sys
sys.path.append('.')

import pyrealsense2 as rs

import os, os.path as osp
import shutil
import numpy as np
import cv2
import shutil
import glob

from utils import pointcloud_xyz_rgb, depth2xyz, depth2xyzrgb, save_2_pcd, save_2_ply


def main():
    """
        Configure
    """
    if osp.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(f'{save_path}/color') or not os.path.exists(f'{save_path}/depth') or not os.path.exists(f'{save_path}/pointcloud_3d'):
        print('save_path not exists, create it ...')
        os.makedirs(f'{save_path}/data')
        os.makedirs(f'{save_path}/color')
        os.makedirs(f'{save_path}/depth')
        os.makedirs(f'{save_path}/pointcloud_3d')

    ctx = rs.context()
    _devices = ctx.query_devices()
    if len(_devices) == 0:
        print("No device connected, please connect a RealSense device.")
        return -100

    print('Connected devices:', _devices)

    # 1. Configure depth and color streams
    pipeline = rs.pipeline(ctx)
    config = rs.config()

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)

    if config.can_resolve(pipeline):
        config.resolve(pipeline)
    else:
        print('rs.config().resolve() failed. please check rs.config.')
        return -200

    # 2. Start streaming
    pipe_profile = pipeline.start(config)

    _device = pipe_profile.get_device()
    if _device is None:
        print('pipe_profile.get_device() is None .')
        return -400
    assert _device is not None

    depth_sensor = _device.first_depth_sensor()
    g_depth_scale = depth_sensor.get_depth_scale()  # 0.00100...

    device_product_line = str(_device.get_info(rs.camera_info.product_line))
    print(device_product_line)
    found_rgb = False
    for s in _device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    align_to = rs.stream.color  # rs.stream.depth
    align = rs.align(align_to)

    # 3. Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # (Optional) We will be removing the background of objects more than clipping_distance_in_meters meters away
    clipping_distance_in_meters = 6  # 6 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # 4. Processing blocks
    pc = rs.pointcloud()
    g_colorizer = rs.colorizer(0)

    # 5. Configuring post-processing filters: decimation_filter --> HDR Merge --> threshold_filter --> Depth to Disparity 
    #                                        --> spatial_filter--> temporal_filter --> Disparity to Depth
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
    g_rs_depth_postprocess_list = [
        g_rs_downsample_filter,
        g_rs_thres_filter,
        g_rs_depth2disparity_trans,
        g_rs_spatical_filter,
        g_rs_templ_filter,
        g_rs_disparity2depth_trans
    ]


    #########################################################################################################
    """
        Main loop for streaming frames
    """
    i = 0
    while i < MAX_FRAME_NUM:
        i += 1
        ret, frames = pipeline.try_wait_for_frames()
        if not ret:
            print('try_wait_for_frames() failed.')
            break

        if i > 0 and i % SKIP_FRAME != 0:
            continue

        print('frame_num: ', i)

        # 1. align
        align_frames = align.process(frames)
        frames = align_frames

        frame_num = frames.frame_number
        timestamp = frames.timestamp
        profile = frames.profile
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        pose_frame = frames.get_pose_frame()

        # 2. Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue
        
        # 3. Process the depth frame with filters
        depth_frame_filter = depth_frame
        for trans in g_rs_depth_postprocess_list:
            depth_frame_filter = trans.process(depth_frame_filter)
        depth_frame = depth_frame_filter

        # 4. Convert images to numpy arrays
        depth_img = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())
        poses = np.asanyarray(pose_frame.get_data()) if pose_frame else None

        depth_colormap = g_colorizer.colorize(depth_frame)
        depth_colormap = np.asarray(depth_colormap.get_data())

        print('timestramp: ', timestamp)
        print('frame_num: ', frame_num)
        print('color_img.shape: ', color_img.shape)
        print('depth_img.shape: ', depth_img.shape)
        print(f'poses: {poses}')

        # 5. Get camera intrinsics, extrinsics. When depth frame is aligned to color frame, the intrinsics of depth frame may change.
        g_intrinsics = frames.get_profile().as_video_stream_profile().get_intrinsics()
        color_intrin = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        g_color_intrinsics_matrix = np.array([
            [color_intrin.fx, 0., color_intrin.ppx],
            [0., color_intrin.fy, color_intrin.ppy],
            [0, 0, 1.]
        ])

        depth_intrin = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        g_depth_intrinsics_matrix = np.array([
            [depth_intrin.fx, 0., depth_intrin.ppx],
            [0., depth_intrin.fy, depth_intrin.ppy],
            [0, 0, 1.]
        ])

        extrinsics = depth_frame.get_profile().get_extrinsics_to(color_frame.get_profile())
        rotation, translation = extrinsics.rotation, extrinsics.translation
        ppx, ppy = depth_intrin.ppx, depth_intrin.ppy
        fx, fy = depth_intrin.fx, depth_intrin.fy
        coeffs = depth_intrin.coeffs

        # 6. Get pointcloud data
        if PointCloud_Process == 'method1':
            pointcloud_data_ply, pointcloud_data_pcd = pointcloud_xyz_rgb(color_frame, depth_frame, color_img, depth_colormap, pc)
        elif PointCloud_Process == 'method2':
            pointcloud_data_ply = depth2xyz(depth_img, g_depth_intrinsics_matrix, g_depth_scale)
            pointcloud_data_pcd = depth2xyzrgb(color_img, depth_img, g_depth_intrinsics_matrix, g_depth_scale)


        # 7. Get the Img data
        img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        # 8. Save data
        cv2.imwrite(f'{save_path}/color/{frame_num}.png', img_bgr)
        # np.save(f'{save_path}/color/{frame_num}.npy', color_img)
        

        cv2.imwrite(f'{save_path}/depth/{frame_num}.png', depth_colormap)
        # np.save(f'{save_path}/depth/{frame_num}.npy', depth_img)

        save_2_ply(pointcloud_data_ply, f'{save_path}/pointcloud_3d/{frame_num}.ply')
        save_2_pcd(pointcloud_data_pcd, f'{save_path}/pointcloud_3d/{frame_num}.pcd')


        data_dict = {
            'frame_num': frame_num,
            'timestamp': timestamp,
            'color_img': color_img,
            'depth_img': depth_img,
        }
        np.savez(f'{save_path}/data/{frame_num}.npz', **data_dict)


        # 9. Show images
        if img_bgr.shape[:2] != depth_colormap.shape[:2]:
            _H,_W = depth_colormap.shape[:2]
            img_bgr = cv2.resize(img_bgr, (_W,_H))
        images = np.hstack((img_bgr, depth_colormap))
        if images.shape[1] > 1200:
            images = cv2.resize(images, dsize=None, fx=0.5, fy=0.5)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

    pipeline.stop()
    print('read stream end ...')







if __name__ == "__main__":
    WIDTH = 640
    HEIGHT = 480
    enable_imu = True
    device_id = 327122075831
    
    
    save_path = './media'

    MAX_FRAME_NUM = 100
    SKIP_FRAME = 3
    FPS = 30
    PointCloud_Process = 'method1' # 'method1' or 'method2'

    main()