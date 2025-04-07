# -*- coding:utf-8 -*-
"""
@FileName  :D435i.py
@Time      :2022/4/16 17:52
@Author    :Jiayu

"""
import numpy as np
import pyrealsense2 as rs
import cv2
import time

# 设定分辨率等相机参数
width = 640
height = 480
fps = 30
device_id = "327122075831"
enable_imu = False

# TODO: enable_depth = True
# TODO: enable_rgb = True
# TODO: enable_ir = True
if __name__ == '__main__':

# 1.初始化
    '============================================================================================'
    # 开始准备IMU
    if enable_imu:
        imu_pipeline = rs.pipeline()
        imu_config = rs.config()
        if None != device_id:
            imu_config.enable_device(device_id)
        imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)  # acceleration 加速度
        imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope    陀螺仪
        # 启用IMU
        imu_pipeline.start(imu_config)

    # 开始准备相机
    pipeline = rs.pipeline()
    # 属性设置
    config = rs.config()
    # 启动指定设备
    if None != device_id:
        config.enable_device(device_id)
    # 深度图
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    # IR左图
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)  # 默认格式 rs.format.y8
    # IR右图
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)  # 默认格式 rs.format.y8
    # 彩色图
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    # 创建对齐目标
    '''
    * rs.align() 允许将深度图和其他图像对齐
    * "align_to" 代表要与深度图对齐的图像类型
    '''
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 启用相机
    profile = pipeline.start(config)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
    # 关闭/打开激光发射器  0 / 1
    sensor.set_option(rs.option.emitter_enabled, 1)
    # 设置相机的增益      min-16  max-248  default-1  step-16
    sensor.set_option(rs.option.gain, 16)
    # 设置相机的曝光      min-1    max-165000    default-1    step-33000
    sensor.set_option(rs.option.exposure, 10000)
    # 设置自动曝光        0 / 1
    sensor.set_option(rs.option.enable_auto_exposure, 0)

    # 记录帧数
    frame_count = 30
    # 记录开始时间
    start_time = time.time()
    frame_time = start_time
    # 获得深度传感器的深度缩放因子 0.0010000000474974513 ~ 0.001
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)


    # 2.获取数据流
    '============================================================================================'
    for i in range(frame_count):
        # 记录停止时间
        last_time = frame_time
        frame_time = time.time() - start_time
        
        # IMU数据流
        if enable_imu:
            # 等待IMU数据
            imu_frames = imu_pipeline.wait_for_frames(200 if (frame_count > 1) else 10000)
            # 获取加速度信息
            accel_frame = imu_frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)
            # 获取陀螺仪信息
            gyro_frame = imu_frames.first_or_default(rs.stream.gyro, rs.format.motion_xyz32f)
            # 打印IMU信息
            print("imu frame {} in {} seconds: \n\taccel = {}, \n\tgyro = {}".format(str(frame_count),
                                                                                     str(frame_time - last_time),
                                                                                     str(accel_frame.as_motion_frame().get_motion_data()),
                                                                                     str(gyro_frame.as_motion_frame().get_motion_data())))
        # 图像数据流
        # 等待图像帧
        frameset = pipeline.wait_for_frames()
        # 创建对齐
        aligned_frames = align.process(frameset)
        # 获取深度图
        frame_depth = frameset.get_depth_frame()  # 没有和RGB对齐的深度图
        frame_depth_aligned = aligned_frames.get_depth_frame()  # 和RGB对齐的深度图
        # 获取RGB图像
        frame_color = frameset.get_color_frame()  # 没有和深度图对齐的RGB图
        frame_color_aligned = aligned_frames.get_color_frame()  # 和深度图对齐的RGB图

        timestamp = frame_depth.get_timestamp()  # 获取时间戳


        # 3.格式转换
        '============================================================================================'
        # 转换图像帧格式，以便后面的显示和储存
        depth_image = np.asarray(frame_depth.get_data())
        RGB_image = np.asarray(frame_color.get_data())
        RGB_image_aligned = np.asarray(frame_color_aligned.get_data())
        depth_image_aligned = np.asarray(frame_depth_aligned.get_data())

        # 给深度图上伪彩色
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_aligned = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_aligned, alpha=0.03), cv2.COLORMAP_JET)

        # 水平拼接图像对
        image_pair_RGBD = np.hstack((RGB_image_aligned, depth_colormap_aligned))  # RGBD_Pairs

        data_dict = {
            'RGB': RGB_image,
            'depth': depth_image,
            'RGB_aligned': RGB_image_aligned,
            'depth_aligned': depth_image_aligned,
            'depth_colormap': depth_colormap,
            'depth_colormap_aligned': depth_colormap_aligned,
            'timestamp': timestamp,
        }
        
        np.savez_compressed(f"./media/{i}.npz", **data_dict)

        # 4.显示与保存
        '============================================================================================'
        # 显示图像
        cv2.imshow("RGBD_Pair", image_pair_RGBD)


# 5.退出
        '============================================================================================'
        frame_count += 1