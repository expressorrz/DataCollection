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
fps = 60
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
    # 记录拍摄图像数
    index = 10
    # 记录帧数
    frame_count = 10
    # 记录开始时间
    start_time = time.time()
    frame_time = start_time
    # 获得深度传感器的深度缩放因子 0.0010000000474974513 ~ 0.001
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # 创建窗口，用来显示读取到的每一帧图像
    cv2.namedWindow("IR_Pair", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("IR_Pair_Bone", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("RGBD_Pair", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Original_RGB", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Original_Depth_colormap", cv2.WINDOW_AUTOSIZE)

    # 创建保存视频的对象，设置编码格式，帧率，图像的宽高等
    fourcc = 0x00000002  # 设置编码格式，一般情况下fourcc = cv2.VideoWriter_fourcc(*'FLV1')   *'FLV1'表示编码方式
    # cv2.VideoWriter_fourcc()返回一个16进制数
    # 示例：out = cv2.VideoWriter('color.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    outVedio_RGB = cv2.VideoWriter('./videoStreams/RGB.flv', fourcc, fps, (width, height))
    outVedio_IR_Left = cv2.VideoWriter('./videoStreams/IR_Left.flv', fourcc, fps, (width, height))
    outVedio_IR_Right = cv2.VideoWriter('./videoStreams/IR_Right.flv', fourcc, fps, (width, height))
    outVedio_Depth = cv2.VideoWriter('./videoStreams/Depth.flv', fourcc, fps, (width, height))
    '''
    * cv2.VideoWriter参数:
        1-需保存视频的文件名
        2-解码参数  0x00000002/cv2.VideoWriter_fourcc(*'FLV1')/cv2.VideoWriter_fourcc('F','L','V','1')都表示Flash视频，如果想用h264编码则可以用0x00000007，不过这需要安装Openh264的库
        3-fps帧率
        4-分辨率
        5-True为彩色视频, False为黑白视频
     * cv2.VideoWriter_fourcc('R', 'G', 'B', 'A') 这个选项是一个未压缩编码，但会产生大文件。文件扩展名应为.avi。
    * cv2.VideoWriter_fourcc('I', '4', '2', '0') 这个选项是一个未压缩的YUV编码，4: 2:0色度子采样。这种编码广泛兼容，但会产生大文件。文件扩展名应为.avi。
    * cv2.VideoWriter_fourcc('P','I','M','1') 此选项为MPEG-1。文件扩展名应为.avi。
    * cv2.VideoWriter_fourcc('X','V','I','D') 此选项是一个相对较旧的MPEG-4编码。视频大小为平均值，MPEG4所需要的空间是MPEG1或M-JPEG的1/10，它对运动物体可以保证有良好的清晰度，时间/画质具有可调性如果要限制结果视频的大小，这是一个很好的选择。文件扩展名应为.avi。
    * cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 此选项是另一个相对较旧的MPEG-4编码。如果要限制结果视频的大小，这是一个很好的选择。文件扩展名应为.m4v。
    * cv2.VideoWriter_fourcc('X','2','6','4')  这个选项是一种比较新的MPEG-4编码方式。如果你想限制结果视频的大小，这可能是最好的选择。文件扩展名应为.mp4。
    * cv2.VideoWriter_fourcc('T','H','E','O') 这个选项是Ogg Vorbis。文件扩展名应为.ogv。
    * cv2.VideoWriter_fourcc('F','L','V','1') 此选项为Flash视频。FLV流媒体格式是一种新的视频格式。由于它形成的文件极小、加载速度极快，使得网络观看视频文件成为可能，文件扩展名应为.flv。
    '''

    while True:
        # 记录停止时间
        last_time = frame_time
        frame_time = time.time() - start_time

# 2.获取数据流
        '============================================================================================'
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
        # 获取IR左右图像对
        frame_left = frameset.get_infrared_frame(1)
        frame_right = frameset.get_infrared_frame(2)

# 3.格式转换
        '============================================================================================'
        # 转换图像帧格式，以便后面的显示和储存
        depth_image = np.asarray(frame_depth.get_data())
        left_image = np.asarray(frame_left.get_data())
        right_image = np.asarray(frame_right.get_data())
        RGB_image = np.asarray(frame_color.get_data())
        RGB_image_aligned = np.asarray(frame_color_aligned.get_data())
        depth_image_aligned = np.asarray(frame_depth_aligned.get_data())

        # 将Y8格式转换为uint8并应用BONE颜色，这样做为了能够保存IR的视频
        left_image_BONE = cv2.applyColorMap(cv2.convertScaleAbs(left_image, alpha=1), cv2.COLORMAP_BONE)
        right_image_BONE = cv2.applyColorMap(cv2.convertScaleAbs(right_image, alpha=1), cv2.COLORMAP_BONE)
        # 给深度图上伪彩色
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_aligned = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_aligned, alpha=0.03),
                                                   cv2.COLORMAP_JET)
        '''
        dst = cv2.convertScaleAbs(src , alpha=0.03 , beta);  
        假设我们需要让我们的深度摄像头感兴趣的距离范围有差别地显示，那么我们就需要确定一个合适的alpha值，
        公式为：有效距离*alpha=255，假设我们想让深度摄像头8m距离内的深度被显示，>8m的与8m的颜色显示相同，
        那么alpha=255/(8*10^3)≈0.03，假设我们想让深度摄像头6m距离内的深度被显示，>6m的与6m的颜色显示相同，
        那么alpha=255/(6*10^3)≈0.0425，cv.convertScaleAbs()函数对输入数组进行如下运算：
        dst = 截取8位(|src*alpha+beta|)

        '''
        # 水平拼接图像对
        image_pair_IR = np.hstack((left_image, right_image))  # IR_Pairs(没有转换格式)
        image_pair_IR_BONE = np.hstack((left_image_BONE, right_image_BONE))  # IR_Pairs_BONE
        image_pair_RGBD = np.hstack((RGB_image_aligned, depth_colormap_aligned))  # RGBD_Pairs

# 4.显示与保存
        '============================================================================================'
        # 显示图像
        cv2.imshow("IR_Pair", image_pair_IR)
        cv2.imshow("IR_Pair_Bone", image_pair_IR_BONE)
        cv2.imshow("RGBD_Pair", image_pair_RGBD)
        cv2.imshow("Original_RGB", RGB_image)
        cv2.imshow("Original_Depth_colormap", depth_colormap)

        # 保存视频
        # 将每一帧图像写入到输出文件中
        outVedio_RGB.write(RGB_image)  # 原RGB视频写入
        outVedio_IR_Left.write(left_image_BONE)  # IR_Left_Colormap_Bone视频写入
        outVedio_IR_Right.write(right_image_BONE)  # IR_Right_Colormap_Bone视频写入
        outVedio_Depth.write(depth_colormap)  # 原Depth视频写入

# 5.退出
        '============================================================================================'
        frame_count += 1
        key = cv2.waitKey(1)
        if (key & 0xFF == ord('q')) | (key & 0xFF == 27):
            pipeline.stop()
            cv2.destroyAllWindows()
            # 释放资源
            outVedio_RGB.release()
            outVedio_IR_Left.release()
            outVedio_IR_Right.release()
            outVedio_Depth.release()
            cv2.destroyAllWindows()
            break
        elif key & 0xFF == ord('s'):
            cv2.imwrite("./IR_Pairs/IR_Pair_{}.png".format(index), image_pair_IR)
            cv2.imwrite("./RGBD_Pairs/RGBD_Pair_{}.png".format(index), image_pair_RGBD)
            index += 1