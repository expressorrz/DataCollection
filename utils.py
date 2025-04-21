import os
import cv2
import numpy as np

from common.cv2_util import get_image_transform
color_transform = get_image_transform(
    input_res=(640, 480),
    output_res=(320, 240), 
    bgr_to_rgb=False)

def transform(data):
    data['color'] = color_transform(data['color'])
    data['depth'] = color_transform(data['depth'])
    data['depth_colormap'] = color_transform(data['depth_colormap'])
    return data


def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = str(len(dirs))

    return log_dir_index

def get_pointcloud(color_frame, depth_frame, color_image, depth_image, depth_colormap, depth_scale, pc, PointCloud_Process='method1'):
        depth_intrin = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        g_depth_intrinsics_matrix = np.array([
            [depth_intrin.fx, 0., depth_intrin.ppx],
            [0., depth_intrin.fy, depth_intrin.ppy],
            [0, 0, 1.]
        ])

        if PointCloud_Process == 'method1':
            pointcloud_data_ply, pointcloud_data_pcd = pointcloud_xyz_rgb(color_frame, depth_frame, color_image, depth_colormap, pc)
        elif PointCloud_Process == 'method2':
            pointcloud_data_ply = depth2xyz(depth_image, g_depth_intrinsics_matrix, depth_scale)
            pointcloud_data_pcd = depth2xyzrgb(color_image, depth_image, g_depth_intrinsics_matrix, depth_scale)

        return pointcloud_data_ply, pointcloud_data_pcd


def pointcloud_xyz_rgb(color_frame, depth_frame, color_img, depth_colormap, pc):
    # mapped_frame, color_source = depth_frame, depth_colormap
    mapped_frame, color_source = color_frame, color_img
    points = pc.calculate(depth_frame)
    pc.map_to(mapped_frame)
    v, t = points.get_vertices(), points.get_texture_coordinates()
    pointcloud_xyz = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz    shape: [N,3]

    # texcoords are [0..1] and relative to top-left pixel corner, multiply by size and add 0.5 to center
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv 色彩
    cw, ch = color_img.shape[:2][::-1]
    v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch - 1, out=u)
    np.clip(v, 0, cw - 1, out=v)
    pointcloud_rgb = color_img[u, v]    # rgb   shape: [N,3]

    pointcloud_data_ply = pointcloud_xyz
    pointcloud_data_pcd = np.concatenate([pointcloud_xyz, pointcloud_rgb], axis=1)
    
    return pointcloud_data_ply, pointcloud_data_pcd


def depth2xyzrgb(color_map, depth_map, depth_cam_matrix, depth_scale=0.0010):
    """
    https://blog.csdn.net/tycoer/article/details/106761886
    # 深度转点云 https://blog.csdn.net/FUTEROX/article/details/126128581
    color_map = np.random.randint(0,255,(720, 1280, 3))
    depth_map = np.random.randint(0,10000,(720, 1280))
    depth_cam_matrix = np.array([[540, 0,  640],
                                 [0,   540,360],
                                 [0,   0,    1]])
    pc = depth2xyzrgb(color_map, depth_map, depth_cam_matrix)
    """
    if depth_map.shape[:2] != color_map.shape[:2]:
        _h, _w = depth_map.shape[:2]
        color_map = cv2.resize(color_map, (_w, _h))

    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]  # [H,W,2]
    z = depth_map * depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    height, width = color_map.shape[:2]
    data_ply = np.zeros((6, width * height), dtype=np.float32)
    data_ply[0] = x.reshape(-1)
    data_ply[1] = y.reshape(-1)
    data_ply[2] = z.reshape(-1)
    data_ply[3] = color_map[:, :, 0].reshape(-1)
    data_ply[4] = color_map[:, :, 1].reshape(-1)
    data_ply[5] = color_map[:, :, 2].reshape(-1)

    data_ply = data_ply.T  # [N,6]
    # ################ pcl 坐标轴是 右手坐标系, 需要把realsense的坐标轴转换下  ################
    # ## realsense坐标系：  X代表水平方向右  Y代表垂直方向下    Z代表深度距离内
    # ## 右手坐标系: X代表水平方向右  Y代表垂直向上   Z代表深度距离向外
    # ## 左手坐标系: X代表水平方向右  Y代表垂直向上   Z代表深度距离向内
    data_ply[:, [1, 2]] *= -1.
    # ################ ################ ################ ################ ################

    return data_ply


def depth2xyz(depth_map, depth_cam_matrix, flatten=True, depth_scale=0.0010):
    """
    https://blog.csdn.net/tycoer/article/details/106761886
    # 深度转点云 https://blog.csdn.net/FUTEROX/article/details/126128581

    depth_map = np.random.randint(0,10000,(720, 1280))
    depth_cam_matrix = np.array([[540, 0,  640],
                                 [0,   540,360],
                                 [0,   0,    1]])
    pc = depth2xyz(depth_map, depth_cam_matrix)
    """
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    z = depth_map * depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy

    xyz = np.dstack((x, y, z)).reshape(-1, 3) if flatten else np.dstack((x, y, z))
    # xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)

    # ################ pcl 坐标轴是 右手坐标系, 需要把realsense的坐标轴转换下  ################
    # ## realsense坐标系：  X代表水平方向右  Y代表垂直方向下    Z代表深度距离内
    # ## 右手坐标系: X代表水平方向右  Y代表垂直向上   Z代表深度距离向外
    # ## 左手坐标系: X代表水平方向右  Y代表垂直向上   Z代表深度距离向内
    xyz[:, [1, 2]] *= -1.
    # ################ ################ ################ ################ ################

    return xyz  # [N,3]



def save_2_ply(pointcloud_xyz, save_filepath):
    data_ply = pointcloud_xyz
    ##################### save *.ply ###########################
    # data_ply.shape:  [N,3]  or  [N,6]
    assert isinstance(data_ply, np.ndarray)
    is_color = data_ply.shape[1] == 6
    float_formatter = lambda x: "%.4f" % x
    points = []
    for i in data_ply:
        if is_color:
            if np.alltrue(i[:3] == 0): continue
            points.append("{} {} {} {} {} {} 0\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           int(i[3]), int(i[4]), int(i[5])))
        else:
            if np.alltrue(i == 0): continue
            points.append("{} {} {}\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           ))

    file = open(save_filepath, "w")
    if is_color:
        file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(points)))
    else:
        file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    end_header
    %s
    ''' % (len(points), "".join(points)))

    file.close()
    ######################################################################


def save_2_pcd(pointcloud_data_pcd, save_filepath):
    data_pcd = pointcloud_data_pcd
    assert isinstance(data_pcd, np.ndarray)
    # [N, 6]
    is_color = data_pcd.shape[1] == 6
    float_formatter = lambda x: "%.4f" % x
    points = []
    for i in data_pcd:
        if is_color:
            r, g, b = list(map(int, i[3:]))
            points.append("{} {} {} {}\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           (int(r) << 16) | (int(g) << 8) | int(b),
                           ))
        else:
            points.append("{} {} {}\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           ))

    file = open(save_filepath, "w")
    if is_color:
        file.write('''# .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z rgb
    SIZE 4 4 4 4
    TYPE F F F U
    COUNT 1 1 1 1
    WIDTH %d
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS %d
    DATA ascii
    %s
    ''' % (len(points), len(points), "".join(points)))
    else:
        file.write('''# .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z
    SIZE 4 4 4
    TYPE F F F
    COUNT 1 1 1
    WIDTH %d
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS %d
    DATA ascii
    %s
    ''' % (len(points), len(points), "".join(points)))
    file.close()