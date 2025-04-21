import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import json
import time
import numpy as np
from real_world.multi_realsense import MultiRealsense
from utils import get_log_dir_index

import zarr

# def transform(data):
#     color = data['color']
#     h,w,_ = color.shape
#     factor = 2
#     color = cv2.resize(color, (w//factor,h//factor), interpolation=cv2.INTER_AREA)
#     data['color'] = color
#     return data

# 1280, 720
# 640, 480




from common.cv2_util import get_image_transform
color_transform = get_image_transform(
    input_res=(1280, 720),
    output_res=(640, 360), 
    bgr_to_rgb=False)

def transform(data):
    data['color'] = color_transform(data['color'])
    data['depth'] = color_transform(data['depth'])
    data['depth_colormap'] = color_transform(data['depth_colormap'])
    return data


def test(save_path):
    # 1. load config
    config = json.load(open('/home/ipu/codes/DP4HRC/data_collection/src/real_world/realsense_config/415_high_accuracy_mode.json', 'r'))
    resolution = [1280, 720]
    fps = 30
    serial_numbers = ['327122075831'] # 247122071632,  239722072823

    with MultiRealsense(
            serial_numbers=serial_numbers,
            resolution=resolution,
            capture_fps=fps,
            record_fps=fps,
            enable_color=True,
            enable_depth=True,
            enable_filter=True,
            enable_pc=True,
            transform=transform,
            vis_transform=transform,
            recording_transform=transform,
            # verbose=True
        ) as realsense:
        cv2.setNumThreads(1)

        n_cams = realsense.n_cameras
        recording = False

        if transform is not None:
            resolution = [resolution[0] // 2, resolution[1] // 2]

         # 2. create data and metadata for zarr
        if save_path is not None:
            root = zarr.open_group(os.path.join(save_path, 'data_buffer.zarr'), mode='w')

            data_groups = []
            meta_groups = []
            for i in range(n_cams):
                camera_group = root.require_group(f"cam_{i}")

                # create data group
                data_group = camera_group.require_group("data")
                data_group.require_dataset("color", shape=(0, resolution[1], resolution[0], 3), dtype='uint8', chunks=(1, resolution[1], resolution[0], 3), overwrite=False)
                data_group.require_dataset("depth", shape=(0, resolution[1], resolution[0]), dtype='uint16', chunks=(1, resolution[1], resolution[0]), overwrite=False)
                data_group.require_dataset("depth_colormap", shape=(0, resolution[1], resolution[0], 3), dtype='uint8', chunks=(1, resolution[1], resolution[0], 3), overwrite=False)
                data_group.require_dataset("pc_ply", shape=(0, resolution[1] * resolution[0], 3), dtype='float32', chunks=(1, resolution[1] * resolution[0], 3), overwrite=False)
                data_group.require_dataset("pc_pcd", shape=(0, resolution[1] * resolution[0], 6), dtype='float32', chunks=(1, resolution[1] * resolution[0], 6), overwrite=False)
                data_group.require_dataset("timestamp", shape=(0,), dtype='float64', chunks=(1024,), overwrite=False)
                data_group.require_dataset("step_idx", shape=(0,), dtype='int32', chunks=(1024,), overwrite=False)

                meta_group = camera_group.require_group("meta")
                meta_group.require_dataset("serial_number", shape=(1,), dtype='S32', overwrite=True)
                meta_group.require_dataset("intrinsics", shape=(3,3), dtype='float64', overwrite=True)
                meta_group.require_dataset("depth_scale", shape=(1,), dtype='float64', overwrite=True)

                data_groups.append(data_group)
                meta_groups.append(meta_group)
        else:
            raise RuntimeError('save_path is None.')


        realsense.set_exposure(exposure=200, gain=5)
        intrs = realsense.get_intrinsics()
        serial_numbers = realsense.get_serial_numbers()
        depth_scales = realsense.get_depth_scale()

        # save the metadata of the cameras
        for i in range(n_cams):
            meta_groups[i]["serial_number"][...] = serial_numbers[i].encode('utf-8')
            meta_groups[i]["intrinsics"][...] = intrs[i]
            meta_groups[i]["depth_scale"][...] = depth_scales[i]
            
            
        rec_start_time = time.time() + 5

        out = None
        cv2.namedWindow("default", cv2.WINDOW_AUTOSIZE)
        
        while True:
            out = realsense.get(out=out)

            # save the data to zarr
            if recording:
                rec_data = out
                for i in range(n_cams):
                    rec_data = out[i]
                    data_groups[i]["color"].append(rec_data["color"][None, ...])
                    data_groups[i]["depth"].append(rec_data["depth"][None, ...])
                    data_groups[i]["depth_colormap"].append(rec_data["depth_colormap"][None, ...])
                    data_groups[i]["pc_ply"].append(rec_data["pc_ply"][None, ...])
                    data_groups[i]["pc_pcd"].append(rec_data["pc_pcd"][None, ...])
                    data_groups[i]["timestamp"].append(np.array([rec_data["timestamp"]]))
                    data_groups[i]["step_idx"].append(np.array([rec_data["step_idx"]], dtype=np.int32))

            bgr = []
            for i in range(realsense.n_cameras):
                if i not in out:
                    continue
                if out[i]['color'] is None:
                    continue
                bgr.append(out[i]['color'])
            bgr = np.concatenate(bgr, axis=1)

            cv2.imshow('default', bgr)

            key = cv2.pollKey()
            if key == ord('r'):
                recording = True
                video_path = save_path + '/video'
                os.makedirs(video_path, exist_ok=True)
                realsense.start_recording(video_path, start_time=rec_start_time)
            if key == ord('q'):
                if recording:
                    recording = False
                    realsense.stop_recording()
                break

            time.sleep(1/60)
            if time.time() > (rec_start_time + 1000.0):
                break


if __name__ == "__main__":
    save_dir = '/home/ipu/codes/DP4HRC/data_collection/src/data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    idx = get_log_dir_index(save_dir)
    save_path = '%s/%s' % (save_dir, idx)
    os.makedirs(save_path, exist_ok=True)


    print(f"save_path: {save_path}")

    test(save_path)
