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

import zarr

def transform(data):
    color = data['color']
    h,w,_ = color.shape
    factor = 4
    color = cv2.resize(color, (w//factor,h//factor), interpolation=cv2.INTER_AREA)
    # color = color[:,140:500]
    data['color'] = color
    return data

from common.cv2_util import get_image_transform
color_transform = get_image_transform(
    input_res=(1280,720),
    output_res=(640,480), 
    bgr_to_rgb=False)

def transform(data):
    data['color'] = color_transform(data['color'])
    return data


def test(save_path):
    # 1. load config
    config = json.load(open('/home/qiangubuntu/research/data_collection/src/real_world/realsense_config/415_high_accuracy_mode.json', 'r'))
    resolution = [640, 480]
    fps = 30
    serial_numbers = ['239722072823', '327122075831'] # 247122071632

    with MultiRealsense(
            serial_numbers=serial_numbers,
            resolution=resolution,
            capture_fps=fps,
            record_fps=fps,
            enable_color=True,
            enable_depth=True,
            # verbose=True
        ) as realsense:
        cv2.setNumThreads(1)

        n_cams = realsense.n_cameras
        recording = False

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
                data_group.require_dataset("timestamp", shape=(0,), dtype='float64', chunks=(1024,), overwrite=False)
                data_group.require_dataset("step_idx", shape=(0,), dtype='int32', chunks=(1024,), overwrite=False)

                meta_group = camera_group.require_group("meta")
                meta_group.require_dataset("serial_number", shape=(1,), dtype='S32', overwrite=True)
                meta_group.require_dataset("intrinsics", shape=(7,), dtype='float64', overwrite=True)

                data_groups.append(data_group)
                meta_groups.append(meta_group)
        else:
            raise RuntimeError('save_path is None.')


        realsense.set_exposure(exposure=150, gain=5)
        intrs = realsense.get_intrinsics()
        serial_numbers = realsense.get_serial_numbers()

        # save the metadata of the cameras
        # for i in range(n_cams):
        #     meta_group[i]["serial_number"][...] = serial_numbers[i].encode('utf-8')
        #     meta_group[i]["intrinsics"][...] = intrs[i]
            
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
            if key == ord('q'):
                break
            elif key == ord('r'):
                recording = True
                video_path = save_path + '/video'
                os.makedirs(video_path, exist_ok=True)
                realsense.start_recording(video_path, start_time=rec_start_time)
            elif key == ord('s'):
                realsense.stop_recording()

            time.sleep(1/60)
            if time.time() > (rec_start_time + 1000.0):
                break


if __name__ == "__main__":
    task_id = input("Task No.").strip()
    operator_id = input("Human No.").strip()
    run_id = input("ID No.").strip()

    save_path = '/home/qiangubuntu/research/data_collection/src/data'
    save_path = f'{save_path}/task_{task_id}/op_{operator_id}/id_{run_id}'

    os.makedirs(save_path, exist_ok=True)

    test(save_path)
