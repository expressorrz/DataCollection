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

def test(video_path, data_path):
    config = json.load(open('/home/ipu/codes/DP4HRC/data_collection_realsense/real_world/realsense_config/415_high_accuracy_mode.json', 'r'))

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


    with MultiRealsense(
            resolution=(640,480),
            capture_fps=15,
            record_fps=15,
            enable_color=True,
            enable_depth=True,
            save_path=data_path,
            # verbose=True
        ) as realsense:

        cv2.setNumThreads(1) 

        realsense.set_exposure(exposure=150, gain=5)
        intr = realsense.get_intrinsics()

        rec_start_time = time.time() + 8

        out = None
        cv2.namedWindow("default", cv2.WINDOW_AUTOSIZE)
        
        while True:
            out = realsense.get(out=out)
            bgr = []
            for i in range(realsense.n_cameras):
                if i not in out:
                    continue
                if out[i]['color'] is None:
                    continue
                bgr.append(out[i]['color'])

            
            bgr = np.concatenate(bgr, axis=1)


            # bgr = out[0]['color']

            cv2.imshow('default', bgr)

            key = cv2.pollKey()
            if key == ord('q'):
                break
            elif key == ord('r'):
                video_path = video_path
                realsense.start_recording(video_path, start_time=rec_start_time)
            elif key == ord('s'):
                realsense.stop_recording()

            time.sleep(1/60)
            if time.time() > (rec_start_time + 20.0):
                break


if __name__ == "__main__":
    task_id = input("Task No.").strip()
    operator_id = input("Human No.").strip()
    run_id = input("ID No.").strip()

    save_path = './data'
    video_path = f'{save_path}/task_{task_id}/op_{operator_id}/id_{run_id}'
    data_path = f'{save_path}/task_{task_id}/op_{operator_id}/id_{run_id}'

    for path in [video_path, data_path]:
        os.makedirs(path, exist_ok=True)

    test(video_path, data_path)
