import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import json
import time
from multiprocessing.managers import SharedMemoryManager
from real_world.single_realsense import SingleRealsense

def test(video_path, data_path):
    
    serials = SingleRealsense.get_connected_devices_serial()
    # import pdb; pdb.set_trace()
    serial = serials[0]
    config = json.load(open('/home/qiangubuntu/research/data_collection/src/real_world/realsense_config/415_high_accuracy_mode.json', 'r'))

    def transform(data):
        color = data['color']
        h,w,_ = color.shape
        factor = 2
        color = cv2.resize(color, (w//factor,h//factor), interpolation=cv2.INTER_AREA)
        # color = color[:,140:500]
        data['color'] = color
        return data

    with SharedMemoryManager() as shm_manager:
        with SingleRealsense(
            shm_manager=shm_manager,
            serial_number=serial,
            resolution=(640,480),
            capture_fps=15,
            enable_color=True,
            enable_depth=True,
            save_path=data_path,
            # transform=transform,
            # recording_transform=transform
            # verbose=True
            ) as realsense:

            
            cv2.setNumThreads(1) 

            realsense.set_exposure(exposure=150, gain=5)
            intr = realsense.get_intrinsics()

            rec_start_time = time.time() + 2

            cv2.startWindowThread()
            cv2.namedWindow("default", cv2.WINDOW_NORMAL)

            data = None
            while True:
                data = realsense.get(out=data)

                bgr = data['color']

                cv2.imshow('default', bgr)
                key = cv2.pollKey()
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    video_path = video_path + '/video.mp4'
                    realsense.start_recording(video_path)
                elif key == ord('s'):
                    realsense.stop_recording()
                
                time.sleep(1/60)
                if time.time() > (rec_start_time + 20.0):
                    break


if __name__ == "__main__":
    task_id = input("Task No.").strip()
    operator_id = input("Human No.").strip()
    run_id = input("ID No.").strip()

    save_path = './home/qiangubuntu/research/data_collection/src/data'
    video_path = f'{save_path}/task_{task_id}/op_{operator_id}/id_{run_id}/video'
    data_path = f'{save_path}/task_{task_id}/op_{operator_id}/id_{run_id}'

    for path in [video_path, data_path]:
        os.makedirs(path, exist_ok=True)

    test(video_path, data_path)