import dv_processing as dv
import os
import cv2 as cv
from datetime import timedelta
import numpy as np
import pandas as pd
import time
import zarr

def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = str(len(dirs) - 1)

    return log_dir_index



camera = dv.io.CameraCapture()
eventsAvailable = camera.isEventStreamAvailable()
framesAvailable = camera.isFrameStreamAvailable()

# --------------------------------------------------------

save_dir = '/home/qiangubuntu/research/data_collection/src/data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

idx = get_log_dir_index(save_dir)
save_path = '%s/%s' % (save_dir, idx)
os.makedirs(save_path, exist_ok=True)

print(f'save_path: {save_path}')
# --------------------------------------------------------


# save data to zarr
root = zarr.open_group(os.path.join(save_path, 'event_buffer.zarr'), mode='w')

data_group = root.create_group('events', overwrite=True)
data_group.create_dataset('event_img', shape=(0, 260, 346, 3), chunks=(1, 260, 346, 3), dtype='uint8', overwrite=False)
data_group.create_dataset('timestamps', shape=(0,), chunks=(1024,), dtype='uint64', overwrite=False)
data_group.create_dataset('timestamps_final', shape=(0,), chunks=(1024,), dtype='uint64', overwrite=False)


slicer = dv.EventMultiStreamSlicer("events")
# slicer.addFrameStream("frames")
# camera.getEventResolution() = (346, 260)
visualizer = dv.visualization.EventVisualizer((346, 260), dv.visualization.colors.white(),
                                              dv.visualization.colors.green(), dv.visualization.colors.red()) # bac, pos, neg
# cv.namedWindow("Preview", cv.WINDOW_NORMAL)
counter = 0
def saveData(data):
    events = data.getEvents("events")
    if events is None:
        return
    
    # 将events保存为csv文件
    events_packets = []
    global counter
    events_csv_path = f'{save_path}/event_csv'
    os.makedirs(events_csv_path, exist_ok=True)
    events_csv_path = os.path.join(events_csv_path, f"{counter}.csv")

    events_packets.append(pd.DataFrame(events.numpy()))
    events_pandas = pd.concat(events_packets)
    events_pandas.to_csv(events_csv_path)
    # ******************************
    first_time = events[0].timestamp()
    last_time = events[events.size() - 1].timestamp()
    event_image = visualizer.generateImage(events)

    data_group['event_img'].append(event_image[None, ...])
    data_group['timestamps'].append(np.array([first_time], dtype=np.uint64))
    data_group['timestamps_final'].append(np.array([last_time], dtype=np.uint64))

    # # 将events保存为png文件
    # events_png_path = f'{save_path}/event_png'
    # os.makedirs(events_png_path, exist_ok=True)
    # events_png_path = os.path.join(events_png_path, f"{counter}.png")
    # cv.imwrite(events_png_path, event_image)

    counter += 1
        

    # cv.imshow("Preview", visualizer.generateImage(events))
    # if cv.waitKey(2) == 27: # code 27 is escape key
    #     exit(0)

slicer.doEveryTimeInterval(timedelta(milliseconds=33), saveData)

while camera.isRunning():
    events = camera.getNextEventBatch()
    if events is not None:
        slicer.accept("events", events)
        
    # frame = camera.getNextFrame()
    # if frame is not None:
    #     slicer.accept("frames", [frame])
