import dv_processing as dv
import os
import cv2 as cv
from datetime import timedelta
import numpy as np
import pandas as pd
import time

camera = dv.io.CameraCapture()
eventsAvailable = camera.isEventStreamAvailable()
framesAvailable = camera.isFrameStreamAvailable()

# --------------------------------------------------------

# save_dir = '/home/ipu/codes/DP4HRC/data_collection/src/data'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# idx = get_log_dir_index(save_dir)
# save_path = '%s/%s' % (save_dir, idx)
# os.makedirs(save_path, exist_ok=True)

# print(f'save_path: {save_path}')
# --------------------------------------------------------

# print (f"dv.visualization.colors.green(): {dv.visualization.colors.green()}") (0.0, 128.0, 0.0)
# print (f"dv.visualization.colors.red(): {dv.visualization.colors.red()}") (0.0, 0.0, 255.0)
task_id = input("Task No.").strip()
operator_id = input("Operator No.").strip()
run_id = input("Run No.").strip()

save_path = '/home/qiangubuntu/research/data_collection/src/data'
save_path = f'{save_path}/task_{task_id}/op_{operator_id}/id_{run_id}'
os.makedirs(save_path, exist_ok=True)

# aedat4_path = os.path.join(save_path, f"event.aedat4")

events_folderpath = f'{save_path}/events'
os.makedirs(events_folderpath, exist_ok=True)

png_folderpath = f'{save_path}/png'
os.makedirs(png_folderpath, exist_ok=True)

# base, ext = os.path.splitext(aedat4_path)
# counter = 1
# while os.path.exists(aedat4_path):
#     aedat4_path = f"{base}_{counter}{ext}"
#     counter += 1

slicer = dv.EventMultiStreamSlicer("events")
# slicer.addFrameStream("frames")
# camera.getEventResolution() = (346, 260)
visualizer = dv.visualization.EventVisualizer((346, 260), dv.visualization.colors.white(),
                                              dv.visualization.colors.green(), dv.visualization.colors.red())
# cv.namedWindow("Preview", cv.WINDOW_NORMAL)

# Initialize the counter variable globally
counter = 0

def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = str(len(dirs) - 1)

    return log_dir_index

def saveData(data):
    events = data.getEvents("events")
    events_packets = []
    global counter 
    events_csv_path = os.path.join(events_folderpath, f"{counter}.csv")
    

    if events is not None:
        # 将events保存为csv文件
        events_packets.append(pd.DataFrame(events.numpy()))
        events_pandas = pd.concat(events_packets)
        events_pandas.to_csv(events_csv_path)

        # 将events保存为png文件
        first_time = events[0].timestamp()
        last_time = events[events.size() - 1].timestamp()
        filename = os.path.join(png_folderpath, f"{counter}.png")
        cv.imwrite(filename, visualizer.generateImage(events))

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


# counter = 0
# try:
#     print("Ready to record")
#     while camera.isConnected():
#         if eventsAvailable:
#             events = camera.getNextEventBatch()
#             events_packets = []
#             events_path = os.path.join(events_folderpath, f"{counter}.csv")
#             counter += 1
#             if events is not None:
#                 events_packets.append(pd.DataFrame(events.numpy()))
#                 

#         # if framesAvailable:
#         #     # Get Frame
#         #     frame = camera.getNextFrame()  # 346 x 260
#         #     # Write Frame
#         #     if frame is not None:
#         #         writer.writeFrame(frame, streamName='frames')


# except KeyboardInterrupt:
#     print("Ending recording")
#     pass

