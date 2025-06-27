import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import os

# Open a file
# reader = dv.io.MonoCameraRecording("/home/qiangubuntu/research/dv-processing/test_davis/data/collect_data_0415/task_0_op_0_run_0.aedat4")

# Retrieve all .aedat4 files in the specified directory and its subdirectories
# data_directory = "/home/qiangubuntu/research/dv-processing/test_davis/data"
data_directory = "/home/qiangubuntu/research/"
aedat4_files = [
    os.path.join(root, f)
    for root, _, files in os.walk(data_directory)
    for f in files if f.endswith(".aedat4")
]
if aedat4_files is None:
    raise FileNotFoundError("No .aedat4 files found in the specified directory.")

# Print the list of .aedat4 files with their indices
for idx, file in enumerate(aedat4_files):
    print(f"{idx}: {file}")


selected_index = int(input("Enter the index of the file you want to open: "))

# Open the selected .aedat4 file
if 0 <= selected_index < len(aedat4_files):
    reader = dv.io.MonoCameraRecording(aedat4_files[selected_index])
else:
    raise IndexError("Invalid index selected.")

print(f"Opened an AEDAT4 file which contains data from [{reader.getCameraName()}] camera")


def slicing_callback(events: dv.EventStore):
    frame = visualizer.generateImage(events)
    cv.imshow("Preview", frame)
    cv.waitKey(33)

slicer = dv.EventStreamSlicer()
cv.namedWindow("Preview", cv.WINDOW_NORMAL)
visualizer = dv.visualization.EventVisualizer(reader.getEventResolution())
visualizer.setBackgroundColor(dv.visualization.colors.white())
visualizer.setPositiveColor(dv.visualization.colors.red())
visualizer.setNegativeColor(dv.visualization.colors.green())
slicer.doEveryTimeInterval(timedelta(milliseconds=33), slicing_callback)

# Read events from the file
while reader.isRunning():
    events = reader.getNextEventBatch()
    if events is not None:
        slicer.accept(events)