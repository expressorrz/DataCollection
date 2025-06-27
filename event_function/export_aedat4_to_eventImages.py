import dv_processing as dv
import cv2 as cv
import os
from datetime import timedelta


# Open a file
file_dir = "/home/qiangubuntu/research/dv-processing/test_davis/data/0408/5.aedat4"
# Extract directory and filename
output_dir = os.path.dirname(file_dir)
filename = os.path.basename(file_dir) # 5.aedat4
filename_no_ext = os.path.splitext(filename)[0]  # Remove file extension
# print(f"filename_no_ext: {filename_no_ext}") # filename_no_ext: 5
event_image_dir = os.path.join(output_dir, f"{filename_no_ext}_eventFrame")
os.makedirs(event_image_dir, exist_ok=True)

reader = dv.io.MonoCameraRecording(file_dir)

# # 获取分辨率和帧率
# resolution = reader.getEventResolution()
# fps = 30 

def slicing_callback(events: dv.EventStore):
    frame = visualizer.generateImage(events)
    cv.imshow("Preview", frame)

    # # Save the frame to the output directory
    timestamp = events.timestamps
    timestamp = str(timestamp).split('[')[1].split(';')[0]
    filename = os.path.join(event_image_dir, f"{timestamp}.png")
    cv.imwrite(filename, frame)
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
