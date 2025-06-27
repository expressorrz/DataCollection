import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import argparse

parser = argparse.ArgumentParser(description='Show a preview of an iniVation event camera input.')
args = parser.parse_args()

# Open any camera
capture = dv.io.CameraCapture()

# frameRate = capture.getFrameRate()
# print(f"Frame rate: {frameRate}")

# Make sure it supports event stream output, throw an error otherwise
if not capture.isEventStreamAvailable():
    raise RuntimeError("Input camera does not provide an event stream.")

visualizer = dv.visualization.EventVisualizer(capture.getEventResolution())
visualizer.setBackgroundColor(dv.visualization.colors.white())
visualizer.setPositiveColor(dv.visualization.colors.red())
visualizer.setNegativeColor(dv.visualization.colors.green())

# static const cv::Scalar black    = cv::Scalar(0, 0, 0);
# static const cv::Scalar white    = cv::Scalar(255, 255, 255);
# static const cv::Scalar red      = cv::Scalar(0, 0, 255); BGR
# static const cv::Scalar lime     = cv::Scalar(0, 255, 0);
# static const cv::Scalar blue     = cv::Scalar(255, 0, 0);
# static const cv::Scalar yellow   = cv::Scalar(0, 255, 255);
# static const cv::Scalar cyan     = cv::Scalar(255, 255, 0);
# static const cv::Scalar magenta  = cv::Scalar(255, 0, 255);
# static const cv::Scalar silver   = cv::Scalar(192, 192, 192);
# static const cv::Scalar gray     = cv::Scalar(128, 128, 128);
# static const cv::Scalar navy     = cv::Scalar(128, 0, 0);
# static const cv::Scalar green    = cv::Scalar(0, 128, 0);
# static const cv::Scalar iniBlue  = cv::Scalar(183, 93, 0);
# static const cv::Scalar darkGrey = cv::Scalar(43, 43, 43);

# Declare the callback method for slicer
def slicing_callback(events: dv.EventStore):
    frame = visualizer.generateImage(events)
    cv.imshow("Preview", frame)
    cv.waitKey(2)

slicer = dv.EventStreamSlicer()
cv.namedWindow("Preview", cv.WINDOW_NORMAL)
slicer.doEveryTimeInterval(timedelta(milliseconds=33), slicing_callback)
# 1000 / 16 milliseconds ≈ 62.5 FPS
# 1000 / 33 milliseconds ≈ 30 FPS

# Run the event processing while the camera is connected
while capture.isRunning():
    events = capture.getNextEventBatch()
    if events is not None:
        slicer.accept(events)
