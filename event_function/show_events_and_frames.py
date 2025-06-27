import dv_processing as dv
import cv2 as cv
from datetime import timedelta

# capture = dv.io.CameraCapture("", dv.io.CameraCapture.CameraType.DAVIS)
capture = dv.io.CameraCapture()

# Callback method for time based slicing
def display_preview(data):
    frames = data.getFrames("frames")
    events = data.getEvents("events")

    # Retrieve and color convert the latest frame of retrieved frames
    latest_image = None
    if len(frames) > 0:
        latest_image = frames[-1].image
        if len(frames[-1].image.shape) != 3:
            latest_image = cv.cvtColor(latest_image, cv.COLOR_GRAY2BGR)
        # Create a copy of the latest image to ensure it is not modified
        # because the visualizer.generateImage(events, latest_image) will modify the latest_image
        frame_copy = latest_image.copy()
        # cv.imshow("Frame Preview", frame_copy)
    else:
        return

    if len(events) > 0:
        event_image = visualizer.generateImage(events)
        event_image = cv.resize(event_image, (latest_image.shape[1], latest_image.shape[0]))
        # cv.imshow("Event Preview", event_image)
    else:
        return

    blended_image = visualizer.generateImage(events, latest_image)
    blended_image = cv.resize(blended_image, (latest_image.shape[1], latest_image.shape[0]))
    # cv.imshow("Blended Preview", blended_image)
    cv.imshow("Preview", cv.hconcat([blended_image, event_image, frame_copy]))

    if cv.waitKey(2) == 27:
        exit(0)

slicer = dv.EventMultiStreamSlicer("events")
slicer.addFrameStream("frames")
visualizer = dv.visualization.EventVisualizer(capture.getEventResolution())
visualizer.setBackgroundColor(dv.visualization.colors.white())
visualizer.setPositiveColor(dv.visualization.colors.red())
visualizer.setNegativeColor(dv.visualization.colors.green())
cv.namedWindow("Preview", cv.WINDOW_NORMAL)
# cv.namedWindow("Blended Preview", cv.WINDOW_NORMAL)
# cv.namedWindow("Event Preview", cv.WINDOW_NORMAL)
# cv.namedWindow("Frame Preview", cv.WINDOW_NORMAL)
slicer.doEveryTimeInterval(timedelta(milliseconds=33), display_preview)

# Continue the loop while both cameras are connected
while capture.isRunning():
    events = capture.getNextEventBatch()
    if events is not None:
        slicer.accept("events", events)

    frame = capture.getNextFrame()
    if frame is not None:
        slicer.accept("frames", [frame])