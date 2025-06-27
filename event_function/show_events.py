import dv_processing as dv
import cv2 as cv
from datetime import timedelta

# Open the camera, just use first detected DAVIS camera
camera = dv.io.CameraCapture("", dv.io.CameraCapture.CameraType.DAVIS)

# Initialize a multi-stream slicer
slicer = dv.EventMultiStreamSlicer("events")

# Add a frame stream to the slicer
slicer.addFrameStream("frames")

# Initialize a visualizer for the overlay
# camera.getEventResolution() = (346, 260)
visualizer = dv.visualization.EventVisualizer(camera.getEventResolution(), dv.visualization.colors.white(),
                                              dv.visualization.colors.green(), dv.visualization.colors.red())

# Create a window for image display
cv.namedWindow("Preview", cv.WINDOW_NORMAL)


# Callback method for time based slicing
def display_preview(data):
    # Retrieve frame data using the named method and stream name
    frames = data.getFrames("frames")

    # Retrieve event data
    events = data.getEvents("events")

    # Retrieve and color convert the latest frame of retrieved frames
    latest_image = None
    if len(frames) > 0:
        if len(frames[-1].image.shape) == 3:
            # We already have colored image, no conversion
            latest_image = frames[-1].image
        else:
            # Image is grayscale, convert to color (BGR image)
            latest_image = cv.cvtColor(frames[-1].image, cv.COLOR_GRAY2BGR)
    else:
        return

    # Generate a preview and show the final image
    cv.imshow("Preview", visualizer.generateImage(events, latest_image))

    # If escape button is pressed (code 27 is escape key), exit the program cleanly
    if cv.waitKey(2) == 27:
        exit(0)


# Register a job to be performed every 33 milliseconds
slicer.doEveryTimeInterval(timedelta(milliseconds=33), display_preview)

# Continue the loop while both cameras are connected
while camera.isRunning():
    events = camera.getNextEventBatch()
    if events is not None:
        slicer.accept("events", events)

    frame = camera.getNextFrame()
    if frame is not None:
        slicer.accept("frames", [frame])
