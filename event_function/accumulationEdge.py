import dv_processing as dv
import cv2 as cv
from datetime import timedelta

# Open any camera
capture = dv.io.CameraCapture()

# Make sure it supports event stream output, throw an error otherwise
if not capture.isEventStreamAvailable():
    raise RuntimeError("Input camera does not provide an event stream.")

# Initialize an accumulator with some resolution
accumulator = dv.EdgeMapAccumulator(capture.getEventResolution())

# Apply configuration, these values can be modified to taste
accumulator.setNeutralPotential(0.0)
accumulator.setContribution(0.25)
accumulator.setDecay(1.0)
accumulator.setIgnorePolarity(False)

# Initialize a preview window
cv.namedWindow("Preview", cv.WINDOW_NORMAL)

# Initialize a slicer
slicer = dv.EventStreamSlicer()


# Declare the callback method for slicer
def slicing_callback(events: dv.EventStore):
    # Pass events into the accumulator and generate a preview frame
    accumulator.accept(events)
    frame = accumulator.generateFrame()

    # Show the accumulated image
    cv.imshow("Preview", frame.image)
    cv.waitKey(2)


# Register callback to be performed every 33 milliseconds
slicer.doEveryTimeInterval(timedelta(milliseconds=33), slicing_callback)

# Run the event processing while the camera is connected
while capture.isRunning():
    # Receive events
    events = capture.getNextEventBatch()

    # Check if anything was received
    if events is not None:
        # If so, pass the events into the slicer to handle them
        slicer.accept(events)