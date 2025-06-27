import dv_processing as dv
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Show a preview of an AEDAT4 recording.')

# Example usage command:
# python3 test_davis/play_aedat4_acc.py -f ./test_davis/data/dvSave-2025_01_13_18_24_11.aedat4
parser.add_argument('-f,--file',
                    dest='file',
                    type=str,
                    required=False,
                    default='/home/qiangubuntu/research/dv-processing/test_davis/data/0408/2.aedat4',
                    metavar='path/to/file',
                    help='Path to an AEDAT4 file')

args = parser.parse_args()

# Open the recording file
recording = dv.io.MonoCameraRecording(args.file)

# Make sure the streams are available
assert recording.isEventStreamAvailable()
assert recording.isFrameStreamAvailable()
assert recording.getFrameResolution() == recording.getEventResolution()

# Initialize event accumulator with the known resolution
acc = dv.Accumulator(recording.getEventResolution())

# Some accumulation parameters
acc.setMaxPotential(1.0)
acc.setEventContribution(0.12)

# Create the preview window
cv.namedWindow("Preview", cv.WINDOW_NORMAL)

# Last frame is None for the first iteration
lastFrame = None

# Read the first frame from the file
frame = recording.getNextFrame()

# While not end-of-file, None is used to determine the last recorded frame
# print_flag = True
while frame is not None:
    # We have more than one frame
    if lastFrame is not None:
        # Time delay for the image to be displayed
        delay = frame.timestamp - lastFrame.timestamp

        # Read intermediate events that are available between the last and current frame
        events = recording.getEventsTimeRange(lastFrame.timestamp, frame.timestamp)
        print(f"events: {events}")
        if events is not None:
            # Accumulate the events
            acc.accept(events)

        # Retrieve accumulated image
        print(f"acc: {acc}")
        accumulatedFrame = acc.generateFrame()

        # If frames have more than 2 dimensions, convert the accumulated image into BGR colorspace
        # if print_flag:
        #     print(f"frame.image.shape: {frame.image.shape}")
        #     print(f"length of Frame.image.shape: {len(frame.image.shape)}")
        #     print_flag = False
        if len(lastFrame.image.shape) > 2:
            accumulatedFrame.image = cv.cvtColor(accumulatedFrame.image, cv.COLOR_GRAY2BGR)

        # Concat the images and preview them
        cv.imshow("Preview", cv.hconcat([lastFrame.image, accumulatedFrame.image]))
        cv.waitKey(int(delay / 1000))

    # Cache the last read frame and move to the next frame
    lastFrame = frame
    frame = recording.getNextFrame()
