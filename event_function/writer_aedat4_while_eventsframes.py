import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import argparse

# Example usage command:
# python3 test_davis/writer_mono_dvs.py -c DAVIS346_00001074 -o ./test_davis/data/getAllDataFromCamera_1.aedat4
parser = argparse.ArgumentParser(description='Record data from a single iniVation camera to a file.')
parser.add_argument("-c",
                    "--camera_name",
                    dest='camera_name',
                    default="DAVIS346_00001074",
                    type=str,
                    help="Camera name (e.g. DVXplorer_DXA00093). The application will open any supported camera "
                    "if no camera name is provided.")
parser.add_argument("-o",
                    "--output_path",
                    dest='output_path',
                    default="/home/qiangubuntu/research/dv-processing/test_davis/data/0408/1.aedat4",
                    type=str,
                    help="Path to an output aedat4 file for writing.")
args = parser.parse_args()

# Open any camera that is discovered in the system
camera = dv.io.CameraCapture(args.camera_name)

# Check whether frames are available
eventsAvailable = camera.isEventStreamAvailable()
framesAvailable = camera.isFrameStreamAvailable()
imuAvailable = camera.isImuStreamAvailable()
triggersAvailable = camera.isTriggerStreamAvailable()

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
    cv.waitKey(33)

    if cv.waitKey(2) == 27:
        exit(0)

slicer = dv.EventMultiStreamSlicer("events")
slicer.addFrameStream("frames")
visualizer = dv.visualization.EventVisualizer(camera.getEventResolution())
visualizer.setBackgroundColor(dv.visualization.colors.white())
visualizer.setPositiveColor(dv.visualization.colors.red())
visualizer.setNegativeColor(dv.visualization.colors.green())
cv.namedWindow("Preview", cv.WINDOW_NORMAL)
slicer.doEveryTimeInterval(timedelta(milliseconds=33), display_preview)

try:
    # Open a file to write, will allocate streams for all available data types
    writer = dv.io.MonoCameraWriter(args.output_path, camera)

    print("Start recording")
    while camera.isConnected():
        if eventsAvailable:
            # Get Events
            events = camera.getNextEventBatch()
            # Write Events
            if events is not None:
                writer.writeEvents(events, streamName='events')
                slicer.accept("events", events)

        if framesAvailable:
            # Get Frame
            frame = camera.getNextFrame()
            # Write Frame
            if frame is not None:
                writer.writeFrame(frame, streamName='frames')
                slicer.accept("frames", [frame])

        if imuAvailable:
            # Get IMU data
            imus = camera.getNextImuBatch()
            # Write IMU data
            if imus is not None:
                writer.writeImuPacket(imus, streamName='imu')

        if triggersAvailable:
            # Get trigger data
            triggers = camera.getNextTriggerBatch()
            # Write trigger data
            if triggers is not None:
                writer.writeTriggerPacket(triggers, streamName='triggers')

except KeyboardInterrupt:
    print("Ending recording")
    pass
