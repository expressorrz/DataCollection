import dv_processing as dv
import cv2 as cv
import time

# Open any camera
capture = dv.io.CameraCapture()
cv.namedWindow("Preview", cv.WINDOW_NORMAL)

while capture.isRunning():
    frame = capture.getNextFrame()
    events = capture.getNextEventBatch()
    imu_batch = capture.getNextImuBatch()

    if frame is not None and events is not None:
        # Print received packet time range
        print(f"********************************************** \n [{frame}]")
        print(f"Received a frame at time [{frame.timestamp}]")



        print(f"********************************************** \n [{events}]")
        print(f"Received events within time range [{events.getLowestTime()}; {events.getHighestTime()}]")
        # for ev in events:
        #     print(f"Sliced event [{ev.timestamp()}, {ev.x()}, {ev.y()}, {ev.polarity()}]")



        # print(f"********************************************** \n [{imu_batch}]")
        # print(f"Received imu data within time range [{imu_batch[0].timestamp}; {imu_batch[-1].timestamp}]")
        # break

        cv.imshow("Preview", frame.image)

    else:
        # No data has arrived yet, short sleep to reduce CPU load
        time.sleep(0.001)
    cv.waitKey(2)
