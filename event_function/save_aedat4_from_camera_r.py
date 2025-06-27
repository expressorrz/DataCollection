import dv_processing as dv
import os
import threading

camera = dv.io.CameraCapture()

# print(f"Is event stream available? {str(camera.isEventStreamAvailable())}")
# print(f"Is frame stream available? {str(camera.isFrameStreamAvailable())}")
# print(f"Is imu stream available? {str(camera.isImuStreamAvailable())}")
# print(f"Is trigger stream available? {str(camera.isTriggerStreamAvailable())}")
eventsAvailable = camera.isEventStreamAvailable()
framesAvailable = camera.isFrameStreamAvailable()
imuAvailable = camera.isImuStreamAvailable()
triggersAvailable = camera.isTriggerStreamAvailable()

# camera.setDavisColorMode(dv.io.CameraCapture.DavisColorMode.Color)
# camera.setDavisColorMode(dv.io.CameraCapture.DavisColorMode.Grayscale)

# Open a file to write, will allocate streams for all available data types
task_id = input("Task No.").strip()
operator_id = input("Operator No.").strip()
run_id = input("Run No.").strip()

save_path = '/home/qiangubuntu/research/data_collection/src/data'
save_path = f'{save_path}/task_{task_id}/op_{operator_id}/id_{run_id}'
os.makedirs(save_path, exist_ok=True)
save_aedat4_path = os.path.join(save_path, f"event.aedat4")
base, ext = os.path.splitext(save_aedat4_path)
counter = 1
while os.path.exists(save_aedat4_path):
    save_aedat4_path = f"{base}_{counter}{ext}"
    counter += 1



recording = False

def toggle_recording():
    global recording
    while True:
        command = input("Enter 'r' to start recording, 's' to stop recording: ").strip().lower()
        if command == 'r':
            recording = True
            print("Recording started")
        elif command == 's':
            recording = False
            print("Recording stopped")

# Start a thread to listen for keyboard input
thread = threading.Thread(target=toggle_recording, daemon=True)
thread.start()

# try:
writer = dv.io.MonoCameraWriter(save_aedat4_path, camera)
print("Ready to record")
while camera.isConnected():
    if recording:
        if eventsAvailable:
            # Get Events
            events = camera.getNextEventBatch()
            # Write Events
            if events is not None:
                writer.writeEvents(events, streamName='events')

        if framesAvailable:
            # Get Frame
            frame = camera.getNextFrame()  # 346 x 260
            # Write Frame
            if frame is not None:
                writer.writeFrame(frame, streamName='frames')

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

# except KeyboardInterrupt:
#     print("Ending recording")
#     pass

