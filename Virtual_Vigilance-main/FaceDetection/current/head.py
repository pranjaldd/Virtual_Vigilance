import threading
import cv2
import numpy as np

# Video capture object to capture frames from the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)      # Set the frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)     # Set the frame height

# Initialize variables such as a counter to keep track of frame counts and variables related to head warnings
counter = 0
head_warning = False
prev_frame = None

def check_head_movement(frame):
    global head_warning, prev_frame

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Handle the case of the first frame by storing it as the previous frame and returning from the function
    if prev_frame is None:
        prev_frame = gray
        return

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude and angle of the 2D vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold to detect motion
    motion_threshold = 5  # Reduced motion threshold
    motion_mask = magnitude > motion_threshold

    # Calculate the percentage of motion in the frame
    motion_percentage = np.mean(motion_mask)

    # Set head warning if motion percentage exceeds a threshold
    head_warning = motion_percentage > 0.05  # Reduced motion percentage threshold

    prev_frame = gray  # Update the previous frame

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            threading.Thread(target=check_head_movement, args=(frame.copy(),)).start()
        counter += 1

        if head_warning:
            cv2.putText(frame, "HEAD MOVEMENT WARNING!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Head Movement Warning", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
