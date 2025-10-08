import threading
import cv2
import numpy as np

#video capture object to capture frames from the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)      #set the frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#initialize variables such as a counter to keep track of frame counts and variables related to lip warnings
counter = 0
lip_warning = False
prev_frame_gray = None

def check_lips(frame):
    global lip_warning, prev_frame_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #frame to grayscale
    gray = cv2.GaussianBlur(gray, (21, 21), 0)#reduce noise

#handle the case of the first frame by storing it as the previous frame and returning from the function
    if prev_frame_gray is None:
        prev_frame_gray = gray
        return

    frame_diff = cv2.absdiff(prev_frame_gray, gray) #abs diff
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)   # threshold to the frame difference to create a binary image
    thresh = cv2.dilate(thresh, None, iterations=2) #perform dilation on the thresholded image to fill in gaps and make the detected regions more understanable

#find contours(change in the frame) in the thresholded image to identify significant regions of motion
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#iterate through the contours and check if any contour represents any lip movement
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        lip_warning = True
        break
    else:
        lip_warning = False

    prev_frame_gray = gray  #update the previous frame

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            threading.Thread(target=check_lips, args=(frame.copy(),)).start()
        counter += 1

        if lip_warning:
            cv2.putText(frame, "LIP WARNING!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Lip Warning", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
