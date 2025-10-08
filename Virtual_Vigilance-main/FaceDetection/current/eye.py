import threading
import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
eye_warning = False
prev_frame_gray = None
prev_pts = None


def check_eyes(frame):
    global eye_warning, prev_frame_gray, prev_pts
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#converts to gray scale


    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))#in gray converted by this 3 factors

    if len(eyes) > 0:
        for (x, y, w, h) in eyes:#x,y and w,h is width and height
            roi_gray = gray[y:y + h, x:x + w]# detected eye from the grayscale frame.
            roi_color = frame[y:y + h, x:x + w]#detected eye from the original color frame.
            if prev_frame_gray is not None and prev_pts is not None:
                # Calculate optical flow
                pts, st, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, gray, prev_pts, None)#optical is calculated using prev and current gray scale

                # Filter points based on status
                good_new = pts[st == 1]
                good_old = prev_pts[st == 1]

                # Calculate the mean displacement of points using lucas-kanade
                mean_displacement = np.mean(np.sqrt(np.sum(np.square(good_new - good_old), axis=1)))#mean diff calculated form new and old

                if mean_displacement > 5:  # Adjust this threshold as needed
                    eye_warning = True
                    break
                else:
                    eye_warning = False
            prev_frame_gray = gray.copy()
            prev_pts = cv2.goodFeaturesToTrack(roi_gray, maxCorners=20, qualityLevel=0.01, minDistance=10)
    else:
        eye_warning = True


while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:#threshold
            threading.Thread(target=check_eyes, args=(frame.copy(),)).start()
        counter += 1

        if eye_warning:
            cv2.putText(frame, "EYE WARNING!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "NO EYE WARNING", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
