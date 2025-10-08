import threading
import cv2
import numpy as np
from deepface import DeepFace
import tkinter as tk
from tkinter import messagebox

# Initialize video capture object to capture frames from the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

# Set the frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables such as a counter to keep track of frame counts and variables related to warnings
counter = 0
face_match = False
eye_warning = False
lip_warning = False
head_warning = False
prev_frame_gray = None
prev_pts = None
prev_frame = None
total_eye_warnings = 0
total_lip_warnings = 0
total_head_warnings = 0
lip_warning_popup_shown = False

reference_img = cv2.imread("ref1.jpg")

def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        pass

def check_eyes(frame):
    global eye_warning, prev_frame_gray, prev_pts, total_eye_warnings
    print("Calculating eye movement...")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    if len(eyes) > 0:
        for (x, y, w, h) in eyes:
            print("Eye coordinates (x, y, w, h):", x, y, w, h)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            if prev_frame_gray is not None and prev_pts is not None:
                pts, st, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, gray, prev_pts, None)
                good_new = pts[st == 1]
                good_old = prev_pts[st == 1]
                mean_displacement = np.mean(np.sqrt(np.sum(np.square(good_new - good_old), axis=1)))
                print("Mean displacement for eyes:", mean_displacement)
                if mean_displacement > 5:
                    eye_warning = True
                    total_eye_warnings += 1
                    break
            prev_frame_gray = gray.copy()
            prev_pts = cv2.goodFeaturesToTrack(roi_gray, maxCorners=20, qualityLevel=0.01, minDistance=10)
    else:
        eye_warning = True
        total_eye_warnings += 1

def check_lips(frame):
    global lip_warning, prev_frame_gray, total_lip_warnings, lip_warning_popup_shown
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame_gray is None:
        prev_frame_gray = gray
        return

    frame_diff = cv2.absdiff(prev_frame_gray, gray)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        lip_warning = True
        total_lip_warnings += 1
        break
    else:
        lip_warning = False

    if total_lip_warnings == 6 and not lip_warning_popup_shown:
        show_warning_popup("lip", 2)
        lip_warning_popup_shown = True

    prev_frame_gray = gray

def check_head_movement(frame):
    global head_warning, prev_frame, total_head_warnings
    print("Calculating head movement...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        prev_frame = gray
        return

    flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_threshold = 5
    motion_mask = magnitude > motion_threshold
    motion_percentage = np.mean(motion_mask)
    print("Motion percentage for head movement:", motion_percentage)
    head_warning = motion_percentage > 0.05
    if head_warning:
        total_head_warnings += 1

    prev_frame = gray

def show_warning_popup(warning_type, remaining_warnings):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Warning", f"Only {remaining_warnings} {warning_type} warning(s) remaining!")

def close_window():
    root.destroy()

while True:
    ret, frame = cap.read()

    if ret:
        print("Processing frame:", counter)
        if counter % 30 == 0:
            threading.Thread(target=check_face, args=(frame.copy(),)).start()
            threading.Thread(target=check_eyes, args=(frame.copy(),)).start()
            threading.Thread(target=check_lips, args=(frame.copy(),)).start()
            threading.Thread(target=check_head_movement, args=(frame.copy(),)).start()
        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NOT MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        if eye_warning:
            cv2.putText(frame, "EYE WARNING!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Changed color to green

        if lip_warning:
            cv2.putText(frame, "LIP WARNING!", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Changed color to green

        if head_warning:
            cv2.putText(frame, "HEAD MOVEMENT WARNING!", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Changed color to green

        cv2.imshow("video", frame)

        if total_eye_warnings == 10 and total_lip_warnings == 8 and total_head_warnings == 5:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("All Warnings Performed", "All warnings have been performed.")
            close_window()
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(0)
            break

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()