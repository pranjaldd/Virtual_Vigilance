import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading

# Initialize video capture object to capture frames from the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

# Set the frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Global variable to track whether webcam should start
webcam_started = False

# Function to start webcam functionality
def start_webcam():
    global webcam_started
    while not webcam_started:  # Wait until webcam_started becomes True
        continue
    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imshow("video", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to start exam and initiate webcam
def start_exam():
    global webcam_started
    webcam_started = True  # Set webcam_started to True, allowing webcam to start
    instruction_window.destroy()  # Close the instruction window
    threading.Thread(target=start_webcam).start()  # Start the webcam functionality in a separate thread

# Function to create instruction window
def instruction_window():
    global instruction_window
    instruction_window = tk.Toplevel()
    instruction_window.title("Instructions")
    set_background(instruction_window, "bgimage.jpeg")

    instructions_text = """
    Instructions:
    1. Read all questions carefully before answering.
    2. You have 60 minutes to complete the exam.
    3. Click the 'Start Exam' button to begin.
    """

    instructions_label = tk.Label(instruction_window, text=instructions_text, justify=tk.LEFT, padx=20, pady=20)
    instructions_label.pack()

    start_button = tk.Button(instruction_window, text="Start Exam", command=start_exam)
    start_button.pack(pady=10)

# Function to set background image for windows
def set_background(window, image_path):
    background_image = Image.open(image_path)
    background_photo = ImageTk.PhotoImage(background_image)
    background_label = tk.Label(window, image=background_photo)
    background_label.image = background_photo
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create the main application window
screen = tk.Tk()
screen.title("Login")
screen.geometry("1000x650")

# Labels
tk.Label(screen, text="Virtual Vigilance", font="impact 21 bold").place(relx=0.5, rely=0.2, anchor=tk.CENTER)
tk.Label(screen, text="Username", font="impact 15").place(relx=0.3, rely=0.4, anchor=tk.CENTER)
tk.Label(screen, text="Password", font="impact 15").place(relx=0.3, rely=0.5, anchor=tk.CENTER)

# Entry fields
entry1 = tk.Entry(screen, bd=4)
entry1.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
entry2 = tk.Entry(screen, bd=4, show="*")
entry2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Buttons
tk.Button(screen, text="Login", command=login).place(relx=0.4, rely=0.6, anchor=tk.CENTER)
tk.Button(screen, text="Sign Up", command=signup).place(relx=0.6, rely=0.6, anchor=tk.CENTER)
tk.Button(screen, text="Forgot Password", command=forgot_password).place(relx=0.5, rely=0.7, anchor=tk.CENTER)

# Start the GUI
screen.mainloop()
