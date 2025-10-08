import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Function to create CNN model
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Define image dimensions
IMG_WIDTH, IMG_HEIGHT = 100, 100
input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)  # Grayscale image

# Create the model
model = create_model(input_shape)
model.summary()  # Print model summary


# Function to preprocess image
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    normalized = resized / 255.0  # Normalize pixel values to [0, 1]
    return normalized.reshape((-1, IMG_WIDTH, IMG_HEIGHT, 1))  # Add batch dimension


# Capture webcam images
cap = cv2.VideoCapture(0)  # 0 is the default camera

lip_movement_detected = False  # Flag to track lip movement detection

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)

    # Perform prediction with the model
    prediction = model.predict(preprocessed_frame)

    # Determine the label based on prediction
    label = "Moving" if prediction > 0.5 else "Still"
    color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)

    # Display the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # If lip movement is detected, set the flag
    if prediction > 0.5:
        lip_movement_detected = True
    else:
        lip_movement_detected = False

    # If lip movement is detected, display a warning
    if lip_movement_detected:
        cv2.putText(frame, "Warning: Lip Movement Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
