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

    # Display the frame
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

'''

Of course! Let's break down the example of a Convolutional Neural Network (CNN) in a simpler way:

1. **What's a CNN?**
   - A CNN is a type of artificial intelligence model, specifically designed for tasks involving images, like recognizing objects in pictures.

2. **The Model Architecture**:
   - Imagine the CNN as a series of layers, each performing a specific task to understand the image better.

3. **Convolutional Layers**:
   - Think of these layers as filters that look for specific patterns or features in the image, like edges or textures.
   - The numbers like `32` and `64` represent how many of these filters the model has learned to use.

4. **Pooling Layers**:
   - After each convolutional layer, we use pooling layers to shrink down the information, making it easier to process. It's like zooming out to see the bigger picture.
   - Max-pooling picks the maximum value from a group of values, simplifying the information.

5. **Flatten Layer**:
   - This layer simply takes the output from the convolutional layers and turns it into a flat list. It's like spreading out all the information to prepare it for the next step.

6. **Fully Connected Layers**:
   - These layers work like traditional neural network layers, where each neuron is connected to every neuron in the previous layer.
   - They learn to combine all the patterns and features detected by the convolutional layers to make predictions about what's in the image.

7. **Activation Functions**:
   - These are mathematical functions that decide whether a neuron "fires" or not based on its input. ReLU (Rectified Linear Unit) is one common activation function used in CNNs.

8. **Output Layer**:
   - The last layer of the network, it gives the final prediction. In this case, we have 10 neurons because we're trying to recognize digits (0 to 9). The softmax function here gives us the probabilities of each digit.

9. **Compilation**:
   - This step configures how the model learns from the data. We specify the optimizer (how the model updates itself), the loss function (how the model measures its performance), and any additional metrics we want to track.

10. **Summary**:
   - Finally, we summarize the model to see its architecture and the number of parameters it has learned.

That's the basic breakdown of a CNN! It's like a smart stack of filters and layers that learns to understand images by detecting patterns and combining them to make predictions.

'''