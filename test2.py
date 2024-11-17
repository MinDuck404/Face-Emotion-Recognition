import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

# Load the model
model = load_model('cnn_emotion_detection.h5')

# Emotion labels
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# Function to predict emotion
def predict_emotion(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    return emotion_labels[emotion_index]  # Return the emotion name

# Function to handle the Browse button
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        emotion = predict_emotion(file_path)
        messagebox.showinfo("Prediction", f'The predicted emotion is: {emotion}')

# Create the main application window
root = tk.Tk()
root.title("Emotion Prediction")

# Create a Browse button
browse_button = tk.Button(root, text="Browse", command=browse_image)
browse_button.pack(pady=20)

# Run the application
root.mainloop()
