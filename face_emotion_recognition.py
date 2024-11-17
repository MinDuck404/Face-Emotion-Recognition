import cv2
import numpy as np
import os
from datetime import datetime
from random import randint
from tensorflow.keras.models import load_model
from tkinter import Tk, Button, Label, filedialog, Frame, Toplevel
from PIL import Image, ImageTk, ImageDraw, ImageFont
from PIL import Image as PILImage

# Tải mô hình đã huấn luyện
model = load_model('emotion_recognition_model.h5')

# Tải Haar Cascade để nhận diện gương mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral', 'Surprise']

# Tạo thư mục "saves" nếu chưa tồn tại
if not os.path.exists("saves"):
    os.makedirs("saves")

def detect_emotion_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        face = cv2.resize(roi_gray, (48, 48))
        face = face.reshape(1, 48, 48, 1) / 255.0
        emotion = model.predict(face)
        emotion_label = emotion_labels[np.argmax(emotion)]
        confidence_score = np.max(emotion) * 100
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        original_height = img.shape[0]
        font_size = max(int(original_height / 15), 16)
        text_y_position = y - (original_height // 10)
        img = put_text_pil(img, f'{emotion_label}: {confidence_score:.2f}%', (x, max(text_y_position, 10)), font_size=font_size, color=(0, 255, 0))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = resize_image(img, 600, 400)
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk

def resize_image(image, max_width, max_height):
    width, height = image.size
    aspect_ratio = width / height
    if width > max_width or height > max_height:
        if width / max_width > height / max_height:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def put_text_pil(img, text, position, font_size=60, color=(0, 255, 0)):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("arial.ttf", font_size)
    draw.text(position, text, font=font, fill=color)
    return np.array(pil_img)

def detect_emotion_from_video():
    cap = cv2.VideoCapture(0)
    video_window = Toplevel(root)
    video_window.title("Live Camera")
    frame_cam = Frame(video_window)
    frame_cam.pack()
    video_label = Label(frame_cam)
    video_label.pack()

    global frame_with_emotion

    def capture_image():
        if frame_with_emotion is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file = "saves/emotion_log.txt"
            gray = cv2.cvtColor(frame_with_emotion, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            with open(log_file, "a") as file:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    face = cv2.resize(roi_gray, (48, 48))
                    face = face.reshape(1, 48, 48, 1) / 255.0
                    emotion = model.predict(face)
                    emotion_label = emotion_labels[np.argmax(emotion)]
                    confidence_score = np.max(emotion) * 100
                    file.write(f"{timestamp} - Emotion: {emotion_label} - Confidence: {confidence_score:.2f}%\n")
            print(f"Emotion log saved to {log_file}")

    btn_capture = Button(video_window, text="Capture", command=capture_image, bg="#FF5722", fg="white", font=("Helvetica", 14))
    btn_capture.pack()

    def update_video():
        global frame_with_emotion
        ret, frame = cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            face = cv2.resize(roi_gray, (48, 48))
            face = face.reshape(1, 48, 48, 1) / 255.0
            emotion = model.predict(face)
            emotion_label = emotion_labels[np.argmax(emotion)]
            confidence_score = np.max(emotion) * 100
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            original_height = frame.shape[0]
            font_size = max(int(original_height / 30), 16)
            text_y_position = y - (original_height // 20)
            frame = put_text_pil(frame, f'{emotion_label}: {confidence_score:.2f}%', (x, max(text_y_position, 10)), font_size=font_size, color=(0, 255, 0))
        frame_with_emotion = frame.copy()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        video_label.config(image=img_tk)
        video_label.image = img_tk
        video_label.after(10, update_video)

    update_video()
    video_window.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), video_window.destroy()))
    video_window.mainloop()

def browse_image():
    image_path = filedialog.askopenfilename()
    if image_path:
        detect_emotion_from_image(image_path)

root = Tk()
root.title("Nhận diện cảm xúc")
root.geometry("800x700")

bg_image = PILImage.open("bg.jpg")
bg_image = bg_image.resize((800, 700), PILImage.Resampling.LANCZOS)
background_image = ImageTk.PhotoImage(bg_image)
background_label = Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

frame_image = Frame(root, width=600, height=400, bg="white")
frame_image.pack(pady=(180, 0))
frame_image.pack_propagate(False)

label = Label(frame_image)
label.pack()

btn_browse = Button(root, text="Browse", command=browse_image, bg="#4CAF50", fg="white", font=("Helvetica", 14), relief="raised")
btn_browse.pack(pady=(10, 10))

btn_video = Button(root, text="Live Video Cam", command=detect_emotion_from_video, bg="#2196F3", fg="white", font=("Helvetica", 14), relief="raised")
btn_video.pack(pady=(10, 20))

root.mainloop()
