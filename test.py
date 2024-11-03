import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk

# Tải mô hình
model = tf.keras.models.load_model('model.h5')

# Danh sách các lớp cảm xúc
emotion_labels = ['surprise', 'sad', 'neutral', 'happy', 'fear', 'disgust', 'angry']

# Hàm nhận diện cảm xúc từ hình ảnh
def recognize_emotion(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img)
    emotion_index = np.argmax(predictions)
    return emotion_labels[emotion_index]

# Hàm để mở hộp thoại và chọn ảnh
def browse_image():
    img_path = filedialog.askopenfilename(title="Chọn hình ảnh", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if img_path:
        img = Image.open(img_path)
        img.thumbnail((250, 250))  # Thay đổi kích thước hình ảnh để hiển thị
        img_display = ImageTk.PhotoImage(img)

        # Cập nhật hình ảnh gốc
        original_image_label.config(image=img_display)
        original_image_label.image = img_display  # Giữ tham chiếu đến hình ảnh

        # Nhận diện cảm xúc
        emotion = recognize_emotion(img_path)
        result_label.config(text=f"Cảm xúc nhận diện: {emotion}")

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Nhận diện cảm xúc từ hình ảnh")

# Nút Browse
browse_button = tk.Button(root, text="Chọn hình ảnh", command=browse_image)
browse_button.pack()

# Nhãn để hiển thị hình ảnh gốc
original_image_label = tk.Label(root)
original_image_label.pack()

# Nhãn để hiển thị kết quả
result_label = tk.Label(root, text="")
result_label.pack()

# Chạy ứng dụng
root.mainloop()
