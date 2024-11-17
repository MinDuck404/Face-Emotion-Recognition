import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
import keras

# Đường dẫn tới thư mục chứa model
MODEL_DIR = 'my_model'  # Thay đổi đường dẫn cho phù hợp

# Tải model bằng TFSMLayer
model = keras.layers.TFSMLayer(MODEL_DIR, call_endpoint='serving_default')

# Kích thước của hình ảnh đầu vào
IMG_HEIGHT, IMG_WIDTH = 48, 48  # Thay đổi kích thước cho phù hợp với mô hình của bạn

# Danh sách cảm xúc
emotions = ['😡', '🤢', '😱', '😊', '😐', '😔', '😲']  # Thay số thành chữ

def preprocess_image(image_path):
    # Đọc và tiền xử lý hình ảnh
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))  # Thay đổi kích thước hình ảnh
    img = img.astype('float32') / 255.0  # Chia cho 255 để chuẩn hóa
    img = np.expand_dims(img, axis=0)  # Thêm chiều batch
    return img

def predict():
    # Chọn file hình ảnh
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:
        img = preprocess_image(file_path)
        preds = model(img)
        
        preds = preds['dense']  # Lấy tensor dự đoán từ dictionary
        class_index = np.argmax(preds)  # Tìm chỉ số của lớp có xác suất cao nhất

        # Hiển thị kết quả dự đoán
        emotion = emotions[class_index]  # Lấy cảm xúc từ danh sách
        messagebox.showinfo("Dự đoán", f"Kết quả dự đoán: {emotion}")

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Dự đoán Hình ảnh với Model")

# Tạo nút Browse
browse_button = tk.Button(root, text="Browse", command=predict)
browse_button.pack(pady=20)

# Chạy vòng lặp chính
root.mainloop()
