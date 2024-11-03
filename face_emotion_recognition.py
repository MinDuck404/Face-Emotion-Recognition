import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, Button, Label, filedialog, Frame
from PIL import Image, ImageTk, ImageDraw, ImageFont
from PIL import Image as PILImage 

# Tải mô hình đã huấn luyện
model = load_model('emotion_recognition_model.h5')

# Tải Haar Cascade để nhận diện gương mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral', 'Surprise']

# Hàm thay đổi kích thước ảnh để vừa với giao diện
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

# Hàm nhận diện cảm xúc từ ảnh
def detect_emotion_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện gương mặt
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        face = cv2.resize(roi_gray, (48, 48))
        face = face.reshape(1, 48, 48, 1) / 255.0

        # Dự đoán cảm xúc
        emotion = model.predict(face)
        emotion_label = emotion_labels[np.argmax(emotion)]
        confidence_score = np.max(emotion) * 100  

        # Vẽ khung và hiển thị cảm xúc với độ chính xác
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Tính toán kích thước phông chữ
        original_height = img.shape[0]
        font_size = max(int(original_height / 15), 16)  

        # Tính toán vị trí văn bản dựa trên kích thước ảnh
        text_y_position = y - (original_height // 10)  
        img = put_text_pil(img, f'{emotion_label}: {confidence_score:.2f}%', (x, max(text_y_position, 10)), font_size=font_size, color=(0, 255, 0))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)


    img = img.resize((int(img.width * (400 / img.height)), 400), Image.Resampling.LANCZOS)
    
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk

# Hàm nhận diện cảm xúc từ webcam
def detect_emotion_from_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện gương mặt
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            face = cv2.resize(roi_gray, (48, 48))
            face = face.reshape(1, 48, 48, 1) / 255.0

            # Dự đoán cảm xúc
            emotion = model.predict(face)
            emotion_label = emotion_labels[np.argmax(emotion)]
            confidence_score = np.max(emotion) * 100  # Chuyển đổi thành %

            # Vẽ khung và hiển thị cảm xúc với độ chính xác
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Tính toán kích thước phông chữ
            original_height = frame.shape[0]
            font_size = max(int(original_height / 30), 16)  # Đặt kích thước tối thiểu cho phông chữ

            # Tính toán vị trí văn bản
            text_y_position = y - (original_height // 20)  # Giảm khoảng cách cho ảnh nhỏ hơn
            frame = put_text_pil(frame, f'{emotion_label}: {confidence_score:.2f}%', (x, max(text_y_position, 10)), font_size=font_size, color=(0, 255, 0))

        cv2.imshow('live camera', frame)

        # Dừng lại khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm chọn file ảnh
def browse_image():
    image_path = filedialog.askopenfilename()
    if image_path:
        detect_emotion_from_image(image_path)

# Xây dựng giao diện bằng Tkinter
root = Tk()
root.title("Nhận diện cảm xúc")
root.geometry("800x700")

bg_image = PILImage.open("bg.jpg")  
bg_image = bg_image.resize((800, 700), PILImage.Resampling.LANCZOS) 
background_image = ImageTk.PhotoImage(bg_image)  
background_label = Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

# Tạo frame để chứa ảnh
frame_image = Frame(root, width=600, height=400, bg="white")
frame_image.pack(pady=(180, 0))  # Thay đổi padding xuống dưới
frame_image.pack_propagate(False)  

# Thêm label chứa ảnh vào frame
label = Label(frame_image)
label.pack()

# Nút chọn file ảnh
btn_browse = Button(root, text="Browse", command=browse_image, bg="#4CAF50", fg="white", font=("Helvetica", 14), relief="raised")
btn_browse.pack(pady=(10, 10))

# Nút mở live camera
btn_video = Button(root, text="Live Video Cam", command=detect_emotion_from_video, bg="#2196F3", fg="white", font=("Helvetica", 14), relief="raised")
btn_video.pack(pady=(10, 20))

# Khởi chạy giao diện
root.mainloop()
