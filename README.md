# Face Emotion Recognition

Face Emotion Recognition là một ứng dụng nhận diện cảm xúc từ khuôn mặt bằng mô hình học sâu. Dự án sử dụng OpenCV, TensorFlow và Tkinter để nhận diện và hiển thị cảm xúc theo thời gian thực.

## 🚀 Tính năng nổi bật
- **Nhận diện cảm xúc theo thời gian thực** từ webcam.
- **Nhận diện từ ảnh tĩnh** với giao diện chọn ảnh.
- **Sử dụng mô hình học sâu** huấn luyện trên dữ liệu khuôn mặt.
- **Hiển thị kết quả trên giao diện trực quan**.
- **Ghi log cảm xúc vào file** để theo dõi lịch sử nhận diện.

## 🎥 Demo
Xem video demo để hiểu rõ hơn cách VN-SubMaker hoạt động:  
[Video Demo VN-SubMaker](https://youtu.be/Tsa7WcBIg1E?si=LPreN4TgEYEIsc_h)

## 🖥️ Cài đặt
### 1️⃣ Yêu cầu hệ thống
- Python 3.x
- TensorFlow/Keras
- OpenCV
- Tkinter
- Pillow
- NumPy

### 2️⃣ Cài đặt thư viện cần thiết
```sh
pip install -r requirements.txt
```

### 3️⃣ Chạy chương trình
```sh
python face_emotion_recognition.py
```

## 📌 Cấu trúc dự án
```
Face-Emotion-Recognition/
│── models/                         # Chứa mô hình đã huấn luyện
│── face_emotion_recognition.py      # Chương trình chính
│── bg.jpg                           # Hình nền giao diện
│── requirements.txt                 # Danh sách thư viện cần thiết
│── saves/                           # Lưu log cảm xúc
│── README.md                        # Tài liệu hướng dẫn sử dụng
```

## 🎯 Hướng dẫn sử dụng
1. **Nhận diện từ ảnh**:
   - Chọn ảnh bằng nút "Browse".
   - Chương trình sẽ hiển thị cảm xúc nhận diện được.
2. **Nhận diện theo thời gian thực**:
   - Nhấn "Live Video Cam" để bật webcam.
   - Chương trình sẽ nhận diện cảm xúc và hiển thị kết quả trên hình ảnh trực tiếp.
3. **Lưu log cảm xúc**:
   - Khi chụp ảnh từ webcam, cảm xúc nhận diện sẽ được lưu vào `saves/emotion_log.txt`.

## 🏗️ Công nghệ sử dụng
- **OpenCV**: Nhận diện khuôn mặt bằng Haar Cascade.
- **TensorFlow/Keras**: Dự đoán cảm xúc bằng mô hình học sâu.
- **Tkinter**: Tạo giao diện đồ họa.
- **Pillow**: Xử lý ảnh với Python Imaging Library.

## 🤝 Đóng góp
Chúng tôi hoan nghênh mọi đóng góp từ cộng đồng! Nếu bạn muốn cải thiện dự án, vui lòng gửi pull request hoặc báo lỗi trên GitHub.

## 📜 Giấy phép
Face Emotion Recognition được phát hành dưới giấy phép MIT.

