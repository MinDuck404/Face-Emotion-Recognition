import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Định nghĩa các đường dẫn đến các thư mục test
test_dir = 'faces_data/test'

# Khởi tạo ImageDataGenerator cho việc chuẩn hóa dữ liệu
test_datagen = ImageDataGenerator(rescale=1./255)

# Sử dụng flow_from_directory để đọc ảnh từ thư mục test
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False  # Đảm bảo rằng các dữ liệu test không bị xáo trộn
)

# Kiểm tra số lượng mẫu trong test_generator
print(f'Số lượng mẫu trong bộ dữ liệu kiểm thử: {test_generator.samples}')

# Tải mô hình đã huấn luyện
model = load_model('emotion_recognition_model.h5')

# Dự đoán nhãn cho dữ liệu kiểm thử
steps = int(np.ceil(test_generator.samples / test_generator.batch_size))  # Chuyển đổi sang kiểu integer
y_pred = model.predict(test_generator, steps=steps, verbose=1)

# Chuyển đổi nhãn dự đoán thành nhãn lớp (số)
y_pred_classes = np.argmax(y_pred, axis=1)

# Lấy nhãn thực tế từ test_generator
y_true = test_generator.classes

# Kiểm tra lại số lượng nhãn thực tế và nhãn dự đoán
print(f'Số lượng nhãn thực tế: {len(y_true)}')
print(f'Số lượng nhãn dự đoán: {len(y_pred_classes)}')

# Tính toán Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Load lịch sử huấn luyện từ file training_history.pkl
with open('training_history.pkl', 'rb') as file:
    history = pickle.load(file)

# Hiển thị các biểu đồ trong cùng một cửa sổ
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Biểu đồ Accuracy
axs[0].plot(history['accuracy'], label='Training Accuracy')
axs[0].plot(history['val_accuracy'], label='Validation Accuracy')
axs[0].set_title('Training and Validation Accuracy')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Accuracy')
axs[0].legend()

# Biểu đồ Loss
axs[1].plot(history['loss'], label='Training Loss')
axs[1].plot(history['val_loss'], label='Validation Loss')
axs[1].set_title('Training and Validation Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].legend()

# Biểu đồ Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices)
disp.plot(cmap='Blues', ax=axs[2])
axs[2].set_title('Confusion Matrix')

# Hiển thị các biểu đồ
plt.tight_layout()
plt.show()
