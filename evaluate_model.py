from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = 'faces_data/train'  
test_dir = 'faces_data/test' 
# Khởi tạo ImageDataGenerator cho việc chuẩn hóa và tiền xử lý dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Sử dụng flow_from_directory để đọc ảnh từ các thư mục
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical'
)

# Tải mô hình đã huấn luyện
model = load_model('emotion_recognition_model.h5')

# Đánh giá mô hình trên dữ liệu test
score = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {score[1] * 100}%')
