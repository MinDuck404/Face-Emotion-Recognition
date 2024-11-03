from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Định nghĩa các đường dẫn đến các thư mục train và test
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

# In thông tin về dữ liệu
print("Train Data Info:", train_generator.class_indices)
print("Test Data Info:", test_generator.class_indices)
