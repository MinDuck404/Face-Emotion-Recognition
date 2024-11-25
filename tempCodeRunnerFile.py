ef detect_emotion_from_image(image_path):
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