import os
import cv2
import numpy as np
import joblib
import json
from .facial_fer_model import FacialExpressionRecog

# ==== Hàm trích xuất đặc trưng từ ảnh ====
def extract_softmax_feature(img_path, recognizer):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Lỗi đọc ảnh:", img_path)
        return None

    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32)

    result = recognizer.infer(img)  # Trả về vector softmax
    return result

# ==== Hàm dự đoán cảm xúc ====
def predict_emotion(recognizer, classifier, label_names, image_path):
    feat = extract_softmax_feature(image_path, recognizer)
    if feat is None:
        return "Unknown"
    pred_idx = classifier.predict([feat])[0]
    return label_names[pred_idx]

# ==== Chạy thử ====
if __name__ == "__main__":
    test_img = "test.jpg"  # Đường dẫn ảnh cần dự đoán

    if os.path.exists(test_img):
        result = predict_emotion(test_img)
        print(f"🧠 Dự đoán cảm xúc cho ảnh '{test_img}': {result}")
    else:
        print("❌ File ảnh không tồn tại.")
