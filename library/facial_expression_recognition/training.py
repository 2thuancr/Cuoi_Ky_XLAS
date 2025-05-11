import os
import cv2
import numpy as np
import joblib
import json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==== Load ONNX model ====
from facial_fer_model import FacialExpressionRecog  # from OpenCV Zoo

onnx_path = "../facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx"
recognizer = FacialExpressionRecog(onnx_path)

# ==== Trích xuất đặc trưng (7 lớp softmax) ====
def extract_softmax_feature(img_path, recognizer):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Lỗi đọc ảnh:", img_path)
        return None

    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32)

    result = recognizer.infer(img)

    return result  # output softmax (7 lớp)

# ==== Gộp tập dữ liệu đặc trưng (bao gồm lớp mới) ====
def build_dataset(images_dir, recognizer):
    X = []
    y = []
    label_names = []

    # Thêm các lớp cảm xúc gốc từ 7 lớp
    for idx, class_name in enumerate(sorted(os.listdir(images_dir))):
        class_dir = os.path.join(images_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        label_names.append(class_name)

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue
            path = os.path.join(class_dir, fname)
            feat = extract_softmax_feature(path, recognizer)
            if feat is not None:
                X.append(feat)
                y.append(idx)

    # Thêm các lớp cảm xúc mới (Happy và Sleepy)
    new_classes = ['Happy', 'Sleepy']
    for idx, class_name in enumerate(new_classes, start=len(label_names)):
        label_names.append(class_name)
        class_dir = os.path.join(images_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue
            path = os.path.join(class_dir, fname)
            feat = extract_softmax_feature(path, recognizer)
            if feat is not None:
                X.append(feat)
                y.append(idx)

    return np.array(X), np.array(y), label_names

# ==== Huấn luyện classifier 9 lớp ====
def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)

    model = SVC(probability=True, kernel='linear', class_weight='balanced')
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Accuracy: {acc:.4f}")
    return model

# ==== Lưu model và nhãn ====
def save_model(model, label_names, out_dir='result'):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, 'emotion_classifier.pkl'))
    with open(os.path.join(out_dir, 'label_names.json'), 'w') as f:
        json.dump(label_names, f)
    print("✅ Model và nhãn đã được lưu.")

# ==== Inference thử ====
def predict_emotion(image_path, recognizer, classifier, label_names):
    feat = extract_softmax_feature(image_path, recognizer)
    if feat is None:
        return "Unknown"
    pred = classifier.predict([feat])[0]
    return label_names[pred]

# ==== MAIN ====
if __name__ == "__main__":
    images_dir = "images"  # Thư mục chứa các thư mục con: Happy/, Sleepy/, ...
    print("📁 Loading dataset...")
    X, y, label_names = build_dataset(images_dir, recognizer)

    print("🎓 Training classifier...")
    model = train_classifier(X, y)

    print("💾 Saving model...")
    save_model(model, label_names)

    # Inference thử
    test_img = "sleep.jpg"  # Thay bằng ảnh thử của bạn
    if os.path.exists(test_img):
        pred = predict_emotion(test_img, recognizer, model, label_names)
        print(f"🧠 Dự đoán cảm xúc cho ảnh '{test_img}': {pred}")
    else:
        print("📸 File test không tồn tại — bỏ qua inference.")
