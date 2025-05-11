import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms


# Nhãn cảm xúc (phải đúng thứ tự)
# class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
class_names = ['Happy', 'Sleepy', 'Surprise']

# Load ONNX model
session = ort.InferenceSession("./model/emotion_cnn.onnx", providers=['CPUExecutionProvider'])

# Hàm tiền xử lý
def preprocess_image(img: Image.Image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    tensor = transform(img).unsqueeze(0).numpy()  # shape: (1, 1, 48, 48)
    return tensor

# Hàm dự đoán
def predict_emotion(img: Image.Image):
    input_tensor = preprocess_image(img)
    outputs = session.run(None, {"input": input_tensor})
    scores = outputs[0][0]
    predicted = np.argmax(scores)
    confidence = float(np.exp(scores[predicted]) / np.sum(np.exp(scores)))
    return class_names[predicted], confidence

def show():
    # Giao diện
    st.title("😃 Nhận dạng cảm xúc")
    st.markdown("<div style='background-color: lightgrey;'>Upload a face image (48x48 grayscale or larger) or use your webcam. <div>",  unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload ảnh khuôn mặt", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Ảnh đã tải lên", width=200)
        emotion, conf = predict_emotion(img)
        st.toast(f"**Emotion:** {emotion} ({conf*100:.2f}%)")

        # Hiển thị label
        label = f"Emotion: {emotion} ({conf*100:.2f}%)"
        st.markdown(f"<h3 style='text-align: center; color: #FF6347; background-color: lightgrey;'>{label}</h3>", unsafe_allow_html=True)

    # # Webcam
    # if st.button("📷 Dùng webcam"):
    #     st.warning("⚠️ Vui lòng chạy file local vì Streamlit cloud không hỗ trợ webcam.")
