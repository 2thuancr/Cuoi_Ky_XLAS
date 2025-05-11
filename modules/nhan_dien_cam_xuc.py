import streamlit as st
import cv2
import numpy as np
import time
import joblib
from library.facial_expression_recognition.predict import predict_emotion
from library.facial_expression_recognition.facial_fer_model import FacialExpressionRecog

# Tải mô hình nhận diện khuôn mặt và mô hình nhận diện cảm xúc
face_detection_model = './model/face_detection_yunet_2023mar.onnx'
face_recognition_model = './model/svc_facial_expression_classifier.onnx'

# Các tham số của mô hình
score_threshold = 0.9
nms_threshold = 0.3
top_k = 5000

# Mã hóa các nhãn cảm xúc
mydict = ['Thuan', 'Tien']
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]

def show():
    st.title("Nhận diện cảm xúc khuôn mặt")

    # Tạo checkbox để bật/tắt nhận diện khuôn mặt
    run = st.checkbox("Nhận diện cảm xúc khuôn mặt", value=st.session_state.get("run", False))

    # Lựa chọn nguồn video (webcam hoặc video file)
    video_source = st.selectbox(
        "Video",
        ("webcam", "video"),
    )

    # Nếu video là file, cho phép upload
    video_file = None
    if video_source == "video":
        video_file = st.file_uploader("Chọn video", type=["mp4", "avi", "mov"])

    # Hiển thị khung hình video
    FRAME_WINDOW = st.image([])

    # Biến cap sẽ là nguồn video được mở
    cap = None
    if run:  # Nếu checkbox được bật
        if video_source == "webcam":
            cap = cv2.VideoCapture(0)
        elif video_source == "video" and video_file is not None:
            with open("temp_video.mp4", "wb") as f:
                f.write(video_file.read())
            cap = cv2.VideoCapture("temp_video.mp4")
        else:
            st.warning("Hãy chọn file video để tiếp tục.")
            st.stop()

        # Khởi tạo các đối tượng detector và recognizer
        detector = cv2.FaceDetectorYN.create(
            face_detection_model,
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k
        )

        # Khởi tạo mô hình nhận diện cảm xúc
        recognizer = FacialExpressionRecog(face_recognition_model)
        
        tm = cv2.TickMeter()

        # Thiết lập các tham số video
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize([frameWidth, frameHeight])

    while run:
        ret, frame = cap.read()
        if not ret:
            if video_source == "video":
                # Nếu hết video, tua lại đầu
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                st.write("Không thể đọc camera")
                break

        # Nhận diện khuôn mặt
        tm.start()
        faces = detector.detect(frame)  # faces is a tuple
        tm.stop()

        value = []
        scores = []
        if faces[1] is not None:
            for x in range(len(faces[1])):
                # Cắt khuôn mặt ra từ frame
                face_align = recognizer.alignCrop(frame, faces[1][x])

                # Trích xuất đặc trưng và dự đoán cảm xúc
                feat = recognizer.extract_softmax_feature(face_align)
                if feat is not None:
                    result = predict_emotion(recognizer, classifier=svc, label_names=mydict, image_path=face_recognition_model)
                    value.append(result)
                    scores.append(1.0)  # Để đơn giản, ta giả sử độ tin cậy luôn là 1

        # Hiển thị kết quả
        visualize(frame, faces, tm.getFPS(), value=value, scores=scores)

        # Chuyển BGR → RGB (Streamlit cần ảnh RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame, channels='RGB')
        time.sleep(0.03)  # Giới hạn tốc độ khung hình
    if cap:
        cap.release()

def visualize(input, faces, fps, thickness=2, value=None, scores=None):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            x, y, w, h = coords[0], coords[1], coords[2], coords[3]

            # Mặc định label và màu
            label = "Unknown"
            color = (200, 200, 200)

            # Nếu có đủ thông tin và confidence cao thì cập nhật label + màu
            if value and scores and idx < len(value):
                label = value[idx]
                color = colors[idx % len(colors)]

            # Hiển thị label phía trên khung
            cv2.putText(input, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(input, (x, y), (x + w, y + h), color, thickness)

            # Vẽ landmark
            cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

    # Hiển thị FPS
    cv2.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == "__main__":
    show()
