import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

def show():
    st.markdown("<div style='text-align: center; font-size: 24px; font-weight: 600;'>🖐️ Nhận diện ngôn ngữ ký hiệu + ghép câu</div>", unsafe_allow_html=True)
    st.write("Sử dụng webcam để nhận diện ký hiệu tay và ghép thành câu.")

    # Load mô hình và encoder
    model = joblib.load("./model/model.pkl")
    label_encoder = joblib.load("./model/labels.pkl")

    # MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    run = st.checkbox("Bắt đầu nhận diện")
    frame_window = st.empty()  # Khởi tạo vùng trống để cập nhật hình ảnh

    # Biến trạng thái
    sentence = ""
    prev_label = ""
    frames_same = 0
    threshold_frames = 30
    confidence_threshold = 0.7

    cap = None
    guide_img = None

    if run:
        cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Không lấy được hình từ webcam!")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        label = None
        landmarks = []

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                probs = model.predict_proba([landmarks])[0]
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]
                label = label_encoder.inverse_transform([pred_idx])[0]

                cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if confidence >= confidence_threshold else (0, 0, 255), 2)

                if label == prev_label and confidence >= confidence_threshold:
                    frames_same += 1
                else:
                    frames_same = 0
                prev_label = label

                if frames_same >= threshold_frames:
                    if label == "SPACE":
                        sentence += "_"
                    elif label == "DELETE":
                        sentence = sentence[:-1]
                    else:
                        sentence += label
                    frames_same = 0

        # Ảnh hướng dẫn
        if guide_img is None:
            if os.path.exists("./model/guide.png"):
                guide_original = cv2.imread("./model/guide.png")
                scale = w / guide_original.shape[1]
                new_h = int(guide_original.shape[0] * scale)
                guide_img = cv2.resize(guide_original, (w, new_h))
            else:
                guide_img = np.ones((240, w, 3), dtype=np.uint8) * 255

        guide_panel = guide_img.copy()

        # Ghép ảnh camera và hướng dẫn
        frame = cv2.resize(frame, (w, h))
        final_img = np.vstack((frame, guide_panel))

        # Thanh trắng chứa câu
        cv2.rectangle(final_img, (0, 0), (final_img.shape[1], 40), (255, 255, 255), -1)
        cv2.putText(final_img, f"Sentence: {sentence}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Resize nếu ảnh quá to
        max_width = 1280
        final_h, final_w = final_img.shape[:2]
        if final_w > max_width:
            scale = max_width / final_w
            final_img = cv2.resize(final_img, (int(final_w * scale), int(final_h * scale)))

        # Cập nhật ảnh trong Streamlit
        frame_window.image(final_img, channels="BGR", use_container_width=True)

    if cap:
        cap.release()
