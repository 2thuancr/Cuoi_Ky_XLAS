import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np


def show():
    # Load mô hình
    model = joblib.load('./model/hand_gesture_model.pkl')
    labels = model.classes_

    # Khởi tạo MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    st.markdown("<div style='text-align: center; font-size: 24px; font-weight: 600;'>🤚 Nhận diện cử chỉ tay bằng MediaPipe + ML</div>", unsafe_allow_html=True)
    st.write("Sử dụng webcam để nhận diện cử chỉ tay theo thời gian thực.")

    run = st.checkbox('Bắt đầu nhận diện')
    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Không lấy được hình từ webcam!")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Trích xuất (x, y, z)
                landmark = []
                for lm in hand_landmarks.landmark:
                    landmark.extend([lm.x, lm.y, lm.z])

                # if len(landmark) == 63:
                #     prediction = model.predict([landmark])[0]
                #     cv2.putText(frame, f'Gesture: {prediction}', (10, 50),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if len(landmark) == 63:
                    probs = model.predict_proba([landmark])[0]
                    max_prob = np.max(probs)
                    predicted_label = model.classes_[np.argmax(probs)]

                    if max_prob > 0.7:
                        cv2.putText(frame, f'Gesture: {predicted_label} ({max_prob:.2f})', (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị lên Streamlit
        frame_window.image(frame, channels="BGR")

    cap.release()