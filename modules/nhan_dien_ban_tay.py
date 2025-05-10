import cv2
import streamlit as st
import mediapipe as mp
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def show():
    # Load mô hình
    model = joblib.load('./model/hand_gesture_model.pkl')
    labels = model.classes_

    # Tạo từ điển ánh xạ nhãn tiếng Anh sang tiếng Việt
    label_mapping = {
        'thumbs_down': 'Ngón tay cái xuống',
        'thumbs_up': 'Ngón tay cái lên',
        'peace': 'Dấu hòa bình',
        'rock': 'Dấu đá',
        'open_hand': 'Bàn tay mở',
        'fist': 'Nắm tay',
        'pointing': 'Chỉ tay',
        # Thêm các nhãn khác tùy theo mô hình của bạn
    }

    # Khởi tạo MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    st.markdown("<div style='text-align: center; font-size: 24px; font-weight: 600;'>🤚 NHẬN DIỆN BÀN TAY BẰNG MEDIAPIPE + ML</div>", unsafe_allow_html=True)
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

                if len(landmark) == 63:
                    probs = model.predict_proba([landmark])[0]
                    max_prob = np.max(probs)
                    predicted_label = model.classes_[np.argmax(probs)]

                    # Chuyển nhãn sang tiếng Việt
                    predicted_label_vietnamese = label_mapping.get(predicted_label, predicted_label)

                    if max_prob > 0.7:
                        # Convert frame to Image (Pillow)
                        img_pil = Image.fromarray(frame)
                        draw = ImageDraw.Draw(img_pil)

                        # Tải font hỗ trợ tiếng Việt (chẳng hạn font "Arial" hoặc bạn có thể thay bằng font khác hỗ trợ tiếng Việt)
                        font = ImageFont.truetype("arial.ttf", 30)

                        # Vẽ text với tiếng Việt
                        draw.text((10, 50), f'Cử chỉ: {predicted_label_vietnamese} ({max_prob:.2f})', font=font, fill=(0, 255, 0))

                        # Convert Image back to OpenCV format
                        frame = np.array(img_pil)

        # Hiển thị lên Streamlit
        frame_window.image(frame, channels="BGR")

    cap.release()
