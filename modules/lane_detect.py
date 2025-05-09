from library import lane
import streamlit as st
import cv2
import numpy as np



def show():
    st.markdown("<div style='text-align: center; font-size: 24px; font-weight: 600;'>LANE DETECTION</div><br><br>", unsafe_allow_html=True)

    # Trạng thái session
    if 'running' not in st.session_state:
        st.session_state.running = False

    # Nút điều khiển
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start"):
            st.session_state.running = True
    with col2:
        if st.button("⏹ Stop"):
            st.session_state.running = False
    frame_window = st.image([])

    cap = cv2.VideoCapture("./images/lane/lane.mp4")

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning(f"Không lấy được hình từ video. Lỗi: {ret}")
            break
        _canny = lane.canny(frame)
        roi = lane.region_of_interest(_canny)
        lines = lane.detect_lines(roi)
        averaged = lane.average_slope_intercept(frame, lines)
        line_img = lane.display_lines(frame, averaged)
        combo = lane.combine_images(frame, line_img)

        frame_window.image(combo, channels="BGR")

    cap.release()
