import os
import streamlit as st
import cv2
import numpy as np
import time
import argparse
import joblib
from library.face_detection.get_face import get_face_recognition_model

def show():
    
    st.title("Thu thập khuôn mặt")

    # Thêm 2 button: Get Faces và Train, khi nhấn vào sẽ gọi hàm get_faces và train
    col1, col2 = st.columns(2)
    with col1:
        text_label = st.text_input("Tên", key="text_label")

    with col1:
        btn_run = st.button("Lấy khuôn mặt")

    if btn_run:
        get_faces()  # Gọi hàm nhận diện khuôn mặt


def get_faces():

    # Lấy tên từ input text_label
    text_label = st.session_state.get("text_label", "")
    if text_label == "":
        st.toast("Vui lòng nhập tên trước khi thu thập khuôn mặt.")
        return
    
    # Tạo thư mục để lưu ảnh nếu chưa tồn tại
    if not os.path.exists("./images/face_detection/" + text_label):
        os.makedirs("./images/face_detection/" + text_label)

    get_face_recognition_model(label=text_label)  # Gọi hàm nhận diện khuôn mặt
    st.session_state.run = True
