import streamlit as st
import cv2
import os

# Hàm lấy và lưu dữ liệu khuôn mặt
def get_face_recognition_model(label):
    cap = cv2.VideoCapture(0)
    st.info(f"Đang lấy dữ liệu khuôn mặt cho: {label}")

    count = 0
    max_images = 50
    FRAME_WINDOW = st.image([])

    # Tạo thư mục theo tên label
    folder = os.path.join("dataset", label)
    os.makedirs(folder, exist_ok=True)

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            st.warning("Không thể mở webcam.")
            break

        # Hiển thị khung hình
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Hình {count+1}/{max_images}", use_column_width=True)

        # Lưu ảnh
        img_path = os.path.join(folder, f"{count}.jpg")
        cv2.imwrite(img_path, frame)

        count += 1
        cv2.waitKey(100)  # đợi 100ms mỗi khung hình

    cap.release()
    st.success(f"Đã lưu {count} ảnh khuôn mặt cho '{label}'")


# Hàm chính hiển thị giao diện
def show():
    st.title("Hệ thống Nhận diện khuôn mặt")

    # Giao diện nút chức năng
    col1, col2 = st.columns(2)
    with col1:
        btn_get_faces = st.button("Lấy khuôn mặt")
    with col2:
        btn_train = st.button("Huấn luyện")

    # Xử lý khi nhấn các nút chức năng
    if btn_get_faces:
        st.session_state["mode"] = "get_faces"
        st.session_state["label"] = ""
        st.session_state["started"] = False

    if btn_train:
        st.session_state["mode"] = "train"
        st.session_state["label"] = ""
        st.session_state["started"] = False

    # Nếu đang ở chế độ lấy khuôn mặt
    if st.session_state.get("mode") == "get_faces":
        label = st.text_input("Nhập tên để gán nhãn khuôn mặt", key="input_label")
        if label:
            if st.button("Bắt đầu"):
                st.session_state["started"] = True
                st.session_state["label"] = label

    # Khi nhấn "Bắt đầu"
    if st.session_state.get("started", False):
        label = st.session_state.get("label", "")
        get_face_recognition_model(label)


# Gọi hàm giao diện chính
if __name__ == "__main__":
    show()
