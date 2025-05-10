import streamlit as st
import cv2
import numpy as np
import time
import argparse
import joblib
from library.face_detection.get_face import get_face_recognition_model
from library.face_detection.training import training_model

def show():
    
    st.title("Nhận diện khuôn mặt")

    # Thêm 2 button: Get Faces và Train, khi nhấn vào sẽ gọi hàm get_faces và train
    col1, col2 = st.columns(2)
    with col1:
        btn_get_faces = st.button("Lấy khuôn mặt")
    with col2:
        btn_train = st.button("Huấn luyện")

    # Cập nhật trạng thái khi nhấn nút Get Faces
    if btn_get_faces:
        st.session_state["get_faces"] = True
        st.session_state["train"] = False
        st.session_state["recognize"] = False
        st.session_state["video_source"] = None
        st.session_state["video_file"] = None
        st.session_state["run"] = False

    # Cập nhật trạng thái khi nhấn nút Train
    if btn_train:
        st.session_state["get_faces"] = False
        st.session_state["train"] = True
        st.session_state["recognize"] = False
        st.session_state["video_source"] = None
        st.session_state["video_file"] = None
        st.session_state["run"] = False

    # Tạo checkbox để bật/tắt nhận diện khuôn mặt
    run = st.checkbox("Nhận diện khuôn mặt", value=st.session_state.get("run", False))

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

    # Nếu đã nhấn nút Get Faces, cập nhật trạng thái
    if st.session_state.get("get_faces", False):
        get_face_recognition_model()  # Gọi hàm nhận diện khuôn mặt

    # Kiểm tra xem người dùng có nhấn nút "Huấn luyện"
    if st.session_state.get("train", False):
        train_model()  # Gọi hàm huấn luyện mô hình khi "train" = True

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
            args.face_detection_model,
            "",
            (320, 320),
            args.score_threshold,
            args.nms_threshold,
            args.top_k
        )
        recognizer = cv2.FaceRecognizerSF.create(
        args.face_recognition_model,"")

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
        
        # Đọc khung hình tiếp theo
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Nhận diện khuôn mặt
        tm.start()
        faces = detector.detect(frame)  # faces is a tuple
        tm.stop()

        value = []
        scores = []
        if faces[1] is not None:
            for x in range(len(faces[1])):

                face_align = recognizer.alignCrop(frame, faces[1][x])
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]

                value.append(test_predict[0])

                score = svc.decision_function(face_feature)
                best_idx = np.argmax(score)
                if 0 <= test_predict[0] < len(mydict):
                    confidence = float(score) if np.isscalar(score) else score[best_idx]
                    scores.append(confidence)

                cv2.putText(frame, result, (1, 50 + 20 * x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[test_predict[0]], 2)

        # Hiển thị kết quả
        visualize(frame, faces, tm.getFPS(), value=value, scores=scores)

        # Chuyển BGR → RGB (Streamlit cần ảnh RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame, channels='RGB')
        time.sleep(0.03)  # Giới hạn tốc độ khung hình
    if cap:
        cap.release()



def train_model():
    """Hàm huấn luyện mô hình, sẽ được gọi khi người dùng nhấn nút 'Huấn luyện'."""
    st.toast("Đang huấn luyện mô hình...")
    training_model()  # Gọi hàm huấn luyện mô hình
    st.toast("Huấn luyện xong!")


def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='./model/face_detection_yunet_2023mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='./model/face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()

svc = joblib.load('./model/svc_face_detection.pkl')
mydict = ['Thuan', 'Tien']
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]

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
                label = mydict[value[idx]]
                color = colors[value[idx]]

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
