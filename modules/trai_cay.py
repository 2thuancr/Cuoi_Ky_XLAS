# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
from typing import Any
import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS


class Inference:

    def __init__(self, **kwargs: Any):
        check_requirements("streamlit>=1.29.0")
        import streamlit as st

        self.st = st
        self.source = None
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind = []
        self.model = None

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Nhận diện trái cây</h1></div>"""
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)

    def sidebar(self):
        self.st.sidebar.title("User Configuration")
        self.source = self.st.sidebar.selectbox("Video", ("webcam", "video"))
        col1, col2 = self.st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()

    def source_upload(self):
        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())
                with open("ultralytics.mp4", "wb") as out:
                    out.write(g.read())
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0

    def configure(self):
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")
            class_names = list(self.model.names.values())
        self.st.success("Model loaded successfully!")

        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]
        if not isinstance(self.selected_ind, list):
            self.selected_ind = list(self.selected_ind)

    def run_inference(self):
        cap = cv2.VideoCapture(self.vid_file_name)
        if not cap.isOpened():
            self.st.error("Could not open webcam or video source.")
            return

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                self.st.warning("Failed to read frame.")
                break

            results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
            annotated_frame = results[0].plot()

            self.org_frame.image(frame, channels="BGR")
            self.ann_frame.image(annotated_frame, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()


def show():
    import sys
    args = len(sys.argv)
    model = "./model/best.pt"
    inf = Inference(model=model)

    # Giao diện và cấu hình
    inf.web_ui()
    inf.sidebar()
    inf.source_upload()
    inf.configure()

    # Chờ người dùng nhấn Start
    if inf.st.sidebar.button("Start"):
        inf.run_inference()


if __name__ == "__main__":
    show()
