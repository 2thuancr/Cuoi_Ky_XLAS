import streamlit as st
from modules import GioiThieu, Chuong3, Chuong4, Chuong9, NhanDienKhuonMat, trai_cay, nhan_dien_ban_tay, nhan_dien_chu_ki_hieu, lane_detect
import streamlit as st
import os


st.set_page_config(page_title="Ứng dụng xử lý ảnh")

# Khởi tạo trạng thái nếu chưa có
if 'selected' not in st.session_state:
    st.session_state.selected = "GioiThieu"

# Hàm xử lý sự kiện khi nhấn nút
def set_selection(choice):
    st.session_state.selected = choice

# Sidebar với các nút riêng biệt
with st.sidebar:  
    logo = "https://tracuuxettuyen.hcmute.edu.vn/assets/img/logo/ute_logo.png"
    st.image(logo, width=250)
st.sidebar.title("Menu")
st.sidebar.button("Giới thiệu", on_click=set_selection, args=("GioiThieu",))
st.sidebar.button("Chương 3", on_click=set_selection, args=("Chuong3",))
st.sidebar.button("Chương 4", on_click=set_selection, args=("Chuong4",))
st.sidebar.button("Chương 9", on_click=set_selection, args=("Chuong9",))
st.sidebar.button("Nhận diện khuôn mặt", on_click=set_selection, args=("NhanDienKhuonMat",))
st.sidebar.button("Nhận diện trái cây", on_click=set_selection, args=("TraiCay",))
st.sidebar.button("Nhận diện bàn tay", on_click=set_selection, args=("BanTay",))
st.sidebar.button("Nhận diện ngôn ngữ kí hiệu, ghép câu", on_click=set_selection, args=("GhepCauKiHieu",))
st.sidebar.button("Lane Detection", on_click=set_selection, args=("LaneDetection",))

# Hiển thị nội dung tương ứng
selected = st.session_state.selected

if selected == "GioiThieu":
    GioiThieu.show()
elif selected == "Chuong3":
    Chuong3.show()
elif selected == "Chuong4":
    Chuong4.show()
elif selected == "Chuong9":
    Chuong9.show()
elif selected == "NhanDienKhuonMat":
    NhanDienKhuonMat.show()
elif selected == "TraiCay":
    trai_cay.show()
elif selected == "BanTay":
    nhan_dien_ban_tay.show()
elif selected == "GhepCauKiHieu":
    nhan_dien_chu_ki_hieu.show()
elif selected == "LaneDetection":
    lane_detect.show()



page_bg = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Open Sans', sans-serif;
    background-image: linear-gradient(
        rgba(0, 0, 0, 0.4), 
        rgba(0, 0, 0, 0.4)
    ), url("https://images.unsplash.com/photo-1465101162946-4377e57745c3?q=80&w=2078&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center;
    color: white;
}

[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0);
}

h1, h2, h3 {
    color: #f2f2f2;
}

/* Style cho nút secondary */
button[data-testid="stBaseButton-secondary"] {
    background: linear-gradient(to right, rgba(0, 0, 0, 0.7), rgba(50, 0, 100, 0.7));
    color: white;
    border: 1px solid white;
    padding: 10px 30px;
    border-radius: 10px;
    width: 260px;
    font-size: 16px;
    transition: all 0.3s ease;
}

button[data-testid="stBaseButton-secondary"]:hover {
    background: linear-gradient(to right, rgba(0, 0, 0, 0.9), rgba(50, 0, 100, 0.9));
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    cursor: pointer;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)