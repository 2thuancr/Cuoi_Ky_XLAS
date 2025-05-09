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
st.sidebar.button("GIỚI THIỆU", on_click=set_selection, args=("GioiThieu",))
st.sidebar.button("CHƯƠNG 3", on_click=set_selection, args=("Chuong3",))
st.sidebar.button("CHƯƠNG 4", on_click=set_selection, args=("Chuong4",))
st.sidebar.button("CHƯƠNG 9", on_click=set_selection, args=("Chuong9",))
st.sidebar.button("NHẬN DIỆN KHUÔN MẶT", on_click=set_selection, args=("NhanDienKhuonMat",))
st.sidebar.button("NHẬN DIỆN TRÁI CÂY", on_click=set_selection, args=("TraiCay",))
st.sidebar.button("NHẬN DIỆN BÀN TAY", on_click=set_selection, args=("BanTay",))
st.sidebar.button("NHẬN DIỆN KÝ HIỆU NGÔN NGỮ, NGÓN TAY", on_click=set_selection, args=("GhepCauKiHieu",))
st.sidebar.button("LANE DETECTION", on_click=set_selection, args=("LaneDetection",))

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
    ), url("https://scontent.fsgn19-1.fna.fbcdn.net/v/t39.30808-6/471306834_1312910153458292_2571871794578179435_n.jpg?_nc_cat=100&ccb=1-7&_nc_sid=cc71e4&_nc_eui2=AeEQNuxU4BSL5zbmuTz-nePFkRcS5UjhLz2RFxLlSOEvPSHeiK1bsd0TUiDK4tLbMWXoUkSNOQS3lTG7D6m9eeLn&_nc_ohc=XfRkOoWeDocQ7kNvwHbHgOU&_nc_oc=Adl-IQJb4v5BhBdrHhClzSQtYTwqju1Pv9LyqqBvDIBH0_qMNMIzAnGSkf72IQMRELk&_nc_zt=23&_nc_ht=scontent.fsgn19-1.fna&_nc_gid=NDPGMwpY50TH_vJSMW38eg&oh=00_AfLfB84pWJhnTp-Yq0v8tpfME-Oxn2VuXIHZNHs5tG_6lA&oe=68238A31");
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
    background: linear-gradient(to right, rgba(0, 80, 200, 0.7), rgba(0, 180, 200, 0.7)); /* Màu xanh */
    color: white;
    border: 1px solid white;
    padding: 10px 30px;
    border-radius: 10px;
    width: 260px;
    font-size: 16px;
    transition: all 0.3s ease;
}

button[data-testid="stBaseButton-secondary"]:hover {
    background: linear-gradient(to right, rgba(0, 80, 200, 0.9), rgba(0, 180, 200, 0.9)); /* Màu xanh đậm khi hover */
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    cursor: pointer;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)