import streamlit as st
from modules import (
    GioiThieu, 
    Chuong3, Chuong4, Chuong9,
    ThuThapKhuonMat, NhanDienKhuonMat, 
    nhan_dien_cam_xuc,
    trai_cay,
    nhan_dien_ban_tay, nhan_dien_chu_ki_hieu,
)

# Cấu hình trang
st.set_page_config(page_title="Ứng dụng xử lý ảnh", layout="wide")

# --- Sidebar ---
with st.sidebar:
    logo = "https://itute.github.io/img/logo/logo.png"
    st.image(logo, width=128)

    st.markdown("### 22110006")
    if st.button("⭐ GIỚI THIỆU"):
        st.query_params.clear()
        st.query_params.update({"menu": "GioiThieu"})
    if st.button("3️⃣ CHƯƠNG 3"):
        st.query_params.clear()
        st.query_params.update({"menu": "Chuong3"})
    if st.button("4️⃣ CHƯƠNG 4"):
        st.query_params.clear()
        st.query_params.update({"menu": "Chuong4"})
    if st.button("9️⃣ CHƯƠNG 9"):
        st.query_params.clear()
        st.query_params.update({"menu": "Chuong9"})
    if st.button("📸 THU THẬP KHUÔN MẶT"):
        st.query_params.clear()
        st.query_params.update({"menu": "ThuThapKhuonMat"})
    if st.button("👥 NHẬN DIỆN KHUÔN MẶT"):
        st.query_params.clear()
        st.query_params.update({"menu": "NhanDienKhuonMat"})
    if st.button("😁 NHẬN DIỆN CẢM XÚC"):
        st.query_params.clear()
        st.query_params.update({"menu": "NhanDienCamXuc"})
    if st.button("🍎 NHẬN DIỆN TRÁI CÂY"):
        st.query_params.clear()
        st.query_params.update({"menu": "TraiCay"})
    if st.button("👋 NHẬN DIỆN BÀN TAY"):
        st.query_params.clear()
        st.query_params.update({"menu": "BanTay"})
    if st.button("🤏 NHẬN DIỆN NGÔN NGỮ KÝ HIỆU"):
        st.query_params.clear()
        st.query_params.update({"menu": "GhepCauKiHieu"})

# --- Routing ---

pages = {
    "GioiThieu": GioiThieu.show,
    "Chuong3": Chuong3.show,
    "Chuong4": Chuong4.show,
    "Chuong9": Chuong9.show,
    "ThuThapKhuonMat": ThuThapKhuonMat.show,
    "NhanDienKhuonMat": NhanDienKhuonMat.show,
    "NhanDienCamXuc": nhan_dien_cam_xuc.show,
    "TraiCay": trai_cay.show,
    "BanTay": nhan_dien_ban_tay.show,
    "GhepCauKiHieu": nhan_dien_chu_ki_hieu.show,
}

# Lấy route từ URL
menu = st.query_params.get("menu", "GioiThieu")

# Gọi hàm tương ứng nếu có
if menu in pages:
    pages[menu]()
else:
    st.error("Trang không tồn tại.")

# --- Giao diện nền ---
page_bg = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Open Sans', sans-serif;
    background-image: linear-gradient(
        rgba(0, 0, 0, 0.4), 
        rgba(0, 0, 0, 0.4)
    ), url("https://itute.github.io/img/hcmute_bg.jpg");
    background-size: cover;
    background-position: center;
}

[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0);
}

h1, h2, h3 {
    color: #f2f2f2;
}

/* Style cho nút trong sidebar */
button[data-testid="stBaseButton"] {
    width: 100% !important;
    background: linear-gradient(to right, rgba(0, 80, 200, 0.7), rgba(0, 180, 200, 0.7)); /* Màu xanh */
    border: 1px solid white;
    padding: 10px 0;
    border-radius: 10px;
    font-size: 16px;
    transition: all 0.3s ease;
}

button[data-testid="stBaseButton"]:hover {
    background: linear-gradient(to right, rgba(0, 80, 200, 0.9), rgba(0, 180, 200, 0.9)); /* Màu xanh đậm khi hover */
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    cursor: pointer;
}

button[data-testid="stBaseButton-secondary"] {
    width: 100% !important;
}

/* Style cho logo trong sidebar */
div[data-testid="stImage"] {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
}

/* Style cho select */
div[data-testid="stTextInput"], 
div[data-testid="stSelectbox"], 
div[data-testid="stMultiSelect"],
div[data-testid="stCheckbox"] {
    background: linear-gradient(to right, rgb(255 255 255 / 70%), rgba(0, 180, 200, 0.7));
    border: 1px solid white;
    padding: 10px;
    border-radius: 10px;
    font-size: 16px;
    color: #fff;
}

section[data-testid="stMain"] {
    background: rgba(255, 255, 255, 0.25);
    padding: 20px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)