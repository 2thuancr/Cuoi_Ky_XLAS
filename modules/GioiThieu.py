import streamlit as st

def show():
    html_code = """
    <div style="font-family: Arial, sans-serif; text-align: center;">
        <h2 style="color: #FFFFFF; margin: 0; padding: 0;">BÁO CÁO CUỐI KỲ</h2>
        <h1 style="color: #FFFFFF; margin: 0; padding: 0;">MÔN XỬ LÝ ẢNH SỐ</h1>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    page_bg = """
    <style>
    .functions-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1em;
        margin-bottom: 2em;
    }
    .function-item {
        background: linear-gradient(to right, rgba(0, 80, 200, 0.7), rgba(0, 180, 200, 0.7));
        padding: 1em;
        text-align: center;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        display: flex; 
        justify-content: center; 
        align-items: center;
        height: 200px;
        width: 200px;
    }
    .function-item:hover {
        background: linear-gradient(to right, rgba(0, 80, 200, 0.9), rgba(0, 180, 200, 0.9));
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .function-item h3 {
        margin-top: 0;
        color: #E0E0FF;
    }
    .members {
        padding: 1em;
        background-color: transparent;
        border-radius: 8px;
        text-align: center;
        font-size: 20px;

        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .members h3 {
        font-size: 26px;
        color: #FFFFFF;
    }
    .member-item {
        margin-bottom: 0.5em;
        color: #FFFFFF;
        font-size: 22px;
    }
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

    with st.container():
        st.markdown(
        """
        <div class="members" style="margin-top: 5px;">
            <h3>Thành viên:</h3>
            <div class="member-item">
                <strong>VI QUỐC THUẬN</strong> - MSSV: 22110006
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.container():
        st.markdown(
        """
        <div class="functions-grid">
            <div class="function-item">
                <h3>Xử lý ảnh số Chương 3</h3>
            </div>
            <div class="function-item">
                <h3>Xử lý ảnh số Chương 4</h3>
            </div>
            <div class="function-item">
                <h3>Xử lý ảnh số Chương 9</h3>
            </div>
            <div class="function-item">
                <h3>Nhận diện khuôn mặt</h3>
            </div>
            <div class="function-item">
                <h3>Nhận diện trái cây</h3>
            </div>
            <div class="function-item">
                <h3>Nhận diện bàn tay</h3>
            </div>
            <div class="function-item">
                <h3>Nhận diện ngôn ngữ kí hiệu</h3>
            </div>
            <div class="function-item">
                <h3>Lane detection</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
