import streamlit as st

def show():
    html_code = """
    <div style="font-family: Arial, sans-serif; text-align: center; padding: 20px;">
        <h1 style="color: #FFFFFF; ">Xử lý ảnh số cuối kỳ</h1>
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
        background: linear-gradient(to right, rgba(0, 0, 0, 0.7), rgba(50, 0, 100, 0.7));
        padding: 1em;
        text-align: center;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        display: flex; 
        justify-content: center; 
        align-items: center;
        height:220px;
        width:220px;
    }
    .function-item h3 {
        margin-top: 0;
        color: #E0E0FF;
    }

    .function-item:hover {
        background: linear-gradient(to right, rgba(0, 0, 0, 0.9), rgba(50, 0, 100, 0.9));
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        
    }

    .members {
        padding: 1em;
        background-color: transparent;
        border-radius: 8px;
    }
    .member-item {
        margin-bottom: 0.5em;
    }

    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

    with st.container():
        st.markdown(
        """
        <div class="functions-grid">
            <div class="function-item">
                <h3>Xử lý ảnh số Chương 3, 4, 9</h3>
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
                <h3>Nhận diện ngôn ngữ kí hiệu, ghép câu</h3>
            </div>
            <div class="function-item">
                <h3>Lane detection</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown(
        """
        <h3>Thành viên:</h3>
        <div class="members">
            <div class="member-item">
                <strong>Phạm Danh Hưởng</strong> - MSSV: 22110344
            </div>
            <div class="member-item">
                <strong>Lê Đăng Hiếu</strong> - MSSV: 22110322
            </div>
        </div>
        """, unsafe_allow_html=True)
