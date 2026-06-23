import streamlit as st
import cv2
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(page_title="ITS - Nhóm 10", layout="wide")
st.title("🚗 Hệ Thống Phân Tích Độ Lún Đường Nhựa (ITS Cloud)")

# Cấu hình Sidebar & Upload
with st.sidebar:
    st.header("⚙️ Cấu hình")
    uploaded_files = st.file_uploader("Chọn video (MP4)", type=["mp4"], accept_multiple_files=True)
    limit_distance = st.number_input("Khoảng cách quét (m)", value=500.0)
    analyze_btn = st.button("Bắt đầu phân tích")

# Chia cột: Bản đồ bên trái, Video/Số liệu bên phải
col_map, col_data = st.columns([1, 1])

# --- Xử lý Bản đồ ---
with col_map:
    st.subheader("🗺️ Bản đồ định vị")
    m = folium.Map(location=[21.0055, 105.9334], zoom_start=15)
    map_data = st_folium(m, width=600, height=400)

# --- Xử lý Logic OpenCV & Dữ liệu ---
if analyze_btn and uploaded_files:
    for uploaded_file in uploaded_files:
        vid_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(vid_path, "wb") as f: f.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(vid_path)
        frame_placeholder = col_data.empty()
        table_placeholder = col_data.empty()
        
        results = []
        frame_count = 0
        current_segment = 0
        seg_area = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # --- Logic OpenCV cũ của bạn ---
            frame_disp = cv2.resize(frame, (640, 360))
            h, w = frame_disp.shape[:2]
            gray = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY)
            # ... (Giữ nguyên logic contour của bạn) ...
            
            # Hiển thị Live Video
            frame_placeholder.image(frame_disp, channels="BGR", use_container_width=True)
            
            # Logic phân đoạn 50m
            if frame_count % 150 == 0: # Giả lập 50m
                results.append({
                    'Đoạn': f"{current_segment*50}m",
                    'Diện tích': 2.5,
                    'Trạng thái': "Nặng"
                })
                table_placeholder.table(pd.DataFrame(results))
                current_segment += 1
            frame_count += 1
        cap.release()
