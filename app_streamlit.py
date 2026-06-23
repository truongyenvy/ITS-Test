import streamlit as st
import cv2
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
import os
import time

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Phân Tích Lún Vệt Bánh Xe - Nhóm 10", layout="wide")

# Khởi tạo Session State để lưu dữ liệu Bản đồ không bị mất khi load lại
if 'start_gps' not in st.session_state: st.session_state.start_gps = [21.0055, 105.9334]
if 'end_gps' not in st.session_state: st.session_state.end_gps = [21.0125, 105.9385]
if 'route_coords' not in st.session_state: st.session_state.route_coords = []
if 'route_distance' not in st.session_state: st.session_state.route_distance = 0.0

# Hàm lấy đường đi thực tế từ OSRM API
def lay_duong_cong_osrm(start, end):
    url = f"https://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
    try:
        res = requests.get(url).json()
        if res.get('routes'):
            coords = [[c[1], c[0]] for c in res['routes'][0]['geometry']['coordinates']]
            dist = res['routes'][0]['distance']
            return coords, dist
    except Exception as e:
        pass
    return [], 0.0

# --- HEADER GIAO DIỆN ---
st.markdown("<h1 style='text-align: center; color: #b91c1c;'>🚗 Hệ Thống Phân Tích Độ Lún Đường Nhựa Qua Video</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569;'>Đồ án môn học ITS - Nhóm 10 | Chạy trên Streamlit Cloud</p>", unsafe_allow_html=True)
st.divider()

# --- KHU VỰC CẤU HÌNH (Tương đương Card Upload & Cấu hình cũ) ---
with st.container():
    st.subheader("⚙️ Cấu Hình & Tải Video")
    col_up1, col_up2, col_up3 = st.columns([1, 1, 1])
    
    with col_up1:
        uploaded_file = st.file_uploader("Tải video mặt đường (MP4)", type=["mp4", "avi"])
    with col_up2:
        start_str = st.text_input("Tọa độ ĐIỂM ĐẦU (Lat, Lng):", value=f"{st.session_state.start_gps[0]}, {st.session_state.start_gps[1]}")
    with col_up3:
        end_str = st.text_input("Tọa độ ĐIỂM CUỐI (Lat, Lng):", value=f"{st.session_state.end_gps[0]}, {st.session_state.end_gps[1]}")
        
    btn_col1, btn_col2 = st.columns([1, 5])
    with btn_col1:
        analyze_btn = st.button("🚀 Bắt Đầu Truyền Luồng & Quét OpenCV", type="primary", use_container_width=True)

st.divider()

# --- KHÔNG GIAN LÀM VIỆC CHÍNH (Chia 2 cột giống giao diện cũ) ---
col_map, col_results = st.columns([4.5, 5.5], gap="large")

with col_map:
    st.markdown("<h4 style='color: #1e3a8a;'>🗺️ BẢN ĐỒ GIÁM SÁT TRỰC TUYẾN</h4>", unsafe_allow_html=True)
    
    # Cập nhật tọa độ từ ô nhập liệu
    try:
        s_lat, s_lng = map(float, start_str.split(','))
        e_lat, e_lng = map(float, end_str.split(','))
        if [s_lat, s_lng] != st.session_state.start_gps or [e_lat, e_lng] != st.session_state.end_gps:
            st.session_state.start_gps = [s_lat, s_lng]
            st.session_state.end_gps = [e_lat, e_lng]
            # Gọi API lấy đường vẽ
            st.session_state.route_coords, st.session_state.route_distance = lay_duong_cong_osrm(st.session_state.start_gps, st.session_state.end_gps)
    except: pass

    # Vẽ Bản đồ Folium
    m = folium.Map(location=st.session_state.start_gps, zoom_start=14)
    folium.Marker(st.session_state.start_gps, popup="Điểm Đầu", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(st.session_state.end_gps, popup="Điểm Cuối", icon=folium.Icon(color='red')).add_to(m)
    
    if st.session_state.route_coords:
        # Đã fix lỗi dash_array thành dashArray='5 5' để không bị màn hình đỏ
        folium.Polyline(st.session_state.route_coords, color='#64748b', weight=3, dashArray='5 5').add_to(m)
        
    st_folium(m, width="100%", height=500)

with col_results:
    st.markdown("<h4 style='color: #1e3a8a;'>🎬 KẾT QUẢ QUÉT OPENCV</h4>", unsafe_allow_html=True)
    
    if not uploaded_file:
        st.info("ℹ️ Vui lòng tải video và bấm Bắt đầu quét.")
    
    frame_placeholder = st.empty() # Khung chứa Video Live
    st.markdown("<h4 style='color: #1e3a8a; margin-top: 20px;'>📊 SỐ LIỆU LIVE</h4>", unsafe_allow_html=True)
    table_placeholder = st.empty() # Khung chứa Bảng số liệu Live

    # =====================================================================
    # GIỮ NGUYÊN 100% LOGIC OPENCV GỐC CỦA BẠN BÊN DƯỚI NÀY
    # =====================================================================
    if analyze_btn and uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        video_path = os.path.join("uploads", uploaded_file.name)
        with open(video_path, "wb") as f: f.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        speed_m_s = 15.0  # Vận tốc xe giả lập 15m/s
        estimated_length_meters = (total_frames / fps) * speed_m_s
        
        # Giới hạn lấy từ bản đồ hoặc mặc định
        limit_distance = st.session_state.route_distance if st.session_state.route_distance > 0 else estimated_length_meters
        num_segments = max(1, int(limit_distance / 50))
        frames_per_segment = int((50 / speed_m_s) * fps)

        current_segment = 0
        frame_count = 0
        seg_len, seg_wid, seg_area, seg_pos = [], [], [], []
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            current_distance_m = int((frame_count / fps) * speed_m_s)
            
            # TỰ ĐỘNG NGẮT: Nếu xe đã chạy hết đoạn đường
            if current_distance_m > limit_distance: break

            frame_disp = cv2.resize(frame, (640, 360))
            height, width = frame_disp.shape[:2]
            
            # Vẽ vùng ROI mặt đường
            mask = np.zeros((height, width), dtype=np.uint8)
            road_polygon = np.array([
                [int(width * 0.1), int(height * 0.5)],
                [int(width * 0.9), int(height * 0.5)],
                [int(width * 1.0), int(height * 1.0)],
                [int(width * 0.0), int(height * 1.0)]
            ], np.int32)
            cv2.fillPoly(mask, [road_polygon], 255)

            roi_gray = cv2.bitwise_and(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY), mask=mask)

            # Nhận diện vệt lún bằng OpenCV
            contours, _ = cv2.findContours(cv2.Canny(cv2.GaussianBlur(roi_gray, (7, 7), 0), 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > 1500:  
                    x, y, w, h = cv2.boundingRect(cnt)
                    L = h * 0.05  
                    W = w * 0.02  
                    A = L * W     
                    
                    center_x = x + (w / 2)
                    if center_x < width / 3: pos = "Trái"
                    elif center_x < 2 * (width / 3): pos = "Giữa"
                    else: pos = "Phải"

                    seg_len.append(L)
                    seg_wid.append(W)
                    seg_area.append(A)
                    seg_pos.append(pos)

                    cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 165, 255), 2)

            # Hiển thị khoảng cách thực tế so với giới hạn tọa độ
            cv2.putText(frame_disp, f"LOG: {current_distance_m}m / Limit: {int(limit_distance)}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Đẩy frame lên Streamlit Giao Diện
            frame_placeholder.image(frame_disp, channels="BGR", use_container_width=True)
            frame_count += 1
            time.sleep(0.01) # Sleep nhỏ để giao diện mượt

            # Tổng hợp dữ liệu phân đoạn 50m
            if frame_count >= frames_per_segment * (current_segment + 1):
                t_len = sum(seg_len)
                a_wid = sum(seg_wid) / len(seg_wid) if seg_wid else 0
                t_area = sum(seg_area)
                
                unique_positions = list(set(seg_pos))
                pos_str = ", ".join(unique_positions) if unique_positions else "-"
                
                status = "Tốt"
                if t_area > 15.0: status = "Nặng"
                elif t_area > 5.0: status = "Trung bình"
                elif t_area > 0: status = "Nhẹ"

                results.append({
                    'Phân Đoạn': f"{current_segment*50} - {(current_segment+1)*50}m",
                    'Dài (m)': round(min(t_len, 50.0), 1),
                    'Rộng (m)': round(a_wid, 2),
                    'Diện tích (m²)': round(t_area, 2),
                    'Vị trí': pos_str,
                    'Mức độ': status
                })
                
                # Cập nhật Bảng số liệu Live
                df = pd.DataFrame(results)
                table_placeholder.dataframe(df, use_container_width=True, hide_index=True)
                
                current_segment += 1
                seg_len, seg_wid, seg_area, seg_pos = [], [], [], []

        cap.release()
        st.success("🎉 Hoàn thành quét Video!")
        
        # Thêm nút Tải Báo Cáo Excel sau khi quét xong
        if results:
            csv = pd.DataFrame(results).to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 Xuất File Excel (CSV)", data=csv, file_name="Bao_Cao_Nhom_10.csv", mime="text/csv")
