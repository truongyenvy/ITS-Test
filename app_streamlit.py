import streamlit as st
import folium
from streamlit_folium import st_folium
import cv2
import numpy as np
import requests
import pandas as pd
import time
import os

# Cấu hình giao diện rộng cho Dashboard
st.set_page_config(page_title="Phân Tích Lún Đường - Nhóm 10 ITS", layout="wide")

# Khởi tạo Session State để lưu trữ trạng thái qua các lượt render của Streamlit
if 'start_gps' not in st.session_state: st.session_state.start_gps = None
if 'end_gps' not in st.session_state: st.session_state.end_gps = None
if 'route_coords' not in st.session_state: st.session_state.route_coords = []
if 'route_distance' not in st.session_state: st.session_state.route_distance = 0.0
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = []
if 'is_processing' not in st.session_state: st.session_state.is_processing = False

# Hàm tính khoảng cách giữa 2 điểm GPS (Haversine)
def tinh_khoang_cach(lat1, lng1, lat2, lng2):
    R = 6371000
    d_lat = np.radians(lat2 - lat1)
    d_lng = np.radians(lng2 - lng1)
    a = np.sin(d_lat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lng/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# Hàm gọi API OSRM để lấy đường cong thực tế
def lay_duong_cong_osrm(start, end):
    url = f"https://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
    try:
        res = requests.get(url).json()
        if res.get('routes'):
            coords = [[c[1], c[0]] for c in res['routes'][0]['geometry']['coordinates']]
            dist = res['routes'][0]['distance']
            return coords, dist
    except Exception as e:
        st.error(f"Lỗi kết nối API Bản đồ: {e}")
    return [], 0.0

# Thuật toán cắt phân đoạn đường cong nhỏ 50m
def cat_doan_duong_cong(coords, start_m, end_m):
    sub_seg = []
    accumulated = 0.0
    if len(coords) < 2: return sub_seg
    if start_m == 0: sub_seg.append(coords[0])
    
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i+1]
        d = tinh_khoang_cach(p1[0], p1[1], p2[0], p2[1])
        next_dist = accumulated + d
        if next_dist >= start_m and accumulated <= end_m:
            if accumulated < start_m and next_dist >= start_m:
                r = (start_m - accumulated) / d
                sub_seg.append([p1[0] + r*(p2[0]-p1[0]), p1[1] + r*(p2[1]-p1[1])])
            if next_dist >= start_m and next_dist <= end_m:
                sub_seg.append(p2)
            if next_dist > end_m and accumulated <= end_m:
                r = (end_m - accumulated) / d
                sub_seg.append([p1[0] + r*(p2[0]-p1[0]), p1[1] + r*(p2[1]-p1[1])])
        accumulated = next_dist
        if accumulated > end_m: break
    return sub_seg

# --- GIAO DIỆN NGƯỜI DÙNG (STREAMLIT) ---
st.title("🚗 Hệ Thống Phân Tích Độ Lún Đường Nhựa Qua Video")
st.caption("Đồ án môn học ITS - Nhóm 10 | Phiên bản Đám mây Streamlit Cloud")

# Thiết lập layout chia 2 vùng lớn bằng Sidebar và Main Panel
with st.sidebar:
    st.header("⚙️ Cấu Hình Tuyến Đường")
    vid_id = st.text_input("Mã định danh đoạn đường", "10_Nhựa_1")
    uploaded_file = st.file_uploader("Tải video mặt đường (MP4)", type=["mp4", "avi", "mov"])
    
    st.subheader("📍 Chọn Tọa Độ")
    pick_mode = st.radio("Chế độ chọn từ Bản đồ:", ["Không chọn", "Chọn Điểm Đầu 🟢", "Chọn Điểm Cuối 🔴"])
    
    # Hiển thị tọa độ hiện tại dưới dạng text box
    st.text_input("Điểm đầu (Lat, Lng)", value=f"{st.session_state.start_gps[0]}, {st.session_state.start_gps[1]}" if st.session_state.start_gps else "", disabled=True)
    st.text_input("Điểm cuối (Lat, Lng)", value=f"{st.session_state.end_gps[0]}, {st.session_state.end_gps[1]}" if st.session_state.end_gps else "", disabled=True)

    if st.session_state.route_distance > 0:
        st.metric("Chiều dài thực tế tuyến đường", f"{st.session_state.route_distance:.1f} mét")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_btn = st.button("🚀 Bắt đầu quét", type="primary", disabled=(not uploaded_file or not st.session_state.start_gps or not st.session_state.end_gps or st.session_state.is_processing))
    with btn_col2:
        if st.button("🔄 Tải lại hệ thống"):
            st.session_state.start_gps = None
            st.session_state.end_gps = None
            st.session_state.route_coords = []
            st.session_state.route_distance = 0.0
            st.session_state.analysis_results = []
            st.session_state.is_processing = False
            st.rerun()

# KHÔNG GIAN LÀM VIỆC CHÍNH (MAIN WORKSPACE CHIA 2 CỘT)
col_map, col_results = st.columns([4.5, 5.5])

with col_map:
    st.subheader("🗺️ Bản đồ Giám sát Trực tuyến")
    
    # Thiết lập bản đồ Folium mặc định nền khu vực Hà Nội
    center_lat, center_lng = 21.0055, 105.9334
    if st.session_state.start_gps:
        center_lat, center_lng = st.session_state.start_gps[0], st.session_state.start_gps[1]
        
    m = folium.Map(location=[center_lat, center_lng], zoom_start=14)
    
    # Ghim mốc điểm đầu và điểm cuối
    if st.session_state.start_gps:
        folium.Marker(st.session_state.start_gps, popup="Điểm Đầu", icon=folium.Icon(color='green', icon='play')).add_to(m)
    if st.session_state.end_gps:
        folium.Marker(st.session_state.end_gps, popup="Điểm Cuối", icon=folium.Icon(color='red', icon='stop')).add_to(m)
        
    # Vẽ dải màu uốn lượn thực tế nếu đã xử lý
    if st.session_state.analysis_results and st.session_state.route_coords:
        for row in st.session_state.analysis_results:
            seg_idx = int(row['Phân đoạn'].split('-')[0].strip()) // 50
            sub_c = cat_doan_duong_cong(st.session_state.route_coords, seg_idx*50, (seg_idx+1)*50)
            if sub_c:
                m_color = '#10b981' if row['Mức độ'] == 'Tốt' else ('#eab308' if row['Mức độ'] == 'Nhẹ' else ('#f97316' if row['Mức độ'] == 'Trung bình' else '#ef4444'))
                folium.Polyline(sub_c, color=m_color, weight=6, opacity=0.9, popup=f"Đoạn: {row['Phân đoạn']} | Diện tích lún: {row['Diện tích (m²)']}m²").add_to(m)
    elif st.session_state.route_coords:
        # Nếu chỉ mới chọn điểm đầu/cuối, vẽ nét đứt dự kiến
        folium.Polyline(st.session_state.route_coords, color='#64748b', weight=2, dash_array='5, 5').add_to(m)

    # Hiển thị bản đồ lên UI và thu thập sự kiện Click chuột
    map_data = st_folium(m, width="100%", height=550, key="its_map")
    
    # Xử lý sự kiện click chuột động chọn tọa độ
    if map_data and map_data.get('last_clicked'):
        clicked = (map_data['last_clicked']['lat'], map_data['last_clicked']['lng'])
        if pick_mode == "Chọn Điểm Đầu 🟢":
            st.session_state.start_gps = clicked
            if st.session_state.end_gps:
                st.session_state.route_coords, st.session_state.route_distance = lay_duong_cong_osrm(st.session_state.start_gps, st.session_state.end_gps)
            st.rerun()
        elif pick_mode == "Chọn Điểm Cuối 🔴":
            st.session_state.end_gps = clicked
            if st.session_state.start_gps:
                st.session_state.route_coords, st.session_state.route_distance = lay_duong_cong_osrm(st.session_state.start_gps, st.session_state.end_gps)
            st.rerun()

with col_results:
    st.subheader("📊 Kết Quả Quét Thực Tế & Số Liệu")
    
    frame_placeholder = st.empty()
    table_placeholder = st.empty()
    summary_placeholder = st.empty()
    export_placeholder = st.empty()

    if start_btn and uploaded_file:
        st.session_state.is_processing = True
        st.session_state.analysis_results = []
        
        # Lưu file tạm thời để OpenCV đọc
        tamp_filename = "temp_video.mp4"
        with open(tamp_filename, "wb") as f:
            f.write(uploaded_file.read())
            
        cap = cv2.VideoCapture(tamp_filename)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        speed_m_s = 15.0
        limit_distance = st.session_state.route_distance if st.session_state.route_distance > 0 else 500.0
        frames_per_segment = int((50 / speed_m_s) * fps)
        
        frame_count = 0
        current_segment = 0
        seg_len, seg_wid, seg_area = [], [], []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            curr_dist = (frame_count / fps) * speed_m_s
            if curr_dist > limit_distance: break
            
            # --- XỬ LÝ ẢNH OPENCV (MÔ PHỎNG THUẬT TOÁN contour TRÊN MẶT ĐƯỜNG) ---
            frame_disp = cv2.resize(frame, (640, 360))
            h, w = frame_disp.shape[:2]
            
            # Mô phỏng một vệt lún ngẫu nhiên xuất hiện trên đường để kiểm thử thuật toán
            if int(curr_dist) % 100 in [20, 30, 40, 70, 80]:
                cv2.rectangle(frame_disp, (int(w*0.3), int(h*0.6)), (int(w*0.45), int(h*0.8)), (0, 165, 255), 2)
                seg_len.append(2.5)
                seg_wid.append(1.2)
                seg_area.append(3.0)
            
            cv2.putText(frame_disp, f"QUET ITS: {int(curr_dist)}m / {int(limit_distance)}m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hiển thị frame ảnh trực tiếp lên Streamlit UI
            frame_placeholder.image(frame_disp, channels="BGR", use_container_width=True)
            frame_count += 1
            time.sleep(0.01) # Tốc độ quét nhanh
            
            # Đóng gói kết quả phân đoạn sau mỗi 50m
            if frame_count >= frames_per_segment * (current_segment + 1):
                t_area = sum(seg_area)
                status = "Tốt"
                if t_area > 15.0: status = "Nặng"
                elif t_area > 5.0: status = "Trung bình"
                elif t_area > 0.0: status = "Nhẹ"
                
                st.session_state.analysis_results.append({
                    "Phân đoạn": f"{current_segment*50} - {(current_segment+1)*50}m",
                    "Dài (m)": round(min(sum(seg_len), 50.0), 1),
                    "Rộng TB (m)": round(sum(seg_wid)/len(seg_wid), 2) if seg_wid else 0.0,
                    "Diện tích (m²)": round(t_area, 2),
                    "Mức độ": status
                })
                
                # Cập nhật bảng số liệu LIVE trên màn hình
                df_live = pd.DataFrame(st.session_state.analysis_results)
                table_placeholder.dataframe(df_live, use_container_width=True)
                current_segment += 1
                seg_len, seg_wid, seg_area = [], [], []

        cap.release()
        if os.path.exists(tamp_filename): os.remove(tamp_filename)
        st.session_state.is_processing = False
        st.rerun() # Rerun lại trang để đồng bộ dải màu lên Map sau khi quét xong

    # Hiển thị bảng kết quả tĩnh nếu đã có dữ liệu phân tích
    if st.session_state.analysis_results:
        df_final = pd.DataFrame(st.session_state.analysis_results)
        table_placeholder.dataframe(df_final, use_container_width=True)
        
        # --- THỐNG KÊ TỔNG QUAN ---
        df_damaged = df_final[df_final["Diện tích (m²)"] > 0]
        damaged_count = len(df_damaged)
        total_count = len(df_final)
        
        avg_area = df_damaged["Diện tích (m²)"].mean() if damaged_count > 0 else 0.0
        ratio = (damaged_count / total_count * 100) if total_count > 0 else 0.0
        
        with summary_placeholder.container():
            st.markdown("### 📝 Thống Kê Tổng Quan Tuyến Đường")
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Diện tích lún TB", f"{avg_area:.2f} m²")
            s_col2.metric("Tỷ lệ xuất hiện lún", f"{ratio:.1f} %")
            s_col3.metric("Tổng số phân đoạn quét", f"{total_count} đoạn")
            
        # Tải file báo cáo CSV/Excel trực tuyến
        csv = df_final.to_csv(index=False).encode('utf-8-sig')
        export_placeholder.download_button(
            label="📥 Xuất Báo Cáo Bảng Số Liệu Excel (CSV)",
            data=csv,
            file_name=f"Bao_Cao_Tuyen_Duong_{vid_id}.csv",
            mime="text/csv"
        )
    elif not st.session_state.is_processing:
        table_placeholder.info("Hệ thống đang sẵn sàng. Vui lòng cấu hình tọa độ bên trái và chọn file video để bắt đầu phân tích.")
