from flask import Flask, request, jsonify, render_template, Response
import cv2
import numpy as np
import os
import time

app = Flask(__name__)

# Tạo thư mục lưu trữ video để tránh lỗi đường dẫn và hỗ trợ tính năng tải lại
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

processing_status = {}
latest_analysis_results = {}
video_max_distances = {} # Lưu trữ giới hạn mét chọn từ bản đồ của từng video

@app.route('/')
def index():
    return render_template('demo.html')

def gen_frames(video_path, vid_index):
    global latest_analysis_results, processing_status, video_max_distances
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    speed_m_s = 15.0  # Vận tốc xe giả lập 15m/s
    estimated_length_meters = (total_frames / fps) * speed_m_s
    
    # Lấy giới hạn mét được truyền từ bản đồ (nếu không chọn thì chạy hết video)
    limit_distance = video_max_distances.get(vid_index, estimated_length_meters)
    
    num_segments = max(1, int(limit_distance / 50))
    frames_per_segment = int((50 / speed_m_s) * fps)

    current_segment = 0
    frame_count = 0
    
    seg_len, seg_wid, seg_area, seg_pos = [], [], [], []
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_distance_m = int((frame_count / fps) * speed_m_s)
        
        # TỰ ĐỘNG NGẮT: Nếu xe đã chạy hết đoạn đường tọa độ bạn chọn
        if current_distance_m > limit_distance:
            break

        frame_disp = cv2.resize(frame, (640, 360))
        height, width = frame_disp.shape[:2]
        
        # V vùng ROI mặt đường
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

        frame_count += 1
        time.sleep(0.02)

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
                'id': f"{current_segment*50} - {(current_segment+1)*50}m",
                'length': round(min(t_len, 50.0), 1),
                'width': round(a_wid, 2),
                'area': round(t_area, 2),
                'position': pos_str,
                'status': status
            })
            
            latest_analysis_results[vid_index] = list(results)
            current_segment += 1
            seg_len, seg_wid, seg_area, seg_pos = [], [], [], []

        ret, buffer = cv2.imencode('.jpg', frame_disp)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    latest_analysis_results[vid_index] = results
    processing_status[vid_index] = True

@app.route('/video_feed/<int:vid_index>')
def video_feed(vid_index):
    video_filename = request.args.get('filename')
    video_path = os.path.join(UPLOAD_FOLDER, video_filename) if video_filename else ""
    if os.path.exists(video_path):
        return Response(gen_frames(video_path, vid_index), mimetype='multipart/x-mixed-replace; boundary=frame')
    return jsonify({'error': 'File không tồn tại'}), 400

@app.route('/upload_init', methods=['POST'])
def upload_init():
    global processing_status, latest_analysis_results, video_max_distances
    files = request.files.getlist('video')
    distances = request.form.getlist('distances') # Nhận mảng khoảng cách từ Map gửi lên
    
    if not files or files[0].filename == '': 
        return jsonify({'error': 'Chưa chọn file'}), 400
        
    processing_status = {i: False for i in range(len(files))}
    latest_analysis_results = {i: [] for i in range(len(files))}
    
    saved_filenames = []
    for i, f in enumerate(files):
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path)
        saved_filenames.append(f.filename)
        # Đồng bộ khoảng cách giới hạn quét cho từng video
        video_max_distances[i] = float(distances[i]) if i < len(distances) else 999999.0
        
    return jsonify({'filenames': saved_filenames})

@app.route('/check_status', methods=['GET'])
def check_status():
    global processing_status, latest_analysis_results
    all_done = all(processing_status.values()) if processing_status else False
    return jsonify({'done': all_done, 'results': latest_analysis_results})

if __name__ == '__main__':
    # BẬT THREADED=TRUE: Bắt buộc để Flask chạy song song luồng video và luồng bảng số liệu không bị nghẽn
    app.run(debug=True, port=8080, threaded=True)