import os
import cv2
import uuid
import numpy as np
import threading
import time
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Configuración
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "static/uploads")
app.config["PROCESSED_FOLDER"] = os.path.join(BASE_DIR, "static/processed")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)

processing_tasks = {}

# --- Lógica de Efectos (Igual que antes) ---
def apply_effects(frame, params):
    # 1. Filtros
    filter_type = params.get('filterType', 'none')
    if filter_type == 'bilateral':
        d = int(params.get('bilateral_d', 9))
        sigma = int(params.get('bilateral_sigma', 75))
        frame = cv2.bilateralFilter(frame, d, sigma, sigma)
    elif filter_type == 'gaussian':
        k = int(params.get('gaussian_k', 5))
        if k % 2 == 0: k += 1
        frame = cv2.GaussianBlur(frame, (k, k), 0)
    elif filter_type == 'median':
        k = int(params.get('median_k', 5))
        if k % 2 == 0: k += 1
        frame = cv2.medianBlur(frame, k)

    # 2. Mejoras
    if params.get('enable_clahe') == 'true':
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(params.get('clahe_clip', 2.0)), tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    if params.get('enable_sharpen') == 'true':
        strength = float(params.get('sharpen_strength', 1.0))
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        frame = cv2.addWeighted(frame, 1 - strength/10, sharpened, strength/10, 0)

    # 3. Escala
    interp_method = params.get('interpMethod', 'linear')
    h, w = frame.shape[:2]
    methods = {
        'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC, 'lanczos': cv2.INTER_LANCZOS4
    }
    cv_method = methods.get(interp_method, cv2.INTER_LINEAR)
    scale = float(params.get('scale_factor', 1.0))
    
    if scale != 1.0 or interp_method != 'linear':
        temp = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv_method)
        frame = cv2.resize(temp, (w, h), interpolation=cv_method)

    return frame

# --- Análisis Estructural ---
def analyze_video_structure(video_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_map = list(range(total_frames))
    unique_count = 0
    prev_frame = None
    reference_idx = 0
    idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        small = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            prev_frame = gray
            unique_count += 1
            reference_idx = idx
        else:
            diff = cv2.absdiff(prev_frame, gray)
            if np.mean(diff) < threshold:
                frame_map[idx] = reference_idx
            else:
                prev_frame = gray
                reference_idx = idx
                unique_count += 1
        idx += 1

    cap.release()
    return frame_map, unique_count, total_frames, fps, w, h

# --- Procesamiento Background ---
def process_video_thread(task_id, video_path, params):
    try:
        frame_map, unique_c, total_f, fps, w, h = analyze_video_structure(video_path)
        
        output_filename = f"final_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h)) # Aquí deberíamos ajustar w,h si hay escala
        
        # Ajuste de tamaño si hay escala en params, necesitamos procesar el primer frame para saber el tamaño final
        cap = cv2.VideoCapture(video_path)
        ret, sample = cap.read()
        if ret:
            processed_sample = apply_effects(sample, params)
            ph, pw = processed_sample.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (pw, ph))
        cap.release()
        
        cap = cv2.VideoCapture(video_path)
        last_unique_processed = None
        current_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            source_idx = frame_map[current_idx]
            
            if source_idx == current_idx:
                processed_frame = apply_effects(frame, params)
                last_unique_processed = processed_frame
                out.write(processed_frame)
            else:
                if last_unique_processed is not None:
                    out.write(last_unique_processed)
                else:
                    proc = apply_effects(frame, params)
                    out.write(proc)

            current_idx += 1
            if current_idx % 10 == 0:
                processing_tasks[task_id]['progress'] = int((current_idx / total_f) * 100)

        cap.release()
        out.release()
        
        processing_tasks[task_id]['progress'] = 100
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['url'] = output_filename
        
    except Exception as e:
        processing_tasks[task_id]['status'] = 'error'
        print(f"Error: {e}")

# --- Rutas ---

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file: return jsonify({"error": "No file"}), 400
    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)
    return jsonify({"video_path": filename})

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    path = os.path.join(app.config["UPLOAD_FOLDER"], data.get("video_name"))
    _, unique, total, fps, w, h = analyze_video_structure(path)
    return jsonify({
        "width": w, "height": h, "fps": fps, 
        "total_frames": total, "unique_frames": unique,
        "duration": total/fps if fps else 0
    })

@app.route("/preview", methods=["POST"])
def preview():
    data = request.json
    video_name = data.get("video_name")
    mode = data.get("mode", "frame")
    time_point = float(data.get("time_point", 0))
    params = data.get("params", {})
    
    path = os.path.join(app.config["UPLOAD_FOLDER"], video_name)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    start_frame = int(time_point * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    output_proc = f"proc_{uuid.uuid4().hex[:6]}"
    output_orig = f"orig_{uuid.uuid4().hex[:6]}"
    
    response_data = {}
    
    # Datos para calcular área (Resolución original)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_area = original_w * original_h
    
    if mode == 'frame':
        ret, frame = cap.read()
        if ret:
            # Medimos tiempo de 1 frame
            t0 = time.time()
            frame_proc = apply_effects(frame, params)
            t1 = time.time()
            
            # Guardamos
            cv2.imwrite(os.path.join(app.config["PROCESSED_FOLDER"], output_orig + ".jpg"), frame)
            cv2.imwrite(os.path.join(app.config["PROCESSED_FOLDER"], output_proc + ".jpg"), frame_proc)
            
            # Datos de rendimiento
            ph, pw = frame_proc.shape[:2]
            processed_area = pw * ph
            
            response_data = {
                "type": "image",
                "processed": url_for('static', filename=f'processed/{output_proc}.jpg'),
                "original": url_for('static', filename=f'processed/{output_orig}.jpg'),
                "process_time": t1 - t0,
                "frames_count": 1,
                "original_area": original_area,
                "processed_area": processed_area # En modo frame es full res
            }
            
    elif mode == 'gif':
        frames_pil_proc = []
        frames_pil_orig = []
        
        max_duration = 3.0 
        capture_step = int(fps / 10) 
        if capture_step < 1: capture_step = 1
        
        count = 0
        frames_to_check = int(max_duration * fps)
        
        total_process_time = 0
        processed_frames_count = 0
        preview_area = 0
        
        while count < frames_to_check:
            ret, frame = cap.read()
            if not ret: break
            
            if count % capture_step == 0:
                # Resize para GIF Preview
                h, w = frame.shape[:2]
                if w > 500:
                    nh = int(h * (500/w))
                    frame_s = cv2.resize(frame, (500, nh))
                else:
                    frame_s = frame
                
                preview_area = frame_s.shape[0] * frame_s.shape[1]

                # Original
                orig_rgb = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)
                frames_pil_orig.append(Image.fromarray(orig_rgb))
                
                # Procesado (MEDIR TIEMPO AQUI)
                t0 = time.time()
                proc_s = apply_effects(frame_s, params)
                t1 = time.time()
                
                total_process_time += (t1 - t0)
                processed_frames_count += 1
                
                proc_rgb = cv2.cvtColor(proc_s, cv2.COLOR_BGR2RGB)
                frames_pil_proc.append(Image.fromarray(proc_rgb))
                
            count += 1
            
        if frames_pil_proc:
            path_p = os.path.join(app.config["PROCESSED_FOLDER"], output_proc + ".gif")
            path_o = os.path.join(app.config["PROCESSED_FOLDER"], output_orig + ".gif")
            
            # Duración del frame en GIF (ms)
            duration_ms = int(1000 / (fps / capture_step))
            
            frames_pil_proc[0].save(path_p, save_all=True, append_images=frames_pil_proc[1:], duration=duration_ms, loop=0)
            frames_pil_orig[0].save(path_o, save_all=True, append_images=frames_pil_orig[1:], duration=duration_ms, loop=0)
            
            response_data = {
                "type": "gif",
                "processed": url_for('static', filename=f'processed/{output_proc}.gif'),
                "original": url_for('static', filename=f'processed/{output_orig}.gif'),
                "process_time": total_process_time, # Tiempo total acumulado de procesar los frames del GIF
                "frames_count": processed_frames_count,
                "original_area": original_area,
                "processed_area": preview_area # Area pequeña del preview
            }

    cap.release()
    return jsonify(response_data)

@app.route("/start_processing", methods=["POST"])
def start_processing():
    data = request.json
    task_id = uuid.uuid4().hex
    params = data.get("params", {})
    video_name = data.get("video_name")
    processing_tasks[task_id] = {'progress': 0, 'status': 'processing', 'url': None}
    threading.Thread(target=process_video_thread, args=(task_id, os.path.join(app.config["UPLOAD_FOLDER"], video_name), params)).start()
    return jsonify({"task_id": task_id})

@app.route("/progress/<task_id>")
def progress(task_id):
    task = processing_tasks.get(task_id)
    if task:
        res = {"progress": task['progress'], "status": task['status']}
        if task['status'] == 'completed':
            res['download_url'] = url_for('static', filename=f'processed/{task["url"]}')
        return jsonify(res)
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)