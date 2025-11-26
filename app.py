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

# Crear directorios
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)

# Diccionario global para almacenar el progreso de las tareas
# Estructura: {'task_id': {'progress': 0, 'status': 'processing', 'url': ''}}
processing_tasks = {}

# ---------------------------------------------------------------------
# Lógica de Procesamiento (Efectos)
# ---------------------------------------------------------------------

def apply_effects(frame, params):
    """Aplica filtros e interpolación."""
    
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

    # 2. Mejoras (CLAHE / Sharpen)
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

    # 3. Interpolación / Escala
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

# ---------------------------------------------------------------------
# Generadores (GIF y Video Full)
# ---------------------------------------------------------------------

def generate_preview_gif(video_path, start_time, duration, params=None):
    """Genera un GIF de 5 segundos para previsualización."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    start_frame = int(start_time * fps)
    # Limitamos a 5 segundos, pero si FPS es muy alto, el GIF pesará mucho.
    # Limitamos frames máximos para el GIF a 60 frames (aprox 2-3 segs fluidos o 5 segs a 12fps)
    # para que la web no se cuelgue cargándolo.
    frames_to_capture = int(duration * fps)
    
    # Salto de frames para que el GIF no sea gigante si el video es 60fps
    step = 1
    if fps > 15:
        step = int(fps / 15) # Forzamos aprox 15fps para el GIF
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_pil = []
    count = 0
    
    while count < frames_to_capture:
        ret, frame = cap.read()
        if not ret: break
        
        # Solo procesamos 1 de cada 'step' frames
        if count % step == 0:
            if params:
                frame = apply_effects(frame, params)
            
            # Convertir BGR (OpenCV) a RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Redimensionar preview si es muy grande (max width 600px para rapidez)
            h, w = frame_rgb.shape[:2]
            if w > 600:
                new_h = int(h * (600 / w))
                frame_rgb = cv2.resize(frame_rgb, (600, new_h))

            pil_img = Image.fromarray(frame_rgb)
            frames_pil.append(pil_img)
            
        count += 1

    cap.release()
    
    if not frames_pil:
        return None

    output_filename = f"prev_{uuid.uuid4().hex[:8]}.gif"
    output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
    
    # Guardar como GIF
    frames_pil[0].save(
        output_path, 
        save_all=True, 
        append_images=frames_pil[1:], 
        optimize=True, 
        duration=int(1000/15), # 15fps aprox
        loop=0
    )
    
    return output_filename

def process_video_thread(task_id, video_path, params):
    """Función que corre en segundo plano para procesar el video completo."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_filename = f"final_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Aplicar efectos
            frame = apply_effects(frame, params)
            out.write(frame)
            
            processed_count += 1
            
            # Actualizar progreso cada 10 frames para no bloquear variables
            if processed_count % 5 == 0:
                progress = int((processed_count / total_frames) * 100)
                processing_tasks[task_id]['progress'] = progress

        cap.release()
        out.release()
        
        # Finalizar
        processing_tasks[task_id]['progress'] = 100
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['url'] = output_filename
        
    except Exception as e:
        processing_tasks[task_id]['status'] = 'error'
        print(f"Error en thread: {e}")

# ---------------------------------------------------------------------
# Rutas Flask
# ---------------------------------------------------------------------

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
    
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frames / fps if fps > 0 else 0
    cap.release()

    return jsonify({"video_path": filename, "duration": duration})

@app.route("/preview_gif", methods=["POST"])
def preview_gif():
    """Genera dos GIFs: uno original y otro procesado."""
    data = request.json
    video_name = data.get("video_name")
    start_time = float(data.get("start_time", 0))
    params = data.get("params", {})
    
    full_path = os.path.join(app.config["UPLOAD_FOLDER"], video_name)
    
    # 1. GIF Procesado
    processed_gif = generate_preview_gif(full_path, start_time, 5, params)
    # 2. GIF Original (sin params)
    original_gif = generate_preview_gif(full_path, start_time, 5, params=None)

    return jsonify({
        "processed_url": url_for('static', filename=f'processed/{processed_gif}'),
        "original_url": url_for('static', filename=f'processed/{original_gif}')
    })

@app.route("/start_processing", methods=["POST"])
def start_processing():
    """Inicia el thread de procesamiento."""
    data = request.json
    video_name = data.get("video_name")
    params = data.get("params", {})
    
    task_id = uuid.uuid4().hex
    full_path = os.path.join(app.config["UPLOAD_FOLDER"], video_name)
    
    # Inicializar estado
    processing_tasks[task_id] = {
        'progress': 0,
        'status': 'processing',
        'url': None
    }
    
    # Arrancar hilo
    thread = threading.Thread(target=process_video_thread, args=(task_id, full_path, params))
    thread.start()
    
    return jsonify({"task_id": task_id})

@app.route("/progress/<task_id>", methods=["GET"])
def get_progress(task_id):
    """El frontend consulta esto cada segundo."""
    task = processing_tasks.get(task_id)
    if task:
        response = {
            "progress": task['progress'],
            "status": task['status']
        }
        if task['status'] == 'completed':
            response['download_url'] = url_for('static', filename=f'processed/{task["url"]}')
        return jsonify(response)
    else:
        return jsonify({"error": "Task not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)