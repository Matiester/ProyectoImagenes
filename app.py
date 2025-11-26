import os
import cv2
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuración
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "static/uploads")
app.config["PROCESSED_FOLDER"] = os.path.join(BASE_DIR, "static/processed")

# Crear directorios
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)

# ---------------------------------------------------------------------
# Lógica de Procesamiento de Imagen
# ---------------------------------------------------------------------

def apply_effects(frame, params):
    """Aplica filtros e interpolación basados en los parámetros."""
    
    # 1. Filtros de Denoising / Suavizado
    filter_type = params.get('filterType', 'none')
    
    if filter_type == 'bilateral':
        d = int(params.get('bilateral_d', 9))
        sigma = int(params.get('bilateral_sigma', 75))
        frame = cv2.bilateralFilter(frame, d, sigma, sigma)
    elif filter_type == 'gaussian':
        k = int(params.get('gaussian_k', 5))
        if k % 2 == 0: k += 1 # Debe ser impar
        frame = cv2.GaussianBlur(frame, (k, k), 0)
    elif filter_type == 'median':
        k = int(params.get('median_k', 5))
        if k % 2 == 0: k += 1
        frame = cv2.medianBlur(frame, k)

    # 2. Mejoras de Calidad (Contrast / Sharpening)
    if params.get('enable_clahe') == 'true':
        # Convertir a LAB para aplicar CLAHE solo a la luminancia
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(params.get('clahe_clip', 2.0)), tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    if params.get('enable_sharpen') == 'true':
        # Kernel de enfoque básico
        strength = float(params.get('sharpen_strength', 1.0)) # Simulado mezclando
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        frame = cv2.addWeighted(frame, 1 - strength/10, sharpened, strength/10, 0)

    # 3. Interpolación / Redimensionado
    # Nota: Aquí simulamos el redimensionado manteniendo el tamaño original 
    # para no romper el video container, pero aplicando el algoritmo.
    interp_method = params.get('interpMethod', 'linear')
    h, w = frame.shape[:2]
    
    # Mapeo de métodos de OpenCV
    methods = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    cv_method = methods.get(interp_method, cv2.INTER_LINEAR)
    
    # Pequeño truco: Reducir y ampliar para forzar el recálculo de interpolación 
    # (útil si el usuario quiere ver cómo afecta el algoritmo)
    scale = float(params.get('scale_factor', 1.0))
    if scale != 1.0 or interp_method != 'linear':
        # Si escala es > 1 es Upscaling, si es < 1 es Downscaling
        # Para visualizar el efecto en el mismo tamaño de video:
        temp = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv_method)
        # Volver al tamaño original para mantener formato del video
        frame = cv2.resize(temp, (w, h), interpolation=cv_method)

    return frame

def generate_video_segment(video_path, start_time, duration, params=None, is_full_video=False):
    """
    Genera un segmento de video (o el video completo) con efectos aplicados.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_time * fps)
    end_frame = start_frame + int(duration * fps) if not is_full_video else total_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_filename = f"proc_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
    
    # Codec MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = start_frame
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        if params:
            frame = apply_effects(frame, params)
        
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    
    return output_filename

# ---------------------------------------------------------------------
# Rutas Flask
# ---------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)
    
    # Obtener duración
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    return jsonify({
        "video_path": filename, # Solo nombre del archivo
        "duration": duration
    })

@app.route("/preview", methods=["POST"])
def preview():
    """Genera una previsualización de 5 segundos."""
    data = request.json
    video_name = data.get("video_name")
    start_time = float(data.get("start_time", 0))
    params = data.get("params", {})
    
    full_path = os.path.join(app.config["UPLOAD_FOLDER"], video_name)
    
    # Generar snippet procesado
    processed_name = generate_video_segment(full_path, start_time, 5, params)
    
    # Generar snippet original (para comparar exactamente los mismos frames)
    original_snippet_name = generate_video_segment(full_path, start_time, 5, params=None)

    return jsonify({
        "processed_url": url_for('static', filename=f'processed/{processed_name}'),
        "original_url": url_for('static', filename=f'processed/{original_snippet_name}')
    })

@app.route("/process_full", methods=["POST"])
def process_full():
    """Procesa todo el video."""
    data = request.json
    video_name = data.get("video_name")
    params = data.get("params", {})
    
    full_path = os.path.join(app.config["UPLOAD_FOLDER"], video_name)
    
    # Procesar video completo (start=0, duration grande, flag=True)
    processed_name = generate_video_segment(full_path, 0, 0, params, is_full_video=True)
    
    return jsonify({
        "download_url": url_for('static', filename=f'processed/{processed_name}')
    })

if __name__ == "__main__":
    app.run(debug=True)