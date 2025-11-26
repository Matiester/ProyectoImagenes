import os
import cv2
import uuid
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import imagehash

app = Flask(__name__)

# Configuración de carpetas
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
FRAMES_FOLDER = "static/frames"

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["FRAMES_FOLDER"] = FRAMES_FOLDER

# ---------------------------------------------------------------------

def extract_frames(video_path):
    """
    Extrae frames y devuelve la lista de rutas, fps y tamaño original.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Obtener FPS y dimensiones originales para la reconstrucción
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # Fallback por si no detecta fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    idx = 0
    # Limpiamos frames anteriores (opcional, para esta demo simple)
    # En producción idealmente usarías carpetas únicas por sesión.
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Nombre único para evitar colisiones si hay múltiples usuarios simultáneos
        frame_name = f"frame_{uuid.uuid4().hex[:8]}_{idx}.jpg"
        frame_path = os.path.join(app.config["FRAMES_FOLDER"], frame_name)
        
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        idx += 1

    cap.release()
    return frames, fps, (width, height)

# ---------------------------------------------------------------------

def detect_equal_frames(frame_paths):
    """
    Detecta frames duplicados usando hashing.
    """
    hashes = []
    id_map = []
    unique_ids = {}

    for path in frame_paths:
        img = Image.open(path)
        h = imagehash.average_hash(img)

        if h in unique_ids:
            id_map.append(unique_ids[h])
        else:
            new_id = len(unique_ids)
            unique_ids[h] = new_id
            id_map.append(new_id)
            hashes.append(h)

    return id_map, unique_ids

# ---------------------------------------------------------------------

def process_and_create_video(frame_paths, fps, original_size):
    """
    Aplica filtros (Bilateral) e interpolación (Bilineal) y genera un video.
    """
    output_filename = f"processed_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
    
    # Codec para MP4 (mp4v suele ser compatible, h264 requiere licencias/instalación extra)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, original_size)

    for path in frame_paths:
        img = cv2.imread(path)
        
        if img is None:
            continue

        # 1. Filtro Bilateral (Suavizado que preserva bordes)
        # d=9: Diámetro del vecindario de píxeles
        # sigmaColor=75: Cuánto se mezclan los colores
        # sigmaSpace=75: Cuánto influyen los píxeles lejanos
        img_filtered = cv2.bilateralFilter(img, 9, 75, 75)

        # 2. Interpolación Bilineal
        # Aquí redimensionamos al tamaño original usando INTER_LINEAR.
        # Si quisieras escalar el video, cambiarías 'original_size'.
        img_processed = cv2.resize(img_filtered, original_size, interpolation=cv2.INTER_LINEAR)

        # Escribir frame procesado al video
        out.write(img_processed)

    out.release()
    
    # Devolvemos la ruta relativa para usar en HTML
    return output_filename

# ---------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("video")
        if not file or file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(video_path)

        # 1. Extraer Frames
        frames, fps, size = extract_frames(video_path)
        
        # 2. Procesamiento de vectores (Tu código original)
        id_vector, uid_dict = detect_equal_frames(frames)

        # 3. Procesar Frames (Filtros) y Crear Video Nuevo
        processed_video_name = process_and_create_video(frames, fps, size)

        return render_template(
            "index.html",
            uploaded=True,
            total_frames=len(frames),
            vector=id_vector,
            unique=len(uid_dict),
            download_video=processed_video_name # Pasamos el nombre del video
        )

    return render_template("index.html", uploaded=False)

# ---------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)