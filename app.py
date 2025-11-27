"""
Face Detection and Recognition Web Application
Menggunakan InsightFace (RetinaFace + ArcFace) untuk akurasi tinggi.
Fallback ke LBPH jika InsightFace tidak tersedia.
"""

import os
import glob
import sqlite3
import threading
import logging
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash, session
)
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== PATH CONFIG ======
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "database_wajah")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(MODEL_DIR, "Trainer.yml")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ====== FLASK APP ======
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# ====== FACE ENGINE SELECTION ======
# Try to use InsightFace first, fallback to LBPH
USE_INSIGHTFACE = os.environ.get("USE_INSIGHTFACE", "1") == "1"
FACE_ENGINE = None

try:
    if USE_INSIGHTFACE:
        import face_engine
        face_engine.initialize()
        FACE_ENGINE = "insightface"
        logger.info("Using InsightFace engine for face recognition")
except ImportError as e:
    logger.warning(f"InsightFace not available, falling back to LBPH: {e}")
    FACE_ENGINE = "lbph"
except Exception as e:
    logger.warning(f"Failed to initialize InsightFace, falling back to LBPH: {e}")
    FACE_ENGINE = "lbph"

if FACE_ENGINE is None:
    FACE_ENGINE = "lbph"

# ====== ADMIN CREDENTIALS ======
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
_default_plain = os.environ.get("ADMIN_PASSWORD_PLAIN", "Cakra@123")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", generate_password_hash(_default_plain))

def login_required(view_func):
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper

# ====== OpenCV SETUP (for LBPH fallback) ======
def get_cascade_path(fname="haarcascade_frontalface_default.xml"):
    try:
        return os.path.join(cv2.data.haarcascades, fname)
    except Exception:
        return fname

CASCADE_FILE_MAIN = get_cascade_path("haarcascade_frontalface_default.xml")
CASCADE_FILE_ALT2 = get_cascade_path("haarcascade_frontalface_alt2.xml")

# Only check cascade if we might need LBPH fallback
detectors = []
if os.path.isfile(CASCADE_FILE_MAIN):
    det_main = cv2.CascadeClassifier(CASCADE_FILE_MAIN)
    if not det_main.empty():
        detectors.append(det_main)

if os.path.isfile(CASCADE_FILE_ALT2):
    det_alt2 = cv2.CascadeClassifier(CASCADE_FILE_ALT2)
    if not det_alt2.empty():
        detectors.append(det_alt2)

# LBPH recognizer (for fallback)
recognizer = None
if hasattr(cv2, "face"):
    recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8)

model_loaded = False
model_lock = threading.Lock()

# Threshold & voting
LBPH_CONF_THRESHOLD = float(os.environ.get("LBPH_CONF_THRESHOLD", "120"))
VOTE_MIN_SHARE = float(os.environ.get("VOTE_MIN_SHARE", "0.35"))
MIN_VALID_FRAMES = int(os.environ.get("MIN_VALID_FRAMES", "2"))
EARLY_VOTES_REQUIRED = int(os.environ.get("EARLY_VOTES_REQUIRED", "4"))
EARLY_CONF_THRESHOLD = float(os.environ.get("EARLY_CONF_THRESHOLD", "80"))

# ====== DB ======
def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def db_init():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                nik INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                dob TEXT NOT NULL,     -- bebas format dari form (YYYY-MM-DD / DD-MM-YYYY)
                address TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # queue table agar admin dashboard tidak error
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queues(
                poli_name TEXT PRIMARY KEY,
                next_number INTEGER NOT NULL
            )
        """)
        c = conn.execute("SELECT COUNT(*) AS c FROM queues").fetchone()
        if c["c"] == 0:
            for poli in ["Poli Umum", "Poli Gigi", "IGD"]:
                conn.execute("INSERT INTO queues(poli_name, next_number) VALUES(?, ?)", (poli, 0))
        conn.commit()
db_init()

# ====== UTIL ======
def parse_date_flexible(dob_str: str):
    if not dob_str:
        return None
    dob_str = dob_str.strip()
    formats = [
        "%Y-%m-%d", "%d-%m-%Y",
        "%Y/%m/%d", "%d/%m/%Y",
        "%Y.%m.%d", "%d.%m.%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dob_str, fmt)
        except Exception:
            continue
    return None

def calculate_age(dob_str: str) -> str:
    try:
        dt = parse_date_flexible(dob_str)
        if not dt:
            return "N/A"
        today = datetime.now()
        age = today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
        return f"{age} Tahun"
    except Exception:
        return "N/A"

def list_existing_samples(nik: int) -> int:
    # Format baru: nik.index.jpg (tanpa name di depan)
    return len(glob.glob(os.path.join(DATA_DIR, f"{nik}.*.jpg")))

def bytes_to_bgr(image_bytes: bytes):
    np_data = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_COLOR)

def is_blurry(gray_roi, thr: float = 80.0) -> bool:
    fm = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    return fm < thr

def preprocess_roi(gray_roi):
    roi = cv2.resize(gray_roi, (200, 200), interpolation=cv2.INTER_CUBIC)
    roi = cv2.equalizeHist(roi)
    return roi

def center_fallback_crop(gray):
    h, w = gray.shape[:2]
    side = max(60, int(min(h, w) * 0.6))
    cx, cy = w // 2, h // 2
    x = max(0, cx - side // 2)
    y = max(0, cy - side // 2)
    if x + side > w: x = w - side
    if y + side > h: y = h - side
    x = max(0, x); y = max(0, y)
    return gray[y:y+side, x:x+side]

def detect_largest_face(gray):
    """
    Multi-cascade: coba beberapa classifier, pilih wajah terbesar.
    Return: (roi_gray, (x,y,w,h)) atau (None, None)
    """
    best_roi, best_rect, best_area = None, None, -1
    for det in detectors:
        faces = det.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
        )
        if len(faces) == 0:
            continue
        (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
        area = w * h
        if area > best_area:
            best_area = area
            best_rect = (x, y, w, h)
            best_roi = gray[y:y+h, x:x+w]
    if best_roi is None:
        return None, None
    return best_roi, best_rect

def save_face_images_from_frame(img_bgr, name: str, nik: int, idx: int) -> int:
    """
    Simpan 1 gambar dengan validasi ketat dan preprocessing konsisten:
    - Wajah HARUS terdeteksi.
    - Wajah TIDAK BOLEH buram.
    - Gambar di-preprocess dengan cara yang sama seperti saat recognize.
    - Format nama file: nik.index.jpg (tanpa name untuk konsistensi)
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0

    crop, rect = detect_largest_face(gray)

    # 1. Wajib ada wajah
    if crop is None:
        return 0

    # 2. Wajib tidak terlalu buram (threshold lebih rendah agar lebih banyak frame lolos)
    if is_blurry(crop, thr=40.0):  # Turunkan dari 50 ke 40
        return 0

    # 3. Preprocess dengan cara yang sama seperti saat recognize (PENTING!)
    preprocessed = preprocess_roi(crop)

    # 4. Format nama file baru: nik.index.jpg (tanpa name untuk menghindari inkonsistensi)
    out_path = os.path.join(DATA_DIR, f"{nik}.{idx}.jpg")
    cv2.imwrite(out_path, preprocessed)
    return 1

def augment_img(img):
    """Augment grayscale numpy img: flip/bright/rotate small angle."""
    out = img.copy()
    # Don't equalizeHist again since already preprocessed at save time
    out = cv2.convertScaleAbs(out, alpha=1.05, beta=5)
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 3, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return out

def ensure_min_samples(nik: int, min_count: int = 20) -> int:
    """
    Pastikan minimal min_count file untuk NIK tersebut.
    Jika kurang, buat hasil augmentasi dari file yang ada.
    Format nama file: nik.index.jpg
    """
    pattern = os.path.join(DATA_DIR, f"{nik}.*.jpg")
    files = sorted(glob.glob(pattern), key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split(".")[1]))
    saved = len(files)
    if saved == 0:
        return 0

    next_idx = int(os.path.splitext(os.path.basename(files[-1]))[0].split(".")[1]) + 1
    added = 0
    src_imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in files if os.path.isfile(p)]
    src_imgs = [im for im in src_imgs if im is not None and im.size > 0]
    if not src_imgs:
        return 0

    i = 0
    while saved + added < min_count:
        base = src_imgs[i % len(src_imgs)]
        aug = augment_img(base)
        out_path = os.path.join(DATA_DIR, f"{nik}.{next_idx}.jpg")
        cv2.imwrite(out_path, aug)
        next_idx += 1
        added += 1
        i += 1
    return added

def get_images_and_labels():
    """
    Load semua gambar training dan NIK-nya.
    Format nama file: nik.index.jpg
    Gambar sudah ter-preprocess saat disimpan, jadi tidak perlu preprocess lagi.
    """
    faces, ids = [], []
    nik_counts = {}
    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith(".jpg"):
            continue
        fpath = os.path.join(DATA_DIR, fname)
        try:
            pil = Image.open(fpath).convert("L")
            img_np = np.array(pil, "uint8")
            parts = fname.split(".")  # Format baru: nik.index.jpg
            if len(parts) < 3:  # Minimal: nik.index.jpg
                print(f"Skip file dengan format salah: {fname}")
                continue
            
            # Ambil NIK dari parts[0] (format baru)
            nik = int(parts[0])
            
            # Gambar sudah ter-preprocess saat save, tidak perlu preprocess lagi
            faces.append(img_np)
            ids.append(nik)
            nik_counts[nik] = nik_counts.get(nik, 0) + 1
        except Exception as e:
            print("Skip:", fpath, e)
    
    if faces:
        print(f"[TRAINING] Loaded {len(faces)} images for {len(nik_counts)} unique NIKs from {DATA_DIR}")
        # Print first 10 NIKs to avoid excessive console output
        for i, (nik, count) in enumerate(sorted(nik_counts.items())):
            if i < 10:
                print(f"  - NIK {nik}: {count} images")
            elif i == 10:
                print(f"  - ... and {len(nik_counts) - 10} more NIKs")
    return faces, ids

def train_model_blocking():
    faces, ids = get_images_and_labels()
    if not faces:
        print("[TRAINING] No training data found!")
        return False, "Tidak ada data untuk training!"
    try:
        print(f"[TRAINING] Starting training with {len(faces)} images...")
        recognizer.train(faces, np.array(ids))
        recognizer.save(MODEL_PATH)
        print(f"[TRAINING] Model saved to {MODEL_PATH}")
        return True, "Training selesai."
    except Exception as e:
        print(f"[TRAINING] Error: {e}")
        return False, f"Error training: {e}"

def load_model_if_exists():
    global model_loaded
    if os.path.isfile(MODEL_PATH):
        try:
            recognizer.read(MODEL_PATH)
            model_loaded = True
            print(f"[MODEL] Successfully loaded model from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[MODEL] Failed to load model: {e}")
            model_loaded = False
            return False
    print(f"[MODEL] No model file found at {MODEL_PATH}")
    return False

def retrain_after_change():
    global model_loaded
    with model_lock:
        jpgs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".jpg")]
        if not jpgs:
            if os.path.isfile(MODEL_PATH):
                try:
                    os.remove(MODEL_PATH)
                except Exception as e:
                    print("Gagal hapus model:", e)
            model_loaded = False
            return True, "Semua data dihapus. Model direset."
        ok, msg = train_model_blocking()
        if ok:
            try:
                recognizer.read(MODEL_PATH)
                model_loaded = True
            except Exception as e:
                model_loaded = False
                return False, f"Model tersimpan tetapi gagal dimuat: {e}"
        return ok, msg

# Load model at startup
load_model_if_exists()

# ====== ROUTES (pages tetap) ======
@app.get("/")
def index():
    return render_template("user.html", active_page="home")

@app.get("/user/register")
def user_register():
    return render_template("user.html", active_page="daftar")

@app.get("/user/recognize")
def user_recognize():
    return render_template("user.html", active_page="verif")

@app.get("/admin/login")
def admin_login():
    return render_template("admin_login.html")

@app.post("/admin/login")
def admin_login_post():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
        session["admin_logged_in"] = True
        session["admin_name"] = username
        return redirect(url_for("admin_dashboard"))
    flash("Username atau password salah.", "danger")
    return redirect(url_for("admin_login"))

@app.get("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))

@app.get("/admin")
@login_required
def admin_dashboard():
    with db_connect() as conn:
        rows = conn.execute("SELECT nik, name, dob, address, created_at FROM patients ORDER BY created_at DESC").fetchall()
        queues = conn.execute("SELECT poli_name, next_number FROM queues").fetchall()
    
    # Get data counts
    data_count = len([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".jpg")])
    
    # Get engine status
    engine_info = {
        'name': FACE_ENGINE.upper(),
        'model_loaded': model_loaded
    }
    
    if FACE_ENGINE == "insightface":
        try:
            status = face_engine.get_engine_status()
            engine_info['embeddings_count'] = status.get('total_embeddings', 0)
            engine_info['model_loaded'] = status.get('insightface_available', False)
        except Exception:
            pass
    
    return render_template(
        "admin_dashboard.html",
        patients=rows,
        model_loaded=engine_info['model_loaded'],
        model_name=engine_info['name'],
        foto_count=data_count,
        total_patients=len(rows),
        queues=queues,
        admin_name=session.get("admin_name", "Admin"),
        face_engine=FACE_ENGINE
    )

# ====== API: ENGINE STATUS ======
@app.get("/api/engine/status")
def api_engine_status():
    """Get face recognition engine status"""
    status = {
        'engine': FACE_ENGINE,
        'model_loaded': model_loaded
    }
    
    if FACE_ENGINE == "insightface":
        try:
            status.update(face_engine.get_engine_status())
        except Exception as e:
            status['error'] = str(e)
    
    return jsonify(ok=True, status=status)

# ====== API: PATIENTS (READ) untuk tabel admin ======
@app.get("/api/patients")
def api_patients():
    with db_connect() as conn:
        rows = conn.execute("""
            SELECT nik, name, dob, address, created_at
            FROM patients
            ORDER BY created_at DESC
        """).fetchall()
    out = []
    for r in rows:
        out.append({
            "nik": r["nik"],
            "name": r["name"],
            "dob": r["dob"],
            "address": r["address"],
            "created_at": r["created_at"],
            "age": calculate_age(r["dob"])
        })
    return jsonify(ok=True, patients=out)

@app.get("/api/patient/<int:nik>")
def api_patient_detail(nik: int):
    with db_connect() as conn:
        r = conn.execute("""
            SELECT nik, name, dob, address, created_at
            FROM patients WHERE nik = ?
        """, (nik,)).fetchone()
    if not r:
        return jsonify(ok=False, msg="Pasien tidak ditemukan."), 404
    return jsonify(ok=True, patient={
        "nik": r["nik"],
        "name": r["name"],
        "dob": r["dob"],
        "address": r["address"],
        "created_at": r["created_at"],
        "age": calculate_age(r["dob"])
    })

# ====== API: REGISTER ======
@app.post("/api/register")
def api_register():
    """
    Register a new patient with face data.
    Uses InsightFace for high-accuracy face embedding when available.
    """
    nik_str = request.form.get("nik", "").strip()
    name = (request.form.get("nama") or request.form.get("name") or "").strip()
    dob = (request.form.get("ttl") or request.form.get("dob") or "").strip()
    address = (request.form.get("alamat") or request.form.get("address") or "").strip()

    files = request.files.getlist("files[]")
    if not files:
        files = request.files.getlist("frames[]")

    if not (nik_str and name and dob and address):
        return jsonify(ok=False, msg="Semua field wajib diisi."), 400
    try:
        nik = int(nik_str)
    except ValueError:
        return jsonify(ok=False, msg="NIK harus angka."), 400
    if not files:
        return jsonify(ok=False, msg="Tidak ada gambar dari webcam."), 400

    now_iso = datetime.now().isoformat(timespec="seconds")
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO patients(nik, name, dob, address, created_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(nik) DO UPDATE SET name=excluded.name, dob=excluded.dob, address=excluded.address
        """, (nik, name, dob, address, now_iso))
        conn.commit()

    # Convert uploaded files to BGR images
    frames = []
    for f in files:
        try:
            img = bytes_to_bgr(f.read())
            if img is not None:
                frames.append(img)
        except Exception as e:
            logger.warning(f"Failed to process frame: {e}")

    if not frames:
        with db_connect() as conn:
            conn.execute("DELETE FROM patients WHERE nik = ?", (nik,))
            conn.commit()
        return jsonify(ok=False, msg="Tidak ada frame yang valid."), 400

    # Use InsightFace engine if available
    if FACE_ENGINE == "insightface":
        try:
            enrolled, msg = face_engine.enroll_multiple_frames(frames, nik, min_embeddings=5)
            if enrolled == 0:
                with db_connect() as conn:
                    conn.execute("DELETE FROM patients WHERE nik = ?", (nik,))
                    conn.commit()
                logger.warning(f"[REGISTER] InsightFace failed for NIK {nik}: {msg}")
                return jsonify(ok=False, msg=f"Registrasi gagal: {msg}"), 400
            
            logger.info(f"[REGISTER] InsightFace success for NIK {nik}: {enrolled} embeddings")
            return jsonify(ok=True, msg=f"Registrasi OK (InsightFace). {enrolled} embedding berhasil disimpan.")
        except Exception as e:
            logger.error(f"[REGISTER] InsightFace error: {e}")
            # Fall through to LBPH fallback

    # LBPH fallback
    existing = list_existing_samples(nik)
    next_idx = existing + 1
    saved_total = 0
    
    for img in frames:
        try:
            saved = save_face_images_from_frame(img, name, nik, next_idx + saved_total)
            saved_total += saved
            if saved_total >= 20:
                break
        except Exception as e:
            logger.warning(f"Failed to save frame: {e}")

    # Ensure minimum samples with augmentation
    if saved_total > 0 and saved_total < 20:
        try:
            added = ensure_min_samples(nik, 20)
            saved_total += added
        except Exception as e:
            logger.warning(f"Pad samples error: {e}")

    if saved_total == 0:
        with db_connect() as conn:
            conn.execute("DELETE FROM patients WHERE nik = ?", (nik,))
            conn.commit()
        logger.warning(f"[REGISTER] LBPH failed for NIK {nik}: No valid frames")
        return jsonify(ok=False, msg="Registrasi gagal: Tidak ada frame yang lolos validasi."), 400

    logger.info(f"[REGISTER] LBPH success for NIK {nik}: {saved_total} frames")
    ok, msg = retrain_after_change()
    return jsonify(ok=True, msg=f"Registrasi OK (LBPH). {saved_total} frame disimpan. {msg}")


# ====== API: RECOGNIZE ======
@app.post("/api/recognize")
def api_recognize():
    """
    Recognize a face from uploaded frames.
    Uses InsightFace for high-accuracy recognition when available.
    """
    files = request.files.getlist("files[]")
    if not files:
        files = request.files.getlist("frames[]")

    if not files:
        return jsonify(ok=False, msg="Tidak ada gambar yang dikirim."), 400

    # Convert uploaded files to BGR images
    frames = []
    for f in files:
        try:
            img = bytes_to_bgr(f.read())
            if img is not None:
                frames.append(img)
        except Exception:
            pass

    if not frames:
        return jsonify(ok=True, found=False, msg="Tidak ada frame yang valid.")

    # Use InsightFace engine if available
    if FACE_ENGINE == "insightface":
        try:
            result = face_engine.recognize_face_multi_frame(frames)
            if result is None:
                logger.info("[RECOGNIZE] InsightFace: No match found")
                return jsonify(ok=True, found=False, msg="Tidak dikenali.")
            
            nik = result['nik']
            with db_connect() as conn:
                row = conn.execute(
                    "SELECT nik, name, dob, address FROM patients WHERE nik = ?",
                    (nik,)
                ).fetchone()
            
            if not row:
                return jsonify(ok=True, found=False, msg="NIK tidak ditemukan di database.")
            
            age = calculate_age(row["dob"])
            confidence = result.get('confidence', int(result['similarity'] * 100))
            
            logger.info(f"[RECOGNIZE] InsightFace success: NIK={nik}, sim={result['similarity']:.3f}")
            return jsonify(
                ok=True, found=True,
                nik=row["nik"], name=row["name"], dob=row["dob"], address=row["address"],
                age=age, confidence=confidence,
                engine="insightface",
                similarity=result['similarity']
            )
        except Exception as e:
            logger.error(f"[RECOGNIZE] InsightFace error: {e}")
            # Fall through to LBPH fallback

    # LBPH fallback
    if not model_loaded or not os.path.isfile(MODEL_PATH):
        return jsonify(ok=False, msg="Model belum tersedia. Silakan register dulu."), 400

    if recognizer is None:
        return jsonify(ok=False, msg="LBPH recognizer tidak tersedia."), 500

    from collections import defaultdict, Counter
    votes = defaultdict(list)
    processed = 0
    best_nik, best_avg = None, 99999.0

    for img in frames:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi_raw, rect = detect_largest_face(gray)
            
            if roi_raw is None or is_blurry(roi_raw, 25.0):
                continue

            roi = preprocess_roi(roi_raw)
            Id_pred, conf = recognizer.predict(roi)
            votes[int(Id_pred)].append(float(conf))
            processed += 1

            # Early stop check
            for nk, cfs in votes.items():
                avg = sum(cfs) / len(cfs)
                if avg < best_avg:
                    best_avg = avg
                    best_nik = nk
            if best_nik is not None:
                share = len(votes[best_nik]) / max(1, processed)
                if share >= VOTE_MIN_SHARE and len(votes[best_nik]) >= EARLY_VOTES_REQUIRED and best_avg <= EARLY_CONF_THRESHOLD:
                    break

        except Exception as e:
            logger.warning(f"LBPH predict error: {e}")

    if processed == 0 or not votes:
        return jsonify(ok=True, found=False, msg="Tidak ada wajah terdeteksi.")

    all_preds = [(nk, c) for nk, lst in votes.items() for c in lst]
    major = Counter([nk for nk, _ in all_preds]).most_common(1)[0][0]
    confs_for_major = votes.get(major, [])
    
    if not confs_for_major:
        return jsonify(ok=True, found=False, msg="Tidak dikenali.")

    vote_share = len(confs_for_major) / processed
    median_conf = float(np.median(confs_for_major))

    if vote_share < VOTE_MIN_SHARE or len(confs_for_major) < MIN_VALID_FRAMES or median_conf >= LBPH_CONF_THRESHOLD:
        return jsonify(ok=True, found=False, msg="Tidak dikenali.")

    with db_connect() as conn:
        row = conn.execute(
            "SELECT nik, name, dob, address FROM patients WHERE nik = ?",
            (major,)
        ).fetchone()

    if not row:
        return jsonify(ok=True, found=False, msg="Tidak dikenali.")

    confidence_percent = int(max(0, min(100, 100 - median_conf)))
    age = calculate_age(row["dob"])
    
    logger.info(f"[RECOGNIZE] LBPH success: NIK={major}, conf={median_conf:.2f}")
    return jsonify(
        ok=True, found=True,
        nik=row["nik"], name=row["name"], dob=row["dob"], address=row["address"],
        age=age, confidence=confidence_percent,
        engine="lbph"
    )

# ====== API: QUEUE (untuk sinkron Admin <-> User, tidak diubah) ======
@app.post("/api/queue/assign")
def api_queue_assign():
    data = request.json if request.is_json else {}
    poli = (data.get("poli") or "").strip()
    if poli not in ["Poli Umum", "Poli Gigi", "IGD"]:
        return jsonify(ok=False, msg="Poli tidak valid."), 400
    with db_connect() as conn:
        row = conn.execute("SELECT next_number FROM queues WHERE poli_name=?", (poli,)).fetchone()
        if not row:
            return jsonify(ok=False, msg="Poli tidak ditemukan."), 404
        last_number = row["next_number"]
        nomor = last_number + 1  # nomor baru untuk user
        conn.execute("UPDATE queues SET next_number=? WHERE poli_name=?", (nomor, poli))
        conn.commit()
    return jsonify(ok=True, poli=poli, nomor=nomor)

@app.post("/api/queue/set")
@login_required
def api_queue_set():
    data = request.json if request.is_json else {}
    poli = (data.get("poli") or "").strip()
    nomor = data.get("nomor")
    if poli not in ["Poli Umum", "Poli Gigi", "IGD"]:
        return jsonify(ok=False, msg="Poli tidak valid."), 400
    try:
        n = int(nomor)
        if n < 0: raise ValueError
    except:
        return jsonify(ok=False, msg="Nomor harus >= 0."), 400
    with db_connect() as conn:
        conn.execute("UPDATE queues SET next_number=? WHERE poli_name=?", (n, poli))
        conn.commit()
    return jsonify(ok=True, msg=f"Nomor terakhir {poli} di-set ke {n}.")

# ====== ADMIN: RETRAIN / DELETE (tidak diubah) ======
@app.post("/admin/retrain")
@login_required
def admin_retrain():
    ok, msg = retrain_after_change()
    flash(("Retrain sukses." if ok else f"Retrain gagal: {msg}"), "success" if ok else "danger")
    return redirect(url_for("admin_dashboard"))

@app.post("/admin/patient/<int:nik>/delete")
@login_required
def admin_delete_patient(nik: int):
    """Delete patient and their face data"""
    with db_connect() as conn:
        conn.execute("DELETE FROM patients WHERE nik = ?", (nik,))
        conn.commit()
    
    removed = 0
    # Delete image files (LBPH format)
    for path in glob.glob(os.path.join(DATA_DIR, f"{nik}.*.jpg")):
        try:
            os.remove(path)
            removed += 1
        except Exception as e:
            logger.warning(f"Failed to delete file {path}: {e}")
    
    # Delete embeddings (InsightFace format)
    if FACE_ENGINE == "insightface":
        try:
            deleted_emb = face_engine.delete_embeddings_for_nik(nik)
            logger.info(f"Deleted {deleted_emb} embeddings for NIK {nik}")
        except Exception as e:
            logger.warning(f"Failed to delete embeddings: {e}")
    
    ok, msg = retrain_after_change()
    flash(f"Hapus NIK {nik}: {removed} file dihapus. {msg}", "success" if ok else "danger")
    return redirect(url_for("admin_dashboard"))

@app.post("/admin/patient/update")
@login_required
def admin_update_patient():
    """Update patient information"""
    try:
        old_nik_str = request.form.get("old_nik", "").strip()
        nik_str = request.form.get("nik", "").strip()
        dob = request.form.get("dob", "").strip()
        address = request.form.get("address", "").strip()

        if not all([old_nik_str, nik_str, dob, address]):
            return jsonify(ok=False, msg="Semua field wajib diisi."), 400

        old_nik = int(old_nik_str)
        nik = int(nik_str)

        with db_connect() as conn:
            if nik != old_nik and conn.execute("SELECT 1 FROM patients WHERE nik = ?", (nik,)).fetchone():
                return jsonify(ok=False, msg=f"NIK {nik} sudah terdaftar untuk pasien lain."), 409

            conn.execute("""
                UPDATE patients SET nik=?, dob=?, address=? WHERE nik=?
            """, (nik, dob, address, old_nik))
            conn.commit()

        if nik != old_nik:
            renamed_count = 0
            # Update image files (LBPH)
            pattern = os.path.join(DATA_DIR, f"{old_nik}.*.jpg")
            for old_path in glob.glob(pattern):
                fname = os.path.basename(old_path)
                parts = fname.split('.')
                if len(parts) >= 3:
                    new_fname = f"{nik}.{'.'.join(parts[1:])}"
                    new_path = os.path.join(DATA_DIR, new_fname)
                    try:
                        os.rename(old_path, new_path)
                        renamed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to rename {old_path}: {e}")
            
            # Update embeddings (InsightFace)
            if FACE_ENGINE == "insightface":
                try:
                    updated_emb = face_engine.update_nik_in_embeddings(old_nik, nik)
                    logger.info(f"Updated {updated_emb} embeddings from NIK {old_nik} to {nik}")
                except Exception as e:
                    logger.warning(f"Failed to update embeddings: {e}")
            
            ok_retrain, msg_retrain = retrain_after_change()
            msg_rename = f"{renamed_count} file gambar di-rename. {msg_retrain}"
            if not ok_retrain:
                return jsonify(ok=False, msg=f"Data diupdate tapi retrain gagal: {msg_retrain}"), 500
        else:
            msg_rename = "NIK tidak berubah, tidak perlu retrain."

        return jsonify(ok=True, msg=f"Data pasien NIK {old_nik} berhasil diupdate. {msg_rename}")

    except ValueError:
        return jsonify(ok=False, msg="NIK harus berupa angka."), 400
    except Exception as e:
        print(f"Error update patient: {e}")
        return jsonify(ok=False, msg=f"Terjadi error di server: {e}"), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)