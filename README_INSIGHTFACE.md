# Face Recognition System - InsightFace + ArcFace

Sistem deteksi dan pengenalan wajah dengan akurasi tinggi menggunakan InsightFace (RetinaFace + ArcFace).

## 🚀 Fitur Utama

- **Deteksi Wajah Akurat**: Menggunakan RetinaFace untuk deteksi wajah real-time dengan akurasi tinggi
- **Pengenalan Wajah Modern**: Menggunakan ArcFace embedding (512 dimensi) untuk pengenalan identitas
- **Multi-Frame Voting**: Meningkatkan akurasi dengan analisis multiple frame
- **Face Alignment**: Normalisasi posisi wajah untuk hasil optimal
- **Auto-Threshold Tuning**: Penyesuaian threshold otomatis berdasarkan distribusi embedding
- **Fallback ke LBPH**: Jika InsightFace tidak tersedia, sistem otomatis menggunakan LBPH

## 📁 Struktur Folder

```
web-face/
├── app.py                    # Aplikasi Flask utama
├── face_engine.py            # Engine deteksi dan pengenalan wajah
├── requirements.txt          # Dependensi Python
├── database.db               # Database SQLite untuk data pasien
├── data/
│   └── database_wajah/       # Folder penyimpanan gambar wajah (LBPH)
├── model/
│   ├── embeddings.db         # Database embedding (InsightFace)
│   ├── Trainer.yml           # Model LBPH (fallback)
│   └── buffalo_l/            # Model InsightFace (auto-download)
├── templates/
│   ├── user.html             # Halaman user (registrasi & verifikasi)
│   ├── admin_login.html      # Halaman login admin
│   └── admin_dashboard.html  # Dashboard admin
├── static/
│   └── js/
│       ├── user.js           # JavaScript untuk halaman user
│       └── admin.js          # JavaScript untuk halaman admin
└── README_INSIGHTFACE.md     # Dokumentasi ini
```

## 🛠️ Instalasi

### Persyaratan Sistem
- Python 3.8+
- pip
- Webcam (untuk registrasi dan verifikasi)

### Langkah Instalasi

1. **Clone repository**
```bash
git clone https://github.com/lustresense/web-face.git
cd web-face
```

2. **Buat virtual environment**
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Jalankan aplikasi**
```bash
python app.py
```

5. **Akses aplikasi**
- User: http://127.0.0.1:5000/
- Admin: http://127.0.0.1:5000/admin/login
  - Username: `admin`
  - Password: `Cakra@123`

## ⚙️ Konfigurasi

### Environment Variables

| Variable | Default | Deskripsi |
|----------|---------|-----------|
| `USE_INSIGHTFACE` | `1` | Set ke `0` untuk paksa gunakan LBPH |
| `DETECTION_THRESHOLD` | `0.5` | Threshold deteksi wajah (0-1) |
| `RECOGNITION_THRESHOLD` | `0.4` | Threshold similarity untuk match (0-1) |
| `MIN_FACE_SIZE` | `60` | Ukuran minimum wajah dalam pixel |
| `VOTE_MIN_SHARE` | `0.35` | Minimum vote share untuk recognize |
| `MIN_VALID_FRAMES` | `2` | Minimum frame valid untuk recognize |
| `SECRET_KEY` | `dev-secret-key` | Secret key Flask |
| `ADMIN_USERNAME` | `admin` | Username admin |
| `ADMIN_PASSWORD_PLAIN` | `Cakra@123` | Password admin |

### Contoh penggunaan:
```bash
export USE_INSIGHTFACE=1
export RECOGNITION_THRESHOLD=0.45
python app.py
```

## 🔄 Alur Kerja (Pipeline)

### 1. Registrasi Wajah
```
Input Webcam → Deteksi Wajah (RetinaFace) → Face Alignment → 
Extract Embedding (ArcFace) → Normalize (L2) → Simpan ke Database
```

### 2. Verifikasi/Pengenalan
```
Input Webcam → Deteksi Wajah (RetinaFace) → Face Alignment →
Extract Embedding (ArcFace) → Normalize (L2) → 
Compare dengan Database (Cosine Similarity) → Multi-Frame Voting → Output Identitas
```

## 📊 Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  [Webcam/Video] ──► [Frame Capture] ──► [BGR Image]             │
└────────────────────────────────────┬────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────┐
│                      DETECTION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  [RetinaFace] ──► [Face BBox] + [5 Landmarks] + [Det Score]     │
│       │                                                         │
│  [Fallback: Haar Cascade] (jika InsightFace tidak tersedia)     │
└────────────────────────────────────┬────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────┐
│                      ALIGNMENT LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  [5-Point Landmarks] ──► [Affine Transform] ──► [Aligned Face]  │
│  (112x112 normalized)                                           │
└────────────────────────────────────┬────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────┐
│                      EMBEDDING LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  [ArcFace Model] ──► [512-dim Embedding] ──► [L2 Normalize]     │
│       │                                                         │
│  [Fallback: LBPH] (jika InsightFace tidak tersedia)             │
└────────────────────────────────────┬────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────┐
│                      MATCHING LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  [Query Embedding] vs [Database Embeddings]                     │
│       │                                                         │
│  [Cosine Similarity] ──► [Threshold Check] ──► [Match/No Match] │
└────────────────────────────────────┬────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────┐
│                       VOTING LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  [Multi-Frame Results] ──► [Vote Counting] ──► [Best Match]     │
│       │                                                         │
│  Early Stop: jika confidence tinggi, stop lebih awal            │
└────────────────────────────────────┬────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────┐
│                       OUTPUT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  [NIK] + [Name] + [Confidence %] + [Similarity Score]           │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Threshold Optimal

### InsightFace (ArcFace)
| Skenario | Threshold | Deskripsi |
|----------|-----------|-----------|
| High Security | 0.50 | Sangat ketat, FAR rendah |
| **Balanced (Default)** | **0.40** | Seimbang antara akurasi dan usability |
| High Usability | 0.35 | Lebih toleran, FRR rendah |

### LBPH (Fallback)
| Skenario | Threshold | Deskripsi |
|----------|-----------|-----------|
| High Security | 80 | Sangat ketat |
| **Balanced (Default)** | **120** | Seimbang |
| High Usability | 150 | Lebih toleran |

## 📈 Tips Meningkatkan Akurasi

1. **Lighting**: Pastikan pencahayaan merata dan cukup terang
2. **Face Position**: Posisikan wajah di tengah frame dan menghadap langsung ke kamera
3. **Multiple Samples**: Saat registrasi, pastikan minimal 5-10 frame berkualitas
4. **Quality Check**: Hindari gambar blur, tertutup, atau terlalu gelap
5. **Database Size**: Semakin banyak embedding per orang, semakin akurat
6. **Threshold Tuning**: Sesuaikan threshold berdasarkan kebutuhan keamanan

## 🧪 Testing

### Test Basic
```bash
python test_basic.py
```

### Test Recognition Workflow
```bash
python test_recognition_workflow.py
```

### Test dengan Curl
```bash
# Register
curl -X POST http://localhost:5000/api/register \
  -F "nik=1234567890123456" \
  -F "name=John Doe" \
  -F "dob=2000-01-01" \
  -F "address=Jakarta" \
  -F "frames[]=@face1.jpg" \
  -F "frames[]=@face2.jpg"

# Recognize
curl -X POST http://localhost:5000/api/recognize \
  -F "frames[]=@test_face.jpg"

# Engine Status
curl http://localhost:5000/api/engine/status
```

## 📚 API Reference

### POST /api/register
Registrasi pasien baru dengan foto wajah.

**Request (multipart/form-data):**
- `nik`: NIK 16 digit (required)
- `name`: Nama lengkap (required)
- `dob`: Tanggal lahir YYYY-MM-DD (required)
- `address`: Alamat (required)
- `frames[]`: File gambar wajah (multiple)

**Response:**
```json
{
  "ok": true,
  "msg": "Registrasi OK (InsightFace). 10 embedding berhasil disimpan."
}
```

### POST /api/recognize
Verifikasi wajah dari frame.

**Request (multipart/form-data):**
- `frames[]`: File gambar wajah (multiple)

**Response:**
```json
{
  "ok": true,
  "found": true,
  "nik": 1234567890123456,
  "name": "John Doe",
  "dob": "2000-01-01",
  "address": "Jakarta",
  "age": "24 Tahun",
  "confidence": 85,
  "engine": "insightface",
  "similarity": 0.85
}
```

### GET /api/engine/status
Mendapatkan status engine pengenalan wajah.

**Response:**
```json
{
  "ok": true,
  "status": {
    "engine": "insightface",
    "model_loaded": true,
    "insightface_available": true,
    "embeddings_loaded": true,
    "total_embeddings": 50,
    "unique_niks": 5,
    "recognition_threshold": 0.4,
    "detection_threshold": 0.5
  }
}
```

## 🔒 Keamanan

- Password admin di-hash menggunakan Werkzeug security
- Session management dengan Flask session
- Input validation untuk semua endpoint
- No sensitive data in logs

## 📝 Changelog

### v2.0.0 (Current)
- ✅ Migrasi dari LBPH ke InsightFace (RetinaFace + ArcFace)
- ✅ Face alignment dengan 5-point landmarks
- ✅ L2 normalization untuk embedding
- ✅ Cosine similarity matching
- ✅ Multi-frame voting dengan early stop
- ✅ Auto-threshold suggestion
- ✅ SQLite embedding storage
- ✅ Fallback ke LBPH jika InsightFace tidak tersedia
- ✅ API untuk engine status

### v1.0.0 (Legacy)
- Haar Cascade untuk deteksi
- LBPH untuk pengenalan
- Basic voting mechanism

## 🤝 Kontribusi

1. Fork repository
2. Buat branch fitur (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📄 Lisensi

Internal / Sesuai kebutuhan proyek.

## 🆘 Troubleshooting

### InsightFace tidak terinstall
```bash
pip install insightface onnxruntime
```

### Model tidak terdownload
Model InsightFace akan otomatis download saat pertama kali dijalankan. Pastikan koneksi internet tersedia.

### Error "No module named 'cv2'"
```bash
pip install opencv-contrib-python
```

### LBPH fallback tidak bekerja
```bash
pip install opencv-contrib-python
```

### Database error
```bash
rm database.db
rm model/embeddings.db
python app.py  # Will recreate databases
```
