# WEB-FACE (Haar + DeepFace Upgrade)

## Ringkasan
Aplikasi web Flask untuk registrasi dan verifikasi wajah pasien rumah sakit. Deteksi wajah memakai Haar Cascade dari OpenCV, pengenalan wajah memakai DeepFace (model VGG-Face + cosine distance). Setiap pasien hanya menyimpan **1 foto representatif** terbaik di folder `data/database_wajah/`.

## Fitur
- Registrasi pasien: kirim form + batch frame webcam (±10 frame), pilih 1 wajah terbaik, simpan 1 foto.
- Verifikasi wajah: ambil batch frame (±6 frame), pilih 1 foto terbaik, DeepFace.find terhadap database.
- Threshold cosine default: `< 0.40` dianggap match.
- Admin Dashboard:
  - Lihat daftar pasien (sorting + paginasi)
  - Edit / Hapus pasien
  - Statistik total pasien & total foto
  - Manajemen antrian poli (Poli Umum, Poli Gigi, IGD) via tabel `queues`
- API Queue untuk nomor antrian.

## Struktur Direktori
```
WEB-FACE/
├── app.py
├── data/
│   └── database_wajah/    # 1 foto per NIK
├── templates/
│   ├── user.html
│   ├── admin_login.html   # login (tidak diubah di contoh)
│   └── admin_dashboard.html
├── static/
│   └── js/
│       ├── user.js
│       └── admin.js
├── database.db
├── requirements.txt
└── README.md
```

## Instalasi
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Akses:
- User: http://127.0.0.1:5000/
- Admin: http://127.0.0.1:5000/admin/login (default: admin / Cakra@123)

## Catatan Teknis
- Tanggal lahir dikirim dari `<input type="date">` (format ISO YYYY-MM-DD) lalu dikonversi ke `DD-MM-YYYY` di backend.
- DeepFace cache embedding (.pkl) dihapus setiap registrasi/hapus pasien agar embedding segar.
- Jika ingin mengganti threshold: set env `DEEPFACE_THRESHOLD`, misal `export DEEPFACE_THRESHOLD=0.38`.

## API Ringkas
| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| /api/register | POST (multipart) | Registrasi pasien + frames[] |
| /api/recognize | POST (multipart) | Verifikasi wajah (frames[]) |
| /api/patients | GET | List pasien |
| /api/patient/<nik> | GET | Detail pasien |
| /api/patient/<nik> | PUT (JSON) | Edit pasien |
| /api/patient/<nik> | DELETE | Hapus pasien |
| /api/queue/status | GET | Status nomor berikutnya |
| /api/queue/assign | POST (JSON {"poli":"Poli Umum"}) | Ambil nomor antrian |
| /api/queue/set | POST (admin only) | Set/Reset nomor berikutnya |

## Pengembangan
- Jika ingin menambah detektor lain (MediaPipe), cukup ubah fungsi `detect_largest_face`.
- Untuk alignment lanjutan, bisa normalisasi crop wajah sebelum simpan/recognize.

## Lisensi
Internal / Sesuai kebutuhan proyek.