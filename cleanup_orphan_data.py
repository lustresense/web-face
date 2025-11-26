#!/usr/bin/env python3
"""
Script untuk membersihkan data yang tidak konsisten:
1. Pasien di database tapi tidak ada file gambar training
2. File gambar training tanpa pasien di database
3. Model lama yang tidak valid
"""

import os
import glob
import sqlite3

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "database_wajah")
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "Trainer.yml")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def cleanup():
    print("=" * 60)
    print("PEMBERSIHAN DATA ORPHAN")
    print("=" * 60)
    
    # 1. Ambil semua NIK dari database
    with get_db_connection() as conn:
        rows = conn.execute("SELECT nik, name FROM patients").fetchall()
        db_niks = {row["nik"]: row["name"] for row in rows}
    
    print(f"\n1. Pasien di database: {len(db_niks)}")
    for nik, name in db_niks.items():
        print(f"   - NIK {nik}: {name}")
    
    # 2. Ambil NIK dari file gambar
    if not os.path.exists(DATA_DIR):
        print(f"\nFolder {DATA_DIR} tidak ditemukan!")
        return
    
    files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    file_niks = set()
    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.split(".")
        try:
            # Format baru: nik.index.jpg
            if len(parts) >= 3 and parts[0].isdigit():
                file_niks.add(int(parts[0]))
            # Format lama: name.nik.index.jpg
            elif len(parts) >= 4 and parts[1].isdigit():
                file_niks.add(int(parts[1]))
        except:
            pass
    
    print(f"\n2. NIK dengan file gambar: {len(file_niks)}")
    for nik in sorted(file_niks):
        count = len(glob.glob(os.path.join(DATA_DIR, f"{nik}.*.jpg")))
        if count == 0:
            count = len(glob.glob(os.path.join(DATA_DIR, f"*.{nik}.*.jpg")))
        print(f"   - NIK {nik}: {count} file")
    
    # 3. Temukan pasien tanpa file (orphan di DB)
    orphan_db = set(db_niks.keys()) - file_niks
    if orphan_db:
        print(f"\n3. ORPHAN: Pasien di DB tapi tidak ada file gambar:")
        for nik in sorted(orphan_db):
            print(f"   - NIK {nik}: {db_niks[nik]}")
        
        resp = input("\nHapus pasien orphan dari database? (y/n): ").strip().lower()
        if resp == 'y':
            with get_db_connection() as conn:
                for nik in orphan_db:
                    conn.execute("DELETE FROM patients WHERE nik = ?", (nik,))
                    print(f"   Dihapus: NIK {nik}")
                conn.commit()
            print("   Pasien orphan berhasil dihapus!")
    else:
        print("\n3. Tidak ada pasien orphan di database.")
    
    # 4. Temukan file tanpa pasien (orphan files)
    orphan_files = file_niks - set(db_niks.keys())
    if orphan_files:
        print(f"\n4. ORPHAN: File gambar tanpa pasien di DB:")
        for nik in sorted(orphan_files):
            count = len(glob.glob(os.path.join(DATA_DIR, f"{nik}.*.jpg")))
            if count == 0:
                count = len(glob.glob(os.path.join(DATA_DIR, f"*.{nik}.*.jpg")))
            print(f"   - NIK {nik}: {count} file")
        
        resp = input("\nHapus file orphan? (y/n): ").strip().lower()
        if resp == 'y':
            deleted = 0
            for nik in orphan_files:
                # Format baru
                for fpath in glob.glob(os.path.join(DATA_DIR, f"{nik}.*.jpg")):
                    os.remove(fpath)
                    deleted += 1
                # Format lama
                for fpath in glob.glob(os.path.join(DATA_DIR, f"*.{nik}.*.jpg")):
                    os.remove(fpath)
                    deleted += 1
            print(f"   {deleted} file orphan berhasil dihapus!")
    else:
        print("\n4. Tidak ada file orphan.")
    
    # 5. Cek model
    if os.path.exists(MODEL_PATH):
        print(f"\n5. Model LBPH ditemukan: {MODEL_PATH}")
        if len(file_niks) == 0:
            resp = input("\nTidak ada data training. Hapus model? (y/n): ").strip().lower()
            if resp == 'y':
                os.remove(MODEL_PATH)
                print("   Model dihapus!")
        else:
            print("   Model tetap dipertahankan (ada data training).")
    else:
        print("\n5. Model belum ada.")
    
    print("\n" + "=" * 60)
    print("PEMBERSIHAN SELESAI!")
    print("=" * 60)

if __name__ == "__main__":
    cleanup()
