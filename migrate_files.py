#!/usr/bin/env python3
"""
Script untuk migrasi file gambar dari format lama ke format baru.
Format lama: name.nik.index.jpg
Format baru: nik.index.jpg

Juga memastikan semua gambar sudah ter-preprocess dengan benar.
"""

import os
import glob
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "database_wajah")

def preprocess_roi(gray_roi):
    """Sama seperti di app.py"""
    roi = cv2.resize(gray_roi, (200, 200), interpolation=cv2.INTER_CUBIC)
    roi = cv2.equalizeHist(roi)
    return roi

def migrate_files():
    """Migrasi file dari format lama ke format baru"""
    if not os.path.exists(DATA_DIR):
        print(f"Folder {DATA_DIR} tidak ditemukan!")
        return
    
    files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    if not files:
        print("Tidak ada file untuk dimigrasi.")
        return
    
    print(f"Ditemukan {len(files)} file gambar.")
    migrated = 0
    already_new = 0
    preprocessed = 0
    errors = 0
    
    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.split(".")
        
        try:
            # Cek apakah sudah format baru (nik.index.jpg)
            if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
                # Format baru: nik.index.jpg
                already_new += 1
                
                # Cek apakah perlu preprocess ulang
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"  ERROR: Tidak bisa load {fname}")
                    errors += 1
                    continue
                
                # Cek apakah ukurannya sudah 200x200
                if img.shape != (200, 200):
                    print(f"  Preprocess ulang: {fname} (size: {img.shape})")
                    preprocessed_img = preprocess_roi(img)
                    cv2.imwrite(fpath, preprocessed_img)
                    preprocessed += 1
                
            elif len(parts) >= 4:
                # Format lama: name.nik.index.jpg
                name = parts[0]
                nik = parts[1]
                index = parts[2]
                
                if not nik.isdigit() or not index.isdigit():
                    print(f"  SKIP: Format tidak valid {fname}")
                    continue
                
                # Load dan preprocess gambar
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"  ERROR: Tidak bisa load {fname}")
                    errors += 1
                    continue
                
                # Preprocess jika belum
                if img.shape != (200, 200):
                    img = preprocess_roi(img)
                else:
                    # Sudah 200x200, tapi pastikan sudah equalized
                    img = cv2.equalizeHist(img)
                
                # Nama file baru
                new_fname = f"{nik}.{index}.jpg"
                new_fpath = os.path.join(DATA_DIR, new_fname)
                
                # Hindari overwrite jika file baru sudah ada
                if os.path.exists(new_fpath):
                    print(f"  SKIP: {new_fname} sudah ada, tidak overwrite {fname}")
                    continue
                
                # Simpan dengan nama baru
                cv2.imwrite(new_fpath, img)
                
                # Hapus file lama
                os.remove(fpath)
                
                print(f"  Migrasi: {fname} -> {new_fname}")
                migrated += 1
            
            else:
                print(f"  SKIP: Format tidak dikenali {fname}")
        
        except Exception as e:
            print(f"  ERROR processing {fname}: {e}")
            errors += 1
    
    print("\n=== SUMMARY ===")
    print(f"Total file: {len(files)}")
    print(f"Sudah format baru: {already_new}")
    print(f"Dimigrasi: {migrated}")
    print(f"Di-preprocess ulang: {preprocessed}")
    print(f"Error: {errors}")
    print("\nMigrasi selesai!")

if __name__ == "__main__":
    print("=" * 60)
    print("MIGRASI FILE GAMBAR KE FORMAT BARU")
    print("=" * 60)
    migrate_files()
