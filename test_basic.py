#!/usr/bin/env python3
"""
Test basic untuk validasi bahwa semua fungsi core berjalan tanpa error
"""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test bahwa semua import berjalan"""
    print("Test 1: Import modules...")
    try:
        import app
        print("  ✓ app.py imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error importing app.py: {e}")
        return False

def test_database_init():
    """Test database initialization"""
    print("\nTest 2: Database initialization...")
    try:
        from app import db_connect, db_init
        db_init()
        with db_connect() as conn:
            # Check tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t[0] for t in tables]
            
            required = ['patients', 'queues']
            for t in required:
                if t in table_names:
                    print(f"  ✓ Table '{t}' exists")
                else:
                    print(f"  ✗ Table '{t}' missing")
                    return False
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_cascade_classifiers():
    """Test cascade classifiers loaded"""
    print("\nTest 3: Cascade classifiers...")
    try:
        from app import detectors
        if len(detectors) > 0:
            print(f"  ✓ {len(detectors)} cascade classifier(s) loaded")
            return True
        else:
            print("  ✗ No cascade classifiers loaded")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_lbph_recognizer():
    """Test LBPH recognizer initialized"""
    print("\nTest 4: LBPH recognizer...")
    try:
        from app import recognizer
        print("  ✓ LBPH recognizer initialized")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_directories():
    """Test required directories exist"""
    print("\nTest 5: Required directories...")
    try:
        from app import DATA_DIR, MODEL_DIR
        dirs = {
            'DATA_DIR': DATA_DIR,
            'MODEL_DIR': MODEL_DIR
        }
        all_ok = True
        for name, path in dirs.items():
            if os.path.exists(path):
                print(f"  ✓ {name}: {path}")
            else:
                print(f"  ✗ {name} not found: {path}")
                all_ok = False
        return all_ok
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_preprocessing_function():
    """Test preprocessing function works"""
    print("\nTest 6: Preprocessing function...")
    try:
        import cv2
        import numpy as np
        from app import preprocess_roi
        
        # Create dummy grayscale image
        dummy_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocess_roi(dummy_img)
        
        if result.shape == (200, 200):
            print(f"  ✓ preprocess_roi works correctly (output: {result.shape})")
            return True
        else:
            print(f"  ✗ preprocess_roi output wrong shape: {result.shape}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_file_naming():
    """Test new file naming format"""
    print("\nTest 7: File naming format...")
    try:
        from app import list_existing_samples
        
        # Test with a dummy NIK (16 digits)
        TEST_NIK = 1234567890123456
        count = list_existing_samples(TEST_NIK)
        print(f"  ✓ list_existing_samples works (found: {count})")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("BASIC FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_database_init,
        test_cascade_classifiers,
        test_lbph_recognizer,
        test_directories,
        test_preprocessing_function,
        test_file_naming
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"❌ {total - passed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
