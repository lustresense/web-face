#!/usr/bin/env python3
"""
Test basic untuk validasi bahwa semua fungsi core berjalan tanpa error
Mendukung InsightFace dan LBPH fallback
"""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Disable InsightFace auto-init for faster testing
os.environ["FACE_ENGINE_INIT"] = "0"

def test_imports():
    """Test bahwa semua import berjalan"""
    print("Test 1: Import modules...")
    try:
        import app
        print("  ✓ app.py imported successfully")
        print(f"  ✓ Face engine: {app.FACE_ENGINE}")
        return True
    except Exception as e:
        print(f"  ✗ Error importing app.py: {e}")
        return False

def test_face_engine_import():
    """Test face_engine module import"""
    print("\nTest 2: Face engine module...")
    try:
        import face_engine
        print("  ✓ face_engine.py imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error importing face_engine.py: {e}")
        return False

def test_database_init():
    """Test database initialization"""
    print("\nTest 3: Database initialization...")
    try:
        from app import db_connect, db_init
        db_init()
        with db_connect() as conn:
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
    """Test cascade classifiers loaded (for LBPH fallback)"""
    print("\nTest 4: Cascade classifiers...")
    try:
        from app import detectors
        count = len(detectors)
        if count > 0:
            print(f"  ✓ {count} cascade classifier(s) loaded")
        else:
            print(f"  ⚠ No cascade classifiers (OK if InsightFace is primary)")
        return True  # Not a failure if InsightFace is available
    except Exception as e:
        print(f"  ⚠ Warning: {e}")
        return True  # Not critical

def test_recognizer():
    """Test recognizer availability"""
    print("\nTest 5: Recognizer...")
    try:
        from app import recognizer, FACE_ENGINE
        if FACE_ENGINE == "insightface":
            print("  ✓ InsightFace engine selected")
        elif recognizer is not None:
            print("  ✓ LBPH recognizer initialized")
        else:
            print("  ⚠ No recognizer available")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_directories():
    """Test required directories exist"""
    print("\nTest 6: Required directories...")
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
    print("\nTest 7: Preprocessing function...")
    try:
        import cv2
        import numpy as np
        from app import preprocess_roi
        
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
    print("\nTest 8: File naming format...")
    try:
        from app import list_existing_samples
        
        TEST_NIK = 1234567890123456
        count = list_existing_samples(TEST_NIK)
        print(f"  ✓ list_existing_samples works (found: {count})")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_embedding_functions():
    """Test embedding database functions"""
    print("\nTest 9: Embedding functions...")
    try:
        import face_engine
        import numpy as np
        
        # Test embedding normalization (using public API)
        test_emb = np.random.randn(512).astype(np.float32)
        normalized = face_engine.normalize_embedding(test_emb)
        norm = np.linalg.norm(normalized)
        
        if abs(norm - 1.0) < 0.001:
            print(f"  ✓ L2 normalization works (norm: {norm:.4f})")
        else:
            print(f"  ✗ L2 normalization failed (norm: {norm:.4f})")
            return False
        
        # Test cosine similarity (using public API)
        sim = face_engine.cosine_similarity(normalized, normalized)
        if abs(sim - 1.0) < 0.001:
            print(f"  ✓ Cosine similarity works (self-sim: {sim:.4f})")
        else:
            print(f"  ✗ Cosine similarity failed (self-sim: {sim:.4f})")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_flask_routes():
    """Test Flask routes are defined"""
    print("\nTest 10: Flask routes...")
    try:
        from app import app
        
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        required_routes = [
            '/api/register',
            '/api/recognize',
            '/api/patients',
            '/api/engine/status',
            '/admin'
        ]
        
        all_ok = True
        for route in required_routes:
            if route in routes:
                print(f"  ✓ Route '{route}' exists")
            else:
                print(f"  ✗ Route '{route}' missing")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("BASIC FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_face_engine_import,
        test_database_init,
        test_cascade_classifiers,
        test_recognizer,
        test_directories,
        test_preprocessing_function,
        test_file_naming,
        test_embedding_functions,
        test_flask_routes
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
