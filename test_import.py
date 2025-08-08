#!/usr/bin/env python3
"""Test if dependencies work"""

print("Testing imports...")

try:
    from PIL import Image
    print("✓ Pillow imported successfully")
except ImportError as e:
    print(f"✗ Pillow import failed: {e}")

try:
    import numpy as np
    print(f"✓ NumPy imported successfully (version {np.__version__})")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    from sklearn.cluster import KMeans
    print("✓ scikit-learn imported successfully")
except Exception as e:
    print(f"✗ scikit-learn import failed: {e}")
    print("\nTrying alternative: using only NumPy for simple quantization...")
    
print("\nNote: The script can work without scikit-learn using simpler color quantization methods.")