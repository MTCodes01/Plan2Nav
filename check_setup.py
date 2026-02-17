import sys
import os
import logging

try:
    import cv2
    import numpy as np
    import pytesseract
    print("✅ Libraries imported successfully (cv2, numpy, pytesseract)")
except ImportError as e:
    print(f"❌ Library import failed: {e}")
    sys.exit(1)

# Check Tesseract
print("\nChecking Tesseract OCR...")
try:
    version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract version: {version}")
except Exception as e:
    print(f"❌ Tesseract check failed: {e}")
    print("  Please ensure Tesseract is installed and in your PATH.")
    print("  Win: https://github.com/UB-Mannheim/tesseract/wiki")

# Check imports from our modules
print("\nChecking application modules...")
try:
    from text_detector import TextDetector
    from room_detector import RoomDetector
    from preprocess_architectural import remove_text_from_floorplan
    print("✅ Application modules imported successfully")
except Exception as e:
    print(f"❌ Module import failed: {e}")
    sys.exit(1)

print("\nReady to run: python main.py --config config_architectural.yaml")
