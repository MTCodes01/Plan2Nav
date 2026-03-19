"""
Text Detector Module

This module handles text detection and recognition from floor plan images using Tesseract OCR.
It identifies room labels and their locations to assist in room classification.
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import platform

# Configure module logger
logger = logging.getLogger(__name__)

@dataclass
class DetectedText:
    """
    Represents a detected text element.
    
    Attributes:
        text: The recognized text string
        confidence: OCR confidence score (0-100)
        bounds: Bounding box (x, y, w, h)
        centroid: Center point (x, y)
    """
    text: str
    confidence: float
    bounds: Tuple[int, int, int, int]
    centroid: Tuple[int, int]


class TextDetector:
    """
    Detects and recognizes text in images using Tesseract OCR.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TextDetector.
        
        Args:
            config: Configuration dictionary containing 'text_detection' settings
        """
        self.config = config.get('text_detection', {})
        self.enabled = self.config.get('enabled', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 40.0)
        
        # Set tesseract command path if provided
        tesseract_cmd = self.config.get('tesseract_cmd')
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Check if Tesseract is available or use Swift/EasyOCR
        self.use_easyocr = False
        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False) # Default to CPU
            self.use_easyocr = True
            self.enabled = True
            logger.info("EasyOCR loaded successfully for text detection")
        except ImportError:
            logger.info("EasyOCR not found. Falling back to platform default.")

        if platform.system() == "Darwin":
            self._compile_swift_ocr()
        
        if not getattr(self, 'use_swift', False) and not self.use_easyocr:
            self._check_tesseract_availability()

    def _compile_swift_ocr(self):
        """Compile the swift script for faster execution on Mac."""
        swift_script = Path(__file__).parent / 'swift_ocr2.swift'
        self.swift_bin = Path(__file__).parent / 'swift_ocr_bin'
        if swift_script.exists():
            try:
                import subprocess
                # Compile to binary if not present
                if not self.swift_bin.exists():
                    logger.info(f"Compiling Swift OCR script: {swift_script}")
                    subprocess.run(['swiftc', str(swift_script), '-o', str(self.swift_bin)], check=True)
                self.use_swift = True
                self.enabled = True
                logger.info("Swift OCR loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to compile Swift OCR script: {e}")
                self.use_swift = False
                self._check_tesseract_availability()
        else:
            self.use_swift = False
            self._check_tesseract_availability()

    def _check_tesseract_availability(self):
        """Check if Tesseract is installed and available."""
        if not self.enabled:
            return

        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available")
        except Exception as e:
            logger.warning(f"Tesseract OCR not found or not working: {e}")
            logger.warning("Text detection will be disabled. Please install Tesseract-OCR.")
            if platform.system() == "Windows":
                logger.warning("On Windows, make sure Tesseract is installed and added to PATH, or configured in config.yaml")
            self.enabled = False

    def detect_text(self, image: np.ndarray, image_path: Optional[str] = None) -> List[DetectedText]:
        """
        Detect text in the given image.
        
        Args:
            image: Input image (BGR or Grayscale)
            image_path: Optional full path to image file (Required for Swift OCR on Mac)
            
        Returns:
            List of DetectedText objects
        """
        if not self.enabled or image is None:
            return []
            
        # Use Swift if enabled and path provided
        if getattr(self, 'use_swift', False) and image_path:
            return self.detect_text_swift(image, image_path)
        elif getattr(self, 'use_easyocr', False):
            return self.detect_text_easyocr(image)

        try:
            # Convert to RGB for Tesseract (if BGR)
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image # Assume grayscale is fine or convert to RGB
                if len(image.shape) == 2:
                     image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Preprocessing for better OCR (optional but recommended for floor plans)
            # 1. Resize if image is too small (Tesseract works better on larger text)
            # 2. Thresholding to isolate text
            
            # Simple preprocessing: Convert to grayscale and threshold
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            # Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Run Tesseract with data output
            # psm 11: Sparse text. Find as much text as possible in no particular order.
            # psm 3: Fully automatic page segmentation, but no OSD. may be better for large labels
            custom_config = r'--oem 3 --psm 11' 
            
            data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)
            
            detected_texts = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = float(data['conf'][i])
                
                # Filter out empty text and low confidence
                if not text or conf < self.confidence_threshold:
                    continue
                    
                # Filter out noise (single characters that are likely noise, unless simple labels like 'A', 'B')
                if len(text) < 2 and text.lower() not in ['a', 'b', 'c', 'd', '1', '2', '3']:
                     continue

                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                centroid = (x + w // 2, y + h // 2)
                
                detected_texts.append(DetectedText(
                    text=text,
                    confidence=conf,
                    bounds=(x, y, w, h),
                    centroid=centroid
                ))
            
            logger.info(f"Detected {len(detected_texts)} text elements")
            return detected_texts

        except Exception as e:
            logger.error(f"Error during text detection: {e}")
            return []

    def detect_text_swift(self, image: np.ndarray, image_path: str) -> List[DetectedText]:
        """Run swift OCR and parse output."""
        if not hasattr(self, 'swift_bin') or not self.swift_bin.exists():
            logger.warning("Swift binary not found or failed compiling")
            return []

        try:
            import subprocess
            logger.info(f"Running Swift OCR on {image_path}")
            result = subprocess.run([str(self.swift_bin), image_path], capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Swift OCR failed: {result.stderr}")
                return []

            detected_texts = []
            height, width = image.shape[:2]

            for line in result.stdout.splitlines():
                if '|' not in line:
                    continue
                parts = line.split('|')
                if len(parts) == 5:
                    text_str, x_str, y_str, w_str, h_str = parts
                    text_str = text_str.strip()
                    if not text_str:
                        continue
                    try:
                        x_norm = float(x_str)
                        y_norm = float(y_str)
                        w_norm = float(w_str)
                        h_norm = float(h_str)

                        # Swift coordinate transform (origin bottom-left, height grows UP)
                        # OpenCV coordinate transform (origin top-left, height grows DOWN)
                        p_x = int(x_norm * width)
                        # Flipped Y math: (1.0 - (y + h)) * size
                        p_y = int((1.0 - (y_norm + h_norm)) * height)
                        p_w = int(w_norm * width)
                        p_h = int(h_norm * height)

                        centroid = (p_x + p_w // 2, p_y + p_h // 2)

                        # Filter Single Characters noise
                        if len(text_str) < 2 and text_str.lower() not in ['a', 'b', 'c', 'd', '1', '2', '3']:
                            continue

                        detected_texts.append(DetectedText(
                            text=text_str,
                            confidence=100.0,
                            bounds=(p_x, p_y, p_w, p_h),
                            centroid=centroid
                        ))
                    except Exception as parse_err:
                        logger.warning(f"Failed to parse OCR line: '{line}': {parse_err}")

            logger.info(f"Swift OCR detected {len(detected_texts)} text items")
            return detected_texts

        except Exception as e:
            logger.error(f"Error executing Swift OCR: {e}")
            return []

    def detect_text_easyocr(self, image: np.ndarray) -> List[DetectedText]:
        """Run EasyOCR and parse output."""
        if not getattr(self, 'use_easyocr', False):
            return []

        try:
            # Convert to RGB (EasyOCR uses RGB)
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            logger.info("Running EasyOCR text detection")
            results = self.easyocr_reader.readtext(image_rgb)
            
            detected_texts = []
            for bbox, text_str, conf in results:
                text_str = text_str.strip()
                if not text_str or conf < (self.confidence_threshold / 100.0): # EasyOCR uses 0.0-1.0
                    continue

                if len(text_str) < 2 and text_str.lower() not in ['a', 'b', 'c', 'd', '1', '2', '3']:
                    continue

                # bbox is a list of 4 points [[x,y],[x,y],[x,y],[x,y]]
                pts = np.array(bbox, dtype=np.int32)
                x_min = int(np.min(pts[:, 0]))
                y_min = int(np.min(pts[:, 1]))
                x_max = int(np.max(pts[:, 0]))
                y_max = int(np.max(pts[:, 1]))
                w = x_max - x_min
                h = y_max - y_min

                centroid = (x_min + w // 2, y_min + h // 2)

                detected_texts.append(DetectedText(
                    text=text_str,
                    confidence=conf * 100.0,
                    bounds=(x_min, y_min, w, h),
                    centroid=centroid
                ))

            logger.info(f"EasyOCR detected {len(detected_texts)} text items")
            return detected_texts

        except Exception as e:
            logger.error(f"Error during EasyOCR detection: {e}")
            return []

    def draw_text_debug(self, image: np.ndarray, detections: List[DetectedText]) -> np.ndarray:
        """
        Draw detected text bounding boxes and labels on an image.
        """
        debug_image = image.copy()
        if len(debug_image.shape) == 2:
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
            
        for dt in detections:
            x, y, w, h = dt.bounds
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(debug_image, dt.text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        return debug_image
