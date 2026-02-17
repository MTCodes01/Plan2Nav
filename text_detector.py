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
        
        # Check if Tesseract is available
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

    def detect_text(self, image: np.ndarray) -> List[DetectedText]:
        """
        Detect text in the given image.
        
        Args:
            image: Input image (BGR or Grayscale)
            
        Returns:
            List of DetectedText objects
        """
        if not self.enabled or image is None:
            return []
            
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
