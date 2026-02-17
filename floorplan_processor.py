"""
Floor Plan Processor Module

This module handles image loading, preprocessing, and basic image operations
for floor plan analysis. It provides the FloorPlanProcessor class that manages
the complete image preprocessing pipeline.

Author: Floor Plan to 3D GeoJSON Converter
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

# Configure module logger
logger = logging.getLogger(__name__)


class FloorPlanProcessor:
    """
    Handles floor plan image loading and preprocessing.
    
    This class provides methods for:
    - Loading images from various formats (PNG, JPG)
    - Converting to grayscale
    - Applying thresholding (global or adaptive)
    - Denoising with Gaussian blur
    - Morphological operations for wall enhancement
    
    Attributes:
        config (dict): Configuration parameters for image processing
        original_image (np.ndarray): Original loaded image
        processed_image (np.ndarray): Preprocessed binary image
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FloorPlanProcessor with configuration.
        
        Args:
            config: Dictionary containing image processing parameters
                   Expected keys under 'image_processing':
                   - threshold: Global threshold value (0-255)
                   - adaptive_threshold: Use adaptive thresholding (bool)
                   - blur_kernel_size: Gaussian blur kernel size (odd int)
                   - morph_kernel_size: Morphological kernel size (int)
        """
        self.config = config.get('image_processing', {})
        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.grayscale_image: Optional[np.ndarray] = None
        self._image_path: Optional[Path] = None
        
        # Set default values if not in config
        self._threshold = self.config.get('threshold', 127)
        self._adaptive_threshold = self.config.get('adaptive_threshold', True)
        self._blur_kernel_size = self.config.get('blur_kernel_size', 5)
        self._morph_kernel_size = self.config.get('morph_kernel_size', 3)
        
        logger.debug(f"FloorPlanProcessor initialized with config: {self.config}")
    
    def load_image(self, image_path: str | Path) -> np.ndarray:
        """
        Load an image from the specified path.
        
        Args:
            image_path: Path to the floor plan image (PNG, JPG, JPEG)
            
        Returns:
            np.ndarray: Loaded image in BGR format
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image format is not supported or loading fails
        """
        self._image_path = Path(image_path)
        
        if not self._image_path.exists():
            raise FileNotFoundError(f"Image not found: {self._image_path}")
        
        # Check supported formats
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        if self._image_path.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported image format: {self._image_path.suffix}. "
                f"Supported formats: {supported_formats}"
            )
        
        # Load the image using OpenCV
        self.original_image = cv2.imread(str(self._image_path))
        
        if self.original_image is None:
            raise ValueError(f"Failed to load image: {self._image_path}")
        
        logger.info(
            f"Loaded image: {self._image_path.name} "
            f"(size: {self.original_image.shape[1]}x{self.original_image.shape[0]})"
        )
        
        return self.original_image
    
    def to_grayscale(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Optional input image. If None, uses the loaded original image.
            
        Returns:
            np.ndarray: Grayscale image
            
        Raises:
            ValueError: If no image is available to process
        """
        if image is None:
            image = self.original_image
            
        if image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Check if already grayscale
        if len(image.shape) == 2:
            self.grayscale_image = image.copy()
        else:
            self.grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        logger.debug("Converted image to grayscale")
        return self.grayscale_image
    
    def apply_blur(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise.
        
        The blur kernel size is determined by the configuration.
        Larger kernel sizes result in more smoothing.
        
        Args:
            image: Optional input image. If None, uses grayscale image.
            
        Returns:
            np.ndarray: Blurred image
        """
        if image is None:
            image = self.grayscale_image
            
        if image is None:
            raise ValueError("No grayscale image available. Call to_grayscale() first.")
        
        # Ensure kernel size is odd
        kernel_size = self._blur_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        logger.debug(f"Applied Gaussian blur with kernel size {kernel_size}")
        return blurred
    
    def apply_threshold(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply thresholding to create a binary image.
        
        Uses either adaptive thresholding (better for varying lighting)
        or global thresholding based on configuration.
        
        Args:
            image: Optional input image. If None, uses grayscale image.
            
        Returns:
            np.ndarray: Binary (thresholded) image where walls are white
        """
        if image is None:
            image = self.grayscale_image
            
        if image is None:
            raise ValueError("No grayscale image available.")
        
        if self._adaptive_threshold:
            # Adaptive thresholding adapts to local brightness
            # Good for floor plans with uneven lighting or scanning artifacts
            binary = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,  # Invert so walls become white
                11,  # Block size for local threshold calculation
                2    # Constant subtracted from mean
            )
            logger.debug("Applied adaptive thresholding")
        else:
            # Global thresholding with Otsu's method for automatic threshold
            _, binary = cv2.threshold(
                image,
                self._threshold,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            logger.debug(f"Applied global thresholding (threshold={self._threshold})")
        
        return binary
    
    def apply_morphology(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up the binary image.
        
        Performs closing (dilation followed by erosion) to:
        - Fill small gaps in walls
        - Connect nearby wall segments
        - Remove small noise
        
        Args:
            image: Binary input image
            
        Returns:
            np.ndarray: Cleaned binary image
        """
        # Create a kernel for morphological operations
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self._morph_kernel_size, self._morph_kernel_size)
        )
        
        # Closing operation: fills small holes and connects nearby regions
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Opening operation: removes small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        logger.debug(f"Applied morphological operations (kernel size={self._morph_kernel_size})")
        return opened
    
    def preprocess(self, image_path: Optional[str | Path] = None) -> np.ndarray:
        """
        Run the complete preprocessing pipeline.
        
        Pipeline steps:
        1. Load image (if path provided)
        2. Convert to grayscale
        3. Apply Gaussian blur for denoising
        4. Apply thresholding to create binary image
        5. Apply morphological operations to clean up
        
        Args:
            image_path: Optional path to load image from.
                       If None, uses already loaded image.
            
        Returns:
            np.ndarray: Preprocessed binary image ready for room detection
        """
        # Load image if path provided
        if image_path is not None:
            self.load_image(image_path)
        
        if self.original_image is None:
            raise ValueError("No image loaded. Provide image_path or call load_image() first.")
        
        logger.info("Starting preprocessing pipeline...")
        
        if self.config.get('remove_text', False):
             logger.info("Using advanced text removal preprocessing...")
             from preprocess_architectural import remove_text_from_floorplan
             
             # remove_text_from_floorplan returns black walls on white background
             # We need white walls on black background for room detection
             cleaned = remove_text_from_floorplan(self.original_image)
             self.processed_image = cv2.bitwise_not(cleaned)
             logger.info("Advanced preprocessing completed")
             return self.processed_image

        # Step 1: Convert to grayscale
        grayscale = self.to_grayscale()
        
        # Step 2: Apply Gaussian blur to reduce noise
        blurred = self.apply_blur(grayscale)
        
        # Step 3: Apply thresholding to create binary image
        binary = self.apply_threshold(blurred)
        
        # Step 4: Apply morphological operations
        self.processed_image = self.apply_morphology(binary)
        
        logger.info("Preprocessing pipeline completed")
        return self.processed_image
    
    def get_image_dimensions(self) -> Tuple[int, int]:
        """
        Get the dimensions of the loaded image.
        
        Returns:
            Tuple[int, int]: (width, height) in pixels
            
        Raises:
            ValueError: If no image is loaded
        """
        if self.original_image is None:
            raise ValueError("No image loaded.")
        
        height, width = self.original_image.shape[:2]
        return width, height
    
    def save_debug_image(
        self,
        output_path: str | Path,
        image: Optional[np.ndarray] = None
    ) -> None:
        """
        Save an image for debugging purposes.
        
        Args:
            output_path: Path where the debug image will be saved
            image: Optional image to save. If None, saves processed_image.
        """
        if image is None:
            image = self.processed_image
            
        if image is None:
            raise ValueError("No image to save.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), image)
        logger.debug(f"Saved debug image to: {output_path}")


def create_processor(config: Dict[str, Any]) -> FloorPlanProcessor:
    """
    Factory function to create a FloorPlanProcessor instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FloorPlanProcessor: Configured processor instance
    """
    return FloorPlanProcessor(config)
