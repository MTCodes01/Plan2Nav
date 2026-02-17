"""
Floor Plan to 3D GeoJSON Converter - Main Entry Point

This script processes floor plan images and converts them to 3D GeoJSON files
for rendering in MapLibre GL JS.

Usage:
    python main.py --input input/ --output output/
    python main.py --input sample_input/ --output output/ --config config.yaml

Author: Floor Plan to 3D GeoJSON Converter
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import yaml
from tqdm import tqdm

import json

# Import our custom modules
from floorplan_processor import FloorPlanProcessor
from room_detector import RoomDetector
from geojson_converter import GeoJSONConverter


def setup_logging(log_level: str = "INFO", log_to_file: bool = False, log_file: str = None):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to a file
        log_file: Path to log file
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_to_file and log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from: {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise


def find_floor_plan_images(input_dir: Path) -> List[Path]:
    """
    Find all floor plan images in the input directory.
    
    Args:
        input_dir: Directory to search for images
        
    Returns:
        List of image file paths
    """
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    image_files = []
    for ext in supported_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def process_floor_plan(
    image_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
    generate_debug: bool = True
) -> bool:
    """
    Process a single floor plan image.
    
    Args:
        image_path: Path to the floor plan image
        output_dir: Directory to save output files
        config: Configuration dictionary
        generate_debug: Whether to generate debug images
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Processing: {image_path.name}")
        
        # Step 1: Load and preprocess the image
        processor = FloorPlanProcessor(config)
        binary_image = processor.preprocess(image_path)
        image_height = processor.get_image_dimensions()[1]
        
        # Step 2: Detect rooms
        detector = RoomDetector(config)
        # Pass original image for OCR
        rooms = detector.detect_rooms(binary_image, processor.original_image)
        
        if not rooms:
            logger.warning(f"No rooms detected in {image_path.name}")
            return False
        
        # Log statistics
        stats = detector.get_room_statistics()
        logger.info(
            f"Detected {stats['total_rooms']} rooms: {stats['room_types']}"
        )
        
        # Step 3: Convert to GeoJSON
        converter = GeoJSONConverter(config)
        output_filename = f"{image_path.stem}_3d.geojson"
        output_path = output_dir / output_filename
        
        feature_collection = converter.convert_and_save(
            rooms,
            image_height,
            output_path
        )

        # Save overlay configuration for viewer
        try:
             # Calculate image bounds (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
            image_h, image_w = processor.original_image.shape[:2]
            overlay_bounds = converter.get_image_bounds(image_w, image_h)
            
            # Use relative path suitable for viewer.html in root
            # image_path is absolute or relative to CWD (d:\VScode...)
            # We want path relative to viewer.html location (root)
            # If running from root, input/example.png is correct.
            # config.json acts as a single source of truth for the viewer.
            
            overlay_config = {
                "image_url": str(image_path.as_posix()), 
                "coordinates": overlay_bounds
            }
            
            overlay_config_path = output_dir / "overlay_config.json"
            with open(overlay_config_path, 'w') as f:
                json.dump(overlay_config, f, indent=2)
                
            logger.info(f"Saved overlay config to {overlay_config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save overlay config: {e}")
            
        # Step 4: Generate debug images if enabled
        if generate_debug and config.get('output', {}).get('generate_debug_images', True):
            debug_dir = output_dir / 'debug'
            debug_dir.mkdir(exist_ok=True)
            
            # Save preprocessed binary image
            debug_binary_path = debug_dir / f"{image_path.stem}_binary.png"
            processor.save_debug_image(debug_binary_path, binary_image)
            
            # Save room detection visualization
            debug_rooms_image = detector.draw_rooms_debug(
                processor.original_image,
                show_contours=False,
                show_simplified=True
            )
            debug_rooms_path = debug_dir / f"{image_path.stem}_rooms.png"
            processor.save_debug_image(debug_rooms_path, debug_rooms_image)
            
            logger.debug(f"Saved debug images to: {debug_dir}")
        
        logger.info(f"[OK] Successfully processed {image_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to process {image_path.name}: {e}", exc_info=True)
        return False


def main():
    """Main entry point for the application."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Convert 2D floor plan images to 3D GeoJSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in input/ folder
  python main.py --input input/ --output output/
  
  # Use custom config file
  python main.py --input sample_input/ --output output/ --config my_config.yaml
  
  # Enable debug mode
  python main.py --input input/ --output output/ --debug
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='input',
        help='Input directory containing floor plan images (default: input/)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory for GeoJSON files (default: output/)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--no-debug-images',
        action='store_true',
        help='Disable generation of debug images'
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    config_path = Path(args.config)
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Setup logging
    log_level = "DEBUG" if args.debug else config.get('logging', {}).get('level', 'INFO')
    log_to_file = config.get('logging', {}).get('log_to_file', False)
    log_file = output_dir / config.get('logging', {}).get('log_file', 'processing.log')
    
    setup_logging(log_level, log_to_file, log_file)
    logger = logging.getLogger(__name__)
    
    # Banner
    logger.info("=" * 60)
    logger.info("Floor Plan to 3D GeoJSON Converter")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir.absolute()}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Config file: {config_path.absolute()}")
    logger.info("=" * 60)
    
    # Find all floor plan images
    image_files = find_floor_plan_images(input_dir)
    
    if not image_files:
        logger.warning(f"No floor plan images found in {input_dir}")
        print("\nNo images to process. Please add floor plan images to the input directory.")
        sys.exit(0)
    
    logger.info(f"Found {len(image_files)} image(s) to process")
    
    # Process each image with progress bar
    generate_debug = not args.no_debug_images
    success_count = 0
    
    with tqdm(image_files, desc="Processing floor plans", unit="image") as pbar:
        for image_path in pbar:
            pbar.set_description(f"Processing {image_path.name}")
            
            if process_floor_plan(image_path, output_dir, config, generate_debug):
                success_count += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Processing complete: {success_count}/{len(image_files)} successful")
    logger.info(f"Output saved to: {output_dir.absolute()}")
    logger.info("=" * 60)
    
    if success_count < len(image_files):
        logger.warning(f"{len(image_files) - success_count} image(s) failed to process")
        sys.exit(1)
    
    print(f"\n[OK] All images processed successfully!")
    print(f"  GeoJSON files saved to: {output_dir.absolute()}")
    print(f"  Open viewer.html to visualize the results")


if __name__ == '__main__':
    main()
