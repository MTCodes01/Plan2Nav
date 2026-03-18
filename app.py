import os
import uuid
import json
import logging
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import yaml

from floorplan_processor import FloorPlanProcessor
from room_detector import RoomDetector
from geojson_converter import GeoJSONConverter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Directory for final uploads to keep and serve
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)


def load_config() -> dict:
    config_path = Path('config_architectural.yaml')
    if not config_path.exists():
        # Fallback to standard config
        config_path = Path('config.yaml')
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/api/process_floors', methods=['POST'])
def process_floors():
    """
    Expects multipart/form-data:
    - Multiple files with key 'images[]'
    - Multiple integers with key 'floor_numbers[]' 
      (indexes should correlate with 'images[]')
    """
    if 'images[]' not in request.files or 'floor_numbers[]' not in request.form:
        return jsonify({"error": "Missing images or floor numbers"}), 400

    files = request.files.getlist('images[]')
    try:
        floor_numbers = [int(f) for f in request.form.getlist('floor_numbers[]')]
    except ValueError:
        return jsonify({"error": "Floor numbers must be integers"}), 400

    if len(files) != len(floor_numbers):
        return jsonify({"error": "Mismatch between number of images and floor numbers"}), 400

    # Sort files by floor number to process them in order (optional, but good practice)
    sorted_pairs = sorted(zip(floor_numbers, files), key=lambda x: x[0])

    base_config = load_config()
    all_features = []
    floors_data = []

    # Process each floor plan
    for floor_num, file_obj in sorted_pairs:
        if file_obj.filename == '':
            continue
            
        logger.info(f"Processing {file_obj.filename} for floor {floor_num}")

        # Save to temp path
        ext = Path(file_obj.filename).suffix
        temp_path = UPLOAD_FOLDER / f"{uuid.uuid4()}{ext}"
        file_obj.save(temp_path)

        try:
            # Deep copy config to safely modify for this floor
            import copy
            floor_config = copy.deepcopy(base_config)
            
            # Dynamically calc the base height and total height.
            # MapLibre requires fill-extrusion-height to be the absolute top height, not relative.
            default_room_height = floor_config.get('room_properties', {}).get('default_height', 3.0)
            calculated_base_height = floor_num * default_room_height
            calculated_top_height = calculated_base_height + default_room_height
            
            if 'room_properties' not in floor_config:
                floor_config['room_properties'] = {}
            floor_config['room_properties']['default_base_height'] = calculated_base_height
            floor_config['room_properties']['default_height'] = calculated_top_height
            
            # Step 1: Preprocess
            processor = FloorPlanProcessor(floor_config)
            binary_image = processor.preprocess(temp_path)
            image_height = processor.get_image_dimensions()[1]
            
            # Step 2: Detect
            detector = RoomDetector(floor_config)
            rooms = detector.detect_rooms(binary_image, processor.original_image)
            
            # ----------------------------------------------------------------
            # STANDARD GEOJSON (Rooms)
            # ----------------------------------------------------------------
            converter = GeoJSONConverter(floor_config)
            dummy_path_rooms = temp_path.with_suffix('.rooms.json')
            feature_collection_rooms = converter.convert_and_save(rooms, image_height, dummy_path_rooms)
            # all_features.extend(feature_collection_rooms.get("features", []))


            # ----------------------------------------------------------------
            # WALL EXTRACTION
            # ----------------------------------------------------------------
            dummy_path_walls = temp_path.with_suffix('.walls.json')
            try:
                wall_mask = processor.preprocess_walls()
                wall_polygons = detector.detect_wall_polygons_from_lines(wall_mask)

                
                if wall_polygons:
                    wall_coll = converter.convert_wall_polygons(wall_polygons, image_height, dummy_path_walls)
                    for f in wall_coll.get("features", []):
                        if "properties" not in f:
                            f["properties"] = {}
                        f["properties"]["floor_num"] = floor_num
                    all_features.extend(wall_coll.get("features", []))
            except Exception as wall_err:
                logger.error(f"Wall extraction failed for {file_obj.filename}: {wall_err}", exc_info=True)
                
            # ----------------------------------------------------------------
            # ZONE EXTRACTION
            # ----------------------------------------------------------------
            dummy_path_zones = temp_path.with_suffix('.zones.json')
            try:
                zones = detector.detect_rectangular_zones(wall_mask)
                if zones:
                    # Integrate Text Extraction with Swift OCR
                    try:
                        from text_detector import TextDetector
                        text_detector = TextDetector(floor_config)
                        # Run OCR on the original image using the saved image file path
                        texts = text_detector.detect_text(processor.original_image, image_path=str(temp_path))
                        
                        import cv2
                        import numpy as np
                        for room in zones:
                            match_texts = []
                            poly = room.simplified_polygon
                            if len(poly.shape) == 3:
                                poly = poly.reshape(-1, 2)
                            
                            for txt in texts:
                                pt = (float(txt.centroid[0]), float(txt.centroid[1]))
                                is_inside = cv2.pointPolygonTest(poly.astype(np.float32), pt, False)
                                if is_inside >= 0:
                                    # Dedup text if multiple matches are identical
                                    if txt.text not in match_texts:
                                        match_texts.append(txt.text)
                            
                            if match_texts:
                                room.room_type = " / ".join(match_texts)
                                logger.info(f"Updated Zone {getattr(room, 'id', '')} to: {room.room_type}")
                    except Exception as ocr_err:
                        logger.error(f"OCR matching failed: {ocr_err}", exc_info=True)

                    # convert_and_save takes list of objects with simplified_polygon
                    zones_coll = converter.convert_and_save(zones, image_height, dummy_path_zones)
                    for f in zones_coll.get("features", []):
                        if "properties" not in f:
                            f["properties"] = {}
                        f["properties"]["floor_num"] = floor_num
                    all_features.extend(zones_coll.get("features", []))
            except Exception as zone_err:
                logger.error(f"Zone extraction failed for {file_obj.filename}: {zone_err}", exc_info=True)

            # ----------------------------------------------------------------
            # ROOM INTERIOR EXTRACTION

            # ----------------------------------------------------------------
            dummy_path_filled = temp_path.with_suffix('.filled.json')
            try:
                raw_img = processor.original_image
                room_interiors = detector.detect_room_interiors(raw_img)
                
                if room_interiors:
                    room_filled_coll = converter.convert_wall_polygons(room_interiors, image_height, dummy_path_filled)
                    
                    # All features will be appended directly
                    # all_features.extend(room_filled_coll.get("features", []))


            except Exception as room_err:
                logger.error(f"Room interior extraction failed for {file_obj.filename}: {room_err}", exc_info=True)

            # Clean up JSON files
            for p in [dummy_path_walls, dummy_path_rooms, dummy_path_filled]:
                if p.exists():
                    p.unlink()

            # Append the floor data for deckgl rendering
            # Pass width and height to calculate deckgl bounds
            width, height = processor.get_image_dimensions()
            deckgl_bounds = converter.get_deckgl_bounds(width, height, calculated_base_height)
            
            floors_data.append({
                "floor_num": floor_num,
                "imageUrl": f"/uploads/{temp_path.name}",
                "bounds": deckgl_bounds,
                "base_height": calculated_base_height
            })

        except Exception as e:
            logger.error(f"Error processing {file_obj.filename}: {e}", exc_info=True)
            return jsonify({"error": f"Failed to process {file_obj.filename}: {str(e)}"}), 500

    # Build merged FeatureCollection
    merged_geojson = {
        "type": "FeatureCollection",
        "features": all_features
    }

    return jsonify({
        "geojson": merged_geojson,
        "floors": floors_data
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
