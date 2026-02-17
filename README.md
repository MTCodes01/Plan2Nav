# 🏠 Floor Plan to 3D GeoJSON Converter

Convert 2D floor plan images to 3D GeoJSON files for rendering in MapLibre GL JS.

## Features

- 📸 **Image Processing**: Automatic wall and room detection from floor plan images
- 🎯 **Smart Detection**: Uses Hough Transform for wall detection and contour analysis for rooms
- 🌍 **GeoJSON Output**: Generates valid GeoJSON with 3D properties (height, base_height)
- 🎨 **Customizable**: Configurable via YAML file (thresholds, colors, coordinate transformation)
- 📊 **Batch Processing**: Process multiple floor plans at once with progress indicators
- 🐛 **Debug Mode**: Generate debug images to visualize detection results
- 🗺️ **Interactive Viewer**: MapLibre GL JS viewer with 3D visualization

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
cd "d:\VScode\2D to 3D - GEOJson"
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. **Place your floor plan images** in the `input/` folder (PNG, JPG, JPEG supported)

2. **Run the converter**:
   ```bash
   python main.py
   ```

3. **Find the output** in the `output/` folder as `*_3d.geojson` files

### Advanced Usage

```bash
# Specify custom input/output directories
python main.py --input my_plans/ --output results/

# Use a custom configuration file
python main.py --config my_config.yaml

# Enable debug mode for verbose logging
python main.py --debug

# Disable debug image generation
python main.py --no-debug-images
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Input directory with floor plan images | `input/` |
| `--output`, `-o` | Output directory for GeoJSON files | `output/` |
| `--config`, `-c` | Path to configuration YAML file | `config.yaml` |
| `--debug` | Enable debug mode with verbose logging | `False` |
| `--no-debug-images` | Disable debug image generation | `False` |

## Configuration

Edit `config.yaml` to customize the processing pipeline:

### Image Processing
```yaml
image_processing:
  threshold: 127                  # Binary threshold (0-255)
  adaptive_threshold: true        # Use adaptive thresholding
  blur_kernel_size: 5            # Gaussian blur kernel (odd number)
  morph_kernel_size: 3           # Morphological operations kernel
```

### Room Detection
```yaml
room_detection:
  min_area: 1000                 # Minimum room area in pixels²
  max_area: 0                    # Maximum area (0 = no limit)
  contour_epsilon: 0.02          # Polygon simplification factor
```

### Coordinate Transformation
```yaml
coordinate_transform:
  center_lat: 0.0                # Center latitude
  center_lon: 0.0                # Center longitude
  meters_per_pixel: 0.1          # Scale: meters per pixel
  rotation_degrees: 0.0          # Rotation angle
```

### Room Colors
```yaml
room_properties:
  default_height: 3.0            # Default ceiling height (meters)
  colors:
    bedroom: "#FFE4B5"
    living_room: "#98FB98"
    kitchen: "#FFB6C1"
    bathroom: "#ADD8E6"
```

## Visualization

### Using the Web Viewer

1. **Open `viewer.html`** in a web browser

2. **Load a GeoJSON file** using the file picker

3. **Interact with the map**:
   - Toggle between 2D and 3D views
   - Click on rooms to see details
   - Use mouse to pan, zoom, and rotate
   - Reset view to original position

### Viewer Controls

- **3D View Toggle**: Switch between flat (2D) and extruded (3D) visualization
- **Reset View**: Return to the initial camera position
- **Clear Data**: Remove loaded GeoJSON from the map
- **Click on Room**: Display room information (type, area, height)

## Example Output

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon1, lat1], [lon2, lat2], ...]]
      },
      "properties": {
        "height": 3.0,
        "base_height": 0.0,
        "room_type": "bedroom",
        "color": "#FFE4B5",
        "area": 15.5
      }
    }
  ]
}
```

## Project Structure

```
2D to 3D - GEOJson/
├── main.py                    # Entry point
├── floorplan_processor.py     # Image preprocessing
├── room_detector.py           # Room detection algorithms
├── geojson_converter.py       # GeoJSON generation
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── viewer.html                # MapLibre GL JS viewer
├── generate_sample.py         # Sample floor plan generator
├── README.md                  # This file
├── input/                     # Place your floor plans here
├── output/                    # Generated GeoJSON files
│   └── debug/                 # Debug images (if enabled)
└── sample_input/              # Sample floor plan for testing
    └── sample_floorplan.png
```

## How It Works

### Processing Pipeline

1. **Image Loading**: Load floor plan image (PNG, JPG)
2. **Preprocessing**:
   - Convert to grayscale
   - Apply Gaussian blur for denoising
   - Threshold to create binary image (walls = white)
   - Morphological operations to clean up
3. **Room Detection**:
   - Find contours (closed regions)
   - Filter by area
   - Simplify polygons
   - Classify room types
4. **Coordinate Transformation**:
   - Convert pixel coordinates to meters
   - Apply rotation (if specified)
   - Transform to WGS84 lat/lon
5. **GeoJSON Generation**:
   - Create Feature for each room
   - Assign properties (height, color, type)
   - Validate output
   - Save to file

### Algorithms Used

- **Hough Transform**: Detects straight lines (walls) in the image
- **Contour Detection**: Finds enclosed regions (rooms)
- **Douglas-Peucker**: Simplifies polygon shapes
- **Equirectangular Projection**: Converts meters to lat/lon coordinates

## Troubleshooting

### No rooms detected

- **Check image quality**: Ensure walls are clearly visible and continuous
- **Adjust threshold**: Try different `threshold` values in config.yaml
- **Reduce min_area**: Lower the `min_area` parameter if rooms are small
- **Enable debug images**: Use `--debug` to see intermediate processing steps

### Incorrect room boundaries

- **Adjust contour_epsilon**: Lower values = more detailed polygons
- **Check wall thickness**: Ensure `min_thickness` and `max_thickness` are appropriate
- **Use adaptive thresholding**: Set `adaptive_threshold: true` for uneven lighting

### GeoJSON not displaying correctly

- **Verify coordinates**: Check that `center_lat`, `center_lon` are correct
- **Adjust scale**: Modify `meters_per_pixel` to match your floor plan scale
- **Validate GeoJSON**: Use online validators like [geojson.io](https://geojson.io)

## Testing

A sample floor plan is provided in `sample_input/`:

```bash
# Generate the sample floor plan
python generate_sample.py

# Process the sample
python main.py --input sample_input --output output

# View the result
# Open viewer.html and load output/sample_floorplan_3d.geojson
```

## Requirements

- `opencv-python >= 4.8.0` - Image processing
- `numpy >= 1.24.0` - Numerical operations
- `shapely >= 2.0.0` - Geometry operations
- `PyYAML >= 6.0` - Configuration parsing
- `Pillow >= 10.0.0` - Image handling
- `Pillow >= 10.0.0` - Image handling
- `tqdm >= 4.65.0` - Progress bars
- `pytesseract >= 0.3.10` - Text recognition (OCR)

### External Tools

- **Tesseract-OCR**: Required for recognizing room labels.
  - **Windows**: Install from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
  - **Linux**: `sudo apt install tesseract-ocr`
  - Ensure Tesseract is in your system PATH or configured in `config_architectural.yaml`. w

## OCR Configuration

To enable text detection for room labeling, edit `config_architectural.yaml`:

```yaml
text_detection:
  enabled: true
  confidence_threshold: 40.0
  # tesseract_cmd: "C:/Program Files/Tesseract-OCR/tesseract.exe"
```

## License

This project is provided as-is for educational and commercial use.

## Contributing

Contributions are welcome! Areas for improvement:

- Machine learning-based room classification
- Support for curved walls and irregular shapes
- Multi-floor building support
- Automatic door detection
- Furniture detection and placement

## Support

For issues or questions, please check:

1. **Debug images**: Enable debug mode to see processing steps
2. **Configuration**: Review config.yaml parameters
3. **Logs**: Check console output for error messages

---

**Made with ❤️ for the mapping community**
