"""
GeoJSON Converter Module

This module handles the conversion of detected rooms (polygons) into GeoJSON format
for 3D rendering in map applications like MapLibre GL JS.

It supports coordinate transformation from pixel space to geographic coordinates (lat/lon)
and assigning properties for extrusion height and color.
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

class GeoJSONConverter:
    """
    Converts detected room polygons to GeoJSON FeatureCollection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the converter with configuration.
        
        Args:
            config: Configuration dictionary containing:
                   - coordinate_transform settings
                   - room_properties settings
                   - output settings
        """
        self.config = config
        
        # Coordinate transform settings
        coord_config = config.get('coordinate_transform', {})
        self.center_lat = coord_config.get('center_lat', 0.0)
        self.center_lon = coord_config.get('center_lon', 0.0)
        self.meters_per_pixel = coord_config.get('meters_per_pixel', 0.1)
        self.rotation_degrees = coord_config.get('rotation_degrees', 0.0)
        
        # Room properties
        room_props = config.get('room_properties', {})
        self.default_height = room_props.get('default_height', 3.0)
        self.default_base_height = room_props.get('default_base_height', 0.0)
        self.colors = room_props.get('colors', {})
        
        # Output settings
        output_config = config.get('output', {})
        self.pretty_print = output_config.get('pretty_print', True)
        self.precision = output_config.get('coordinate_precision', 8)

    def _pixel_to_latlon(self, x: float, y: float, image_height: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to latitude/longitude.
        
        Assumes the image is a flat plane centered at center_lat, center_lon.
        Y-axis in image grows downwards, while latitude grows North (up).
        
        Args:
            x: Pixel x-coordinate
            y: Pixel y-coordinate
            image_height: Total height of the image (for flipping Y)
            
        Returns:
            Tuple of (longitude, latitude)
        """
        # Flip Y axis to match Cartesian coordinates (y grows up)
        # Assuming (0,0) of image is top-left
        # Convert to local meters relative to bottom-left of the image
        # Wait, usually we want to center the image at the lat/lon.
        # Let's assume center of image corresponds to center_lat/lon for simplicity logic
        # Or let's imply bottom-left is origin?
        # The prompt doesn't specify deeply, but centering is standard.
        
        # Let's pivot around the image center
        # If image width is unknown here, we might just pivot around top-left or use 0,0 as top-left
        # and offset later.
        
        # BUT, standard approach:
        # 1 px = N meters.
        # local_x = x * scale
        # local_y = (image_height - y) * scale  (Flip Y)
        
        local_x = x * self.meters_per_pixel
        local_y = (image_height - y) * self.meters_per_pixel
        
        # Apply rotation if needed (around origin 0,0 which is bottom-left now)
        if self.rotation_degrees != 0:
            rad = math.radians(self.rotation_degrees)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            
            # Simple 2D rotation
            rot_x = local_x * cos_a - local_y * sin_a
            rot_y = local_x * sin_a + local_y * cos_a
            local_x, local_y = rot_x, rot_y

        # Convert meters to degrees
        # Earth Circumference ~ 40,075,000 meters
        # 1 degree lat ~ 111,132 meters
        METERS_PER_DEG_LAT = 111132.954
        
        delta_lat = local_y / METERS_PER_DEG_LAT
        
        # Longitude varies with latitude
        # 1 degree lon ~ 111,132 * cos(lat) meters
        lat_rad = math.radians(self.center_lat)
        meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(lat_rad)
        
        delta_lon = local_x / meters_per_deg_lon
        
        return (self.center_lon + delta_lon, self.center_lat + delta_lat)

    def get_image_bounds(self, width: int, height: int) -> List[List[float]]:
        """
        Calculate the geographic bounds of the image for map overlay.
        
        Returns:
            List of 4 coordinates [lon, lat] in order:
            Top-Left, Top-Right, Bottom-Right, Bottom-Left
            This is the format expected by MapLibre/Mapbox image source.
        """
        # Top-Left (0, 0)
        tl = self._pixel_to_latlon(0, 0, height)
        # Top-Right (width, 0)
        tr = self._pixel_to_latlon(width, 0, height)
        # Bottom-Right (width, height)
        br = self._pixel_to_latlon(width, height, height)
        # Bottom-Left (0, height)
        bl = self._pixel_to_latlon(0, height, height)
        
        return [
            [round(tl[0], self.precision), round(tl[1], self.precision)],
            [round(tr[0], self.precision), round(tr[1], self.precision)],
            [round(br[0], self.precision), round(br[1], self.precision)],
            [round(bl[0], self.precision), round(bl[1], self.precision)]
        ]

    def get_deckgl_bounds(self, width: int, height: int, base_height: float) -> List[List[float]]:
        """
        Calculate the geographic bounds of the image for Deck.gl BitmapLayer.
        Deck.gl expects bounds as [left, bottom, right, top], but for simple
        alignment with extruded features, its better to give a 4 point array
        [[left, bottom, z], [left, top, z], [right, top, z], [right, bottom, z]]
        MapLibre bounds are usually [lon, lat], Deck.gl allows [lon, lat, Z]
        """
        # Top-Left (0, 0)
        tl = self._pixel_to_latlon(0, 0, height)
        # Top-Right (width, 0)
        tr = self._pixel_to_latlon(width, 0, height)
        # Bottom-Right (width, height)
        br = self._pixel_to_latlon(width, height, height)
        # Bottom-Left (0, height)
        bl = self._pixel_to_latlon(0, height, height)
        
        # We need counter-clockwise coordinates starting from bottom-left for standard quad
        # But Deck.gl BitmapLayer bounds property accepts:
        # Array of four coordinates [ [left, bottom], [left, top], [right, top], [right, bottom] ]
        # Each coordinate can be [x, y, z]
        return [
            [round(bl[0], self.precision), round(bl[1], self.precision), base_height],
            [round(tl[0], self.precision), round(tl[1], self.precision), base_height],
            [round(tr[0], self.precision), round(tr[1], self.precision), base_height],
            [round(br[0], self.precision), round(br[1], self.precision), base_height]
        ]

    def convert_and_save(self, rooms: List[Any], image_height: int, output_path: Path) -> Dict[str, Any]:
        """
        Convert detection results to GeoJSON and save to file.
        
        Args:
            rooms: List of Room objects detected by RoomDetector
            image_height: Height of the original image
            output_path: Path to save the GeoJSON file
            
        Returns:
            The generated GeoJSON FeatureCollection dictionary
        """
        features = []
        
        for room in rooms:
            # Skip invalid rooms
            if len(room.simplified_polygon) < 3:
                continue
                
            # Determine which geometry to use
            has_wall = getattr(room, 'wall_polygon', None) is not None
            
            if has_wall:
                # Use wall polygon
                geom_shape = room.wall_polygon
                
                # Handle MultiPolygon or Polygon
                if geom_shape.geom_type == 'Polygon':
                    polys = [geom_shape]
                elif geom_shape.geom_type == 'MultiPolygon':
                    polys = geom_shape.geoms
                else:
                    logger.warning(f"Unsupported geometry type for wall: {geom_shape.geom_type}")
                    continue
                
                # Process each part of the wall
                for poly in polys:
                    # Exterior ring
                    coords_lists = [list(poly.exterior.coords)]
                    # Interior rings (holes)
                    coords_lists.extend([list(interior.coords) for interior in poly.interiors])
                    
                    # Convert all rings to lat/lon
                    geo_rings = []
                    for ring_coords in coords_lists:
                        geo_ring = []
                        for x, y in ring_coords:
                            lon, lat = self._pixel_to_latlon(float(x), float(y), image_height)
                            lon = round(lon, self.precision)
                            lat = round(lat, self.precision)
                            geo_ring.append([lon, lat])
                        geo_rings.append(geo_ring)
                    
                    # Create Feature for this wall segment
                    feature = self._create_feature(room, geo_rings, has_wall=True)
                    features.append(feature)
            
            else:
                # Use original valid simplified polygon (solid room)
                # simplified_polygon is typically a numpy array of shape (N, 1, 2) or (N, 2)
                points = room.simplified_polygon
                if len(points.shape) == 3:
                    points = points.reshape(-1, 2)
                    
                # Convert each vertex to lat/lon
                geo_ring = []
                for point in points:
                    x, y = point[0], point[1]
                    lon, lat = self._pixel_to_latlon(float(x), float(y), image_height)
                    
                    # Round to configured precision
                    lon = round(lon, self.precision)
                    lat = round(lat, self.precision)
                    
                    geo_ring.append([lon, lat])
                
                # Close the polygon ring (first point == last point)
                if geo_ring[0] != geo_ring[-1]:
                    geo_ring.append(geo_ring[0])
                
                # Create Feature (solid room)
                feature = self._create_feature(room, [geo_ring], has_wall=False)
                features.append(feature)
        
        # Create FeatureCollection
        feature_collection = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                indent = 2 if self.pretty_print else None
                json.dump(feature_collection, f, indent=indent)
            
            logger.info(f"Successfully saved GeoJSON to {output_path} ({len(features)} features)")
            
        except Exception as e:
            logger.error(f"Failed to save GeoJSON file: {e}")
            raise
            
        return feature_collection

    def _create_feature(self, room: Any, coordinates: List[List[List[float]]], has_wall: bool) -> Dict[str, Any]:
        """Helper to create a GeoJSON feature from coordinates."""
        # Get room properties
        room_type = getattr(room, 'room_type', 'unknown')
        
        # Color lookup
        color = self.colors.get(room_type, self.colors.get('unknown', '#808080'))
        if room_type.startswith("Zone"):
            try:
                # Extract coordinate index from 'Zone X'
                index = int(room_type.split()[-1])
            except:
                index = 1
            palette = ["#FF8C42", "#D9E2EC", "#4E8588", "#90A959", "#A44A3F", "#F2D06B"]
            color = palette[(index - 1) % len(palette)]

        
        # Construct properties for 3D extrusion
        height = self.default_height
        base_height = self.default_base_height

        if room_type.startswith("Zone") or (not has_wall and room_type != "unknown"):
            height = base_height + 0.1  # Flat zone layout on floor

        props = {
            "room_type": room_type,
            "height": height,
            "base_height": base_height,

            "color": color,
            "area_pixels": float(getattr(room, 'area', 0)),
            "is_wall": has_wall,
            # MapLibre/Mapbox standard properties for 3D extrusion
            "fill-extrusion-color": color,
            "fill-extrusion-height": height,
            "fill-extrusion-base": base_height,
            "fill-extrusion-opacity": 0.8
        }

        
        # Create Feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates
            },
            "properties": props
        }
        
        if hasattr(room, 'id') and room.id > 0:
            feature["id"] = room.id
        
        return feature

    # ------------------------------------------------------------------
    # Wall polygon conversion
    # ------------------------------------------------------------------

    def convert_wall_polygons(
        self,
        wall_polygon_data: list,
        image_height: int,
        output_path: "Path",
    ) -> dict:
        """
        Convert (outer_pts, holes) pixel-coordinate wall polygons to GeoJSON
        and save to file.

        Args:
            wall_polygon_data : Output of RoomDetector.detect_wall_polygons().
                                List of (outer_pts, [hole_pts, ...]).
            image_height      : Pixel height of the source image (for Y-flip).
            output_path       : Destination .geojson file path.

        Returns:
            GeoJSON FeatureCollection dict.
        """
        features = []

        for outer_pts, holes_pts in wall_polygon_data:
            if len(outer_pts) < 3:
                continue

            # Convert outer ring pixel coords → lon/lat
            geo_outer = []
            for x, y in outer_pts:
                lon, lat = self._pixel_to_latlon(float(x), float(y), image_height)
                geo_outer.append([round(lon, self.precision), round(lat, self.precision)])

            # Close the ring
            if geo_outer[0] != geo_outer[-1]:
                geo_outer.append(geo_outer[0])

            # GeoJSON exterior ring must be CCW
            if not _is_ccw(geo_outer):
                geo_outer = geo_outer[::-1]

            # Convert hole rings
            geo_holes = []
            for hole_pts in holes_pts:
                if len(hole_pts) < 3:
                    continue
                geo_hole = []
                for x, y in hole_pts:
                    lon, lat = self._pixel_to_latlon(float(x), float(y), image_height)
                    geo_hole.append([round(lon, self.precision), round(lat, self.precision)])
                if geo_hole[0] != geo_hole[-1]:
                    geo_hole.append(geo_hole[0])
                # GeoJSON hole rings must be CW
                if _is_ccw(geo_hole):
                    geo_hole = geo_hole[::-1]
                geo_holes.append(geo_hole)

            coordinates = [geo_outer] + geo_holes

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordinates,
                },
                "properties": {
                    "type": "wall",
                    "is_wall": True,
                    "height": self.default_height,
                    "base_height": self.default_base_height,
                    "color": "#555555",
                    "fill-extrusion-color": "#555555",
                    "fill-extrusion-height": self.default_height,
                    "fill-extrusion-base": self.default_base_height,
                    "fill-extrusion-opacity": 1.0,
                },
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                import json as _json
                _json.dump(geojson, f, indent=2 if self.pretty_print else None)
            logger.info(
                f"Saved {len(features)} wall feature(s) to {output_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save wall GeoJSON: {e}")
            raise

        return geojson


def _is_ccw(ring: list) -> bool:
    """Return True if coordinate ring is counter-clockwise (shoelace formula)."""
    n = len(ring)
    area = sum(
        (ring[i][0] * ring[(i + 1) % n][1]) - (ring[(i + 1) % n][0] * ring[i][1])
        for i in range(n)
    )
    return area > 0
