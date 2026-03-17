"""
Room Detector Module

This module handles wall and room detection from preprocessed floor plan images.
It uses Hough Transform for wall detection and contour analysis for room identification.

Author: Floor Plan to 3D GeoJSON Converter
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Configure module logger
logger = logging.getLogger(__name__)



@dataclass
class Room:
    """
    Represents a detected room with its properties.
    
    Attributes:
        contour: Numpy array of polygon vertices in pixel coordinates
        area: Area in square pixels
        simplified_polygon: Simplified polygon vertices (fewer points)
        room_type: Classified room type (e.g., 'bedroom', 'kitchen')
        centroid: Center point of the room (x, y)
    """
    contour: np.ndarray
    area: float
    simplified_polygon: np.ndarray
    wall_polygon: Optional[Polygon] = None
    room_type: str = "unknown"
    centroid: Tuple[float, float] = (0.0, 0.0)
    
    def __post_init__(self):
        """Calculate centroid after initialization."""
        if self.centroid == (0.0, 0.0):
            M = cv2.moments(self.contour)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                self.centroid = (cx, cy)


class RoomDetector:
    """
    Detects walls and rooms from preprocessed floor plan images.
    
    This class provides methods for:
    - Wall detection using Hough Transform
    - Room boundary detection using contours
    - Polygon simplification
    - Room classification based on area and shape
    - Room labeling using OCR
    
    Attributes:
        config (dict): Configuration parameters for detection
        detected_rooms (List[Room]): List of detected room objects
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RoomDetector with configuration.
        
        Args:
            config: Dictionary containing detection parameters
                   Expected keys:
                   - wall_detection: Wall detection parameters
                   - room_detection: Room detection parameters
                   - room_properties: Room property settings
                   - text_detection: Text detection settings
        """
        self.config = config
        self.wall_config = config.get('wall_detection', {})
        self.room_config = config.get('room_detection', {})
        self.room_properties = config.get('room_properties', {})
        
        self.detected_rooms: List[Room] = []
        self._binary_image: Optional[np.ndarray] = None
        
        # Load configuration parameters
        self._min_area = self.room_config.get('min_area', 1000)
        self._max_area = self.room_config.get('max_area', 0)
        self._contour_epsilon = self.room_config.get('contour_epsilon', 0.02)
        self._detect_nested = self.room_config.get('detect_nested', False)
        
        # Wall detection parameters
        self._hough_threshold = self.wall_config.get('hough_threshold', 50)
        self._hough_min_line_length = self.wall_config.get('hough_min_line_length', 30)
        self._hough_max_line_gap = self.wall_config.get('hough_max_line_gap', 10)
        
        
        # Text Detector removed
        
        
        # Wall generation parameters
        self._generate_walls = self.config.get('wall_generation', {}).get('generate_walls', False)
        self._wall_thickness = self.config.get('wall_generation', {}).get('wall_thickness', 5)
        
        logger.debug(f"RoomDetector initialized with min_area={self._min_area}")

    def _generate_wall_polygon(self, room_contour: np.ndarray) -> Optional[Polygon]:
        """
        Generate a wall polygon by buffering the room contour.
        """
        try:
            # Convert contour to Shapely Polygon
            # contour shape is (N, 1, 2), needs (N, 2)
            points = room_contour.reshape(-1, 2)
            if len(points) < 3:
                return None
                
            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)
                
            # Create wall by buffering outward
            # buffer returns a polygon that includes the original
            # we want the "rind" or "frame"
            expanded = poly.buffer(self._wall_thickness, join_style=2) # 2 = mitre
            
            # Subtract original to get just the wall
            wall = expanded.difference(poly)
            
            return wall
        except Exception as e:
            logger.warning(f"Failed to generate wall polygon: {e}")
            return None
    
    def detect_walls(self, binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect walls using Hough Line Transform.
        
        The Hough Transform detects straight lines in the binary image,
        which typically correspond to walls in floor plans.
        
        Args:
            binary_image: Preprocessed binary image (walls as white)
            
        Returns:
            List of detected lines as (x1, y1, x2, y2) tuples
        """
        self._binary_image = binary_image
        
        # Apply edge detection to find wall boundaries
        edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
        
        # Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self._hough_threshold,
            minLineLength=self._hough_min_line_length,
            maxLineGap=self._hough_max_line_gap
        )
        
        if lines is None:
            logger.warning("No walls detected using Hough Transform")
            return []
        
        # Convert to list of tuples
        wall_lines = [(int(x1), int(y1), int(x2), int(y2)) 
                      for line in lines for x1, y1, x2, y2 in line]
        
        logger.info(f"Detected {len(wall_lines)} wall segments")
        return wall_lines
    
    def detect_rooms(self, binary_image: np.ndarray, original_image: Optional[np.ndarray] = None) -> List[Room]:
        """
        Detect rooms by finding enclosed contours in the binary image.
        
        Rooms are identified as closed contours (polygons) in the floor plan.
        The method filters contours by area and simplifies their shape.
        
        Args:
            binary_image: Preprocessed binary image
            original_image: Original image for text detection (optional)
            
        Returns:
            List of detected Room objects
        """
        self._binary_image = binary_image
        
        # Text detection removed
        detected_texts = []
        
        
        # Thicken walls to ensure they create complete barriers between rooms
        # This is crucial for floor plans where walls are thin lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thickened = cv2.dilate(binary_image, kernel, iterations=2)
        
        # Invert the image so rooms (white spaces) become foreground
        inverted = cv2.bitwise_not(thickened)
        
        # Find contours (room boundaries)
        # Use RETR_TREE to get nested contours (rooms inside the floor plan outline)
        contours, hierarchy = cv2.findContours(
            inverted,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        logger.info(f"Found {len(contours)} potential room contours")
        
        # Debug: log contour areas
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            logger.info(f"Contour areas: min={min(areas):.0f}, max={max(areas):.0f}, count={len(areas)}")
            for idx, area in enumerate(areas[:10]):  # Log first 10
                logger.debug(f"Contour {idx}: area={area:.0f}px²")
        
        # Filter and process contours
        self.detected_rooms = []
        image_h, image_w = binary_image.shape[:2]
        
        for idx, contour in enumerate(contours):
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < self._min_area:
                continue
            
            # Filter by maximum area (if specified)
            if self._max_area > 0 and area > self._max_area:
                continue
            
            # Filter by aspect ratio (reject very elongated shapes like fixtures)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 1)
            if aspect_ratio > 5:  # Reject very elongated shapes
                logger.debug(f"Skipping contour {idx} (aspect ratio {aspect_ratio:.1f} too high)")
                continue
            
            # Outer space filtering
            # Check if contour is too close to the image boundary
            x, y, w, h = cv2.boundingRect(contour)
            margin = 5  # pixels
            is_boundary = (x <= margin or y <= margin or 
                          (x + w) >= (image_w - margin) or 
                          (y + h) >= (image_h - margin))
            
            # If it's a boundary contour and very large (likely the empty space around the house), skip it
            # Unless we have a specific text label that says "Terrace" or "Garden"
            # We defer this check until we check for text labels
            
            # Simplify the polygon using Douglas-Peucker algorithm
            epsilon = self._contour_epsilon * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            # Create Room object
            room = Room(
                contour=contour,
                area=area,
                simplified_polygon=simplified
            )
            
            # Generate wall polygon if enabled
            if self._generate_walls:
                room.wall_polygon = self._generate_wall_polygon(room.simplified_polygon)
            
            # Filter out background/outer space if no text label confirms it's a room
            if is_boundary:
                if area > (image_h * image_w * 0.5): # Lowered threshold slightly to 50% to be safe
                     logger.debug(f"Skipping room {idx} (Background/Outer space)")
                     continue

            # Heuristic classification based on area
            room.room_type = self._classify_room_by_area(area)
            
            self.detected_rooms.append(room)
            
            logger.debug(
                f"Room {idx}: area={area:.0f}px², "
                f"vertices={len(simplified)}, type={room.room_type}"
            )
        
        logger.info(f"Detected {len(self.detected_rooms)} valid rooms")
        return self.detected_rooms
    

    def _classify_room_by_area(self, area: float) -> str:
        """
        Classify room type based on area.
        """
        # Area-based classification with adjusted thresholds
        if area < 5000:
            return "closet"
        elif area < 10000:
            return "bathroom"
        elif area < 25000:
            return "kitchen"
        elif area < 40000:
            return "bedroom"
        else:
            return "living_room"
    
    def get_room_polygons(self) -> List[np.ndarray]:
        """
        Get simplified polygons for all detected rooms.
        
        Returns:
            List of numpy arrays, each containing polygon vertices
        """
        return [room.simplified_polygon for room in self.detected_rooms]
    
    def draw_rooms_debug(
        self,
        image: np.ndarray,
        show_contours: bool = True,
        show_simplified: bool = True
    ) -> np.ndarray:
        """
        Draw detected rooms on an image for debugging.
        
        Args:
            image: Base image to draw on (will be converted to color if grayscale)
            show_contours: Whether to draw original contours
            show_simplified: Whether to draw simplified polygons
            
        Returns:
            Image with rooms drawn
        """
        # Convert to color if grayscale
        if len(image.shape) == 2:
            debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            debug_image = image.copy()
        
        for idx, room in enumerate(self.detected_rooms):
            # Generate a color for this room
            color = self._get_room_color(idx)
            
            if show_contours:
                # Draw original contour in thin line
                cv2.drawContours(debug_image, [room.contour], -1, color, 1)
            
            if show_simplified:
                # Draw simplified polygon in thick line
                cv2.drawContours(debug_image, [room.simplified_polygon], -1, color, 3)
            
            # Draw centroid
            cx, cy = int(room.centroid[0]), int(room.centroid[1])
            cv2.circle(debug_image, (cx, cy), 5, color, -1)
            
            # Add room label
            label = f"{room.room_type}" # Simplified label
            cv2.putText(
                debug_image,
                label,
                (cx - 30, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, # Increased font size
                (0, 0, 0), # Black text outline
                3
            )
            cv2.putText(
                debug_image,
                label,
                (cx - 30, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1
            )
        
        return debug_image
    
    def _get_room_color(self, index: int) -> Tuple[int, int, int]:
        """
        Generate a distinct color for each room.
        
        Args:
            index: Room index
            
        Returns:
            BGR color tuple
        """
        # Use HSV color space for distinct colors
        hue = (index * 40) % 180
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
        return tuple(map(int, color_bgr[0, 0]))
    
    def draw_walls_debug(
        self,
        image: np.ndarray,
        walls: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Draw detected walls on an image for debugging.
        
        Args:
            image: Base image to draw on
            walls: List of wall lines as (x1, y1, x2, y2) tuples
            
        Returns:
            Image with walls drawn
        """
        # Convert to color if grayscale
        if len(image.shape) == 2:
            debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            debug_image = image.copy()
        
        # Draw each wall line in red
        for x1, y1, x2, y2 in walls:
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return debug_image
    
    def get_room_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about detected rooms.
        
        Returns:
            Dictionary containing room statistics
        """
        if not self.detected_rooms:
            return {
                'total_rooms': 0,
                'total_area': 0,
                'room_types': {}
            }
        
        # Count room types
        room_types = {}
        total_area = 0
        
        for room in self.detected_rooms:
            room_types[room.room_type] = room_types.get(room.room_type, 0) + 1
            total_area += room.area
        
        return {
            'total_rooms': len(self.detected_rooms),
            'total_area': total_area,
            'average_area': total_area / len(self.detected_rooms),
            'room_types': room_types,
            'min_area': min(r.area for r in self.detected_rooms) if self.detected_rooms else 0,
            'max_area': max(r.area for r in self.detected_rooms) if self.detected_rooms else 0
        }


    def detect_wall_polygons(self, wall_mask: np.ndarray, min_area: int = 500) -> List[Tuple]:
        """
        Extract wall polygons directly from the clean wall binary mask.

        Uses RETR_CCOMP (2-level hierarchy) so each wall blob's outer boundary
        becomes the exterior ring while its enclosed room interior becomes a
        hole ring — accurately representing the real wall band geometry.

        Args:
            wall_mask : Clean binary image from FloorPlanProcessor.preprocess_walls().
                        Wall pixels = 255 (white), background = 0 (black).
            min_area  : Minimum contour area in pixels to keep (filters tiny fragments).

        Returns:
            List of (outer_pts, holes) where:
              - outer_pts : list of [x, y] pixel coords for the exterior ring (not closed).
              - holes     : list of lists of [x, y] pixel coords for interior hole rings.
        """
        contours, hierarchy = cv2.findContours(
            wall_mask,
            cv2.RETR_CCOMP,        # 2-level: top-level walls + their interior holes
            cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None or len(contours) == 0:
            logger.warning("No wall contours found in wall mask.")
            return []

        hierarchy = hierarchy[0]  # shape: (N, 4) → [next, prev, first_child, parent]
        wall_polygons = []

        for i, contour in enumerate(contours):
            # Only process top-level contours (no parent)
            if hierarchy[i][3] != -1:
                continue

            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Simplify outer ring — tight epsilon preserves wall corners
            eps = 0.004 * cv2.arcLength(contour, True)
            outer = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)

            # Collect child contours (room interiors inside the wall blob = holes)
            holes = []
            child_idx = hierarchy[i][2]   # index of first child
            while child_idx != -1:
                child_contour = contours[child_idx]
                child_area    = cv2.contourArea(child_contour)
                if child_area > 100:
                    eps_c = 0.004 * cv2.arcLength(child_contour, True)
                    hole  = cv2.approxPolyDP(child_contour, eps_c, True).reshape(-1, 2)
                    holes.append(hole.tolist())
                child_idx = hierarchy[child_idx][0]   # next sibling

            wall_polygons.append((outer.tolist(), holes))
            logger.debug(
                f"Wall polygon {len(wall_polygons)}: area={area:.0f}px\u00b2, "
                f"outer_pts={len(outer)}, holes={len(holes)}"
            )

        logger.info(f"Detected {len(wall_polygons)} wall polygon(s) from mask.")
        return wall_polygons

    def detect_room_interiors(
        self,
        raw_image: np.ndarray,
        min_area: int = 3000,
    ) -> List[Tuple]:
        """
        Extract filled room polygons directly from the floor plan image.

        Rooms in floor plan images are naturally white/light spaces completely
        enclosed by thick dark walls. This method:
        1. Converts to grayscale (if color input).
        2. Thresholds: bright pixels (>180) become white = 'open space'.
        3. Flood-fills from all four image borders to remove the exterior page margin.
        4. The remaining white blobs are the enclosed room interiors.
        5. Finds contours and returns (outer_pts, []) tuples.

        Args:
            raw_image : Original floor plan image (BGR or grayscale numpy array).
            min_area  : Minimum room area in pixels (filters small fixtures/noise).

        Returns:
            List of (outer_pts, holes) where:
              - outer_pts : list of [x, y] pixel coords for the room boundary.
              - holes     : always empty [] for room interior polygons.
        """
        # Step 1: Ensure grayscale
        if len(raw_image.shape) == 3:
            gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = raw_image.copy()

        h, w = gray.shape[:2]

        # Step 2: Mild blur to reduce JPEG noise, then threshold: bright > 180 = white
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bright = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

        # Step 3: Flood-fill from all border pixels to remove exterior page margin.
        # Exterior is the large white region outside the building boundary walls.
        flood = bright.copy()
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        for x in range(w):
            if flood[0, x] == 255:
                cv2.floodFill(flood, ff_mask, (x, 0), 128)
            if flood[h - 1, x] == 255:
                cv2.floodFill(flood, ff_mask, (x, h - 1), 128)
        for y in range(h):
            if flood[y, 0] == 255:
                cv2.floodFill(flood, ff_mask, (0, y), 128)
            if flood[y, w - 1] == 255:
                cv2.floodFill(flood, ff_mask, (w - 1, y), 128)

        # Step 4: White pixels that were NOT flooded = enclosed room interiors
        rooms_only = np.zeros_like(flood)
        rooms_only[flood == 255] = 255

        logger.debug(
            f"Room interior pixels found: {(rooms_only > 0).sum()} "
            f"(out of {h * w} total)"
        )

        # Step 5: Find contours of each room blob
        contours, _ = cv2.findContours(
            rooms_only,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        room_polygons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Simplify contour — preserve room corners
            eps = 0.003 * cv2.arcLength(contour, True)
            pts = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)

            room_polygons.append((pts.tolist(), []))

            logger.debug(
                f"Room interior {len(room_polygons)}: "
                f"area={area:.0f}px\u00b2, pts={len(pts)}"
            )

        logger.info(f"Detected {len(room_polygons)} room interior(s) from floor plan image.")
        return room_polygons


    def _get_skeleton(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Compute the skeleton (centerlines) of the binary image using iterative erosion.
        """
        img = binary_image.copy()
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            if cv2.countNonZero(img) == 0:
                done = True

        return skel

    def detect_wall_polygons_from_lines(self, wall_mask: np.ndarray) -> List[Tuple]:
        """
        Extract wall polygons by skeletonizing the wall mask, running HoughLinesP
        to find single centerlines, and buffering those lines to create solid volumes.
        """
        from shapely.geometry import LineString
        from shapely.ops import unary_union

        # 1. Skeletonize to get single centerlines
        skel = self._get_skeleton(wall_mask)

        # 2. Run HoughLinesP on the skeleton
        edges = cv2.Canny(skel, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            skel,  # Use skeleton directly instead of edges! Edges on singlepx = same
            rho=1,
            theta=np.pi / 180,
            threshold=15,    # adaptive for single-px skeletons
            minLineLength=15, # retains short stubs & division walls 
            maxLineGap=self._hough_max_line_gap

        )

        if lines is None:
            logger.warning("No line paths detected for walls on skeleton.")
            return []

        logger.info(f"HoughLines detected {len(lines)} paths on skeleton.")

        # 3. Dilate and Erode (Morphological Closing) to bridge gaps in junctions
        buffered_polys = []
        thickness = self._wall_thickness if self._wall_thickness > 1 else 3

        # D = Dilate radius to bridge gaps. E = Erode radius to restore thickness.
        # D - E = thickness. 12 - 9 = 3. Bridges gaps up to 24px.
        D = 12
        E = D - thickness if (D - thickness) > 0 else 0

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2 and y1 == y2:
                    continue
                ln = LineString([(x1, y1), (x2, y2)])
                # Buffer with radius D for overlap
                buffered = ln.buffer(D, cap_style=1, join_style=1)
                if not buffered.is_empty:
                    buffered_polys.append(buffered)

        if not buffered_polys:
            return []

        # Merge overlapping dilated lines
        merged_shape = unary_union(buffered_polys)

        # Erode back to restore original wall thickness
        if E > 0:
            shrunk_shape = merged_shape.buffer(-E)
        else:
            shrunk_shape = merged_shape

        wall_polygons = []
        # Handle Multipolygon list or simple single polygon
        if shrunk_shape.geom_type == 'Polygon':
            shapes = [shrunk_shape]
        elif shrunk_shape.geom_type == 'MultiPolygon':
            shapes = shrunk_shape.geoms
        else:
            shapes = []

        for poly in shapes:
            if poly.is_empty:
                continue

            ext_coords = np.array(poly.exterior.coords)
            holes = []
            for hole in poly.interiors:
                 holes.append(np.array(hole.coords).tolist())

            wall_polygons.append((ext_coords.tolist(), holes))

        logger.info(f"Buffered and merged {len(wall_polygons)} connected wall polygon(s).")
        return wall_polygons



def create_detector(config: Dict[str, Any]) -> "RoomDetector":

    """
    Factory function to create a RoomDetector instance.

    Args:
        config: Configuration dictionary

    Returns:
        RoomDetector: Configured detector instance
    """
    return RoomDetector(config)
