import math
import logging
from shapely.geometry import shape

logger = logging.getLogger(__name__)

def build_graph(geojson_obj: dict) -> dict:
    """
    Builds a connectivity graph from a GeoJSON FeatureCollection containing zones.
    Nodes are zones, edges represent pathways (walkable adjacent boundaries or stairs).
    """
    features = geojson_obj.get("features", [])
    nodes = []
    edges = []
    
    # Scale calculation factors 
    # 1 deg lat ~ 111,132 meters
    # A safe buffer in degrees representing ~ 1.5 meters overlap
    BUFFER_THRESHOLD_DEG = 1.5 / 111132.0 
    
    # 1. Gather all zone and wall features
    zones = []
    walls = []
    for f in features:
        props = f.get("properties", {})
        if props.get("is_wall", False):
            walls.append(f)
        elif f.get("id") is not None:
            zones.append(f)
             
    logger.info(f"Building navigation graph with {len(zones)} zone nodes.")
    
    # 2. Setup Nodes
    for f in zones:
        props = f["properties"]
        try:
            poly_node = shape(f["geometry"])
            centroid = [poly_node.centroid.x, poly_node.centroid.y]
        except:
            centroid = None
        nodes.append({
            "id": f["id"],
            "floor": props.get("floor_num", 0),
            "name": props.get("room_type", "unknown"),
            "color": props.get("color"),
            "base_height": props.get("base_height", 0.0),
            "centroid": centroid
        })
        
    # 3. Setup Spatial Indexes / Lookup dictionaries
    zones_by_floor = {}
    for f in zones:
        floor = f["properties"].get("floor_num", 0)
        if floor not in zones_by_floor:
            zones_by_floor[floor] = []
        zones_by_floor[floor].append(f)
        
    # 4. Intra-Floor Navigation (Connecting adjacent rooms)
    from shapely.ops import unary_union
    for floor, floor_zones in zones_by_floor.items():
        # Merge walls for this floor
        floor_walls = [f for f in walls if f["properties"].get("floor_num", 0) == floor]
        merged_walls = None
        if floor_walls:
            try:
                merged_walls = unary_union([shape(f["geometry"]) for f in floor_walls])
            except Exception as e:
                logger.warning(f"Failed to merge walls on floor {floor}: {e}")

        for i in range(len(floor_zones)):
            for j in range(i + 1, len(floor_zones)):
                f_a = floor_zones[i]
                f_b = floor_zones[j]
                
                try:
                    poly_a = shape(f_a["geometry"])
                    poly_b = shape(f_b["geometry"])
                    
                    # Buffer one to check overlap for adjacency
                    # buffer() defaults to degree-like units for lat/lon representation
                    # Compute a THIN shared border strip for precise wall collisions
                    BUFFER_WALL_TEST = 0.8 / 111132.0 # ~80 cm
                    overlap_thin = poly_a.buffer(BUFFER_WALL_TEST).intersection(poly_b.buffer(BUFFER_WALL_TEST))
                    
                    # Original heavy buffer for weight distance heuristics
                    overlap_buff = poly_a.buffer(BUFFER_THRESHOLD_DEG).intersection(poly_b.buffer(BUFFER_THRESHOLD_DEG))

                    if not overlap_thin.is_empty:
                        # Check if wall blocks it
                        if merged_walls:
                             walkway = overlap_thin.difference(merged_walls)
                             if walkway.is_empty:
                                 continue
                             
                             # Erode remaining shard sides (since thin overlap leaves edges)
                             eroded = walkway.buffer(-1.0e-6)
                             
                             if eroded.is_empty or eroded.area < 1.0e-14:
                                 continue # Blocked!
                        
                        # Edges need weights: Distance between centroids in meters
                        c_a = [poly_a.centroid.x, poly_a.centroid.y]
                        c_b = [poly_b.centroid.x, poly_b.centroid.y]
                        if True:
                            dy = (c_a[1] - c_b[1]) * 111132.0
                            dx = (c_a[0] - c_b[0]) * 111132.0 * math.cos(math.radians(c_a[1]))
                            distance = math.sqrt(dx*dx + dy*dy)
                            
                            # Calculate waypoint at shared boundary
                            overlap = poly_a.intersection(poly_b.buffer(BUFFER_THRESHOLD_DEG))
                            waypoint = [overlap.centroid.x, overlap.centroid.y] if not overlap.is_empty else None

                            edges.append({
                                "u": f_a["id"],
                                "v": f_b["id"],
                                "weight": distance,
                                "type": "walk",
                                "waypoint": waypoint
                            })
                except Exception as e:
                    logger.warning(f"Failed spatial adjacency check: {e}")

    # 5. Inter-Floor Navigation (Stairs / Elevators)
    sorted_floors = sorted(zones_by_floor.keys())
    for f_idx in range(len(sorted_floors) - 1):
        floor_a = sorted_floors[f_idx]
        floor_b = sorted_floors[f_idx + 1]
        
        for f_a in zones_by_floor[floor_a]:
            room_type_a = f_a["properties"].get("room_type", "").lower()
            if "stairs" in room_type_a or "lift" in room_type_a or "elevator" in room_type_a:
                for f_b in zones_by_floor[floor_b]:
                    room_type_b = f_b["properties"].get("room_type", "").lower()
                    if "stairs" in room_type_b or "lift" in room_type_b or "elevator" in room_type_b:
                        try:
                            poly_a = shape(f_a["geometry"])
                            poly_b = shape(f_b["geometry"])
                            
                            # Standard staircase overlaps on 2d space
                            if poly_a.intersects(poly_b):
                                elev_diff = abs(f_a["properties"].get("base_height", 0.0) - f_b["properties"].get("base_height", 0.0))
                                weight = elev_diff if elev_diff > 0 else 3.0 # Fallback 
                                
                                overlap = poly_a.intersection(poly_b)
                                waypoint = [overlap.centroid.x, overlap.centroid.y] if not overlap.is_empty else None

                                edges.append({
                                    "u": f_a["id"],
                                    "v": f_b["id"],
                                    "weight": weight,
                                    "type": "stairs",
                                    "waypoint": waypoint
                                })
                        except Exception as e:
                             logger.warning(f"Failed spatial staircase check: {e}")

    return {"nodes": nodes, "edges": edges}
