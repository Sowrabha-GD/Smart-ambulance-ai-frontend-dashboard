"""
Converts raw YOLO detections into vehicle counts, a traffic score, and a
traffic status level. Weights and thresholds are pulled from config.py so
they can be changed without touching this logic or the frontend.
"""
import config

VEHICLE_CLASSES = ["car", "two_wheeler", "auto", "bus", "truck"]
NON_VEHICLE_CLASSES = ["pedestrian", "obstacle"]


def summarize_detections(detections: list) -> dict:
    counts = {cls: 0 for cls in VEHICLE_CLASSES}
    non_vehicle_counts = {cls: 0 for cls in NON_VEHICLE_CLASSES}

    for det in detections:
        cls = det["cls"]
        if cls in counts:
            counts[cls] += 1
        elif cls in non_vehicle_counts:
            non_vehicle_counts[cls] += 1
        # unknown classes are ignored for scoring but not silently dropped
        # from the raw `detections` list returned to the frontend

    total_vehicles = sum(counts.values())
    return {
        "vehicleCounts": counts,
        "totalVehicles": total_vehicles,
        "pedestrians": non_vehicle_counts["pedestrian"] if any(
            d["cls"] == "pedestrian" for d in detections
        ) or non_vehicle_counts["pedestrian"] > 0 else None,
        "obstacles": non_vehicle_counts["obstacle"] if any(
            d["cls"] == "obstacle" for d in detections
        ) or non_vehicle_counts["obstacle"] > 0 else None,
    }


def compute_traffic_score(vehicle_counts: dict) -> float:
    """Project heuristic — NOT an official traffic-engineering formula."""
    score = 0.0
    for cls, count in vehicle_counts.items():
        weight = config.TRAFFIC_WEIGHTS.get(cls, 0.0)
        score += count * weight
    return round(score, 1)


def classify_traffic(score: float):
    """Returns (level, color_state) based on config.TRAFFIC_THRESHOLDS."""
    for upper_bound, label, color in config.TRAFFIC_THRESHOLDS:
        if score <= upper_bound:
            return label, color
    return "SEVERE", "critical_red"
