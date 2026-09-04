"""
Central configuration for SMART AMBULANCE CORRIDOR AI backend.

Everything that should be tunable (model path, scoring weights, routing
coefficients, thresholds) lives here so the rest of the codebase never
hardcodes these values.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# YOLO model configuration
# ---------------------------------------------------------------------------
# Path to fine-tuned BMD-45 weights. If this file does not exist (or the
# `ultralytics` package is not installed), the backend automatically falls
# back to DEMO MODE and clearly labels every response as such.
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "weights" / "best.pt"))

# Confidence threshold used when running real YOLO inference.
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.35"))

# Force demo mode even if weights exist (useful for grading/demo laptops).
FORCE_DEMO_MODE = os.getenv("FORCE_DEMO_MODE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# BMD-45 sample data
# ---------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
ROAD_IMAGE_MAP = {
    "A": DATA_DIR / "road_a.jpg",
    "B": DATA_DIR / "road_b.jpg",
    "C": DATA_DIR / "road_c.jpg",
}

# ---------------------------------------------------------------------------
# Traffic scoring heuristic (PROJECT-DEFINED, not an official formula)
# ---------------------------------------------------------------------------
# Traffic Score = sum(count[class] * weight[class])
TRAFFIC_WEIGHTS = {
    "car": 1.0,
    "two_wheeler": 0.5,
    "auto": 1.0,
    "bus": 3.0,
    "truck": 3.0,
    # non-vehicle classes are tracked but do not contribute to the score
    "pedestrian": 0.0,
    "obstacle": 0.0,
}

# Traffic status thresholds -> (label, dashboard color state)
TRAFFIC_THRESHOLDS = [
    (30, "LOW", "green"),
    (60, "MEDIUM", "yellow"),
    (100, "HIGH", "red"),
    (float("inf"), "SEVERE", "critical_red"),
]

# ---------------------------------------------------------------------------
# Weighted Dijkstra routing coefficients
# ---------------------------------------------------------------------------
# edge_cost = distance_weight*distance + traffic_weight*traffic_score
#           + blockage_weight*blockage_penalty + signal_weight*signal_delay
ROUTING_WEIGHTS = {
    "distance_weight": 1.0,
    "traffic_weight": 1.5,
    "blockage_weight": 4.0,
    "signal_weight": 1.0,
}

# Static road network metadata (distance in km, blockage penalty, signal
# delay in minutes-equivalent units). Traffic score is injected at runtime
# from the AI traffic analysis stage.
ROAD_NETWORK = {
    "A": {"from": "J1", "to": "HOSPITAL", "distance": 6.4, "blockage_penalty": 0, "signal_delay": 3},
    "B": {"from": "J1", "to": "HOSPITAL", "distance": 7.1, "blockage_penalty": 0, "signal_delay": 1},
    "C": {"from": "J1", "to": "HOSPITAL", "distance": 5.8, "blockage_penalty": 1, "signal_delay": 2},
}

# Approx average speed (km/h) used to convert distance + delay into an ETA.
AVERAGE_SPEED_KMPH = 34.0

# ---------------------------------------------------------------------------
# Hospitals (simulated capacity data — NOT a real hospital integration)
# ---------------------------------------------------------------------------
HOSPITALS = [
    {
        "id": "metro_emergency",
        "name": "Metro Emergency Hospital",
        "distance_km": 6.2,
        "beds": 22,
        "icu": 5,
        "doctors": 7,
    },
    {
        "id": "cityview_general",
        "name": "CityView General Hospital",
        "distance_km": 8.9,
        "beds": 40,
        "icu": 9,
        "doctors": 15,
    },
    {
        "id": "sunrise_trauma",
        "name": "Sunrise Trauma Center",
        "distance_km": 4.7,
        "beds": 12,
        "icu": 3,
        "doctors": 5,
    },
]

# Hospital scoring weights (higher capacity / lower distance & traffic = better)
HOSPITAL_WEIGHTS = {
    "beds": 0.6,
    "icu": 2.0,
    "doctors": 0.8,
    "distance": -1.2,
    "route_cost": -0.8,
}

# ---------------------------------------------------------------------------
# Green corridor junction sequence per road
# ---------------------------------------------------------------------------
ROUTE_JUNCTIONS = {
    "A": ["J1", "J2", "HOSPITAL"],
    "B": ["J1", "J2", "J3", "HOSPITAL"],
    "C": ["J1", "J4", "HOSPITAL"],
}
