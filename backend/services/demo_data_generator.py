"""
Generates three SIMULATED road images (standing in for BMD-45 samples) and a
matching JSON detection manifest for each, so that DEMO MODE bounding boxes
always exactly match what's visually drawn on the image.

These are schematic illustrations (sky / buildings / road / lane markings /
colored vehicle blocks), NOT real photographs, and NOT claimed to be live
Bengaluru CCTV. Run this once to (re)generate backend/data/*.

Usage:
    python -m services.demo_data_generator
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 800, 450

CLASS_COLORS = {
    "car": (66, 133, 244),
    "two_wheeler": (255, 159, 28),
    "auto": (255, 214, 10),
    "bus": (239, 68, 68),
    "truck": (168, 85, 247),
    "pedestrian": (34, 197, 94),
    "obstacle": (148, 163, 184),
}

# Deterministic vehicle-count "recipes" per road so the demo consistently
# tells the LOW / MEDIUM / HIGH story described in the project brief.
ROAD_RECIPES = {
    "A": {"car": 20, "two_wheeler": 10, "auto": 5, "bus": 7, "truck": 5,
          "pedestrian": 6, "obstacle": 1, "seed": 101, "label": "MG Road Junction"},
    "B": {"car": 15, "two_wheeler": 8, "auto": 2, "bus": 1, "truck": 0,
          "pedestrian": 3, "obstacle": 0, "seed": 202, "label": "Outer Ring Road Bypass"},
    "C": {"car": 31, "two_wheeler": 10, "auto": 3, "bus": 2, "truck": 2,
          "pedestrian": 5, "obstacle": 1, "seed": 303, "label": "Silk Board Approach"},
}


def _draw_background(draw: ImageDraw.ImageDraw, label: str):
    # sky
    draw.rectangle([0, 0, IMG_W, 140], fill=(20, 28, 42))
    # distant buildings (simple skyline silhouette)
    rng = random.Random(7)
    x = 0
    while x < IMG_W:
        w = rng.randint(30, 70)
        h = rng.randint(30, 90)
        draw.rectangle([x, 140 - h, x + w, 140], fill=(30, 41, 59))
        x += w + rng.randint(4, 14)
    # road surface
    draw.rectangle([0, 140, IMG_W, IMG_H], fill=(38, 42, 48))
    # footpaths
    draw.rectangle([0, 140, IMG_W, 165], fill=(58, 63, 70))
    draw.rectangle([0, IMG_H - 25, IMG_W, IMG_H], fill=(58, 63, 70))
    # lane markings
    lane_y_positions = [215, 280, 345]
    for ly in lane_y_positions:
        for lx in range(10, IMG_W, 46):
            draw.rectangle([lx, ly, lx + 24, ly + 4], fill=(226, 232, 240))
    # road label plate
    draw.rectangle([16, 16, 300, 46], fill=(15, 20, 30))
    draw.text((26, 22), f"BMD-45 SIM: {label}", fill=(148, 233, 189))


def _place_vehicles(draw: ImageDraw.ImageDraw, recipe: dict, seed: int):
    rng = random.Random(seed)
    detections = []
    occupied = []

    def overlaps(box):
        for ob in occupied:
            if not (box[2] < ob[0] or box[0] > ob[2] or box[3] < ob[1] or box[1] > ob[3]):
                return True
        return False

    dims = {
        "car": (46, 26), "two_wheeler": (18, 30), "auto": (28, 30),
        "bus": (78, 34), "truck": (70, 34), "pedestrian": (10, 24),
        "obstacle": (22, 22),
    }

    order = []
    for cls, count in recipe.items():
        if cls in dims:
            order += [cls] * count
    rng.shuffle(order)

    for cls in order:
        w, h = dims[cls]
        placed = False
        for _ in range(60):
            x = rng.randint(20, IMG_W - w - 20)
            y = rng.randint(170, IMG_H - h - 30)
            box = [x, y, x + w, y + h]
            if not overlaps(box):
                occupied.append(box)
                placed = True
                break
        if not placed:
            continue
        color = CLASS_COLORS[cls]
        draw.rectangle(box, fill=color, outline=(15, 20, 30), width=1)
        confidence = round(rng.uniform(0.78, 0.98), 2)
        detections.append({"cls": cls, "confidence": confidence, "box": [float(v) for v in box]})

    return detections


def generate_all():
    for road_id, recipe in ROAD_RECIPES.items():
        img = Image.new("RGB", (IMG_W, IMG_H), (20, 28, 42))
        draw = ImageDraw.Draw(img)
        _draw_background(draw, recipe["label"])
        detections = _place_vehicles(draw, recipe, recipe["seed"])

        img_path = DATA_DIR / f"road_{road_id.lower()}.jpg"
        img.save(img_path, quality=90)

        manifest = {
            "roadId": road_id,
            "label": recipe["label"],
            "detections": detections,
        }
        with open(DATA_DIR / f"road_{road_id.lower()}_demo.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Generated {img_path} with {len(detections)} demo detections")


if __name__ == "__main__":
    generate_all()
