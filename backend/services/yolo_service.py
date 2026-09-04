"""
YOLO vehicle-detection service for SMART AMBULANCE CORRIDOR AI.

MODE SELECTION (checked once at startup, re-checkable via reload_mode()):
    LIVE_YOLO  -> `ultralytics` is installed AND MODEL_PATH exists on disk
                  AND FORCE_DEMO_MODE is not set.
                  Runs real YOLO inference on the given image.
    DEMO       -> otherwise. Loads a pre-generated, clearly-labeled demo
                  detection manifest (see demo_data_generator.py). No
                  detection values are invented at request time; they are
                  fixed sample data that ships with the repo.

Both modes return the exact same response shape so the frontend never needs
to know which mode produced the data (it just reads `mode` for the badge).
"""
import base64
import io
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import config

_YOLO_MODEL = None
_MODE = None  # "LIVE_YOLO" | "DEMO"


def _try_load_yolo():
    """Attempt to load ultralytics + weights. Returns (model, ok, message)."""
    if config.FORCE_DEMO_MODE:
        return None, False, "FORCE_DEMO_MODE is enabled."

    weights_path = Path(config.MODEL_PATH)
    if not weights_path.exists():
        return None, False, f"YOLO weights not found at {weights_path}. Running DEMO MODE."

    try:
        from ultralytics import YOLO
    except ImportError:
        return None, False, "ultralytics package not installed. Running DEMO MODE."

    try:
        model = YOLO(str(weights_path))
        return model, True, f"Loaded YOLO weights from {weights_path}."
    except Exception as exc:  # pragma: no cover - defensive
        return None, False, f"Failed to load YOLO weights ({exc}). Running DEMO MODE."


def get_mode() -> str:
    global _MODE, _YOLO_MODEL
    if _MODE is None:
        model, ok, _ = _try_load_yolo()
        _YOLO_MODEL = model
        _MODE = "LIVE_YOLO" if ok else "DEMO"
    return _MODE


def get_status() -> dict:
    model, ok, message = _try_load_yolo()
    return {
        "aiMode": "LIVE_YOLO" if ok else "DEMO",
        "modelPath": config.MODEL_PATH,
        "weightsFound": Path(config.MODEL_PATH).exists(),
        "message": message,
    }


def _img_to_data_uri(img_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _draw_boxes(img_bgr, detections: list):
    annotated = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = _class_color(det["cls"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f'{det["cls"].upper()} {det["confidence"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (15, 20, 30), 1, cv2.LINE_AA)
    return annotated


_COLOR_MAP = {
    "car": (244, 133, 66), "two_wheeler": (28, 159, 255), "auto": (10, 214, 255),
    "bus": (68, 68, 239), "truck": (247, 85, 168), "pedestrian": (94, 197, 34),
    "obstacle": (184, 163, 148),
}


def _class_color(cls: str):
    return _COLOR_MAP.get(cls, (200, 200, 200))


def _load_demo_manifest(road_id: str) -> dict:
    manifest_path = config.DATA_DIR / f"road_{road_id.lower()}_demo.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Demo manifest missing for road {road_id}. "
            f"Run `python -m services.demo_data_generator` inside backend/."
        )
    with open(manifest_path) as f:
        return json.load(f)


def analyze_road(road_id: str, image_bytes: Optional[bytes] = None) -> dict:
    """
    Analyze a road image and return raw detections + both original and
    annotated images as base64 data URIs. Does NOT compute traffic score
    (that's traffic_service's job) so this stays a pure perception step.
    """
    mode = get_mode()

    if mode == "LIVE_YOLO" and _YOLO_MODEL is not None:
        if image_bytes is None:
            image_path = config.ROAD_IMAGE_MAP.get(road_id.upper())
            if image_path is None or not image_path.exists():
                raise FileNotFoundError(f"No sample image configured for road {road_id}")
            image_bytes = image_path.read_bytes()

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        results = _YOLO_MODEL.predict(pil_img, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
        detections = []
        r = results[0]
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            conf = float(box.conf[0])
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({"cls": cls_name, "confidence": round(conf, 2), "box": xyxy})

        annotated_bgr = _draw_boxes(img_bgr, detections)
        return {
            "mode": "LIVE_YOLO",
            "detections": detections,
            "originalImage": _img_to_data_uri(img_bgr),
            "annotatedImage": _img_to_data_uri(annotated_bgr),
        }

    # ---- DEMO MODE ----
    manifest = _load_demo_manifest(road_id)
    image_path = config.DATA_DIR / f"road_{road_id.lower()}.jpg"
    pil_img = Image.open(image_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    detections = manifest["detections"]
    annotated_bgr = _draw_boxes(img_bgr, detections)

    return {
        "mode": "DEMO",
        "detections": detections,
        "originalImage": _img_to_data_uri(img_bgr),
        "annotatedImage": _img_to_data_uri(annotated_bgr),
    }
