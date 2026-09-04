from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from services import yolo_service, traffic_service
from models.schemas import TrafficAnalysisResponse

router = APIRouter()


@router.post("/analyze-traffic", response_model=TrafficAnalysisResponse)
async def analyze_traffic(road_id: str, image: Optional[UploadFile] = File(None)):
    """
    Runs the perception pipeline for one road:
    BMD-45 image -> YOLO (or DEMO manifest) -> vehicle counts -> traffic score.
    """
    road_id = road_id.upper()
    if road_id not in ("A", "B", "C"):
        raise HTTPException(status_code=400, detail="road_id must be one of A, B, C")

    image_bytes = await image.read() if image is not None else None

    try:
        perception = yolo_service.analyze_road(road_id, image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"YOLO inference failed: {exc}")

    detections = perception["detections"]
    summary = traffic_service.summarize_detections(detections)
    score = traffic_service.compute_traffic_score(summary["vehicleCounts"])
    level, color_state = traffic_service.classify_traffic(score)

    return TrafficAnalysisResponse(
        roadId=road_id,
        mode=perception["mode"],
        vehicleCounts=summary["vehicleCounts"],
        totalVehicles=summary["totalVehicles"],
        pedestrians=summary["pedestrians"],
        obstacles=summary["obstacles"],
        trafficScore=score,
        trafficLevel=level,
        colorState=color_state,
        detections=detections,
        originalImage=perception["originalImage"],
        annotatedImage=perception["annotatedImage"],
    )


@router.get("/status")
async def system_status():
    return JSONResponse(yolo_service.get_status())
