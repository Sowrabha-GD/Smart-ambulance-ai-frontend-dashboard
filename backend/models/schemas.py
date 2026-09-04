from typing import List, Optional, Dict
from pydantic import BaseModel


class Detection(BaseModel):
    cls: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2] in pixels


class TrafficAnalysisResponse(BaseModel):
    roadId: str
    mode: str  # "LIVE_YOLO" | "DEMO"
    vehicleCounts: Dict[str, int]
    totalVehicles: int
    pedestrians: Optional[int] = None
    obstacles: Optional[int] = None
    trafficScore: float
    trafficLevel: str
    colorState: str
    detections: List[Detection]
    originalImage: str  # base64 data URI
    annotatedImage: str  # base64 data URI


class RoadInput(BaseModel):
    id: str
    trafficScore: float


class OptimizeRouteRequest(BaseModel):
    roads: List[RoadInput]


class RouteScore(BaseModel):
    id: str
    trafficScore: float
    edgeCost: float
    distanceKm: float
    etaMinutes: float
    junctions: List[str]


class OptimizeRouteResponse(BaseModel):
    recommendedRoute: str
    routes: List[RouteScore]
    estimatedDelay: float
    reason: str


class GreenCorridorRequest(BaseModel):
    route: str


class GreenCorridorResponse(BaseModel):
    active: bool
    route: str
    junctions: List[str]


class Hospital(BaseModel):
    id: str
    name: str
    distanceKm: float
    beds: int
    icu: int
    doctors: int
    trafficPercent: float
    score: float
    bestMatch: bool


class HospitalListResponse(BaseModel):
    hospitals: List[Hospital]


class SystemStatusResponse(BaseModel):
    aiMode: str  # "LIVE_YOLO" | "DEMO"
    modelPath: str
    weightsFound: bool
    message: str
