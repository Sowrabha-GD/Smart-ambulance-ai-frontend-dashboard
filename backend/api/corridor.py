from fastapi import APIRouter, HTTPException

import config
from models.schemas import GreenCorridorRequest, GreenCorridorResponse
from services import hospital_service

router = APIRouter()


@router.post("/activate-green-corridor", response_model=GreenCorridorResponse)
async def activate_green_corridor(payload: GreenCorridorRequest):
    route = payload.route.upper()
    junctions = config.ROUTE_JUNCTIONS.get(route)
    if junctions is None:
        raise HTTPException(status_code=400, detail=f"Unknown route '{route}'")

    return GreenCorridorResponse(active=True, route=route, junctions=junctions)


@router.get("/hospitals")
async def get_hospitals(route_cost: float = 0.0, route_traffic_score: float = 0.0):
    hospitals = hospital_service.recommend_hospitals(route_cost, route_traffic_score)
    return {"hospitals": hospitals}
