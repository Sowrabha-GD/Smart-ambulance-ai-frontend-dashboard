from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import sumo_service

router = APIRouter()


class StartSimulationRequest(BaseModel):
    roadId: str
    roadTrafficScores: Optional[Dict[str, float]] = None


@router.get("/sumo/status")
async def sumo_status():
    """Reports whether a real SUMO install (sumo binary + traci) is
    available on this machine — LIVE_SUMO or UNAVAILABLE, same pattern as
    /api/status for the YOLO model."""
    return sumo_service.get_status()


@router.get("/sumo/network")
async def sumo_network():
    """Static road-network geometry (edges + junctions) for the frontend
    to draw once. Reads the actual .net.xml via sumolib."""
    status = sumo_service.get_status()
    if status["mode"] != "LIVE_SUMO":
        raise HTTPException(status_code=503, detail=status["message"])
    return sumo_service.get_network_geometry()


@router.post("/sumo/start")
async def sumo_start(payload: StartSimulationRequest):
    road_id = payload.roadId.upper()
    if road_id not in ("A", "B", "C"):
        raise HTTPException(status_code=400, detail=f"Unknown road '{road_id}'")
    result = sumo_service.start_simulation(road_id, payload.roadTrafficScores)
    if not result.get("started"):
        raise HTTPException(status_code=503, detail=result.get("message", "SUMO unavailable"))
    return result


@router.post("/sumo/stop")
async def sumo_stop():
    return sumo_service.stop_simulation()


@router.get("/sumo/state")
async def sumo_state():
    """Polled every ~300ms by the frontend while a simulation is running."""
    return sumo_service.get_state()
