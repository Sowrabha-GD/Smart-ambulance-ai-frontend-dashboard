from fastapi import APIRouter

from services import routing_service
import config
from models.schemas import OptimizeRouteRequest, OptimizeRouteResponse, RouteScore

router = APIRouter()


@router.post("/optimize-route", response_model=OptimizeRouteResponse)
async def optimize_route(payload: OptimizeRouteRequest):
    """
    Runs Weighted Dijkstra over the road graph using the traffic scores
    produced by the AI Traffic Analysis stage, and returns the lowest
    predicted-cost route.
    """
    road_traffic_scores = {r.id.upper(): r.trafficScore for r in payload.roads}

    best_road_id, edge_meta, ranked = routing_service.dijkstra_best_road(road_traffic_scores)

    routes = []
    for road_id, meta in ranked:
        eta = routing_service.estimate_eta_minutes(
            meta["distance"], meta["signal_delay"], meta["trafficScore"]
        )
        routes.append(RouteScore(
            id=road_id,
            trafficScore=meta["trafficScore"],
            edgeCost=round(meta["cost"], 2),
            distanceKm=meta["distance"],
            etaMinutes=eta,
            junctions=config.ROUTE_JUNCTIONS.get(road_id, []),
        ))

    best_meta = edge_meta[best_road_id]
    best_eta = routing_service.estimate_eta_minutes(
        best_meta["distance"], best_meta["signal_delay"], best_meta["trafficScore"]
    )

    return OptimizeRouteResponse(
        recommendedRoute=best_road_id,
        routes=routes,
        estimatedDelay=best_eta,
        reason="Lowest predicted travel cost (distance + traffic + blockage + signal delay)",
    )
