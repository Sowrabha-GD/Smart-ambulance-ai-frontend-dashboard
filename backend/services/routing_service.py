"""
Smart Route Engine — treats the road network as a weighted graph
(junction = node, road segment = edge) and runs Dijkstra where each edge's
cost is a function of distance, live traffic score, blockage penalty, and
signal delay. This deliberately does NOT just take min(trafficScore) —
see edge_cost() below.
"""
import heapq

import config


def edge_cost(distance: float, traffic_score: float, blockage_penalty: float, signal_delay: float) -> float:
    w = config.ROUTING_WEIGHTS
    return (
        w["distance_weight"] * distance
        + w["traffic_weight"] * traffic_score
        + w["blockage_weight"] * blockage_penalty
        + w["signal_weight"] * signal_delay
    )


def _build_graph(road_traffic_scores: dict):
    """
    Builds an adjacency list graph from config.ROAD_NETWORK, injecting the
    live traffic score for each road. Each road is one edge from its
    junction to the hospital (this prototype uses a simple star topology —
    see README for how to extend to a multi-hop graph).
    """
    graph = {}
    edge_meta = {}
    for road_id, meta in config.ROAD_NETWORK.items():
        traffic_score = road_traffic_scores.get(road_id, 0.0)
        cost = edge_cost(meta["distance"], traffic_score, meta["blockage_penalty"], meta["signal_delay"])
        graph.setdefault(meta["from"], []).append((meta["to"], cost, road_id))
        edge_meta[road_id] = {**meta, "trafficScore": traffic_score, "cost": cost}
    return graph, edge_meta


def dijkstra_best_road(road_traffic_scores: dict, source: str = "J1", target: str = "HOSPITAL"):
    """
    Runs Dijkstra over the road graph and returns the lowest-cost road id
    directly connecting source->target, plus full per-road cost breakdown
    (useful for the AI Route Recommendation comparison panel).
    """
    graph, edge_meta = _build_graph(road_traffic_scores)

    # Standard Dijkstra shortest-path-cost computation (kept general so the
    # graph can be extended to multi-hop junctions later without rewriting
    # this function).
    dist = {source: 0.0}
    came_via_road = {}
    pq = [(0.0, source)]
    visited = set()

    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        for neighbor, cost, road_id in graph.get(node, []):
            nd = d + cost
            if neighbor not in dist or nd < dist[neighbor]:
                dist[neighbor] = nd
                came_via_road[neighbor] = road_id
                heapq.heappush(pq, (nd, neighbor))

    best_road_id = came_via_road.get(target)

    # Rank all candidate roads by cost for the comparison UI
    ranked = sorted(edge_meta.items(), key=lambda kv: kv[1]["cost"])
    best_road_id = best_road_id or ranked[0][0]

    return best_road_id, edge_meta, ranked


def estimate_eta_minutes(distance_km: float, signal_delay: float, traffic_score: float) -> float:
    """Rough ETA: base travel time at average speed + signal delay + a
    small congestion penalty proportional to traffic score."""
    base_minutes = (distance_km / config.AVERAGE_SPEED_KMPH) * 60
    congestion_penalty = traffic_score * 0.03
    return round(base_minutes + signal_delay + congestion_penalty, 1)
