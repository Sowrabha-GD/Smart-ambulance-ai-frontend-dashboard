"""
Hospital recommendation — scores each hospital using capacity, ICU
availability, doctor count, distance, and the AI-computed route cost, then
flags the best match. Simulated capacity data, not a real hospital feed.
"""
import config


def recommend_hospitals(route_cost: float = 0.0, route_traffic_score: float = 0.0):
    scored = []
    w = config.HOSPITAL_WEIGHTS

    for h in config.HOSPITALS:
        score = (
            w["beds"] * h["beds"]
            + w["icu"] * h["icu"]
            + w["doctors"] * h["doctors"]
            + w["distance"] * h["distance_km"]
            + w["route_cost"] * route_cost
        )
        traffic_percent = min(100.0, round(route_traffic_score, 1))
        scored.append({
            "id": h["id"],
            "name": h["name"],
            "distanceKm": h["distance_km"],
            "beds": h["beds"],
            "icu": h["icu"],
            "doctors": h["doctors"],
            "trafficPercent": traffic_percent,
            "score": round(score, 2),
            "bestMatch": False,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    if scored:
        scored[0]["bestMatch"] = True
    return scored
