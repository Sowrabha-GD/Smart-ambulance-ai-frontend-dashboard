"""
SUMO (Simulation of Urban MObility) traffic simulation service for
SMART AMBULANCE CORRIDOR AI.

This is a REAL SUMO integration via TraCI — not a canned animation. It
launches an actual `sumo` process (headless) against the network in
backend/sumo/, steps it in a background thread, and:

  1. spawns background traffic on all three roads (A/B/C), scaled from the
     live traffic scores produced by the AI Traffic Analysis stage when
     available;
  2. inserts an ambulance vehicle onto whichever road the Weighted Dijkstra
     stage recommended;
  3. implements real Green Corridor signal preemption: as the ambulance
     approaches a signal-controlled junction (J2/J3/J4), TraCI forces that
     junction's approach link to green, then releases it back to the
     junction's normal actuated program once the ambulance has passed.

MODE SELECTION (mirrors yolo_service.py's LIVE/DEMO pattern):
    LIVE_SUMO    -> the `sumo` binary is on PATH and the `traci` package
                    imports successfully.
    UNAVAILABLE  -> otherwise. The API reports this clearly; it does not
                    fabricate simulation data.

Only one simulation run is supported at a time (a single background
thread + a single TraCI connection), which is sufficient for a demo
dashboard.
"""
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

SUMO_DIR = Path(__file__).resolve().parent.parent / "sumo"
NET_FILE = SUMO_DIR / "corridor.net.xml"
ROUTE_FILE = SUMO_DIR / "corridor.rou.xml"

STEP_LENGTH = 1.0          # seconds of sim time per TraCI step
REALTIME_STEP_DELAY = 0.15 # wall-clock delay between steps (watchable pace)
PREEMPT_DISTANCE_M = 250.0 # how far out the ambulance forces a green light
MAX_SIM_SECONDS = 900      # safety cap so a forgotten run doesn't loop forever

ROAD_ROUTE_MAP = {"A": "route_A", "B": "route_B", "C": "route_C"}
ROAD_ENTRY_EDGE = {"A": "e_J1_J2", "B": "e_J1_J2", "C": "e_J1_J4"}

# Base background-traffic rates (vehsPerHour) per road/vehicle-class,
# matching corridor.rou.xml. Scaled at runtime by the live traffic score.
BASE_FLOWS = {
    "A": {"car": 360, "two_wheeler": 240, "bus": 12},
    "B": {"car": 300, "two_wheeler": 180, "truck": 18},
    "C": {"car": 200, "two_wheeler": 150},
}

_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_stop_flag = threading.Event()
_state = {
    "running": False,
    "mode": "UNAVAILABLE",
    "message": "Simulation not started.",
    "simTime": 0.0,
    "roadId": None,
    "ambulance": None,
    "vehicles": [],
    "trafficLights": [],
    "greenCorridorActive": False,
    "arrived": False,
}


def _try_import_traci():
    try:
        import traci  # noqa: F401
        return traci, True, "traci import OK."
    except ImportError as exc:
        return None, False, f"traci package not installed ({exc})."


def get_status() -> dict:
    sumo_bin = shutil.which("sumo")
    traci_mod, traci_ok, traci_msg = _try_import_traci()
    net_ok = NET_FILE.exists() and ROUTE_FILE.exists()
    ok = bool(sumo_bin) and traci_ok and net_ok
    if not sumo_bin:
        message = "sumo binary not found on PATH. Install with: pip install eclipse-sumo traci sumolib"
    elif not traci_ok:
        message = traci_msg
    elif not net_ok:
        message = f"Missing SUMO network/route files in {SUMO_DIR}"
    else:
        message = f"SUMO ready ({sumo_bin})."
    return {
        "mode": "LIVE_SUMO" if ok else "UNAVAILABLE",
        "sumoBinary": sumo_bin,
        "networkFile": str(NET_FILE),
        "message": message,
    }


def get_network_geometry() -> dict:
    """Static geometry (edges + junctions) for the frontend to draw once."""
    import sumolib
    net = sumolib.net.readNet(str(NET_FILE))

    edges = []
    for edge in net.getEdges():
        shape = edge.getShape()
        edges.append({
            "id": edge.getID(),
            "from": edge.getFromNode().getID(),
            "to": edge.getToNode().getID(),
            "shape": [[round(x, 1), round(y, 1)] for x, y in shape],
            "numLanes": edge.getLaneNumber(),
        })

    junctions = []
    for node in net.getNodes():
        x, y = node.getCoord()
        junctions.append({
            "id": node.getID(),
            "x": round(x, 1),
            "y": round(y, 1),
            "type": node.getType(),
            "hasTLS": node.getType() == "traffic_light",
        })

    xmin, ymin, xmax, ymax = net.getBoundary()
    return {
        "edges": edges,
        "junctions": junctions,
        "boundary": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
        "roadEntryEdges": ROAD_ENTRY_EDGE,
    }


def _traffic_score_to_multiplier(score: Optional[float]) -> float:
    """LOW traffic ~0 score -> 0.5x flow, SEVERE ~120+ -> ~2.2x flow."""
    if score is None:
        return 1.0
    return max(0.4, min(2.5, 0.5 + score / 60.0))


def _run_loop(road_id: str, road_traffic_scores: Optional[dict]):
    import traci

    multipliers = {r: _traffic_score_to_multiplier((road_traffic_scores or {}).get(r)) for r in "ABC"}

    sumo_cmd = [
        "sumo",
        "-n", str(NET_FILE),
        "-r", str(ROUTE_FILE),
        "--no-warnings", "true",
        "--no-step-log", "true",
        "--step-length", str(STEP_LENGTH),
        "--time-to-teleport", "120",
        "--collision.action", "warn",
    ]

    try:
        traci.start(sumo_cmd, label=f"corridor-{time.time()}")
    except Exception as exc:  # pragma: no cover - defensive
        with _lock:
            _state.update(running=False, mode="UNAVAILABLE", message=f"Failed to launch SUMO: {exc}")
        return

    with _lock:
        _state.update(
            running=True, mode="LIVE_SUMO", roadId=road_id, simTime=0.0,
            arrived=False, greenCorridorActive=False,
            message=f"Simulation running on Road {road_id}.",
        )

    # Scale each road's background flow by its live traffic score, since
    # TraCI flows already loaded from corridor.rou.xml keep their file
    # rates — we approximate "more traffic" by periodically injecting a
    # few extra vehicles on congested roads rather than reloading routes.
    extra_inject_counter = 0.0

    ambulance_id = "ambulance_live"
    ambulance_inserted = False   # we've called traci.vehicle.add()
    ambulance_ever_seen = False  # it has actually appeared in getIDList() at least once
    held_tls = {}
    sim_time = 0.0

    try:
        while not _stop_flag.is_set() and sim_time < MAX_SIM_SECONDS:
            traci.simulationStep()
            sim_time += STEP_LENGTH

            # Insert the ambulance a couple of seconds in, so some
            # background traffic is already on the road.
            if not ambulance_inserted and sim_time >= 2.0:
                try:
                    traci.vehicle.add(
                        ambulance_id, ROAD_ROUTE_MAP[road_id],
                        typeID="ambulance", departLane="best", departSpeed="max",
                    )
                    ambulance_inserted = True
                except traci.exceptions.TraCIException:
                    pass

            # Periodically top up background traffic on higher-scored
            # roads so the visual congestion matches the AI analysis.
            extra_inject_counter += STEP_LENGTH
            if extra_inject_counter >= 8.0:
                extra_inject_counter = 0.0
                for r, mult in multipliers.items():
                    if mult > 1.15:
                        try:
                            vid = f"extra_{r}_{int(sim_time)}_{r}"
                            traci.vehicle.add(vid, ROAD_ROUTE_MAP[r], typeID="car", departLane="random")
                        except traci.exceptions.TraCIException:
                            pass

            current_ids = traci.vehicle.getIDList()
            amb_present = ambulance_inserted and ambulance_id in current_ids
            if amb_present:
                ambulance_ever_seen = True
            ambulance_payload = None
            green_active = False

            if amb_present:
                x, y = traci.vehicle.getPosition(ambulance_id)
                edge_id = traci.vehicle.getRoadID(ambulance_id)
                speed = traci.vehicle.getSpeed(ambulance_id)
                route = traci.vehicle.getRoute(ambulance_id)
                route_idx = traci.vehicle.getRouteIndex(ambulance_id)
                progress = 0.0
                if len(route) > 1:
                    progress = max(0.0, min(1.0, route_idx / (len(route) - 1)))

                nxt = traci.vehicle.getNextTLS(ambulance_id)
                next_tls_info = None
                if nxt:
                    tls_id, link_idx, dist, tls_state = nxt[0]
                    next_tls_info = {"id": tls_id, "distanceM": round(dist, 1), "state": tls_state}
                    if dist <= PREEMPT_DISTANCE_M:
                        held_tls[tls_id] = link_idx
                        traci.trafficlight.setLinkState(tls_id, link_idx, "G")
                        green_active = True

                # Release any junction the ambulance has already passed.
                still_next = nxt[0][0] if nxt else None
                for tls_id in list(held_tls):
                    if tls_id != still_next:
                        try:
                            traci.trafficlight.setProgram(tls_id, "0")
                        except traci.exceptions.TraCIException:
                            pass
                        del held_tls[tls_id]

                ambulance_payload = {
                    "present": True,
                    "x": round(x, 1), "y": round(y, 1),
                    "edge": edge_id, "speedMps": round(speed, 1),
                    "routeProgress": round(progress, 3),
                    "nextTLS": next_tls_info,
                    "arrived": False,
                }
            elif ambulance_inserted and ambulance_ever_seen:
                # Was in the sim before, now gone -> it arrived at HOSPITAL.
                for tls_id in list(held_tls):
                    try:
                        traci.trafficlight.setProgram(tls_id, "0")
                    except traci.exceptions.TraCIException:
                        pass
                held_tls.clear()
                ambulance_payload = {"present": False, "arrived": True}
            elif ambulance_inserted:
                # add() was called but it hasn't appeared in getIDList() yet
                # (SUMO inserts on the *next* step) -> still waiting to spawn.
                ambulance_payload = {"present": False, "arrived": False}

            vehicles = []
            for vid in traci.vehicle.getIDList():
                if vid == ambulance_id:
                    continue
                vx, vy = traci.vehicle.getPosition(vid)
                vehicles.append({
                    "id": vid,
                    "type": traci.vehicle.getTypeID(vid),
                    "x": round(vx, 1), "y": round(vy, 1),
                    "angle": round(traci.vehicle.getAngle(vid), 1),
                })
            vehicles = vehicles[:130]  # cap payload size

            tls_states = []
            for tls_id in traci.trafficlight.getIDList():
                tls_states.append({"id": tls_id, "state": traci.trafficlight.getRedYellowGreenState(tls_id)})

            with _lock:
                _state.update(
                    running=True, mode="LIVE_SUMO", simTime=round(sim_time, 1),
                    roadId=road_id, ambulance=ambulance_payload, vehicles=vehicles,
                    trafficLights=tls_states, greenCorridorActive=green_active,
                    arrived=bool(ambulance_payload and ambulance_payload.get("arrived")),
                    message=f"Simulation running on Road {road_id}.",
                )

            if ambulance_ever_seen and not amb_present:
                # Ambulance arrived — keep the loop alive briefly so the
                # frontend can show "arrived", then stop.
                time.sleep(1.0)
                break

            time.sleep(REALTIME_STEP_DELAY)
    finally:
        try:
            traci.close()
        except Exception:
            pass
        with _lock:
            _state["running"] = False
            _state["message"] = _state.get("message", "") + " (stopped)"


def start_simulation(road_id: str, road_traffic_scores: Optional[dict] = None) -> dict:
    global _thread
    road_id = road_id.upper()
    if road_id not in ROAD_ROUTE_MAP:
        raise ValueError(f"Unknown road '{road_id}'")

    status = get_status()
    if status["mode"] != "LIVE_SUMO":
        return {"started": False, **status}

    with _lock:
        already_running = _thread is not None and _thread.is_alive()
    if already_running:
        stop_simulation()

    _stop_flag.clear()
    _thread = threading.Thread(target=_run_loop, args=(road_id, road_traffic_scores), daemon=True)
    _thread.start()
    # give the process a moment to spin up before the first status poll
    time.sleep(0.3)
    return {"started": True, "mode": "LIVE_SUMO", "roadId": road_id}


def stop_simulation() -> dict:
    _stop_flag.set()
    if _thread is not None:
        _thread.join(timeout=5.0)
    with _lock:
        _state["running"] = False
        _state["message"] = "Simulation stopped."
    return {"stopped": True}


def get_state() -> dict:
    with _lock:
        return dict(_state)
