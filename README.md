# Smart Ambulance Corridor AI

A college major-project prototype demonstrating an end-to-end AI-powered
ambulance routing pipeline:

```
BMD-45 road image → YOLO vehicle detection → vehicle counts → traffic score
→ weighted road cost → Weighted Dijkstra → best route → Green Corridor
→ ambulance → hospital arrival
```

This is a **software prototype**. It does **not** connect to real Bengaluru
CCTV, real traffic signals, or a real hospital system — see
"Important limitations" at the bottom.

---

## 1. Architecture

```
smart-ambulance-corridor/
  backend/                  FastAPI + YOLO + routing engine
    main.py                 App entrypoint
    config.py                All tunable weights/paths live here
    api/
      traffic.py             POST /api/analyze-traffic, GET /api/status
      routing.py              POST /api/optimize-route
      corridor.py             POST /api/activate-green-corridor, GET /api/hospitals
    services/
      yolo_service.py         LIVE_YOLO / DEMO mode inference + bounding-box drawing
      traffic_service.py      Vehicle-count -> traffic score -> LOW/MEDIUM/HIGH/SEVERE
      routing_service.py      Weighted Dijkstra over the road graph
      hospital_service.py     Hospital scoring/recommendation
      demo_data_generator.py  Generates the 3 simulated BMD-45-style road images
    models/schemas.py         Pydantic request/response models
    data/                     Generated demo road images + detection manifests
    weights/                  Put your fine-tuned best.pt here for LIVE_YOLO mode
    requirements.txt
    .env.example

  frontend/                 React + Vite dashboard (dark command-center UI)
    src/
      pages/                  DashboardPage, SumoSimulationPage, RoutingPage
                              (React Router — Dashboard links open the other
                              two pages in a new tab, carrying the current
                              road/traffic-score context via URL params)
      components/             Header, EmergencyControls, StatusCards, AmbulanceRoute,
                              TrafficAnalysis, TrafficRoadCard, AIRecommendation,
                              HospitalRecommendation, GreenCorridor, HospitalStatus,
                              LiveSystemFeed, WorkflowSteps, ToolNav
      state/emergencyState.js  Central emergency workflow state machine
      services/api.js          Backend API client
      App.jsx                  Router (/ , /sumo-simulation, /routing)
    package.json
    vite.config.js            Proxies /api -> http://localhost:8000
```

---

## 1a. SUMO Simulation & Routing pages

Two extra pages, reachable from the cards at the top of the dashboard
(each opens in a new tab):

- **SUMO Simulation** (`/sumo-simulation`) — a *real* SUMO/TraCI
  microscopic traffic simulation, not an animation. `backend/sumo/`
  contains an actual road network (`corridor.net.xml`, built with
  `netconvert` from `corridor.nod.xml`/`corridor.edg.xml`) whose junctions
  and distances match `config.ROAD_NETWORK` / `config.ROUTE_JUNCTIONS`,
  plus a route file (`corridor.rou.xml`) with background traffic and an
  `ambulance` vehicle type. `backend/services/sumo_service.py` launches
  `sumo` headless via TraCI in a background thread, steps it, and
  implements real **Green Corridor signal preemption**: as the ambulance
  approaches a signal-controlled junction (J2/J3/J4) it forces that
  junction's approach light green, then releases it back to the normal
  program once the ambulance has passed. The frontend polls
  `/api/sumo/state` every ~300ms and draws the live network, background
  traffic, and ambulance on an SVG canvas.

  Requires `pip install eclipse-sumo traci sumolib` (already in
  `requirements.txt` as optional deps — `eclipse-sumo` bundles the actual
  SUMO binaries for Linux/Windows/macOS, no separate system install
  needed). If SUMO isn't installed, `/api/sumo/status` reports
  `UNAVAILABLE` and the page shows a clear message instead of faking data
  — same LIVE/DEMO pattern as the YOLO integration.

- **Routing** (`/routing`) — a live visualization of the Weighted
  Dijkstra route graph, built entirely from `/api/optimize-route`
  responses (no hardcoded topology). Drag the traffic-score sliders to
  see the recommended route recompute in real time, with a comparison
  table of cost/distance/ETA per road.

---

## 2. Run the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate the 3 simulated BMD-45-style road images + demo detections
# (already included in data/, but re-run any time you want fresh ones)
python -m services.demo_data_generator

# Start the API
uvicorn main:app --reload --port 8000
```

The API is now live at `http://localhost:8000` (interactive docs at `/docs`).

On startup it checks for `weights/best.pt`. If it's missing, the backend
automatically runs in **DEMO MODE** — this is expected and clearly labeled
throughout the API responses and the UI (look for the `mode` field / the
`● LIVE YOLO` vs `○ DEMO MODE` badge in the header).

## 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/api/*` requests
to the backend at `localhost:8000` (see `vite.config.js`), so both servers
must be running.

## 4. Run the demo

1. Click **Start Emergency** — priority, dispatch, and patient pickup fire
   automatically with live-feed events.
2. In the **AI Traffic Analysis** section, click **Run AI Traffic Analysis**.
   Each road (A/B/C) is analyzed in turn: the BMD-45 image loads, YOLO
   bounding boxes are drawn, vehicle counts and a traffic score appear.
3. Once all three roads are analyzed, **Weighted Dijkstra** runs
   automatically and the **AI Route Recommendation** card shows the selected
   route with a live comparison of all three.
4. Click **Select Best Hospital** — hospital cards appear, the best match is
   flagged, and the **Green Corridor** activates: junctions light up green in
   sequence and the ambulance animates along the *selected* route on the map
   (the visual route always matches the algorithm's choice).
5. Click **Complete Arrival** to close out the run. Click **Reset** to run
   the demo again — try re-running traffic analysis to see how a different
   traffic mix can change which road gets recommended.

---

## 5. Switching from DEMO MODE to real YOLO inference

1. Fine-tune (or otherwise obtain) a YOLO model on BMD-45 — a
   `YOLOv12-S`-class model is what this project targets, but any
   `ultralytics`-compatible `.pt` file will work as long as its class names
   line up with `car`, `two_wheeler`, `auto`, `bus`, `truck`, and optionally
   `pedestrian` / `obstacle`.
2. Place the weights file at `backend/weights/best.pt`, or point
   `MODEL_PATH` (in `backend/.env`, copied from `.env.example`) at wherever
   you keep it.
3. Make sure `ultralytics` is installed (`pip install ultralytics` — it's
   already in `requirements.txt`, just commented as optional since it's a
   heavier dependency).
4. Restart the backend. `GET /api/status` and the frontend's header badge
   will flip to `LIVE_YOLO` automatically — no frontend code changes needed,
   since both modes return the exact same response shape.
5. To feed it real BMD-45 images instead of the generated placeholders,
   replace `backend/data/road_a.jpg`, `road_b.jpg`, `road_c.jpg` with your
   own samples (any road photo works). In LIVE_YOLO mode, the demo JSON
   manifests are ignored — the model runs fresh inference on whatever image
   is at that path.
6. You can also skip the sample-image lookup entirely and POST an uploaded
   image directly to `/api/analyze-traffic?road_id=A` with a multipart
   `image` field — the API already supports this.

To force DEMO MODE back on for a presentation (e.g. on a laptop without the
weights file), set `FORCE_DEMO_MODE=true` in `backend/.env`.

---

## 6. Configuration reference

Everything tunable lives in `backend/config.py`:

| What | Where |
|---|---|
| YOLO weights path / confidence threshold | `MODEL_PATH`, `YOLO_CONF_THRESHOLD` |
| Traffic score weights (per vehicle class) | `TRAFFIC_WEIGHTS` |
| Traffic status thresholds (LOW/MEDIUM/HIGH/SEVERE) | `TRAFFIC_THRESHOLDS` |
| Dijkstra edge-cost coefficients | `ROUTING_WEIGHTS` |
| Road network (distance, blockage, signal delay) | `ROAD_NETWORK` |
| Hospital sample data + scoring weights | `HOSPITALS`, `HOSPITAL_WEIGHTS` |
| Green corridor junction sequence per road | `ROUTE_JUNCTIONS` |

None of these are hardcoded into the frontend — the frontend always reads
scores/levels/recommendations from the API response.

---

## 7. Important limitations (please read before presenting)

This is a **software prototype simulating** the Smart Corridor architecture,
not a production system:

- **Traffic perception**: BMD-45-style imagery, either the bundled
  schematic simulated images or your own uploaded photos — **not** a live
  Bengaluru CCTV feed.
- **AI**: YOLO vehicle detection, either real inference (LIVE_YOLO mode) or
  a fixed, clearly-labeled demo detection set (DEMO mode) — never invented
  numbers presented as if they came from a model.
- **Routing**: a traffic-weighted Dijkstra over a small 3-road prototype
  graph — not integrated with any real traffic-signal control system.
- **Hospitals**: simulated capacity/ICU/doctor data — not a real hospital
  integration.
- **Dispatch**: the whole emergency workflow is a client-side state machine
  for demonstration purposes — no real ambulance dispatch system is
  involved.

The traffic-scoring formula (`car×1 + two_wheeler×0.5 + auto×1 + bus×3 +
truck×3`) is this project's own heuristic, not an official traffic-
engineering standard.
