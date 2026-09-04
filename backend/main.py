"""
SMART AMBULANCE CORRIDOR AI — backend entrypoint.

Run with:
    uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import traffic, routing, corridor, sumo

app = FastAPI(
    title="Smart Ambulance Corridor AI",
    description=(
        "AI pipeline: BMD-45 road image -> YOLO vehicle detection -> traffic "
        "score -> weighted Dijkstra routing -> Green Corridor activation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # college prototype — tighten if deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(traffic.router, prefix="/api", tags=["traffic"])
app.include_router(routing.router, prefix="/api", tags=["routing"])
app.include_router(corridor.router, prefix="/api", tags=["corridor"])
app.include_router(sumo.router, prefix="/api", tags=["sumo"])


@app.get("/")
async def root():
    return {
        "project": "Smart Ambulance Corridor AI",
        "status": "running",
        "docs": "/docs",
    }
