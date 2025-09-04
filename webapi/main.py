"""
F1 Teammate Qualifying - FastAPI Backend
Serves predictions, events, and track maps for the React dashboard.
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add the repository root to Python path to import src modules
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from routers import status, events, predictions, trackmap

# Create FastAPI app
app = FastAPI(
    title="F1 Teammate Qualifying API",
    description="Backend API for F1 teammate qualifying predictions and track maps",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
assets_dir = repo_root / "data" / "assets"
if assets_dir.exists():
    app.mount("/static/assets", StaticFiles(directory=str(assets_dir)), name="assets")

# Mount static tracks
tracks_dir = repo_root / "data" / "assets" / "tracks"
if tracks_dir.exists():
    app.mount("/static/tracks", StaticFiles(directory=str(tracks_dir)), name="tracks")

# Include routers
app.include_router(status.router, prefix="/api", tags=["status"])
app.include_router(events.router, prefix="/api", tags=["events"])
app.include_router(predictions.router, prefix="/api", tags=["predictions"])
app.include_router(trackmap.router, prefix="/api", tags=["trackmap"])

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "F1 Teammate Qualifying API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "status": "/api/status",
            "events": "/api/events",
            "predictions": "/api/predictions/{season}/{event_key}",
            "trackmap": "/api/trackmap/{season}/{event_key}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
