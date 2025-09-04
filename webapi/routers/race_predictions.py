"""
FastAPI router for race winner predictions.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from pathlib import Path
import sys
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from race_winner_model import RaceWinnerModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predict", tags=["race_predictions"])

# Pydantic models for API responses
class RaceCandidate(BaseModel):
    driver: str
    team: str
    prob: float
    conf_low: float
    conf_high: float

class RaceWinnerResponse(BaseModel):
    season: int
    event_key: str
    generated_at: str
    candidates: List[RaceCandidate]

@router.get("/race_winner/{season}/{event_key}", response_model=RaceWinnerResponse)
async def predict_race_winner(
    season: int,
    event_key: str,
    include_grid: bool = Query(False, description="Include grid position in response")
):
    """
    Predict race winner probabilities for a specific event.
    
    Returns ranked list of drivers with their win probabilities and confidence intervals.
    """
    try:
        # Load race data
        race_data_path = Path("data/race_features_combined.parquet")
        if not race_data_path.exists():
            raise HTTPException(
                status_code=404, 
                detail="Race data not found. Please collect race data first."
            )
        
        df = pd.read_parquet(race_data_path)
        
        # Filter for the specific season and event
        event_data = df[(df['season'] == season) & (df['event_key'] == event_key)]
        
        if event_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No race data found for {season} {event_key}"
            )
        
        # Load the trained model
        model_path = Path("webapi/ml/models/race_winner.joblib")
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Race winner model not found. Please train the model first."
            )
        
        model = RaceWinnerModel(str(model_path))
        
        if model.model_obj is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to load race winner model"
            )
        
        # Make predictions
        predictions = model.predict_event(df, event_key)
        
        if predictions.empty:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate predictions"
            )
        
        # Convert to API response format
        candidates = []
        for _, row in predictions.iterrows():
            candidate = RaceCandidate(
                driver=row['driver'],
                team=row['team'],
                prob=float(row['probability']),
                conf_low=float(row['conf_low']),
                conf_high=float(row['conf_high'])
            )
            candidates.append(candidate)
        
        # Sort by probability (highest first)
        candidates.sort(key=lambda x: x.prob, reverse=True)
        
        # Create response
        from datetime import datetime
        response = RaceWinnerResponse(
            season=season,
            event_key=event_key,
            generated_at=datetime.now().isoformat(),
            candidates=candidates
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting race winner: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/race_winner/{season}/{event_key}/summary")
async def get_race_winner_summary(
    season: int,
    event_key: str
):
    """
    Get a summary of race winner predictions for an event.
    
    Returns top 3 drivers and overall statistics.
    """
    try:
        # Get full predictions
        full_response = await predict_race_winner(season, event_key)
        
        # Create summary
        top_3 = full_response.candidates[:3]
        total_prob = sum(c.prob for c in full_response.candidates)
        
        summary = {
            "season": season,
            "event_key": event_key,
            "top_3": [
                {
                    "rank": i + 1,
                    "driver": c.driver,
                    "team": c.team,
                    "probability": f"{c.prob:.1%}",
                    "confidence": f"{c.conf_low:.1%} - {c.conf_high:.1%}"
                }
                for i, c in enumerate(top_3)
            ],
            "total_drivers": len(full_response.candidates),
            "total_probability": f"{total_prob:.1%}",
            "generated_at": full_response.generated_at
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
