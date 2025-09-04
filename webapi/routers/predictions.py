"""
Predictions router for getting model predictions and results.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional
import sys

# Import from src modules
try:
    from src.predict import TeammatePredictor
    from src.assets import resolve_image, circularize
except ImportError:
    # Fallback if src modules not available
    TeammatePredictor = None

router = APIRouter()

@router.get("/predictions/{season}/{event_key}")
async def get_predictions(season: int, event_key: str) -> Dict[str, Any]:
    """Get predictions and results for a specific event."""
    try:
        if TeammatePredictor is None:
            raise HTTPException(status_code=500, detail="Prediction module not available")
        
        # Load predictions
        predictor = TeammatePredictor()
        predictions_df = predictor.build_event_prediction_df(event_key, include_actual=True)
        
        if predictions_df.empty:
            raise HTTPException(status_code=404, detail=f"No predictions found for {event_key}")
        
        # Load actual results for evaluation
        labeled_path = Path(__file__).parent.parent.parent / "data" / "interim" / "qual_labeled.parquet"
        actual_df = pd.read_parquet(labeled_path)
        event_actual = actual_df[actual_df['event_key'] == event_key]
        
        # Merge predictions with actual results
        results_df = predictions_df.merge(
            event_actual[['driver_id', 'beats_teammate_q', 'teammate_gap_ms']], 
            on='driver_id', 
            how='left'
        )
        
        # Add model correctness column
        results_df['model_correct'] = (results_df['model_pick'] == results_df['actual_beats_teammate'])
        
        # Group by constructor to show teammate pairs
        display_rows = []
        
        for constructor_id in sorted(results_df['constructor_id'].unique()):
            team_drivers = results_df[results_df['constructor_id'] == constructor_id]
            if len(team_drivers) != 2:
                continue
                
            driver1, driver2 = team_drivers.iloc[0], team_drivers.iloc[1]
            
            # Determine which driver the model picked as winner
            if driver1['model_pick'] == 1:
                winner = driver1
                loser = driver2
            else:
                winner = driver2
                loser = driver1
            
            # Create display row
            row = {
                'team': constructor_id.replace('_', ' ').title(),
                'driver_a': {
                    'id': driver1['driver_id'],
                    'name': driver1['driver_name'],
                    'model_pick': driver1['model_pick'],
                    'model_confidence': float(driver1['model_confidence']),
                    'actual_beats_teammate': int(driver1['actual_beats_teammate']) if pd.notna(driver1['actual_beats_teammate']) else None,
                    'teammate_gap_ms': float(driver1['teammate_gap_ms']) if pd.notna(driver1['teammate_gap_ms']) else None
                },
                'driver_b': {
                    'id': driver2['driver_id'],
                    'name': driver2['driver_name'],
                    'model_pick': driver2['model_pick'],
                    'model_confidence': float(driver2['model_confidence']),
                    'actual_beats_teammate': int(driver2['actual_beats_teammate']) if pd.notna(driver2['actual_beats_teammate']) else None,
                    'teammate_gap_ms': float(driver2['teammate_gap_ms']) if pd.notna(driver2['teammate_gap_ms']) else None
                },
                'model_pick': winner['driver_name'],
                'model_confidence': float(winner['model_confidence']),
                'model_correct': bool(winner['actual_beats_teammate'] == 1) if pd.notna(winner['actual_beats_teammate']) else None
            }
            
            display_rows.append(row)
        
        # Calculate summary metrics
        model_correct = [r['model_correct'] for r in display_rows if r['model_correct'] is not None]
        model_accuracy = sum(model_correct) / len(model_correct) if model_correct else 0
        
        return {
            "season": season,
            "event_key": event_key,
            "predictions": display_rows,
            "summary": {
                "total_teams": len(display_rows),
                "model_accuracy": model_accuracy,
                "model_correct": sum(model_correct) if model_correct else 0,
                "model_incorrect": len(model_correct) - sum(model_correct) if model_correct else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {str(e)}")

@router.get("/predictions/{season}/{event_key}/summary")
async def get_predictions_summary(season: int, event_key: str) -> Dict[str, Any]:
    """Get a summary of predictions for an event."""
    try:
        # Get full predictions first
        predictions_data = await get_predictions(season, event_key)
        
        # Extract just the summary
        return {
            "season": season,
            "event_key": event_key,
            "summary": predictions_data["summary"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load prediction summary: {str(e)}")
