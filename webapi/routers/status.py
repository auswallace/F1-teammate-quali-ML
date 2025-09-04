"""
Status router for API health checks and system status.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from typing import Dict, Any

router = APIRouter()

@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get overall system status and health."""
    try:
        # Check if models exist
        models_dir = Path(__file__).parent.parent.parent / "models"
        model_status = {
            "xgboost": (models_dir / "xgboost_model.joblib").exists(),
            "logistic_regression": (models_dir / "logistic_model.joblib").exists(),
            "calibrator": (models_dir / "xgboost_calibrator.joblib").exists()
        }
        
        # Check if data exists
        data_dir = Path(__file__).parent.parent.parent / "data"
        data_status = {
            "processed": (data_dir / "processed" / "teammate_qual.parquet").exists(),
            "labeled": (data_dir / "interim" / "qual_labeled.parquet").exists(),
            "input_linked": (data_dir / "input").exists() and any((data_dir / "input").iterdir())
        }
        
        return {
            "status": "healthy",
            "models": model_status,
            "data": data_status,
            "api_version": "1.0.0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
