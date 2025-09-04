"""
Events router for listing available seasons and events.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any

router = APIRouter()

@router.get("/events")
async def get_events() -> Dict[str, Any]:
    """Get all available seasons and events."""
    try:
        # Load labeled data to get available events
        labeled_path = Path(__file__).parent.parent.parent / "data" / "interim" / "qual_labeled.parquet"
        
        if not labeled_path.exists():
            raise HTTPException(status_code=404, detail="Labeled data not found. Please run the pipeline first.")
        
        df = pd.read_parquet(labeled_path)
        
        # Group by season and event
        events_by_season = {}
        event_details = {}
        
        for _, row in df.groupby(['season', 'event_key']).first().iterrows():
            season = int(row['season'])
            event_key = row['event_key']
            
            if season not in events_by_season:
                events_by_season[season] = []
            
            events_by_season[season].append(event_key)
            
            # Store event details
            event_details[event_key] = {
                "season": season,
                "event_key": event_key,
                "track_name": row.get('track_name', 'Unknown Track'),
                "location": row.get('location', 'Unknown'),
                "round": row.get('round', None),
                "event_date": row.get('event_date', None)
            }
        
        # Sort seasons descending and events within each season
        for season in events_by_season:
            events_by_season[season].sort()
        
        return {
            "seasons": sorted(events_by_season.keys(), reverse=True),
            "events_by_season": events_by_season,
            "event_details": event_details
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load events: {str(e)}")

@router.get("/events/{season}")
async def get_season_events(season: int) -> Dict[str, Any]:
    """Get events for a specific season."""
    try:
        # Load labeled data
        labeled_path = Path(__file__).parent.parent.parent / "data" / "interim" / "qual_labeled.parquet"
        
        if not labeled_path.exists():
            raise HTTPException(status_code=404, detail="Labeled data not found.")
        
        df = pd.read_parquet(labeled_path)
        season_df = df[df['season'] == season]
        
        if season_df.empty:
            raise HTTPException(status_code=404, detail=f"No events found for season {season}")
        
        # Get unique events for this season
        events = season_df['event_key'].unique().tolist()
        events.sort()
        
        # Get event details
        event_details = {}
        for _, row in season_df.groupby('event_key').first().iterrows():
            event_key = row['event_key']
            event_details[event_key] = {
                "season": season,
                "event_key": event_key,
                "track_name": row.get('track_name', 'Unknown Track'),
                "location": row.get('location', 'Unknown'),
                "round": row.get('round', None),
                "event_date": row.get('event_date', None)
            }
        
        return {
            "season": season,
            "events": events,
            "event_details": event_details
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load season events: {str(e)}")
