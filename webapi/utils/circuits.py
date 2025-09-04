"""
Circuit and event code utilities for F1 data.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import lru_cache

@lru_cache(maxsize=1)
def load_event_codes(config_path: str = "config/event_codes_by_season.yaml") -> Dict:
    """Load event codes configuration with caching."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Event codes config not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def get_codes_for_season(season: int, config_path: str = "config/event_codes_by_season.yaml") -> List[str]:
    """Get all event codes for a specific season."""
    config = load_event_codes(config_path)
    
    for season_data in config.get('seasons', []):
        if season_data.get('season') == season:
            return [event.get('code') for event in season_data.get('events', [])]
    
    return []

def lookup_event(season: int, round: Optional[int] = None, official: Optional[str] = None, 
                config_path: str = "config/event_codes_by_season.yaml") -> Optional[Dict]:
    """Look up event details by season and optional criteria."""
    config = load_event_codes(config_path)
    
    for season_data in config.get('seasons', []):
        if season_data.get('season') == season:
            for event in season_data.get('events', []):
                if round is not None and event.get('round') == round:
                    return event
                if official is not None and official in event.get('official', ''):
                    return event
    
    return None

def resolve_code(season: int, round: Optional[int] = None, country: Optional[str] = None, 
                location: Optional[str] = None, official: Optional[str] = None,
                config_path: str = "config/event_codes_by_season.yaml") -> Optional[str]:
    """Resolve event code by various criteria."""
    event = lookup_event(season, round, official, config_path)
    if event:
        return event.get('code')
    
    # If no exact match, try to find by other criteria
    config = load_event_codes(config_path)
    
    for season_data in config.get('seasons', []):
        if season_data.get('season') == season:
            for event in season_data.get('events', []):
                if country and country.lower() in event.get('country', '').lower():
                    return event.get('code')
                if location and location.lower() in event.get('location', '').lower():
                    return event.get('code')
    
    return None

def get_season_summary(season: int, config_path: str = "config/event_codes_by_season.yaml") -> Dict:
    """Get summary of all events for a season."""
    config = load_event_codes(config_path)
    
    for season_data in config.get('seasons', []):
        if season_data.get('season') == season:
            events = season_data.get('events', [])
            return {
                'season': season,
                'event_count': len(events),
                'events': [
                    {
                        'round': event.get('round'),
                        'code': event.get('code'),
                        'name': event.get('name'),
                        'country': event.get('country'),
                        'location': event.get('location'),
                        'date': event.get('date')
                    }
                    for event in events
                ]
            }
    
    return {'season': season, 'event_count': 0, 'events': []}

def get_available_seasons(config_path: str = "config/event_codes_by_season.yaml") -> List[int]:
    """Get list of available seasons."""
    config = load_event_codes(config_path)
    return [season_data.get('season') for season_data in config.get('seasons', [])]
