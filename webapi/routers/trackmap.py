"""
Track map router for serving static circuit SVG files with neon styling.
"""

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import yaml
from typing import Dict, List, Any, Optional
import re

router = APIRouter()

def load_circuits_config() -> Dict[str, Any]:
    """Load the circuits configuration from YAML file."""
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "circuits_map.yaml"
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Failed to load circuits config: {e}")
        return {}

def resolve_circuit_for_event(event_key: str, circuits_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Resolve which circuit should be used for a given event.
    
    Args:
        event_key: The event key (e.g., '2025_03')
        circuits_config: Loaded circuits configuration
        
    Returns:
        Circuit config dict if found, None otherwise
    """
    # First, try to find by explicit event mapping
    for circuit_id, circuit_data in circuits_config.items():
        if 'events' in circuit_data and event_key in circuit_data['events']:
            return {
                'id': circuit_id,
                **circuit_data
            }
    
    # TODO: Future enhancement - try to match by circuit_id from event metadata
    # For now, return None if no explicit mapping found
    return None

def extract_svg_paths(svg_content: str) -> List[str]:
    """
    Extract path data from SVG content.
    
    Args:
        svg_content: Raw SVG file content
        
    Returns:
        List of path 'd' attributes
    """
    # Find all path elements with 'd' attributes
    path_pattern = r'<path[^>]*d="([^"]*)"'
    matches = re.findall(path_pattern, svg_content, re.IGNORECASE)
    
    return matches

def get_svg_viewbox(svg_content: str) -> Optional[Dict[str, float]]:
    """
    Extract viewBox from SVG content.
    
    Args:
        svg_content: Raw SVG file content
        
    Returns:
        ViewBox dimensions or None if not found
    """
    viewbox_pattern = r'viewBox="([^"]*)"'
    match = re.search(viewbox_pattern, svg_content, re.IGNORECASE)
    
    if match:
        try:
            parts = match.group(1).split()
            if len(parts) >= 4:
                return {
                    'x': float(parts[0]),
                    'y': float(parts[1]),
                    'width': float(parts[2]),
                    'height': float(parts[3])
                }
        except (ValueError, IndexError):
            pass
    
    return None

@router.get("/trackmap/{season}/{event_key}")
async def get_trackmap(season: int, event_key: str) -> Dict[str, Any]:
    """
    Get track map data for an event.
    
    Args:
        season: F1 season year
        event_key: Event identifier
        
    Returns:
        JSON with circuit info and SVG paths
    """
    try:
        # Load circuits configuration
        circuits_config = load_circuits_config()
        
        # Try to resolve circuit for this event
        circuit = resolve_circuit_for_event(event_key, circuits_config)
        
        if not circuit:
            # No circuit mapping found
            return {
                "ok": True,
                "season": season,
                "event_key": event_key,
                "placeholder": True,
                "message": "No circuit mapping found for this event"
            }
        
        # Check if SVG file exists
        svg_filename = circuit['svg']
        svg_path = Path(__file__).parent.parent.parent / "data" / "assets" / "tracks" / svg_filename
        
        if not svg_path.exists():
            return {
                "ok": True,
                "season": season,
                "event_key": event_key,
                "placeholder": True,
                "message": f"SVG file {svg_filename} not found"
            }
        
        # Read and parse SVG content
        try:
            with open(svg_path, 'r') as f:
                svg_content = f.read()
        except Exception as e:
            return {
                "ok": True,
                "season": season,
                "event_key": event_key,
                "placeholder": True,
                "message": f"Failed to read SVG file: {str(e)}"
            }
        
        # Extract path data and viewBox
        paths = extract_svg_paths(svg_content)
        viewbox = get_svg_viewbox(svg_content)
        
        if not paths:
            return {
                "ok": True,
                "season": season,
                "event_key": event_key,
                "placeholder": True,
                "message": "No path data found in SVG"
            }
        
        # Return circuit data
        return {
            "ok": True,
            "season": season,
            "event_key": event_key,
            "circuit": {
                "id": circuit['id'],
                "name": circuit.get('name', 'Unknown Circuit'),
                "svg_url": f"/static/tracks/{svg_filename}",
                "paths": paths,
                "viewbox": viewbox,
                "country_code": circuit.get('country_code', 'xx')
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get track map: {str(e)}")

@router.get("/trackmap/{season}/{event_key}/circuit")
async def get_circuit_info(season: int, event_key: str) -> Dict[str, Any]:
    """Get just the circuit information for an event."""
    try:
        circuits_config = load_circuits_config()
        circuit = resolve_circuit_for_event(event_key, circuits_config)
        
        if not circuit:
            return {
                "ok": True,
                "season": season,
                "event_key": event_key,
                "found": False
            }
        
        return {
            "ok": True,
            "season": season,
            "event_key": event_key,
            "found": True,
            "circuit": {
                "id": circuit['id'],
                "name": circuit.get('name', 'Unknown Circuit'),
                "country_code": circuit.get('country_code', 'xx')
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get circuit info: {str(e)}")
