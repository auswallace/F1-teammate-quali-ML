"""
Asset management for F1 teammate qualifying predictor.
Handles driver headshots, team logos, and country flags with smart caching.
"""

import hashlib
import csv
from pathlib import Path
from typing import Dict, Literal, Optional
import requests
from PIL import Image, ImageDraw
import streamlit as st
from functools import lru_cache

# Asset directories
ASSETS_DIR = Path("data/assets")
CACHE_DIR = ASSETS_DIR / "cache"
PLACEHOLDERS_DIR = ASSETS_DIR / "placeholders"
HEADSHOTS_DIR = ASSETS_DIR / "headshots"
LOGOS_DIR = ASSETS_DIR / "logos"

# Default flag CDN
FLAG_CDN_BASE = "https://flagcdn.com/w40"

def ensure_dirs():
    """Create necessary asset directories."""
    for dir_path in [ASSETS_DIR, CACHE_DIR, PLACEHOLDERS_DIR, HEADSHOTS_DIR, LOGOS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder images if they don't exist
    create_placeholder_images()

def create_placeholder_images():
    """Create simple placeholder images for missing assets."""
    size = 48
    
    # Driver placeholder (gray circle)
    driver_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(driver_img)
    draw.ellipse([2, 2, size-2, size-2], fill=(128, 128, 128, 200))
    driver_img.save(PLACEHOLDERS_DIR / "driver.png")
    
    # Team placeholder (blue circle)
    team_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(team_img)
    draw.ellipse([2, 2, size-2, size-2], fill=(0, 100, 200, 200))
    team_img.save(PLACEHOLDERS_DIR / "team.png")
    
    # Flag placeholder (red circle)
    flag_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(flag_img)
    draw.ellipse([2, 2, size-2, size-2], fill=(200, 0, 0, 200))
    flag_img.save(PLACEHOLDERS_DIR / "flag.png")

@lru_cache(maxsize=100)
def load_driver_map() -> Dict[str, str]:
    """Load driver ID to asset path mapping."""
    csv_path = ASSETS_DIR / "drivers.csv"
    if not csv_path.exists():
        # Create stub file
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['driver_id', 'asset'])
            writer.writerow(['VER', 'local:data/assets/headshots/max_verstappen.png'])
            writer.writerow(['NOR', 'local:data/assets/headshots/lando_norris.png'])
            writer.writerow(['PIS', 'local:data/assets/headshots/oscar_piastri.png'])
        
        st.info("📁 Created `data/assets/drivers.csv` - add your driver headshots there!")
        return {}
    
    mapping = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row['driver_id']] = row['asset']
    except Exception as e:
        st.warning(f"⚠️ Error loading driver assets: {e}")
    
    return mapping

@lru_cache(maxsize=50)
def load_team_map() -> Dict[str, str]:
    """Load constructor ID to asset path mapping."""
    csv_path = ASSETS_DIR / "teams.csv"
    if not csv_path.exists():
        # Create stub file
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['constructor_id', 'asset'])
            writer.writerow(['RED_BULL', 'local:data/assets/logos/red_bull.png'])
            writer.writerow(['MCLAREN', 'local:data/assets/logos/mclaren.png'])
            writer.writerow(['MERCEDES', 'local:data/assets/logos/mercedes.png'])
        
        st.info("📁 Created `data/assets/teams.csv` - add your team logos there!")
        return {}
    
    mapping = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row['constructor_id']] = row['asset']
    except Exception as e:
        st.warning(f"⚠️ Error loading team assets: {e}")
    
    return mapping

@lru_cache(maxsize=100)
def load_flag_map() -> Dict[str, str]:
    """Load country code to flag asset mapping."""
    csv_path = ASSETS_DIR / "flags.csv"
    if not csv_path.exists():
        # Create stub file
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['country_code', 'asset'])
            writer.writerow(['gb', 'https://flagcdn.com/w40/gb.png'])
            writer.writerow(['it', 'https://flagcdn.com/w40/it.png'])
            writer.writerow(['nl', 'https://flagcdn.com/w40/nl.png'])
            writer.writerow(['au', 'https://flagcdn.com/w40/au.png'])
            writer.writerow(['jp', 'https://flagcdn.com/w40/jp.png'])
        
        st.info("📁 Created `data/assets/flags.csv` - customize flag sources there!")
        return {}
    
    mapping = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row['country_code']] = row['asset']
    except Exception as e:
        st.warning(f"⚠️ Error loading flag assets: {e}")
    
    return mapping

@lru_cache(maxsize=200)
def resolve_image(kind: Literal["driver", "team", "flag"], key: str, default_placeholder: str) -> Path:
    """
    Resolve an image asset to a local path.
    
    Args:
        kind: Type of asset (driver, team, flag)
        key: Identifier (driver_id, constructor_id, country_code)
        default_placeholder: Path to default placeholder image
    
    Returns:
        Path to the resolved image file
    """
    if kind == "driver":
        mapping = load_driver_map()
    elif kind == "team":
        mapping = load_team_map()
    elif kind == "flag":
        mapping = load_flag_map()
    else:
        return Path(default_placeholder)
    
    asset_path = mapping.get(key, "")
    
    if not asset_path:
        return Path(default_placeholder)
    
    if asset_path.startswith("local:"):
        # Local file path
        local_path = Path(asset_path.replace("local:", ""))
        if local_path.exists():
            return local_path
        else:
            st.warning(f"⚠️ Local asset not found: {local_path}")
            return Path(default_placeholder)
    
    elif asset_path.startswith("http"):
        # HTTP URL - download and cache
        try:
            # Create cache filename from URL hash
            url_hash = hashlib.sha1(asset_path.encode()).hexdigest()
            extension = Path(asset_path).suffix or ".png"
            cache_filename = f"{url_hash}{extension}"
            cache_path = CACHE_DIR / cache_filename
            
            if cache_path.exists():
                return cache_path
            
            # Download and cache
            response = requests.get(asset_path, timeout=10)
            response.raise_for_status()
            
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            
            return cache_path
            
        except Exception as e:
            st.warning(f"⚠️ Failed to download {asset_path}: {e}")
            return Path(default_placeholder)
    
    else:
        # Assume it's a direct path
        direct_path = Path(asset_path)
        if direct_path.exists():
            return direct_path
        else:
            st.warning(f"⚠️ Asset not found: {direct_path}")
            return Path(default_placeholder)

@lru_cache(maxsize=100)
def circularize(image_path: Path, size: int = 48) -> Path:
    """
    Create a circular version of an image with alpha mask.
    
    Args:
        image_path: Path to source image
        size: Output size in pixels
    
    Returns:
        Path to circularized image
    """
    try:
        # Create cache key for this transformation
        cache_key = f"{image_path.stem}_{size}_circular.png"
        cache_path = CACHE_DIR / cache_key
        
        if cache_path.exists():
            return cache_path
        
        # Open and process image
        with Image.open(image_path) as img:
            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Resize maintaining aspect ratio
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            
            # Create circular mask
            mask = Image.new('L', (size, size), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse([0, 0, size, size], fill=255)
            
            # Apply mask
            output = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            output.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
            output.putalpha(mask)
            
            # Save to cache
            output.save(cache_path)
            return cache_path
            
    except Exception as e:
        st.warning(f"⚠️ Failed to circularize {image_path}: {e}")
        # Return placeholder
        return Path("data/assets/placeholders/driver.png")

def country_code_for_event(event_row) -> Optional[str]:
    """
    Infer country code from event metadata.
    
    Args:
        event_row: Row from events data with location/track info
    
    Returns:
        ISO 3166-1 alpha-2 country code or None
    """
    # Try to extract country from various fields
    location = getattr(event_row, 'location', None)
    track = getattr(event_row, 'track', None)
    
    if not location and not track:
        return None
    
    # Simple mapping of common F1 locations to country codes
    country_mapping = {
        'melbourne': 'au', 'australia': 'au',
        'monaco': 'mc', 'monte carlo': 'mc',
        'montreal': 'ca', 'canada': 'ca',
        'silverstone': 'gb', 'great britain': 'gb', 'england': 'gb',
        'monza': 'it', 'italy': 'it',
        'spa': 'be', 'belgium': 'be',
        'zandvoort': 'nl', 'netherlands': 'nl',
        'red bull ring': 'at', 'austria': 'at',
        'hungaroring': 'hu', 'hungary': 'hu',
        'interlagos': 'br', 'brazil': 'br',
        'suzuka': 'jp', 'japan': 'jp',
        'shanghai': 'cn', 'china': 'cn',
        'singapore': 'sg',
        'abu dhabi': 'ae', 'uae': 'ae',
        'mexico city': 'mx', 'mexico': 'mx',
        'miami': 'us', 'austin': 'us', 'united states': 'us',
        'jeddah': 'sa', 'saudi arabia': 'sa',
        'baku': 'az', 'azerbaijan': 'az',
        'portimao': 'pt', 'portugal': 'pt',
        'imola': 'it', 'emilia romagna': 'it',
        'sochi': 'ru', 'russia': 'ru',
        'istanbul': 'tr', 'turkey': 'tr',
        'nurburgring': 'de', 'germany': 'de',
        'hockenheim': 'de',
        'paul ricard': 'fr', 'france': 'fr',
        'catalunya': 'es', 'spain': 'es',
        'albert park': 'au', 'australia': 'au'
    }
    
    # Search in location and track fields
    search_text = f"{location or ''} {track or ''}".lower()
    
    for key, code in country_mapping.items():
        if key in search_text:
            return code
    
    return None
