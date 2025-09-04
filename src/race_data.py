"""
Race data collection and processing for F1 race winner predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import fastf1
from fastf1 import api
import logging
from typing import Dict, List, Optional, Tuple
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RaceDataCollector:
    """Collects race data for winner prediction modeling."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize with cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))
        
    def collect_race_data(self, season: int, event_key: str) -> pd.DataFrame:
        """Collect race data for a specific event."""
        try:
            # Load event schedule
            schedule = fastf1.get_event_schedule(season)
            event = schedule[schedule['EventName'].str.contains(event_key, case=False, na=False)]
            
            if event.empty:
                logger.warning(f"No event found for {season} {event_key}")
                return pd.DataFrame()
            
            event_name = event.iloc[0]['EventName']
            round_num = event.iloc[0]['RoundNumber']
            
            # Load race session
            race_session = fastf1.get_session(season, round_num, 'R')
            race_session.load()
            
            # Get race results
            race_results = race_session.results
            
            if race_results is None or race_results.empty:
                logger.warning(f"No race results for {season} {event_name}")
                return pd.DataFrame()
            
            # Get qualifying results for grid position
            try:
                quali_session = fastf1.get_session(season, round_num, 'Q')
                quali_session.load()
                quali_results = quali_session.results
            except Exception as e:
                logger.warning(f"Could not load qualifying results: {e}")
                quali_results = None
            
            # Build race dataset
            race_data = []
            
            for _, result in race_results.iterrows():
                driver_code = result['Abbreviation']
                team = result['TeamName']
                final_position = result['Position']
                grid_position = result.get('GridPosition', np.nan)
                best_qual_pos = np.nan
                
                # Get qualifying position if available
                if quali_results is not None:
                    quali_driver = quali_results[quali_results['Abbreviation'] == driver_code]
                    if not quali_driver.empty:
                        # Get the actual qualifying position from the Position column
                        best_qual_pos = quali_driver.iloc[0].get('Position', np.nan)
                
                # Get grid position if not in results
                if pd.isna(grid_position) and quali_results is not None:
                    quali_driver = quali_results[quali_results['Abbreviation'] == driver_code]
                    if not quali_driver.empty:
                        grid_position = quali_driver.iloc[0].get('GridPosition', np.nan)
                
                # Create event key in the format used by the qualifying data
                event_key_formatted = f"{season}_R{round_num:02d}"
                
                race_data.append({
                    'season': season,
                    'event_key': event_key_formatted,
                    'round': round_num,
                    'driver_code': driver_code,
                    'team': team,
                    'grid_position': grid_position,
                    'best_qual_pos': best_qual_pos,
                    'final_position': final_position,
                    'label_winner': 1 if final_position == 1 else 0
                })
            
            return pd.DataFrame(race_data)
            
        except Exception as e:
            logger.error(f"Error collecting race data for {season} {event_key}: {e}")
            return pd.DataFrame()
    
    def collect_season_race_data(self, season: int, event_keys: List[str]) -> pd.DataFrame:
        """Collect race data for multiple events in a season."""
        all_data = []
        
        for event_key in event_keys:
            logger.info(f"Collecting race data for {season} {event_key}")
            event_data = self.collect_race_data(season, event_key)
            if not event_data.empty:
                all_data.append(event_data)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def add_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple features for race winner prediction."""
        if df.empty:
            return df
        
        # Add simple weather (placeholder for now)
        df['weather_simple'] = 'dry'  # Default to dry
        
        # Add some basic derived features
        df['grid_position_filled'] = df['grid_position'].fillna(df['best_qual_pos'])
        df['best_qual_pos_filled'] = df['best_qual_pos'].fillna(df['grid_position'])
        
        return df
    
    def save_race_data(self, df: pd.DataFrame, output_path: str):
        """Save race data to parquet file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved race data to {output_file}")
    
    def load_race_data(self, file_path: str) -> pd.DataFrame:
        """Load race data from parquet file."""
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded race data from {file_path}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading race data from {file_path}: {e}")
            return pd.DataFrame()


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect F1 race data for winner prediction")
    parser.add_argument("--season", type=int, required=True, help="F1 season year")
    parser.add_argument("--events", nargs="+", required=True, help="Event names to collect")
    parser.add_argument("--output", type=str, default="data/interim/race_data.parquet", help="Output file path")
    parser.add_argument("--cache", type=str, default="cache", help="Cache directory")
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = RaceDataCollector(cache_dir=args.cache)
    
    # Collect data
    df = collector.collect_season_race_data(args.season, args.events)
    
    if df.empty:
        logger.error("No race data collected")
        return
    
    # Add simple features
    df = collector.add_simple_features(df)
    
    # Save data
    collector.save_race_data(df, args.output)
    
    logger.info(f"Successfully collected {len(df)} race records")


if __name__ == "__main__":
    main()
