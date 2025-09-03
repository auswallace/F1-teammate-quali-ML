"""
Data loading module for F1 teammate qualifying prediction pipeline.

Loads pre-exported parquet files and normalizes schema into a single dataframe.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1DataLoader:
    """Loads and normalizes F1 data from parquet exports."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the data loader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.input_dir = Path(self.config['data']['input_dir'])
        self.interim_dir = Path(self.config['data']['interim_dir'])
        
        # Ensure directories exist
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        
        # Track loaded data
        self.raw_data = {}
        self.normalized_data = None
        
    def load_all_seasons(self) -> Dict[int, pd.DataFrame]:
        """Load all available seasons from the input directory."""
        logger.info(f"Loading data from {self.input_dir}")
        
        seasons = []
        for item in self.input_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                seasons.append(int(item.name))
        
        seasons.sort()
        logger.info(f"Found seasons: {seasons}")
        
        for season in seasons:
            season_data = self._load_season(season)
            if season_data is not None:
                self.raw_data[season] = season_data
                logger.info(f"Loaded season {season}: {len(season_data)} sessions")
        
        return self.raw_data
    
    def _load_season(self, season: int) -> Optional[pd.DataFrame]:
        """Load all sessions for a specific season."""
        season_dir = self.input_dir / str(season)
        if not season_dir.exists():
            logger.warning(f"Season directory {season_dir} not found")
            return None
        
        sessions = []
        for round_dir in season_dir.iterdir():
            if round_dir.is_dir() and round_dir.name.startswith('round_'):
                round_sessions = self._load_round_sessions(round_dir, season)
                sessions.extend(round_sessions)
        
        if sessions:
            return pd.concat(sessions, ignore_index=True)
        return None
    
    def _load_round_sessions(self, round_dir: Path, season: int) -> List[pd.DataFrame]:
        """Load all session files for a specific round."""
        sessions = []
        
        for session_file in round_dir.glob("session_*.parquet"):
            try:
                session_df = pd.read_parquet(session_file)
                
                # Extract session info from filename
                filename = session_file.stem
                parts = filename.split('_')
                round_num = parts[2]
                session_type = parts[3]
                
                # Add metadata
                session_df['season'] = season
                session_df['round'] = round_num
                session_df['session_type'] = session_type
                session_df['event_key'] = f"{season}_{round_num}"
                
                sessions.append(session_df)
                
            except Exception as e:
                logger.error(f"Error loading {session_file}: {e}")
                continue
        
        return sessions
    
    def normalize_schema(self) -> pd.DataFrame:
        """Normalize the schema into a single dataframe with required columns."""
        logger.info("Normalizing schema across all seasons")
        
        if not self.raw_data:
            raise ValueError("No data loaded. Call load_all_seasons() first.")
        
        normalized_sessions = []
        
        for season, season_data in self.raw_data.items():
            # Filter for qualifying sessions
            quali_sessions = season_data[season_data['session_type'] == 'Q'].copy()
            
            if quali_sessions.empty:
                logger.warning(f"No qualifying sessions found for season {season}")
                continue
            
            # Normalize each qualifying session
            for _, session in quali_sessions.groupby('event_key'):
                normalized = self._normalize_quali_session(session)
                if normalized is not None:
                    normalized_sessions.append(normalized)
        
        if not normalized_sessions:
            raise ValueError("No qualifying sessions could be normalized")
        
        self.normalized_data = pd.concat(normalized_sessions, ignore_index=True)
        logger.info(f"Normalized schema complete: {len(self.normalized_data)} rows")
        
        return self.normalized_data
    
    def _normalize_quali_session(self, session_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Normalize a single qualifying session to required schema."""
        try:
            # Required columns mapping
            column_mapping = {
                'season': 'season',
                'event_key': 'event_key',
                'DriverId': 'driver_id',
                'FullName': 'driver_name',
                'TeamId': 'constructor_id',
                'TeamName': 'constructor_name',
                'Position': 'quali_position',
                'Time': 'quali_best_lap_ms',
                'Status': 'status',
                'round': 'round'
            }
            
            # Create normalized dataframe
            normalized = pd.DataFrame()
            
            # Map existing columns
            for old_col, new_col in column_mapping.items():
                if old_col in session_df.columns:
                    normalized[new_col] = session_df[old_col]
                else:
                    logger.warning(f"Column {old_col} not found in session")
                    normalized[new_col] = None
            
            # Add missing required columns with defaults
            required_columns = [
                'penalties_applied_bool', 'track_id', 'track_name',
                'fp1_best_ms', 'fp2_best_ms', 'fp3_best_ms'
            ]
            
            for col in required_columns:
                if col not in normalized.columns:
                    normalized[col] = None
            
            # Convert time to milliseconds if available
            if 'quali_best_lap_ms' in normalized.columns:
                normalized['quali_best_lap_ms'] = self._convert_time_to_ms(
                    normalized['quali_best_lap_ms']
                )
            
            # Add track information (extract from event_key for now)
            normalized['track_id'] = normalized['event_key'].str.split('_').str[1]
            normalized['track_name'] = normalized['track_id']  # Placeholder
            
            # Add penalties flag (simplified)
            normalized['penalties_applied_bool'] = False  # Placeholder
            
            # Add session_type column (we know this is qualifying data)
            normalized['session_type'] = 'Q'
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing session: {e}")
            return None
    
    def _convert_time_to_ms(self, time_series: pd.Series) -> pd.Series:
        """Convert time series to milliseconds."""
        def time_to_ms(time_val):
            if pd.isna(time_val):
                return None
            try:
                # Handle timedelta format
                if hasattr(time_val, 'total_seconds'):
                    return time_val.total_seconds() * 1000
                # Handle string format
                elif isinstance(time_val, str):
                    # Parse time string (e.g., "00:01:15.096000")
                    parts = time_val.split(':')
                    if len(parts) == 3:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = float(parts[2])
                        total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000
                        return total_ms
                return None
            except:
                return None
        
        return time_series.apply(time_to_ms)
    
    def save_normalized_data(self, filename: str = "qual_base.parquet") -> None:
        """Save normalized data to interim directory."""
        if self.normalized_data is None:
            raise ValueError("No normalized data to save. Call normalize_schema() first.")
        
        output_path = self.interim_dir / filename
        self.normalized_data.to_parquet(output_path, index=False)
        logger.info(f"Saved normalized data to {output_path}")
        
        # Also save schema information
        schema_path = self.interim_dir / "qual_base_schema.json"
        schema_info = {
            'shape': self.normalized_data.shape,
            'columns': list(self.normalized_data.columns),
            'dtypes': self.normalized_data.dtypes.to_dict(),
            'missing_counts': self.normalized_data.isnull().sum().to_dict()
        }
        
        with open(schema_path, 'w') as f:
            import json
            json.dump(schema_info, f, indent=2, default=str)
        
        logger.info(f"Saved schema info to {schema_path}")


def main():
    """Main function to run data loading pipeline."""
    loader = F1DataLoader()
    
    # Load all seasons
    loader.load_all_seasons()
    
    # Normalize schema
    loader.normalize_schema()
    
    # Save normalized data
    loader.save_normalized_data()
    
    logger.info("Data loading pipeline complete!")


if __name__ == "__main__":
    main()
