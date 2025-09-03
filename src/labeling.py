"""
Labeling module for F1 teammate qualifying prediction pipeline.

Creates target variable 'beats_teammate_q' and computes teammate gap features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeammateLabeler:
    """Creates labels for teammate qualifying predictions."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the labeler with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.interim_dir = Path(self.config['data']['interim_dir'])
        self.grid_setting_session = self.config['sprint_weekends']['grid_setting_session']
        
        # Track labeling statistics
        self.labeling_stats = {}
        
    def load_normalized_data(self, filename: str = "qual_base.parquet") -> pd.DataFrame:
        """Load normalized data from interim directory."""
        data_path = self.interim_dir / filename
        if not data_path.exists():
            raise FileNotFoundError(f"Normalized data not found at {data_path}")
        
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded normalized data: {df.shape}")
        return df
    
    def create_teammate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create teammate labels for each constructor-event pair."""
        logger.info("Creating teammate labels")
        
        # Create a copy to avoid pandas warnings
        df_copy = df.copy()
        
        # Initialize new columns
        df_copy['beats_teammate_q'] = None
        df_copy['teammate_gap_ms'] = None
        df_copy['teammate_id'] = None
        df_copy['equal_time_tiebreak'] = False
        df_copy['single_teammate'] = False
        
        # Group by constructor and event
        for (constructor_id, event_key), group_indices in df_copy.groupby(['constructor_id', 'event_key']).groups.items():
            group_mask = df_copy.index.isin(group_indices)
            group_data = df_copy[group_mask]
            
            if len(group_data) < 2:
                # Single driver for this constructor-event
                df_copy.loc[group_indices, 'single_teammate'] = True
                df_copy.loc[group_indices, 'beats_teammate_q'] = None
                df_copy.loc[group_indices, 'teammate_gap_ms'] = None
                df_copy.loc[group_indices, 'teammate_id'] = None
                continue
            
            if len(group_data) > 2:
                logger.warning(f"More than 2 drivers for {constructor_id} at {event_key}: {len(group_data)}")
                # Take the two with best qualifying positions
                best_two = group_data.nsmallest(2, 'quali_position')
                group_indices = best_two.index
            
            # Sort by qualifying position
            sorted_indices = df_copy.loc[group_indices].sort_values('quali_position').index
            driver1_idx = sorted_indices[0]
            driver2_idx = sorted_indices[1]
            
            # Check for equal times
            driver1_time = df_copy.loc[driver1_idx, 'quali_best_lap_ms']
            driver2_time = df_copy.loc[driver2_idx, 'quali_best_lap_ms']
            
            if (pd.notna(driver1_time) and 
                pd.notna(driver2_time) and
                abs(driver1_time - driver2_time) < 1):  # 1ms tolerance
                
                # Equal times - use official qualifying order
                df_copy.loc[driver1_idx, 'equal_time_tiebreak'] = True
                df_copy.loc[driver2_idx, 'equal_time_tiebreak'] = True
                logger.info(f"Equal times at {event_key} for {constructor_id}, using official order")
            
            # Label based on qualifying position
            df_copy.loc[driver1_idx, 'beats_teammate_q'] = 1  # Better position
            df_copy.loc[driver2_idx, 'beats_teammate_q'] = 0  # Worse position
            
            # Set teammate IDs
            df_copy.loc[driver1_idx, 'teammate_id'] = df_copy.loc[driver2_idx, 'driver_id']
            df_copy.loc[driver2_idx, 'teammate_id'] = df_copy.loc[driver1_idx, 'driver_id']
            
            # Calculate teammate gap
            if (pd.notna(driver1_time) and pd.notna(driver2_time)):
                # Gap from driver1's perspective (positive = driver1 is faster)
                gap = driver2_time - driver1_time
                df_copy.loc[driver1_idx, 'teammate_gap_ms'] = gap
                df_copy.loc[driver2_idx, 'teammate_gap_ms'] = -gap
            else:
                df_copy.loc[driver1_idx, 'teammate_gap_ms'] = None
                df_copy.loc[driver2_idx, 'teammate_gap_ms'] = None
        
        # Update statistics
        self._update_labeling_stats(df_copy)
        
        logger.info(f"Labeling complete: {len(df_copy)} rows")
        return df_copy
    
    def _update_labeling_stats(self, df: pd.DataFrame) -> None:
        """Update labeling statistics."""
        total_events = df['event_key'].nunique()
        total_constructors = df['constructor_id'].nunique()
        
        # Count labeled pairs
        labeled_pairs = df[df['beats_teammate_q'].notna()].shape[0]
        single_drivers = df[df['single_teammate']].shape[0]
        equal_times = df[df['equal_time_tiebreak']].shape[0]
        
        # Count by target class
        class_counts = df['beats_teammate_q'].value_counts().to_dict()
        
        self.labeling_stats = {
            'total_events': total_events,
            'total_constructors': total_constructors,
            'labeled_pairs': labeled_pairs,
            'single_drivers': single_drivers,
            'equal_times': equal_times,
            'class_distribution': class_counts
        }
        
        logger.info(f"Labeling statistics: {self.labeling_stats}")
    
    def handle_sprint_weekends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle sprint weekends by using the correct qualifying session."""
        logger.info("Handling sprint weekends")
        
        # For now, we'll use the main qualifying session (Q)
        # In the future, this could be enhanced to detect sprint weekends
        # and use the appropriate session that sets the race grid
        
        sprint_handled = df.copy()
        
        # Add sprint weekend flag (simplified detection)
        # This could be enhanced with actual sprint weekend data
        sprint_handled['is_sprint_weekend'] = False
        
        return sprint_handled
    
    def validate_labels(self, df: pd.DataFrame) -> bool:
        """Validate the created labels."""
        logger.info("Validating labels")
        
        # Check for missing labels
        missing_labels = df[df['beats_teammate_q'].isna() & ~df['single_teammate']]
        if len(missing_labels) > 0:
            logger.error(f"Found {len(missing_labels)} rows with missing labels")
            return False
        
        # Check for invalid labels
        invalid_labels = df[~df['beats_teammate_q'].isin([0, 1, None])]
        if len(invalid_labels) > 0:
            logger.error(f"Found {len(invalid_labels)} rows with invalid labels")
            return False
        
        # Check teammate consistency
        for (constructor_id, event_key), group in df.groupby(['constructor_id', 'event_key']):
            if len(group) >= 2:
                labeled_group = group[group['beats_teammate_q'].notna()]
                if len(labeled_group) == 2:
                    # Should have one 1 and one 0
                    values = labeled_group['beats_teammate_q'].values
                    if not (1 in values and 0 in values):
                        logger.error(f"Invalid label distribution for {constructor_id} at {event_key}: {values}")
                        return False
        
        logger.info("Label validation passed")
        return True
    
    def save_labeled_data(self, df: pd.DataFrame, filename: str = "qual_labeled.parquet") -> None:
        """Save labeled data to interim directory."""
        output_path = self.interim_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved labeled data to {output_path}")
        
        # Save labeling statistics
        stats_path = self.interim_dir / "labeling_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(self.labeling_stats, f, indent=2, default=str)
        
        logger.info(f"Saved labeling statistics to {stats_path}")


def main():
    """Main function to run labeling pipeline."""
    labeler = TeammateLabeler()
    
    # Load normalized data
    df = labeler.load_normalized_data()
    
    # Create teammate labels
    labeled_df = labeler.create_teammate_labels(df)
    
    # Handle sprint weekends
    labeled_df = labeler.handle_sprint_weekends(labeled_df)
    
    # Validate labels
    if labeler.validate_labels(labeled_df):
        # Save labeled data
        labeler.save_labeled_data(labeled_df)
        logger.info("Labeling pipeline complete!")
    else:
        logger.error("Labeling validation failed!")
        raise ValueError("Labeling validation failed")


if __name__ == "__main__":
    main()
