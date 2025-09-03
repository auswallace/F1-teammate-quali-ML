"""
Feature engineering module for F1 teammate qualifying prediction pipeline.

Creates features for driver form, team performance, head-to-head records, and more.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for teammate qualifying predictions."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the feature engineer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.interim_dir = Path(self.config['data']['interim_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.feature_config = self.config['features']
        
        # Ensure directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Track feature creation statistics
        self.feature_stats = {}
        
    def load_labeled_data(self, filename: str = "qual_labeled.parquet") -> pd.DataFrame:
        """Load labeled data from interim directory."""
        data_path = self.interim_dir / filename
        if not data_path.exists():
            raise FileNotFoundError(f"Labeled data not found at {data_path}")
        
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded labeled data: {df.shape}")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for the model."""
        logger.info("Creating features")
        
        # Filter to only labeled pairs (exclude single drivers)
        labeled_df = df[df['beats_teammate_q'].notna()].copy()
        
        # Create driver rolling form features
        labeled_df = self._create_driver_form_features(labeled_df)
        
        # Create team form features
        labeled_df = self._create_team_form_features(labeled_df)
        
        # Create head-to-head features
        labeled_df = self._create_h2h_features(labeled_df)
        
        # Create practice session features
        labeled_df = self._create_practice_features(labeled_df)
        
        # Create track features
        labeled_df = self._create_track_features(labeled_df)
        
        # Create weather features (placeholder)
        labeled_df = self._create_weather_features(labeled_df)
        
        # Data hygiene
        labeled_df = self._apply_data_hygiene(labeled_df)
        
        # Update feature statistics
        self._update_feature_stats(labeled_df)
        
        logger.info(f"Feature creation complete: {labeled_df.shape}")
        return labeled_df
    
    def _create_driver_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create driver rolling form features."""
        logger.info("Creating driver form features")
        
        # Sort by driver, season, event
        df_sorted = df.sort_values(['driver_id', 'season', 'event_key'])
        
        # Rolling qualifying position (last 3 and 5 events)
        for window in self.feature_config['driver_form_windows']:
            rolling_avg = (
                df_sorted.groupby('driver_id')['quali_position']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            df_sorted[f'driver_quali_pos_avg_{window}'] = rolling_avg.values
        
        # Rolling share of beating teammate (last 6 events)
        rolling_share = (
            df_sorted.groupby('driver_id')['beats_teammate_q']
            .rolling(window=6, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        df_sorted['driver_beats_teammate_share_6'] = rolling_share.values
        
        return df_sorted
    
    def _create_team_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team form features."""
        logger.info("Creating team form features")
        
        # Sort by constructor, season, event
        df_sorted = df.sort_values(['constructor_id', 'season', 'event_key'])
        
        # Rolling constructor qualifying average (last 3 and 5 events)
        for window in self.feature_config['team_form_windows']:
            rolling_avg = (
                df_sorted.groupby('constructor_id')['quali_position']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            df_sorted[f'team_quali_pos_avg_{window}'] = rolling_avg.values
        
        # Team pace percentile vs field (rolling)
        pace_percentile = (
            df_sorted.groupby('event_key')['quali_best_lap_ms']
            .rank(pct=True)
            .groupby(df_sorted['constructor_id'])
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        df_sorted['team_pace_percentile_5'] = pace_percentile.values
        
        return df_sorted
    
    def _create_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create head-to-head features."""
        logger.info("Creating head-to-head features")
        
        # Sort by driver, season, event
        df_sorted = df.sort_values(['driver_id', 'season', 'event_key'])
        
        # Rolling record vs current teammate this season
        rolling_record = (
            df_sorted.groupby(['driver_id', 'season'])['beats_teammate_q']
            .rolling(window=6, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        df_sorted['driver_vs_teammate_record_6'] = rolling_record.values
        
        # Average teammate gap last 4 events
        rolling_gap = (
            df_sorted.groupby('driver_id')['teammate_gap_ms']
            .rolling(window=4, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        df_sorted['driver_teammate_gap_avg_4'] = rolling_gap.values
        
        return df_sorted
    
    def _create_practice_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create practice session features."""
        logger.info("Creating practice session features")
        
        # For now, we'll create placeholder features since practice data isn't fully loaded
        # In a full implementation, these would be computed from actual practice session data
        
        # Placeholder practice deltas (median imputation)
        practice_sessions = ['fp1', 'fp2', 'fp3']
        
        for session in practice_sessions:
            # Create placeholder columns
            df[f'{session}_best_ms'] = None
            df[f'{session}_delta_to_median'] = None
            df[f'{session}_was_imputed'] = True
        
        # Add practice consistency flag
        df['practice_consistency'] = 0.5  # Placeholder
        
        return df
    
    def _create_track_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create track-related features."""
        logger.info("Creating track features")
        
        # Track one-hot encoding
        track_dummies = pd.get_dummies(df['track_id'], prefix='track')
        df = pd.concat([df, track_dummies], axis=1)
        
        # Street circuit flag (simplified - could be enhanced with actual track data)
        street_tracks = ['monaco', 'baku', 'singapore', 'jeddah', 'miami', 'las_vegas']
        df['is_street_circuit'] = df['track_id'].isin(street_tracks)
        
        # Historical qualifying variance at this track
        df['track_quali_variance'] = (
            df.groupby('track_id')['quali_best_lap_ms']
            .transform('std')
            .fillna(0)
        )
        
        # Driver familiarity with track (count of previous appearances)
        df['driver_track_familiarity'] = (
            df.groupby(['driver_id', 'track_id'])
            .cumcount() + 1
        )
        
        return df
    
    def _create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather features (placeholder)."""
        logger.info("Creating weather features")
        
        # Placeholder weather features
        df['temperature'] = 20.0  # Placeholder
        df['rain_probability'] = 0.1  # Placeholder
        df['wind_speed'] = 5.0  # Placeholder
        
        # Missingness flags
        df['weather_temp_missing'] = False
        df['weather_rain_missing'] = False
        df['weather_wind_missing'] = False
        
        return df
    
    def _apply_data_hygiene(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data hygiene measures."""
        logger.info("Applying data hygiene")
        
        # Clip outliers based on configuration
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ['quali_best_lap_ms', 'teammate_gap_ms']:
                # Clip based on percentiles
                lower_pct = self.feature_config['outlier_clipping']['quali_time'][0]
                upper_pct = self.feature_config['outlier_clipping']['quali_time'][1]
                
                lower_val = df[col].quantile(lower_pct)
                upper_val = df[col].quantile(upper_pct)
                
                # Add clipping flags
                df[f'{col}_was_clipped'] = (
                    (df[col] < lower_val) | (df[col] > upper_val)
                )
                
                # Clip values
                df[col] = df[col].clip(lower=lower_val, upper=upper_val)
        
        # Impute missing values with medians
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[f'{col}_imputed'] = df[col].isnull()
                df[col] = df[col].fillna(median_val)
        
        # Add missingness flags for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[f'{col}_missing'] = df[col].isnull()
                df[col] = df[col].fillna('unknown')
        
        return df
    
    def _update_feature_stats(self, df: pd.DataFrame) -> None:
        """Update feature creation statistics."""
        total_features = len(df.columns)
        numeric_features = len(df.select_dtypes(include=[np.number]).columns)
        categorical_features = len(df.select_dtypes(include=['object']).columns)
        
        # Count clipping and imputation
        clipping_cols = [col for col in df.columns if col.endswith('_was_clipped')]
        imputation_cols = [col for col in df.columns if col.endswith('_imputed')]
        missing_cols = [col for col in df.columns if col.endswith('_missing')]
        
        total_clipped = sum(df[col].sum() for col in clipping_cols)
        total_imputed = sum(df[col].sum() for col in imputation_cols)
        total_missing_flags = sum(df[col].sum() for col in missing_cols)
        
        self.feature_stats = {
            'total_features': total_features,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'total_clipped_values': total_clipped,
            'total_imputed_values': total_imputed,
            'total_missing_flags': total_missing_flags,
            'feature_columns': list(df.columns)
        }
        
        logger.info(f"Feature statistics: {self.feature_stats}")
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "teammate_qual.parquet") -> None:
        """Save processed data to processed directory."""
        output_path = self.processed_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        # Save feature statistics
        stats_path = self.processed_dir / "feature_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(self.feature_stats, f, indent=2, default=str)
        
        logger.info(f"Saved feature statistics to {stats_path}")
        
        # Save schema
        schema_path = self.processed_dir / "schema.json"
        schema_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_counts': df.isnull().sum().to_dict(),
            'feature_stats': self.feature_stats
        }
        
        with open(schema_path, 'w') as f:
            json.dump(schema_info, f, indent=2, default=str)
        
        logger.info(f"Saved schema to {schema_path}")


def main():
    """Main function to run feature engineering pipeline."""
    engineer = FeatureEngineer()
    
    # Load labeled data
    df = engineer.load_labeled_data()
    
    # Create features
    featured_df = engineer.create_features(df)
    
    # Save processed data
    engineer.save_processed_data(featured_df)
    
    logger.info("Feature engineering pipeline complete!")


if __name__ == "__main__":
    main()
