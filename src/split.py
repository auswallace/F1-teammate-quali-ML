"""
Data splitting module for F1 teammate qualifying prediction pipeline.

Handles temporal splits by season and provides cross-validation splits.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import GroupKFold

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """Handles data splitting for temporal validation."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the data splitter with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.interim_dir = Path(self.config['data']['interim_dir'])
        self.splits_config = self.config['splits']
        
        # Track split statistics
        self.split_stats = {}
        
    def load_processed_data(self, filename: str = "teammate_qual.parquet") -> pd.DataFrame:
        """Load processed data from processed directory."""
        data_path = self.processed_dir / filename
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {data_path}")
        
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded processed data: {df.shape}")
        return df
    
    def create_temporal_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create temporal splits by season."""
        logger.info("Creating temporal splits")
        
        # Extract seasons from configuration
        train_seasons = set(self.splits_config['train_seasons'])
        val_seasons = set(self.splits_config['val_seasons'])
        test_seasons = set(self.splits_config['test_seasons'])
        
        # Validate no overlap
        if train_seasons & val_seasons:
            raise ValueError("Train and validation seasons overlap")
        if train_seasons & test_seasons:
            raise ValueError("Train and test seasons overlap")
        if val_seasons & test_seasons:
            raise ValueError("Validation and test seasons overlap")
        
        # Create splits
        train_df = df[df['season'].isin(train_seasons)].copy()
        val_df = df[df['season'].isin(val_seasons)].copy()
        test_df = df[df['season'].isin(test_seasons)].copy()
        
        # Add split labels
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'val'  # Test is also validation for now
        
        # Combine all splits
        all_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # Update split statistics
        self._update_split_stats(all_splits, train_df, val_df, test_df)
        
        logger.info(f"Temporal splits created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'all': all_splits
        }
    
    def _update_split_stats(self, all_splits: pd.DataFrame, train_df: pd.DataFrame, 
                           val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Update split statistics."""
        # Count by split
        split_counts = all_splits['split'].value_counts().to_dict()
        
        # Count by season
        season_counts = all_splits['season'].value_counts().sort_index().to_dict()
        
        # Count by constructor
        constructor_counts = all_splits['constructor_id'].value_counts().to_dict()
        
        # Target distribution by split
        target_dist_by_split = {}
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            if len(split_df) > 0:
                target_dist_by_split[split_name] = split_df['beats_teammate_q'].value_counts().to_dict()
            else:
                target_dist_by_split[split_name] = {}
        
        self.split_stats = {
            'split_counts': split_counts,
            'season_counts': season_counts,
            'constructor_counts': constructor_counts,
            'target_distribution_by_split': target_dist_by_split,
            'total_samples': len(all_splits),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df)
        }
        
        logger.info(f"Split statistics: {self.split_stats}")
    
    def create_cv_splits(self, df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create cross-validation splits by season groups."""
        logger.info(f"Creating {n_splits}-fold cross-validation splits")
        
        # Use season as group to prevent leakage
        seasons = df['season'].values
        
        # Create GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        cv_splits = list(gkf.split(df, groups=seasons))
        
        # Validate splits
        self._validate_cv_splits(df, cv_splits, seasons)
        
        logger.info(f"Created {len(cv_splits)} CV splits")
        return cv_splits
    
    def _validate_cv_splits(self, df: pd.DataFrame, cv_splits: List[Tuple[np.ndarray, np.ndarray]], 
                           seasons: np.ndarray) -> None:
        """Validate cross-validation splits for leakage."""
        logger.info("Validating CV splits for leakage")
        
        for i, (train_idx, val_idx) in enumerate(cv_splits):
            train_seasons = set(seasons[train_idx])
            val_seasons = set(seasons[val_idx])
            
            # Check for season overlap
            overlap = train_seasons & val_seasons
            if overlap:
                logger.error(f"CV split {i} has season overlap: {overlap}")
                raise ValueError(f"CV split {i} has season overlap: {overlap}")
            
            # Check event key overlap
            train_events = set(df.iloc[train_idx]['event_key'].values)
            val_events = set(df.iloc[val_idx]['event_key'].values)
            
            event_overlap = train_events & val_events
            if event_overlap:
                logger.error(f"CV split {i} has event overlap: {event_overlap}")
                raise ValueError(f"CV split {i} has event overlap: {event_overlap}")
        
        logger.info("CV split validation passed")
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns (exclude target and metadata)."""
        exclude_cols = {
            'beats_teammate_q', 'teammate_gap_ms', 'teammate_id', 'split',
            'season', 'event_key', 'driver_id', 'driver_name', 'constructor_id', 'constructor_name',
            'round', 'status', 'equal_time_tiebreak', 'single_teammate'
        }
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Identified {len(feature_cols)} feature columns")
        
        return feature_cols
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare training data with features and target."""
        logger.info("Preparing training data")
        
        # Filter to training split
        train_df = df[df['split'] == 'train'].copy()
        
        if len(train_df) == 0:
            raise ValueError("No training data found")
        
        # Get feature columns
        feature_cols = self.get_feature_columns(train_df)
        
        # Prepare X and y
        X = train_df[feature_cols].copy()
        y = train_df['beats_teammate_q'].copy()
        
        # Handle missing values in features
        X = self._handle_missing_features(X)
        
        # Convert categorical features to numeric
        X = self._encode_categorical_features(X)
        
        logger.info(f"Training data prepared: X={X.shape}, y={y.shape}")
        logger.info(f"Feature columns: {feature_cols}")
        
        return X, y, feature_cols
    
    def _handle_missing_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature matrix."""
        # For numeric columns, fill with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        # For categorical columns, fill with mode
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                mode_val = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'unknown'
                X[col] = X[col].fillna(mode_val)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical features to numeric."""
        # For boolean columns, convert to int
        bool_cols = X.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            X[col] = X[col].astype(int)
        
        # For object columns, use label encoding (simplified)
        # In production, you might want to use more sophisticated encoding
        object_cols = X.select_dtypes(include=['object']).columns
        for col in object_cols:
            if X[col].nunique() < 100:  # Only encode if not too many unique values
                X[col] = pd.Categorical(X[col]).codes
                logger.info(f"Label encoded {col}")
        
        return X
    
    def save_split_data(self, splits: Dict[str, pd.DataFrame], filename_prefix: str = "split") -> None:
        """Save split data to interim directory."""
        for split_name, split_df in splits.items():
            if len(split_df) > 0:
                output_path = self.interim_dir / f"{filename_prefix}_{split_name}.parquet"
                split_df.to_parquet(output_path, index=False)
                logger.info(f"Saved {split_name} split to {output_path}")
        
        # Save split statistics
        stats_path = self.interim_dir / "split_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(self.split_stats, f, indent=2, default=str)
        
        logger.info(f"Saved split statistics to {stats_path}")


def main():
    """Main function to run data splitting pipeline."""
    splitter = DataSplitter()
    
    # Load processed data
    df = splitter.load_processed_data()
    
    # Create temporal splits
    splits = splitter.create_temporal_splits(df)
    
    # Create CV splits
    cv_splits = splitter.create_cv_splits(splits['train'])
    
    # Prepare training data
    X, y, feature_cols = splitter.prepare_training_data(df)
    
    # Save split data
    splitter.save_split_data(splits)
    
    logger.info("Data splitting pipeline complete!")
    logger.info(f"CV splits created: {len(cv_splits)}")
    logger.info(f"Training features: {len(feature_cols)}")


if __name__ == "__main__":
    main()
