"""
Unit tests for the feature engineering module.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.config_content = """
data:
  interim_dir: "test_data/interim"
  processed_dir: "test_data/processed"
features:
  driver_form_windows: [3, 5]
  team_form_windows: [3, 5]
  h2h_windows: [4, 6]
  practice_imputation: "seasonal_median"
  outlier_clipping:
    quali_time: [0.05, 0.95]
    practice_delta: [0.01, 0.99]
        """
        
        # Create test data with labels
        self.test_data = pd.DataFrame({
            'season': [2023, 2023, 2023, 2023, 2023, 2023],
            'event_key': ['2023_01', '2023_01', '2023_02', '2023_02', '2023_03', '2023_03'],
            'constructor_id': ['mercedes', 'mercedes', 'mercedes', 'mercedes', 'mercedes', 'mercedes'],
            'driver_id': ['hamilton', 'russell', 'hamilton', 'russell', 'hamilton', 'russell'],
            'quali_position': [1, 2, 2, 1, 1, 2],
            'quali_best_lap_ms': [80000, 80100, 80200, 80050, 79900, 80150],
            'beats_teammate_q': [1, 0, 0, 1, 1, 0],
            'teammate_gap_ms': [100, -100, -150, 150, 250, -250],
            'track_id': ['monaco', 'monaco', 'baku', 'baku', 'spa', 'spa'],
            'status': ['Finished'] * 6
        })
        
        # Create test directories
        test_dirs = ["test_data/interim", "test_data/processed"]
        for test_dir in test_dirs:
            Path(test_dir).mkdir(parents=True, exist_ok=True)
        
        # Save test config
        config_dir = Path("test_data/config")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_dir / "test_settings.yaml", 'w') as f:
            f.write(self.config_content)
        
        self.engineer = FeatureEngineer(str(config_dir / "test_settings.yaml"))
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path("test_data").exists():
            shutil.rmtree("test_data")
    
    def test_create_features(self):
        """Test feature creation."""
        featured_df = self.engineer.create_features(self.test_data)
        
        # Check that features were created
        self.assertGreater(len(featured_df.columns), len(self.test_data.columns))
        
        # Check for specific feature types
        self.assertTrue(any('driver_quali_pos_avg' in col for col in featured_df.columns))
        self.assertTrue(any('team_quali_pos_avg' in col for col in featured_df.columns))
        self.assertTrue(any('driver_beats_teammate_share' in col for col in featured_df.columns))
        self.assertTrue(any('track_' in col for col in featured_df.columns))
        self.assertTrue('is_street_circuit' in featured_df.columns)
    
    def test_driver_form_features(self):
        """Test driver form feature creation."""
        featured_df = self.engineer.create_features(self.test_data)
        
        # Check rolling averages
        hamilton_data = featured_df[featured_df['driver_id'] == 'hamilton']
        
        # Should have rolling averages
        self.assertTrue('driver_quali_pos_avg_3' in featured_df.columns)
        self.assertTrue('driver_quali_pos_avg_5' in featured_df.columns)
        
        # Check that values are reasonable
        for col in ['driver_quali_pos_avg_3', 'driver_quali_pos_avg_5']:
            if col in featured_df.columns:
                values = featured_df[col].dropna()
                self.assertTrue(all(values >= 1))  # Positions should be >= 1
    
    def test_team_form_features(self):
        """Test team form feature creation."""
        featured_df = self.engineer.create_features(self.test_data)
        
        # Check team features
        self.assertTrue('team_quali_pos_avg_3' in featured_df.columns)
        self.assertTrue('team_quali_pos_avg_5' in featured_df.columns)
        
        # Check that values are reasonable
        for col in ['team_quali_pos_avg_3', 'team_quali_pos_avg_5']:
            if col in featured_df.columns:
                values = featured_df[col].dropna()
                self.assertTrue(all(values >= 1))  # Positions should be >= 1
    
    def test_track_features(self):
        """Test track feature creation."""
        featured_df = self.engineer.create_features(self.test_data)
        
        # Check track features
        self.assertTrue('is_street_circuit' in featured_df.columns)
        self.assertTrue('driver_track_familiarity' in featured_df.columns)
        
        # Check track one-hot encoding
        track_cols = [col for col in featured_df.columns if col.startswith('track_')]
        self.assertGreater(len(track_cols), 0)
        
        # Check street circuit flag
        # Monaco should be marked as street circuit
        monaco_data = featured_df[featured_df['track_id'] == 'monaco']
        self.assertTrue(all(monaco_data['is_street_circuit'] == True))
    
    def test_data_hygiene(self):
        """Test data hygiene features."""
        featured_df = self.engineer.create_features(self.test_data)
        
        # Check for clipping flags
        clipping_cols = [col for col in featured_df.columns if col.endswith('_was_clipped')]
        self.assertGreater(len(clipping_cols), 0)
        
        # Check for imputation flags
        imputation_cols = [col for col in featured_df.columns if col.endswith('_imputed')]
        self.assertGreater(len(imputation_cols), 0)
        
        # Check for missing flags
        missing_cols = [col for col in featured_df.columns if col.endswith('_missing')]
        self.assertGreater(len(missing_cols), 0)
    
    def test_feature_statistics(self):
        """Test feature statistics tracking."""
        featured_df = self.engineer.create_features(self.test_data)
        
        # Check that statistics were updated
        self.assertIsNotNone(self.engineer.feature_stats)
        self.assertIn('total_features', self.engineer.feature_stats)
        self.assertIn('numeric_features', self.engineer.feature_stats)
        self.assertIn('categorical_features', self.engineer.feature_stats)
        
        # Check that statistics are reasonable
        self.assertGreater(self.engineer.feature_stats['total_features'], 0)
        self.assertGreaterEqual(self.engineer.feature_stats['numeric_features'], 0)
        self.assertGreaterEqual(self.engineer.feature_stats['categorical_features'], 0)


if __name__ == '__main__':
    unittest.main()
