"""
Unit tests for the data splitting module.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from split import DataSplitter


class TestDataSplitter(unittest.TestCase):
    """Test cases for DataSplitter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.config_content = """
data:
  interim_dir: "test_data/interim"
  processed_dir: "test_data/processed"
splits:
  train_seasons: [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
  val_seasons: [2023]
  test_seasons: [2024, 2025]
        """
        
        # Create test data with multiple seasons
        self.test_data = pd.DataFrame({
            'season': [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
            'event_key': ['2020_01', '2020_02', '2021_01', '2021_02', '2022_01', '2022_02', '2023_01', '2023_02', '2024_01', '2024_02'],
            'constructor_id': ['mercedes', 'mercedes', 'mercedes', 'mercedes', 'mercedes', 'mercedes', 'mercedes', 'mercedes', 'mercedes', 'mercedes'],
            'driver_id': ['hamilton', 'russell'] * 5,
            'quali_position': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            'beats_teammate_q': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'track_id': ['monaco', 'baku', 'monaco', 'baku', 'monaco', 'baku', 'monaco', 'baku', 'monaco', 'baku']
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
        
        self.splitter = DataSplitter(str(config_dir / "test_settings.yaml"))
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path("test_data").exists():
            shutil.rmtree("test_data")
    
    def test_create_temporal_splits(self):
        """Test temporal split creation."""
        splits = self.splitter.create_temporal_splits(self.test_data)
        
        # Check that splits were created
        self.assertIn('train', splits)
        self.assertIn('val', splits)
        self.assertIn('test', splits)
        self.assertIn('all', splits)
        
        # Check split sizes
        self.assertEqual(len(splits['train']), 6)  # 2020-2022 (3 seasons × 2 events)
        self.assertEqual(len(splits['val']), 2)   # 2023 (1 season × 2 events)
        self.assertEqual(len(splits['test']), 2)  # 2024 (1 season × 2 events)
        
        # Check that split column was added
        self.assertIn('split', splits['all'].columns)
        
        # Check split values
        self.assertTrue(all(splits['train']['split'] == 'train'))
        self.assertTrue(all(splits['val']['split'] == 'val'))
        self.assertTrue(all(splits['test']['split'] == 'val'))  # Test is also val for now
    
    def test_split_validation(self):
        """Test that splits don't overlap."""
        splits = self.splitter.create_temporal_splits(self.test_data)
        
        # Check no season overlap
        train_seasons = set(splits['train']['season'].unique())
        val_seasons = set(splits['val']['season'].unique())
        test_seasons = set(splits['test']['season'].unique())
        
        self.assertEqual(len(train_seasons & val_seasons), 0)
        self.assertEqual(len(train_seasons & test_seasons), 0)
        self.assertEqual(len(val_seasons & test_seasons), 0)
        
        # Check no event overlap
        train_events = set(splits['train']['event_key'].unique())
        val_events = set(splits['val']['event_key'].unique())
        test_events = set(splits['test']['event_key'].unique())
        
        self.assertEqual(len(train_events & val_events), 0)
        self.assertEqual(len(train_events & test_events), 0)
        self.assertEqual(len(val_events & test_events), 0)
    
    def test_cv_splits(self):
        """Test cross-validation split creation."""
        splits = self.splitter.create_temporal_splits(self.test_data)
        cv_splits = self.splitter.create_cv_splits(splits['train'])
        
        # Check that CV splits were created
        self.assertIsInstance(cv_splits, list)
        self.assertGreater(len(cv_splits), 0)
        
        # Check that each split is a tuple of (train_idx, val_idx)
        for split in cv_splits:
            self.assertIsInstance(split, tuple)
            self.assertEqual(len(split), 2)
            self.assertIsInstance(split[0], np.ndarray)
            self.assertIsInstance(split[1], np.ndarray)
    
    def test_feature_columns(self):
        """Test feature column identification."""
        feature_cols = self.splitter.get_feature_columns(self.test_data)
        
        # Check that target and metadata columns are excluded
        excluded_cols = ['beats_teammate_q', 'teammate_gap_ms', 'split', 'season', 'event_key']
        for col in excluded_cols:
            self.assertNotIn(col, feature_cols)
        
        # Check that feature columns are included
        feature_cols_expected = ['constructor_id', 'driver_id', 'quali_position', 'track_id']
        for col in feature_cols_expected:
            self.assertIn(col, feature_cols)
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        # Add split column to test data
        test_data_with_split = self.test_data.copy()
        test_data_with_split['split'] = 'train'
        
        X, y, feature_cols = self.splitter.prepare_training_data(test_data_with_split)
        
        # Check that X and y have correct shapes
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X.columns), len(feature_cols))
        
        # Check that y contains the target
        self.assertIn('beats_teammate_q', y.name)
        
        # Check that X contains only features
        self.assertNotIn('beats_teammate_q', X.columns)
        self.assertNotIn('split', X.columns)
    
    def test_split_statistics(self):
        """Test split statistics tracking."""
        splits = self.splitter.create_temporal_splits(self.test_data)
        
        # Check that statistics were updated
        self.assertIsNotNone(self.splitter.split_stats)
        self.assertIn('total_samples', self.splitter.split_stats)
        self.assertIn('train_samples', self.splitter.split_stats)
        self.assertIn('val_samples', self.splitter.split_stats)
        self.assertIn('test_samples', self.splitter.split_stats)
        
        # Check that statistics are correct
        self.assertEqual(self.splitter.split_stats['total_samples'], 10)
        self.assertEqual(self.splitter.split_stats['train_samples'], 6)
        self.assertEqual(self.splitter.split_stats['val_samples'], 2)
        self.assertEqual(self.splitter.split_stats['test_samples'], 2)


if __name__ == '__main__':
    unittest.main()
