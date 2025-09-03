"""
Unit tests for the labeling module.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labeling import TeammateLabeler


class TestTeammateLabeler(unittest.TestCase):
    """Test cases for TeammateLabeler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.config_content = """
data:
  interim_dir: "test_data/interim"
sprint_weekends:
  grid_setting_session: "Q"
        """
        
        # Create test data
        self.test_data = pd.DataFrame({
            'season': [2023, 2023, 2023, 2023],
            'event_key': ['2023_01', '2023_01', '2023_02', '2023_02'],
            'constructor_id': ['mercedes', 'mercedes', 'redbull', 'redbull'],
            'driver_id': ['hamilton', 'russell', 'verstappen', 'perez'],
            'quali_position': [1, 2, 1, 3],
            'quali_best_lap_ms': [80000, 80100, 79000, 79200],
            'status': ['Finished', 'Finished', 'Finished', 'Finished']
        })
        
        # Create test directory
        test_dir = Path("test_data/interim")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Save test config
        config_dir = Path("test_data/config")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_dir / "test_settings.yaml", 'w') as f:
            f.write(self.config_content)
        
        self.labeler = TeammateLabeler(str(config_dir / "test_settings.yaml"))
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path("test_data").exists():
            shutil.rmtree("test_data")
    
    def test_create_teammate_labels(self):
        """Test teammate label creation."""
        labeled_df = self.labeler.create_teammate_labels(self.test_data)
        
        # Check that labels were created
        self.assertIn('beats_teammate_q', labeled_df.columns)
        self.assertIn('teammate_gap_ms', labeled_df.columns)
        self.assertIn('teammate_id', labeled_df.columns)
        
        # Check that we have the expected number of labeled pairs
        labeled_pairs = labeled_df[labeled_df['beats_teammate_q'].notna()]
        self.assertEqual(len(labeled_pairs), 4)  # 2 teams × 2 drivers each
        
        # Check Mercedes team
        mercedes_data = labeled_df[labeled_df['constructor_id'] == 'mercedes']
        hamilton = mercedes_data[mercedes_data['driver_id'] == 'hamilton'].iloc[0]
        russell = mercedes_data[mercedes_data['driver_id'] == 'russell'].iloc[0]
        
        # Hamilton should beat Russell (position 1 vs 2)
        self.assertEqual(hamilton['beats_teammate_q'], 1)
        self.assertEqual(russell['beats_teammate_q'], 0)
        
        # Check teammate IDs
        self.assertEqual(hamilton['teammate_id'], 'russell')
        self.assertEqual(russell['teammate_id'], 'hamilton')
        
        # Check teammate gap (Russell is 100ms slower)
        self.assertEqual(hamilton['teammate_gap_ms'], 100)
        self.assertEqual(russell['teammate_gap_ms'], -100)
    
    def test_validate_labels(self):
        """Test label validation."""
        # Create valid labeled data
        labeled_df = self.labeler.create_teammate_labels(self.test_data)
        
        # Validation should pass
        self.assertTrue(self.labeler.validate_labels(labeled_df))
        
        # Test invalid labels
        invalid_df = labeled_df.copy()
        invalid_df.loc[0, 'beats_teammate_q'] = 2  # Invalid value
        
        # Validation should fail
        self.assertFalse(self.labeler.validate_labels(invalid_df))
    
    def test_handle_sprint_weekends(self):
        """Test sprint weekend handling."""
        labeled_df = self.labeler.create_teammate_labels(self.test_data)
        sprint_handled = self.labeler.handle_sprint_weekends(labeled_df)
        
        # Should add sprint weekend flag
        self.assertIn('is_sprint_weekend', sprint_handled.columns)
        
        # All should be False for test data
        self.assertTrue(all(sprint_handled['is_sprint_weekend'] == False))


if __name__ == '__main__':
    unittest.main()
