"""
Baseline functions for F1 teammate qualifying predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


def compute_baseline_a(results_df: pd.DataFrame, event_key: str, season: int) -> pd.DataFrame:
    """Compute Baseline A: Prior head-to-head leader."""
    try:
        # Load labeled data to compute rolling head-to-head
        labeled_path = Path("data/interim/qual_labeled.parquet")
        if not labeled_path.exists():
            # If no labeled data, return results_df with empty baseline columns
            results_df['baseline_a_pick'] = np.nan
            results_df['baseline_a_correct'] = np.nan
            return results_df
            
        all_data = pd.read_parquet(labeled_path)
        
        # Get data strictly before the selected event (same season)
        prior_data = all_data[
            (all_data['season'] == season) & 
            (all_data['event_key'] < event_key)
        ].copy()
        
        # Initialize baseline columns
        results_df['baseline_a_pick'] = np.nan
        results_df['baseline_a_correct'] = np.nan
        
        # For each team, compute rolling head-to-head
        for constructor_id in results_df['constructor_id'].unique():
            team_drivers = results_df[results_df['constructor_id'] == constructor_id]
            if len(team_drivers) != 2:
                continue
                
            driver1, driver2 = team_drivers.iloc[0], team_drivers.iloc[1]
            
            # Get prior head-to-head data for this team
            team_prior = prior_data[prior_data['constructor_id'] == constructor_id]
            if len(team_prior) == 0:
                continue
            
            # Compute rolling head-to-head record
            driver1_wins = 0
            driver2_wins = 0
            
            for _, row in team_prior.iterrows():
                if row['driver_id'] == driver1['driver_id']:
                    if row['beats_teammate_q'] == 1:
                        driver1_wins += 1
                    else:
                        driver2_wins += 1
                elif row['driver_id'] == driver2['driver_id']:
                    if row['beats_teammate_q'] == 1:
                        driver2_wins += 1
                    else:
                        driver1_wins += 1
            
            # Pick the leader (or mark as no prior if tie)
            if driver1_wins > driver2_wins:
                baseline_pick = driver1['driver_id']
            elif driver2_wins > driver1_wins:
                baseline_pick = driver2['driver_id']
            else:
                baseline_pick = None  # Tie or no prior
            
            # Set baseline picks and correctness
            for _, driver in team_drivers.iterrows():
                if baseline_pick is not None:
                    is_baseline_pick = (driver['driver_id'] == baseline_pick)
                    results_df.loc[results_df['driver_id'] == driver['driver_id'], 'baseline_a_pick'] = is_baseline_pick
                    
                    # Check if baseline was correct
                    if pd.notna(driver['actual_beats_teammate']):
                        baseline_correct = (is_baseline_pick == driver['actual_beats_teammate'])
                        results_df.loc[results_df['driver_id'] == driver['driver_id'], 'baseline_a_correct'] = baseline_correct
                else:
                    results_df.loc[results_df['driver_id'] == driver['driver_id'], 'baseline_a_pick'] = None
                    results_df.loc[results_df['driver_id'] == driver['driver_id'], 'baseline_a_correct'] = None
        
        return results_df
        
    except Exception as e:
        print(f"Warning: Could not compute Baseline A: {e}")
        results_df['baseline_a_pick'] = np.nan
        results_df['baseline_a_correct'] = np.nan
        return results_df


def calculate_accuracy_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate accuracy metrics for the event."""
    metrics = {}
    
    # Model accuracy
    if 'model_correct' in results_df.columns:
        model_correct = results_df['model_correct'].dropna()
        metrics['model_accuracy'] = model_correct.mean() if len(model_correct) > 0 else 0
    else:
        metrics['model_accuracy'] = 0
    
    # Baseline A accuracy (only on pairs where baseline had a pick)
    if 'baseline_a_correct' in results_df.columns:
        baseline_a_correct = results_df['baseline_a_correct'].dropna()
        metrics['baseline_a_accuracy'] = baseline_a_correct.mean() if len(baseline_a_correct) > 0 else 0
    else:
        metrics['baseline_a_accuracy'] = 0
    
    return metrics


def create_display_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a display-friendly table for the results."""
    # Group by constructor to show teammate pairs
    display_rows = []
    
    for constructor_id in sorted(results_df['constructor_id'].unique()):
        team_drivers = results_df[results_df['constructor_id'] == constructor_id]
        if len(team_drivers) != 2:
            continue
            
        driver1, driver2 = team_drivers.iloc[0], team_drivers.iloc[1]
        
        # Determine which driver the model picked as winner
        if driver1['model_pick'] == 1:
            winner = driver1
            loser = driver2
        else:
            winner = driver2
            loser = driver1
        
        # Create display row
        row = {
            '🏎️ Team': constructor_id.replace('_', ' ').title(),
            'Driver A': driver1['driver_name'],
            'Driver B': driver2['driver_name'],
            '🤖 Model Pick': winner['driver_name'],
            '🎯 Model Confidence': f"{winner['model_confidence']:.1%}",
            '🏁 Actual Winner': winner['driver_name'] if winner['actual_beats_teammate'] == 1 else loser['driver_name'],
            '📈 H2H Pick': winner['driver_name'] if winner['baseline_a_pick'] == 1 else (loser['driver_name'] if loser['baseline_a_pick'] == 1 else 'No Prior'),
            '✅ Model Correct': winner['actual_beats_teammate'] == 1,
            '✅ H2H Correct': winner.get('baseline_a_correct', None) if pd.notna(winner.get('baseline_a_correct', None)) else None
        }
        
        display_rows.append(row)
    
    return pd.DataFrame(display_rows)
