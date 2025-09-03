
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.predict import TeammatePredictor
from src.eval import ModelEvaluator

# Page config
st.set_page_config(
    page_title="F1 Teammate Qualifying Predictor",
    page_icon="🏎️",
    layout="wide"
)

# Title and description
st.title("🏎️ F1 Teammate Qualifying Predictor")
st.markdown("Predict which driver will beat their teammate in qualifying sessions using machine learning.")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Check if models are available
@st.cache_data
def check_models():
    """Check if trained models are available."""
    models_dir = Path("models")
    xgb_model = models_dir / "xgboost.joblib"
    lr_model = models_dir / "logistic_regression.joblib"
    calibrator = models_dir / "xgb_calibrator.joblib"
    
    return {
        'xgboost': xgb_model.exists(),
        'logistic_regression': lr_model.exists(),
        'calibrator': calibrator.exists(),
        'all_models': xgb_model.exists() and lr_model.exists()
    }

# Check if data is available
@st.cache_data
def check_data():
    """Check if required data is available."""
    processed_data = Path("data/processed/teammate_qual.parquet")
    labeled_data = Path("data/interim/qual_labeled.parquet")
    input_data = Path("data/input")
    
    return {
        'processed': processed_data.exists(),
        'labeled': labeled_data.exists(),
        'input_linked': input_data.exists() and input_data.is_symlink(),
        'input_has_files': input_data.exists() and len(list(input_data.rglob("*.parquet"))) > 0
    }

# Check if salary data is available
@st.cache_data
def check_salary_data():
    """Check if driver salary data is available."""
    salary_path = Path("data/input/driver_salary.csv")
    if salary_path.exists():
        try:
            df = pd.read_csv(salary_path)
            required_cols = ['driver_id', 'salary_usd']
            if all(col in df.columns for col in required_cols):
                return True, df
        except:
            pass
    return False, None

# Load available events
@st.cache_data
def load_available_events():
    """Load available seasons and events from labeled data."""
    labeled_path = Path("data/interim/qual_labeled.parquet")
    if not labeled_path.exists():
        return {}, {}
    
    try:
        df = pd.read_parquet(labeled_path)
        
        # Build season -> events mapping
        season_events = {}
        for season in sorted(df['season'].unique(), reverse=True):
            season_data = df[df['season'] == season]
            events = sorted(season_data['event_key'].unique())
            season_events[season] = events
        
        # Build event details mapping
        event_details = {}
        for _, row in df.iterrows():
            event_key = row['event_key']
            if event_key not in event_details:
                event_details[event_key] = {
                    'season': row['season'],
                    'round': row.get('round', ''),
                    'track_name': row.get('track_name', ''),
                    'is_sprint_weekend': row.get('is_sprint_weekend', False)
                }
        
        return season_events, event_details
    except Exception as e:
        st.error(f"Error loading events: {e}")
        return {}, {}

# Load configuration
@st.cache_data
def load_config():
    """Load configuration from settings.yaml."""
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {}

# Main app logic
def main():
    # Load configuration and check status
    config = load_config()
    model_status = check_models()
    data_status = check_data()
    salary_available, salary_df = check_salary_data()
    
    # Check if models are available
    if not model_status['all_models']:
        st.error("🚨 **Models not found!**")
        st.info("Please run the training pipeline first:")
        st.code("python run_all.py --train")
        st.stop()
    
    # Check if data is available
    if not data_status['processed']:
        st.error("🚨 **Processed data not found!**")
        st.info("Please run the data processing pipeline first:")
        st.code("python run_all.py --build")
        st.stop()
    
    if not data_status['input_has_files']:
        st.warning("⚠️ **Input data not linked or empty!**")
        st.info("Please link your F1 data first:")
        st.code("make link")
        st.stop()
    
    # Load available events
    season_events, event_details = load_available_events()
    if not season_events:
        st.error("🚨 **No events found in labeled data!**")
        st.info("Please run the labeling pipeline first:")
        st.code("python run_all.py --build")
        st.stop()
    
    # Sidebar controls
    st.sidebar.subheader("Select Event")
    
    # Season dropdown
    selected_season = st.sidebar.selectbox(
        "Season",
        options=list(season_events.keys()),
        index=0
    )
    
    # Event dropdown
    if selected_season in season_events:
        events = season_events[selected_season]
        selected_event = st.sidebar.selectbox(
            "Event",
            options=events,
            index=len(events) - 1,  # Default to most recent
            format_func=lambda x: f"{x} ({event_details.get(x, {}).get('track_name', 'Unknown Track')})"
        )
    else:
        selected_event = None
        st.sidebar.warning("No events found for selected season")
    
    # Model info
    st.sidebar.subheader("Model Information")
    st.sidebar.info(f"**XGBoost**: {'✅' if model_status['xgboost'] else '❌'}")
    st.sidebar.info(f"**Logistic Regression**: {'✅' if model_status['logistic_regression'] else '❌'}")
    st.sidebar.info(f"**Calibrator**: {'✅' if model_status['calibrator'] else '❌'}")
    
    # Data info
    st.sidebar.subheader("Data Status")
    st.sidebar.info(f"**Processed**: {'✅' if data_status['processed'] else '❌'}")
    st.sidebar.info(f"**Labeled**: {'✅' if data_status['labeled'] else '❌'}")
    st.sidebar.info(f"**Input Linked**: {'✅' if data_status['input_linked'] else '❌'}")
    
    # Salary baseline info
    if salary_available:
        st.sidebar.success("💰 **Salary baseline available**")
    else:
        st.sidebar.info("💰 **Salary baseline**: Add `data/input/driver_salary.csv` to enable")
    
    # Main content area
    if selected_event:
        st.header(f"Predictions for {selected_event}")
        
        # Event details
        if selected_event in event_details:
            details = event_details[selected_event]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Season", details['season'])
            with col2:
                st.metric("Round", details['round'] if details['round'] else "N/A")
            with col3:
                st.metric("Track", details['track_name'] if details['track_name'] else "Unknown")
            
            if details.get('is_sprint_weekend'):
                st.info("🏁 **Sprint Weekend**: Using qualifying session that sets race grid (not sprint shootout)")
        
        # Load predictions
        try:
            predictor = TeammatePredictor()
            predictions_df = predictor.build_event_prediction_df(selected_event, include_actual=True)
            
            # Load actual results for evaluation
            labeled_path = Path("data/interim/qual_labeled.parquet")
            actual_df = pd.read_parquet(labeled_path)
            event_actual = actual_df[actual_df['event_key'] == selected_event]
            
            # Merge predictions with actual results
            results_df = predictions_df.merge(
                event_actual[['driver_id', 'beats_teammate_q', 'teammate_gap_ms']], 
                on='driver_id', 
                how='left'
            )
            
            # Add model correctness column
            results_df['model_correct'] = (results_df['model_pick'] == results_df['actual_beats_teammate'])
            
            # Compute Baseline A: Prior head-to-head leader
            results_df = compute_baseline_a(results_df, selected_event, details['season'])
            
            # Compute Baseline B: Salary-based (if available)
            if salary_available:
                results_df = compute_baseline_b(results_df, salary_df)
            
            # Calculate accuracy metrics
            metrics = calculate_accuracy_metrics(results_df)
            
            # Display summary metrics
            st.subheader("📊 Event Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Accuracy", f"{metrics['model_accuracy']:.1%}")
            with col2:
                st.metric("Baseline A Accuracy", f"{metrics['baseline_a_accuracy']:.1%}")
            with col3:
                if salary_available:
                    st.metric("Baseline B Accuracy", f"{metrics['baseline_b_accuracy']:.1%}")
                else:
                    st.metric("Baseline B", "N/A")
            
            # Display predictions table
            st.subheader("🏁 Team Predictions")
            
            # Create a more readable table
            display_df = create_display_table(results_df, salary_available)
            
            # Color code the rows based on correctness
            def color_correctness(val):
                if pd.isna(val):
                    return ''
                return 'background-color: lightgreen' if val else 'background-color: lightcoral'
            
            styled_df = display_df.style.applymap(
                color_correctness, 
                subset=['Model Correct', 'Baseline A Correct', 'Baseline B Correct']
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Legend
            st.markdown("---")
            st.markdown("**Legend:**")
            st.markdown("- **Model confidence** = calibrated probability that the predicted teammate beats their teammate in qualifying")
            st.markdown("- **Baseline A** = prior head-to-head leader entering the event")
            st.markdown("- **Baseline B** = higher salary driver (if salary data available)")
            
        except Exception as e:
            st.error(f"Error loading predictions: {e}")
            st.info("Please ensure the pipeline is properly set up and run:")
            st.code("make build && make train")
    
    else:
        st.info("👈 Select a season and event from the sidebar to view predictions")

def compute_baseline_a(results_df, event_key, season):
    """Compute Baseline A: Prior head-to-head leader."""
    try:
        # Load labeled data to compute rolling head-to-head
        labeled_path = Path("data/interim/qual_labeled.parquet")
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
        st.warning(f"Warning: Could not compute Baseline A: {e}")
        results_df['baseline_a_pick'] = np.nan
        results_df['baseline_a_correct'] = np.nan
        return results_df

def compute_baseline_b(results_df, salary_df):
    """Compute Baseline B: Salary-based prediction."""
    try:
        # Initialize baseline columns
        results_df['baseline_b_pick'] = np.nan
        results_df['baseline_b_correct'] = np.nan
        
        # For each team, pick the higher salary driver
        for constructor_id in results_df['constructor_id'].unique():
            team_drivers = results_df[results_df['constructor_id'] == constructor_id]
            if len(team_drivers) != 2:
                continue
                
            driver1, driver2 = team_drivers.iloc[0], team_drivers.iloc[1]
            
            # Get salaries
            salary1 = salary_df[salary_df['driver_id'] == driver1['driver_id']]['salary_usd'].iloc[0] if len(salary_df[salary_df['driver_id'] == driver1['driver_id']]) > 0 else 0
            salary2 = salary_df[salary_df['driver_id'] == driver2['driver_id']]['salary_usd'].iloc[0] if len(salary_df[salary_df['driver_id'] == driver2['driver_id']]) > 0 else 0
            
            # Pick the higher salary driver
            if salary1 > salary2:
                baseline_pick = driver1['driver_id']
            elif salary2 > salary1:
                baseline_pick = driver2['driver_id']
            else:
                baseline_pick = None  # Equal salaries
            
            # Set baseline picks and correctness
            for _, driver in team_drivers.iterrows():
                if baseline_pick is not None:
                    is_baseline_pick = (driver['driver_id'] == baseline_pick)
                    results_df.loc[results_df['driver_id'] == driver['driver_id'], 'baseline_b_pick'] = is_baseline_pick
                    
                    # Check if baseline was correct
                    if pd.notna(driver['actual_beats_teammate']):
                        baseline_correct = (is_baseline_pick == driver['actual_beats_teammate'])
                        results_df.loc[results_df['driver_id'] == driver['driver_id'], 'baseline_b_correct'] = baseline_correct
                else:
                    results_df.loc[results_df['driver_id'] == driver['driver_id'], 'baseline_b_pick'] = None
                    results_df.loc[results_df['driver_id'] == driver['driver_id'], 'baseline_b_correct'] = None
        
        return results_df
        
    except Exception as e:
        st.warning(f"Warning: Could not compute Baseline B: {e}")
        results_df['baseline_b_pick'] = np.nan
        results_df['baseline_b_correct'] = np.nan
        return results_df

def calculate_accuracy_metrics(results_df):
    """Calculate accuracy metrics for the event."""
    metrics = {}
    
    # Model accuracy
    model_correct = results_df['model_correct'].dropna()
    metrics['model_accuracy'] = model_correct.mean() if len(model_correct) > 0 else 0
    
    # Baseline A accuracy (only on pairs where baseline had a pick)
    baseline_a_correct = results_df['baseline_a_correct'].dropna()
    metrics['baseline_a_accuracy'] = baseline_a_correct.mean() if len(baseline_a_correct) > 0 else 0
    
    # Baseline B accuracy (if available)
    baseline_b_correct = results_df['baseline_b_correct'].dropna()
    metrics['baseline_b_accuracy'] = baseline_b_correct.mean() if len(baseline_b_correct) > 0 else 0
    
    return metrics

def create_display_table(results_df, salary_available):
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
            'Constructor': constructor_id.replace('_', ' ').title(),
            'Driver A': driver1['driver_name'],
            'Driver B': driver2['driver_name'],
            'Model Pick': winner['driver_name'],
            'Model Confidence': f"{winner['model_confidence']:.1%}",
            'Actual Winner': winner['driver_name'] if winner['actual_beats_teammate'] == 1 else loser['driver_name'],
            'Baseline A Pick': winner['driver_name'] if winner['baseline_a_pick'] == 1 else (loser['driver_name'] if loser['baseline_a_pick'] == 1 else 'No Prior'),
            'Baseline B Pick': winner['driver_name'] if winner['baseline_b_pick'] == 1 else (loser['driver_name'] if loser['baseline_b_pick'] == 1 else 'Equal Salary') if salary_available else 'N/A',
            'Model Correct': winner['actual_beats_teammate'] == 1,
            'Baseline A Correct': winner['baseline_a_correct'] if pd.notna(winner['baseline_a_correct']) else None,
            'Baseline B Correct': winner['baseline_b_correct'] if pd.notna(winner['baseline_b_correct']) else None
        }
        
        display_rows.append(row)
    
    return pd.DataFrame(display_rows)

if __name__ == "__main__":
    main()
