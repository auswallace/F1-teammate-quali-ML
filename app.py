
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

# Track name mapping (round number to actual track names)
TRACK_MAPPING = {
    'R01': 'Bahrain International Circuit',
    'R02': 'Jeddah Corniche Circuit', 
    'R03': 'Albert Park Circuit',
    'R04': 'Suzuka International Racing Course',
    'R05': 'Miami International Autodrome',
    'R06': 'Circuit de Monaco',
    'R07': 'Circuit de Barcelona-Catalunya',
    'R08': 'Red Bull Ring',
    'R09': 'Silverstone Circuit',
    'R10': 'Hungaroring',
    'R11': 'Circuit de Spa-Francorchamps',
    'R12': 'Circuit Zandvoort',
    'R13': 'Monza Circuit',
    'R14': 'Baku City Circuit',
    'R15': 'Las Vegas Strip Circuit',
    'R16': 'Lusail International Circuit',
    'R17': 'Yas Marina Circuit',
    'R18': 'Circuit of the Americas',
    'R19': 'Autódromo José Carlos Pace',
    'R20': 'Autódromo Hermanos Rodríguez',
    'R21': 'Marina Bay Street Circuit',
    'R22': 'Shanghai International Circuit',
    'R23': 'Mount Fuji Speedway',
    'R24': 'Kyalami Grand Prix Circuit'
}

# Season dates mapping (actual F1 2025 calendar)
SEASON_DATES = {
    2025: {
        'R01': 'March 2, 2025',      # Bahrain
        'R02': 'March 9, 2025',      # Saudi Arabia
        'R03': 'March 23, 2025',     # Australia
        'R04': 'April 6, 2025',      # Japan
        'R05': 'April 20, 2025',     # China
        'R06': 'May 4, 2025',        # Miami
        'R07': 'May 18, 2025',       # Emilia Romagna
        'R08': 'May 25, 2025',       # Monaco
        'R09': 'June 1, 2025',       # Spain
        'R10': 'June 8, 2025',       # Austria
        'R11': 'June 22, 2025',      # Great Britain
        'R12': 'July 6, 2025',       # Hungary
        'R13': 'July 20, 2025',      # Belgium
        'R14': 'July 27, 2025',      # Netherlands
        'R15': 'August 3, 2025',     # Italy
        'R16': 'August 17, 2025',    # Azerbaijan
        'R17': 'September 7, 2025',  # Singapore
        'R18': 'September 21, 2025', # United States
        'R19': 'October 5, 2025',    # Mexico
        'R20': 'October 19, 2025',   # Brazil
        'R21': 'November 2, 2025',   # Las Vegas
        'R22': 'November 16, 2025',  # Qatar
        'R23': 'November 23, 2025',  # Abu Dhabi
    },
    2024: {
        'R01': 'March 2, 2024',      # Bahrain
        'R02': 'March 9, 2024',      # Saudi Arabia
        'R03': 'March 24, 2024',     # Australia
        'R04': 'April 7, 2024',      # Japan
        'R05': 'April 21, 2024',     # China
        'R06': 'May 5, 2024',        # Miami
        'R07': 'May 19, 2024',       # Emilia Romagna
        'R08': 'May 26, 2024',       # Monaco
        'R09': 'June 2, 2024',       # Spain
        'R10': 'June 9, 2024',       # Austria
        'R11': 'June 23, 2024',      # Great Britain
        'R12': 'July 7, 2024',       # Hungary
        'R13': 'July 21, 2024',      # Belgium
        'R14': 'July 28, 2024',      # Netherlands
        'R15': 'August 4, 2024',     # Italy
        'R16': 'August 18, 2024',    # Azerbaijan
        'R17': 'September 8, 2024',  # Singapore
        'R18': 'September 22, 2024', # United States
        'R19': 'October 6, 2024',    # Mexico
        'R20': 'October 20, 2024',   # Brazil
        'R21': 'November 3, 2024',   # Las Vegas
        'R22': 'November 17, 2024',  # Qatar
        'R23': 'November 24, 2024',  # Abu Dhabi
    }
}

# Page config
st.set_page_config(
    page_title="F1 Teammate Qualifying Predictor",
    page_icon="🏎️",
    layout="wide"
)

# Title and description
st.title("🏎️ F1 Teammate Qualifying Predictor")
st.markdown("""
**Predict which driver will beat their teammate in qualifying sessions using machine learning.**

This system uses historical F1 data to predict teammate qualifying outcomes with calibrated probabilities. 
The model considers driver form, team performance, track characteristics, practice sessions, and weather conditions.
""")

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
                round_key = row.get('round', '')
                season = row['season']
                
                # Get proper track name and date
                track_name = TRACK_MAPPING.get(round_key, f"Round {round_key}")
                event_date = SEASON_DATES.get(season, {}).get(round_key, "Date TBD")
                
                event_details[event_key] = {
                    'season': season,
                    'round': round_key,
                    'track_name': track_name,
                    'event_date': event_date,
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
        
        # Create better event descriptions
        def format_event_name(event_key):
            details = event_details.get(event_key, {})
            track_name = details.get('track_name', 'Unknown Track')
            round_num = details.get('round', '')
            event_date = details.get('event_date', '')
            
            # Create a rich description with round, track, and date
            if round_num and track_name != f"Round {round_num}":
                # Extract just the track name without "Circuit" etc. for cleaner display
                clean_track = track_name.replace(' International Circuit', '').replace(' Circuit', '').replace(' Autodrome', '')
                return f"{round_num} - {clean_track} ({event_date})"
            else:
                return f"{event_key} - {track_name} ({event_date})"
        
        selected_event = st.sidebar.selectbox(
            "Event",
            options=events,
            index=len(events) - 1,  # Default to most recent
            format_func=format_event_name
        )
    else:
        selected_event = None
        st.sidebar.warning("No events found for selected season")
    
    # Model info
    st.sidebar.subheader("🤖 Model Information")
    st.sidebar.info(f"**XGBoost (Main Model)**: {'✅ Trained' if model_status['xgboost'] else '❌ Missing'}")
    st.sidebar.info(f"**Logistic Regression**: {'✅ Trained' if model_status['logistic_regression'] else '❌ Missing'}")
    st.sidebar.info(f"**Probability Calibrator**: {'✅ Active' if model_status['calibrator'] else '❌ Missing'}")
    
    if model_status['calibrator']:
        st.sidebar.success("🎯 **Calibrated predictions enabled** - probabilities are trustworthy!")
    
    # Data info
    st.sidebar.subheader("📊 Data Status")
    st.sidebar.info(f"**Processed Features**: {'✅ Ready' if data_status['processed'] else '❌ Missing'}")
    st.sidebar.info(f"**Labeled Data**: {'✅ Ready' if data_status['labeled'] else '❌ Missing'}")
    st.sidebar.info(f"**F1 Data Source**: {'✅ Linked' if data_status['input_linked'] else '❌ Missing'}")
    
    # Salary baseline info
    if salary_available:
        st.sidebar.success("💰 **Salary baseline available**")
    else:
        st.sidebar.info("💰 **Salary baseline**: Add `data/input/driver_salary.csv` to enable")
    
    # Main content area
    if selected_event:
        st.header(f"🏁 Predictions for {selected_event}")
        
        # Event details
        if selected_event in event_details:
            details = event_details[selected_event]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Season", details['season'])
            with col2:
                st.metric("Round", details['round'] if details['round'] else "N/A")
            with col3:
                st.metric("Track", details['track_name'] if details['track_name'] else "Unknown")
            with col4:
                st.metric("Date", details.get('event_date', 'TBD'))
            
            if details.get('is_sprint_weekend'):
                st.info("🏁 **Sprint Weekend**: Using qualifying session that sets race grid (not sprint shootout)")
        
        # Add context about what we're predicting
        st.markdown("""
        **What we're predicting:** For each team, we predict which driver will qualify ahead of their teammate 
        and provide a confidence score. The model considers historical performance, current form, track characteristics, 
        and practice session data.
        """)
        
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
            else:
                # Initialize Baseline B columns as NaN if not available
                results_df['baseline_b_pick'] = np.nan
                results_df['baseline_b_correct'] = np.nan
            
            # Calculate accuracy metrics
            metrics = calculate_accuracy_metrics(results_df)
            
            # Display summary metrics
            st.subheader("📊 Event Performance Summary")
            
            # Add explanation of what each metric means
            st.markdown("""
            **How to read these metrics:** 
            - **Model Accuracy**: How often our ML model correctly predicted the teammate winner
            - **Baseline A**: How often the driver with better head-to-head record won (simple heuristic)
            - **Baseline B**: How often the higher-paid driver won (if salary data available)
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🤖 ML Model Accuracy", f"{metrics['model_accuracy']:.1%}")
            with col2:
                st.metric("📈 H2H Record Accuracy", f"{metrics['baseline_a_accuracy']:.1%}")
            with col3:
                if salary_available:
                    st.metric("💰 Salary-Based Accuracy", f"{metrics['baseline_b_accuracy']:.1%}")
                else:
                    st.metric("💰 Salary Baseline", "N/A")
            
            # Display predictions table
            st.subheader("🏁 Team-by-Team Predictions")
            
            # Add explanation of the table
            st.markdown("""
            **Table explanation:** Each row shows a team's driver pairing. The model predicts which driver will qualify ahead 
            and provides a confidence score. We compare this against what actually happened and against simple baseline rules.
            """)
            
            # Create a more readable table
            display_df = create_display_table(results_df, salary_available)
            
            # Apply better styling with custom CSS
            st.markdown("""
            <style>
            .stDataFrame {
                font-size: 14px;
            }
            .stDataFrame th {
                background-color: #f0f2f6;
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Color code the rows based on correctness with better colors
            def color_correctness(val):
                if pd.isna(val):
                    return ''
                elif val:  # Correct prediction
                    return 'background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;'
                else:  # Incorrect prediction
                    return 'background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;'
            
            styled_df = display_df.style.applymap(
                color_correctness, 
                subset=['✅ Model Correct', '✅ H2H Correct', '✅ Salary Correct']
            ).format({
                '🎯 Model Confidence': '{:.1%}'
            })
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Add prediction explanations
            st.markdown("---")
            st.subheader("🔍 Prediction Explanations")
            
            # Show explanations for each team
            for _, row in display_df.iterrows():
                team_name = row['🏎️ Team']
                model_pick = row['🤖 Model Pick']
                confidence = row['🎯 Model Confidence']
                
                # Generate explanation
                explanation = generate_prediction_explanation(model_pick, confidence, team_name)
                
                with st.expander(f"📋 {team_name} - {model_pick} prediction"):
                    st.markdown(explanation)
                    
                    # Add some context about what the model considers
                    st.info("""
                    **What the model analyzed:**
                    - Recent qualifying performance and consistency
                    - Head-to-head record against current teammate
                    - Track-specific experience and historical performance
                    - Practice session pace relative to field
                    - Team performance trends and car development
                    - Weather conditions and track characteristics
                    """)
            
            # Legend and explanations
            st.markdown("---")
            st.subheader("📚 How to Read the Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🎯 Model Predictions:**
                - **Model Pick**: Which driver our ML system predicts will qualify ahead
                - **Model Confidence**: Calibrated probability (higher = more confident)
                - **Model Correct**: ✅ Green = correct prediction, ❌ Red = incorrect
                """)
            
            with col2:
                st.markdown("""
                **📊 Baseline Comparisons:**
                - **H2H Record**: Driver with better head-to-head record entering the event
                - **Salary-Based**: Higher-paid driver (if salary data available)
                - **Baseline Correct**: Same color coding as model predictions
                """)
            
            st.info("""
            **💡 Key Insight**: The ML model considers hundreds of features including driver form, team performance, 
            track characteristics, practice sessions, and weather. Simple baselines only look at one factor each.
            """)
            
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
    
    # Baseline B accuracy (if available)
    if 'baseline_b_correct' in results_df.columns:
        baseline_b_correct = results_df['baseline_b_correct'].dropna()
        metrics['baseline_b_accuracy'] = baseline_b_correct.mean() if len(baseline_b_correct) > 0 else 0
    else:
        metrics['baseline_b_accuracy'] = 0
    
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
            '🏎️ Team': constructor_id.replace('_', ' ').title(),
            'Driver A': driver1['driver_name'],
            'Driver B': driver2['driver_name'],
            '🤖 Model Pick': winner['driver_name'],
            '🎯 Model Confidence': winner['model_confidence'],
            '🏁 Actual Winner': winner['driver_name'] if winner['actual_beats_teammate'] == 1 else loser['driver_name'],
            '📈 H2H Pick': winner['driver_name'] if winner['baseline_a_pick'] == 1 else (loser['driver_name'] if loser['baseline_a_pick'] == 1 else 'No Prior'),
            '💰 Salary Pick': winner['driver_name'] if winner['baseline_b_pick'] == 1 else (loser['driver_name'] if loser['baseline_b_pick'] == 1 else 'Equal Salary') if salary_available else 'N/A',
            '✅ Model Correct': winner['actual_beats_teammate'] == 1,
            '✅ H2H Correct': winner.get('baseline_a_correct', None) if pd.notna(winner.get('baseline_a_correct', None)) else None,
            '✅ Salary Correct': winner.get('baseline_b_correct', None) if pd.notna(winner.get('baseline_b_correct', None)) else None
        }
        
        display_rows.append(row)
    
    return pd.DataFrame(display_rows)

def generate_prediction_explanation(driver_name, confidence, constructor_name):
    """Generate a human-readable explanation of the prediction."""
    confidence_level = "very high" if confidence >= 0.8 else "high" if confidence >= 0.6 else "moderate" if confidence >= 0.5 else "low"
    
    explanations = [
        f"**{driver_name}** is predicted to qualify ahead of their teammate at **{constructor_name}** with **{confidence_level} confidence** ({confidence:.1%}).",
        f"The model analyzed historical performance, current form, track characteristics, and practice session data to make this prediction.",
        f"Key factors likely include: driver's recent qualifying performance, head-to-head record against teammate, and track-specific experience."
    ]
    
    if confidence >= 0.8:
        explanations.append("🎯 **High confidence**: The model sees strong indicators supporting this prediction.")
    elif confidence >= 0.6:
        explanations.append("📊 **Good confidence**: Multiple factors align to support this prediction.")
    else:
        explanations.append("🤔 **Lower confidence**: The prediction is less certain, suggesting a close battle between teammates.")
    
    return " ".join(explanations)

if __name__ == "__main__":
    main()
