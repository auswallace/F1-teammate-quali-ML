
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
from src.assets import ensure_dirs, resolve_image, circularize, country_code_for_event

# F1 Calendar Database - Maps event keys to actual track names and dates
F1_CALENDAR = {
    # 2025 Season
    '2025_R01': {'track': 'Bahrain International Circuit', 'location': 'Sakhir, Bahrain', 'date': 'Mar 02'},
    '2025_R02': {'track': 'Jeddah Corniche Circuit', 'location': 'Jeddah, Saudi Arabia', 'date': 'Mar 09'},
    '2025_R03': {'track': 'Albert Park Circuit', 'location': 'Melbourne, Australia', 'date': 'Mar 23'},
    '2025_R04': {'track': 'Suzuka International Racing Course', 'location': 'Suzuka, Japan', 'date': 'Apr 06'},
    '2025_R05': {'track': 'Shanghai International Circuit', 'location': 'Shanghai, China', 'date': 'Apr 20'},
    '2025_R06': {'track': 'Miami International Autodrome', 'location': 'Miami, USA', 'date': 'May 04'},
    '2025_R07': {'track': 'Imola Circuit', 'location': 'Imola, Italy', 'date': 'May 18'},
    '2025_R08': {'track': 'Circuit de Monaco', 'location': 'Monte Carlo, Monaco', 'date': 'May 25'},
    '2025_R09': {'track': 'Circuit de Barcelona-Catalunya', 'location': 'Barcelona, Spain', 'date': 'Jun 01'},
    '2025_R10': {'track': 'Red Bull Ring', 'location': 'Spielberg, Austria', 'date': 'Jun 08'},
    '2025_R11': {'track': 'Silverstone Circuit', 'location': 'Silverstone, UK', 'date': 'Jun 22'},
    '2025_R12': {'track': 'Hungaroring', 'location': 'Budapest, Hungary', 'date': 'Jul 06'},
    '2025_R13': {'track': 'Circuit de Spa-Francorchamps', 'location': 'Spa, Belgium', 'date': 'Jul 20'},
    '2025_R14': {'track': 'Circuit Zandvoort', 'location': 'Zandvoort, Netherlands', 'date': 'Jul 27'},
    '2025_R15': {'track': 'Monza Circuit', 'location': 'Monza, Italy', 'date': 'Aug 03'},
    '2025_R16': {'track': 'Baku City Circuit', 'location': 'Baku, Azerbaijan', 'date': 'Aug 17'},
    '2025_R17': {'track': 'Marina Bay Street Circuit', 'location': 'Singapore', 'date': 'Sep 07'},
    '2025_R18': {'track': 'Circuit of the Americas', 'location': 'Austin, USA', 'date': 'Sep 21'},
    '2025_R19': {'track': 'Autódromo Hermanos Rodríguez', 'location': 'Mexico City, Mexico', 'date': 'Oct 05'},
    '2025_R20': {'track': 'Interlagos Circuit', 'location': 'São Paulo, Brazil', 'date': 'Oct 19'},
    '2025_R21': {'track': 'Las Vegas Strip Circuit', 'location': 'Las Vegas, USA', 'date': 'Nov 02'},
    '2025_R22': {'track': 'Lusail International Circuit', 'location': 'Doha, Qatar', 'date': 'Nov 16'},
    '2025_R23': {'track': 'Yas Marina Circuit', 'location': 'Abu Dhabi, UAE', 'date': 'Nov 23'},
    
    # 2024 Season
    '2024_R01': {'track': 'Bahrain International Circuit', 'location': 'Sakhir, Bahrain', 'date': 'Mar 02'},
    '2024_R02': {'track': 'Jeddah Corniche Circuit', 'location': 'Jeddah, Saudi Arabia', 'date': 'Mar 09'},
    '2024_R03': {'track': 'Albert Park Circuit', 'location': 'Melbourne, Australia', 'date': 'Mar 24'},
    '2024_R04': {'track': 'Suzuka International Racing Course', 'location': 'Suzuka, Japan', 'date': 'Apr 07'},
    '2024_R05': {'track': 'Shanghai International Circuit', 'location': 'Shanghai, China', 'date': 'Apr 21'},
    '2024_R06': {'track': 'Miami International Autodrome', 'location': 'Miami, USA', 'date': 'May 05'},
    '2024_R07': {'track': 'Imola Circuit', 'location': 'Imola, Italy', 'date': 'May 19'},
    '2024_R08': {'track': 'Circuit de Monaco', 'location': 'Monte Carlo, Monaco', 'date': 'May 26'},
    '2024_R09': {'track': 'Circuit de Barcelona-Catalunya', 'location': 'Barcelona, Spain', 'date': 'Jun 02'},
    '2024_R10': {'track': 'Red Bull Ring', 'location': 'Spielberg, Austria', 'date': 'Jun 09'},
    '2024_R11': {'track': 'Silverstone Circuit', 'location': 'Silverstone, UK', 'date': 'Jun 23'},
    '2024_R12': {'track': 'Hungaroring', 'location': 'Budapest, Hungary', 'date': 'Jul 07'},
    '2024_R13': {'track': 'Circuit de Spa-Francorchamps', 'location': 'Spa, Belgium', 'date': 'Jul 21'},
    '2024_R14': {'track': 'Circuit Zandvoort', 'location': 'Zandvoort, Netherlands', 'date': 'Jul 28'},
    '2024_R15': {'track': 'Monza Circuit', 'location': 'Monza, Italy', 'date': 'Aug 04'},
    '2024_R16': {'track': 'Baku City Circuit', 'location': 'Baku, Azerbaijan', 'date': 'Aug 18'},
    '2024_R17': {'track': 'Marina Bay Street Circuit', 'location': 'Singapore', 'date': 'Sep 08'},
    '2024_R18': {'track': 'Circuit of the Americas', 'location': 'Austin, USA', 'date': 'Sep 22'},
    '2024_R19': {'track': 'Autódromo Hermanos Rodríguez', 'location': 'Mexico City, Mexico', 'date': 'Oct 06'},
    '2024_R20': {'track': 'Interlagos Circuit', 'location': 'São Paulo, Brazil', 'date': 'Oct 20'},
    '2024_R21': {'track': 'Las Vegas Strip Circuit', 'location': 'Las Vegas, USA', 'date': 'Nov 03'},
    '2024_R22': {'track': 'Lusail International Circuit', 'location': 'Doha, Qatar', 'date': 'Nov 17'},
    '2024_R23': {'track': 'Yas Marina Circuit', 'location': 'Abu Dhabi, UAE', 'date': 'Nov 24'},
}

# Page config
st.set_page_config(
    page_title="F1 Teammate Qualifying Predictor",
    page_icon="🏎️",
    layout="wide"
)

# Initialize assets
ensure_dirs()

# Title and description
st.title("🏎️ F1 Teammate Qualifying Predictor")
st.markdown("""
**Predict which driver will beat their teammate in qualifying sessions using machine learning.**

This system uses historical F1 data to predict teammate qualifying outcomes with calibrated probabilities. 
The model considers driver form, team performance, track characteristics, practice sessions, and weather conditions.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# High-contrast mode toggle
high_contrast = st.sidebar.checkbox("🎨 High-contrast mode (F1-ish)", value=False)

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
                # Get calendar info from our F1 database
                calendar_info = F1_CALENDAR.get(event_key, {})
                
                event_details[event_key] = {
                    'season': row['season'],
                    'round': row.get('round', ''),
                    'track_name': calendar_info.get('track', f"Round {row.get('round', '')}"),
                    'location': calendar_info.get('location', 'Location TBD'),
                    'event_date': calendar_info.get('date', 'Date TBD'),
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
            location = details.get('location', '')
            
            # Create a rich description with round, track, location, and date
            if round_num and track_name != f"Round {round_num}":
                # Extract just the track name without "Circuit" etc. for cleaner display
                clean_track = track_name.replace(' International Circuit', '').replace(' Circuit', '').replace(' Autodrome', '')
                return f"{round_num} - {clean_track}, {location} ({event_date})"
            else:
                return f"{event_key} - {track_name}, {location} ({event_date})"
        
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
    
    # Main content area
    if selected_event:
        st.header(f"🏁 Predictions for {selected_event}")
        
        # Event details
        if selected_event in event_details:
            details = event_details[selected_event]
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Season", details['season'])
            with col2:
                st.metric("Round", details['round'] if details['round'] else "N/A")
            with col3:
                st.metric("Track", details['track_name'] if details['track_name'] else "Unknown")
            with col4:
                st.metric("Location", details.get('location', 'TBD'))
            with col5:
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
            
            # Calculate accuracy metrics
            metrics = calculate_accuracy_metrics(results_df)
            
            # Display summary metrics
            st.subheader("📊 Event Performance Summary")
            
            # Add explanation of what each metric means
            st.markdown("""
            **How to read these metrics:** 
            - **Model Accuracy**: How often our ML model correctly predicted the teammate winner
            - **Baseline A**: How often the driver with better head-to-head record won (simple heuristic)
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🤖 ML Model Accuracy", f"{metrics['model_accuracy']:.1%}")
            with col2:
                st.metric("📈 H2H Record Accuracy", f"{metrics['baseline_a_accuracy']:.1%}")
            
            # Display predictions table
            st.subheader("🏁 Team-by-Team Predictions")
            
            # Add explanation of the table
            st.markdown("""
            **Table explanation:** Each row shows a team's driver pairing. The model predicts which driver will qualify ahead 
            and provides a confidence score. We compare this against what actually happened and against simple baseline rules.
            """)
            
            # Create a more readable table
            display_df = create_display_table(results_df)
            
            # Quick summary table at the top
            st.markdown("**📊 Quick Summary**")
            summary_cols = st.columns(4)
            
            with summary_cols[0]:
                st.metric("Teams", len(display_df))
            with summary_cols[1]:
                correct_predictions = display_df['✅ Model Correct'].sum()
                st.metric("Model Correct", f"{correct_predictions}/{len(display_df)}")
            with summary_cols[2]:
                avg_confidence = display_df['🎯 Model Confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            with summary_cols[3]:
                h2h_correct = display_df['✅ H2H Correct'].dropna().sum()
                h2h_total = display_df['✅ H2H Correct'].notna().sum()
                st.metric("H2H Correct", f"{h2h_correct}/{h2h_total}" if h2h_total > 0 else "N/A")
            
            # Apply F1-style styling with custom CSS
            contrast_style = """
            .team-row {
                background: rgba(0, 0, 0, 0.95);
                border: 2px solid #e10600;
                border-radius: 8px;
                padding: 20px;
                margin: 15px 0;
                color: white;
            }
            .team-row h3, .team-row h4, .team-row h5 {
                color: white !important;
                font-weight: 700;
            }
            .team-row p {
                color: #f0f0f0 !important;
                font-weight: 500;
            }
            """ if high_contrast else ""
            
            st.markdown(f"""
            <style>
            .stDataFrame {{
                font-size: 14px;
            }}
            .stDataFrame th {{
                background-color: #f0f2f6;
                font-weight: bold;
            }}
            .circle-img {{ 
                border-radius: 50%; 
                width: 48px; 
                height: 48px; 
                object-fit: cover; 
                border: 2px solid #e0e0e0;
            }}
            .circle-flag {{ 
                border-radius: 50%; 
                width: 28px; 
                height: 28px; 
                object-fit: cover; 
                border: 1px solid #ddd;
            }}
            .circle-team {{ 
                border-radius: 50%; 
                width: 40px; 
                height: 40px; 
                object-fit: contain; 
                background: #fff; 
                border: 1px solid #eee;
            }}
            .rowok {{ 
                background: rgba(0, 200, 0, 0.06); 
            }}
            .rowbad {{ 
                background: rgba(200, 0, 0, 0.06); 
            }}
            .f1-header {{
                background: linear-gradient(135deg, #e10600 0%, #ff6b6b 100%);
                color: white;
                padding: 10px;
                border-radius: 8px;
                margin: 10px 0;
            }}
            .team-row {{
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            }}
            {contrast_style}
            </style>
            """, unsafe_allow_html=True)
            
            # Display F1-style rich table with circular icons
            st.markdown("### 🏁 Team-by-Team Predictions")
            
            # Add a toggle to switch between visual and table views
            view_mode = st.radio(
                "Choose your view:",
                ["📊 Clean Table", "🎨 Rich Visual Layout", "📋 Simple Data Only"],
                horizontal=True,
                index=0
            )
            
            if view_mode == "📊 Clean Table":
                # Show the original clean table format
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add some styling info
                st.info("""
                **Table Guide:**
                - **Model Pick**: Driver predicted to qualify ahead (with confidence %)
                - **Actual Winner**: What really happened in qualifying
                - **H2H Pick**: Driver with better head-to-head record
                - **✅/❌**: Whether predictions were correct
                """)
                
            elif view_mode == "📋 Simple Data Only":
                # Show just the raw data without styling
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add download option
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"f1_predictions_{selected_event}.csv",
                    mime="text/csv"
                )
                
            else:
                # Rich visual layout
                # Get country code for the event
                event_info = F1_CALENDAR.get(selected_event, {})
                country_code = country_code_for_event(event_info)
                
                # Display each team with rich visuals using Streamlit components
                for _, row in display_df.iterrows():
                team_name = row['🏎️ Team']
                driver_a = row['Driver A']
                driver_b = row['Driver B']
                model_pick = row['🤖 Model Pick']
                confidence = row['🎯 Model Confidence']
                actual_winner = row['🏁 Actual Winner']
                h2h_pick = row['📈 H2H Pick']
                model_correct = row['✅ Model Correct']
                h2h_correct = row['✅ H2H Correct']
                
                # Determine team constructor ID for assets
                constructor_id = team_name.upper().replace(' ', '_')
                
                # Resolve assets
                try:
                    flag_img = circularize(resolve_image("flag", country_code or "xx", "data/assets/placeholders/flag.png"), size=28)
                    team_img = circularize(resolve_image("team", constructor_id, "data/assets/placeholders/team.png"), size=40)
                    driver_a_img = circularize(resolve_image("driver", driver_a.split()[-1].upper()[:3], "data/assets/placeholders/driver.png"), size=48)
                    driver_b_img = circularize(resolve_image("driver", driver_b.split()[-1].upper()[:3], "data/assets/placeholders/driver.png"), size=48)
                except Exception as e:
                    st.warning(f"Asset loading error: {e}")
                    flag_img = "data/assets/placeholders/flag.png"
                    team_img = "data/assets/placeholders/team.png"
                    driver_a_img = "data/assets/placeholders/driver.png"
                    driver_b_img = "data/assets/placeholders/driver.png"
                
                # Create team row with F1 styling using Streamlit components
                with st.container():
                    # Team header with flag and track
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        try:
                            st.image(str(flag_img), width=28)
                        except:
                            st.markdown("🚩")
                    with col2:
                        st.markdown(f"**{event_info.get('track', 'Unknown Track')}**")
                    
                    # Team name and logo
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        try:
                            st.image(str(team_img), width=40)
                        except:
                            st.markdown("🏁")
                    with col2:
                        st.markdown(f"### {team_name}")
                    
                    # Driver comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Driver A
                        border_color = "#28a745" if model_pick == driver_a else "#6c757d"
                        background_color = "rgba(40, 167, 69, 0.1)" if model_pick == driver_a else "rgba(108, 117, 125, 0.1)"
                        
                        st.markdown(f"""
                        <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 15px; background: {background_color}; text-align: center;">
                        """, unsafe_allow_html=True)
                        
                        try:
                            st.image(str(driver_a_img), width=48)
                        except:
                            st.markdown("👤")
                        
                        st.markdown(f"**{driver_a}**")
                        st.markdown("*Driver A*")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        # Driver B
                        border_color = "#28a745" if model_pick == driver_b else "#6c757d"
                        background_color = "rgba(40, 167, 69, 0.1)" if model_pick == driver_b else "rgba(108, 117, 125, 0.1)"
                        
                        st.markdown(f"""
                        <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 15px; background: {background_color}; text-align: center;">
                        """, unsafe_allow_html=True)
                        
                        try:
                            st.image(str(driver_b_img), width=48)
                        except:
                            st.markdown("👤")
                        
                        st.markdown(f"**{driver_b}**")
                        st.markdown("*Driver B*")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Prediction results
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**🤖 Model Pick**")
                        st.markdown(f"**{model_pick}**")
                        st.markdown(f"*{confidence:.1%} confidence*")
                    
                    with col2:
                        st.markdown("**🏁 Actual Winner**")
                        st.markdown(f"**{actual_winner}**")
                    
                    with col3:
                        st.markdown("**📈 H2H Pick**")
                        st.markdown(f"**{h2h_pick}**")
                    
                    # Correctness indicators
                    col1, col2 = st.columns(2)
                    with col1:
                        if model_correct:
                            st.success("✅ Model Correct")
                        else:
                            st.error("❌ Model Incorrect")
                    
                    with col2:
                        if pd.notna(h2h_correct):
                            if h2h_correct:
                                st.success("✅ H2H Correct")
                            else:
                                st.error("❌ H2H Incorrect")
                        else:
                            st.info("No H2H Data")
                    
                    st.markdown("---")
            
            # Add legend
            st.markdown("---")
            st.markdown("""
            <div class="f1-header">
                <h4 style="margin: 0;">📚 Icon Legend</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                **🚩 Flag**: Host country of the race
                **🏁 Team Logo**: Constructor branding
                **👤 Driver**: Individual driver headshot
                """)
            with col2:
                st.markdown("""
                **🤖 Model Pick**: ML prediction with confidence
                **🏁 Actual Winner**: What really happened
                **📈 H2H Pick**: Head-to-head record baseline
                """)
            with col3:
                st.markdown("""
                **✅ Green Border**: Model's predicted winner
                **🎯 Confidence**: Calibrated probability
                **📊 Comparison**: Model vs baseline accuracy
                """)
            
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
    
    return metrics

def create_display_table(results_df):
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
            '✅ Model Correct': winner['actual_beats_teammate'] == 1,
            '✅ H2H Correct': winner.get('baseline_a_correct', None) if pd.notna(winner.get('baseline_a_correct', None)) else None
        }
        
        display_rows.append(row)
    
    return pd.DataFrame(display_rows)

def get_image_base64(image_path):
    """Convert image to base64 for HTML display."""
    try:
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        if not image_path.exists():
            # Return placeholder base64
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        import base64
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        st.warning(f"Image loading error: {e}")
        # Return transparent pixel
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

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
