#!/usr/bin/env python3
"""
F1 Teammate Qualifying Predictor - Streamlit App
Predicts which driver will beat their teammate in qualifying using ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
from datetime import datetime
import base64
from PIL import Image
import io

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.predict import TeammatePredictor
    from src.assets import circularize, resolve_image, country_code_for_event
    from src.baselines import compute_baseline_a, calculate_accuracy_metrics, create_display_table
    from webapi.utils.circuits import load_event_codes
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please ensure all dependencies are installed and the src directory is accessible.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="F1 Teammate Qualifying Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for F1 styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #e10600 0%, #ff6b6b 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid #e10600;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .stDataFrame {
        font-size: 14px;
    }
    .stDataFrame th {
        background-color: #f0f2f6;
        font-weight: bold;
    }
    .circle-img { 
        border-radius: 50%; 
        width: 48px; 
        height: 48px; 
        object-fit: cover; 
        border: 2px solid #e0e0e0;
    }
    .circle-flag { 
        border-radius: 50%; 
        width: 28px; 
        height: 28px; 
        object-fit: cover; 
        border: 1px solid #ddd;
    }
    .circle-team { 
        border-radius: 50%; 
        width: 40px; 
        height: 40px; 
        object-fit: contain; 
        background: #fff; 
        border: 1px solid #eee;
    }
    .rowok { 
        background: rgba(200, 0, 0, 0.06); 
    }
    .rowbad { 
        background: rgba(200, 0, 0, 0.06); 
    }
</style>
""", unsafe_allow_html=True)

def load_available_events_from_data():
    """Load available events directly from the labeled data."""
    try:
        labeled_path = Path("data/interim/qual_labeled.parquet")
        if not labeled_path.exists():
            return {}, {}
        
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
                    'round': event_key.split('_R')[1] if '_R' in event_key else '',
                    'track_name': f"Round {event_key.split('_R')[1] if '_R' in event_key else 'Unknown'}",
                    'location': 'Location TBD',
                    'event_date': 'Date TBD',
                    'circuit_code': '',
                    'is_sprint_weekend': False
                }
        
        return season_events, event_details
        
    except Exception as e:
        st.error(f"Error loading events from data: {e}")
        return {}, {}

def format_event_name(event_key):
    """Format event key to readable name."""
    if '_R' in event_key:
        season, round_num = event_key.split('_R')
        return f"{season} Round {round_num}"
    return event_key

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏎️ F1 Teammate Qualifying Predictor</h1>
        <p>Machine Learning-powered predictions for Formula 1 qualifying head-to-head battles</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Load available events from actual data
    season_events, event_details = load_available_events_from_data()
    
    if not season_events:
        st.error("🚨 **No events found in data!**")
        st.info("Please ensure the pipeline is properly set up and run:")
        st.code("make build && make train")
        st.stop()
    
    # Season and event selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        season = st.selectbox("Season", options=list(season_events.keys()), index=0)
    with col2:
        if season in season_events:
            events = season_events[season]
            selected_event = st.selectbox("Event", options=events, index=len(events)-1, format_func=format_event_name)
        else:
            selected_event = None
            st.warning("No events found for selected season")
    
    # High contrast mode
    high_contrast = st.sidebar.checkbox("🎨 High-contrast mode (F1-ish)", value=False)
    
    if not selected_event:
        st.info("👈 Please select a season and event from the sidebar to view predictions.")
        return
    
    # Get event details
    details = event_details.get(selected_event, {})
    
    # Display event info
    st.header(f"🏁 Predictions for {selected_event}")
    
    # Event details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Season", details.get('season', 'Unknown'))
    with col2:
        st.metric("Round", details.get('round', 'Unknown'))
    with col3:
        st.metric("Track", details.get('track_name', 'Unknown Track'))
    
    # System status
    st.sidebar.subheader("📊 System Status")
    st.sidebar.info(f"**Season**: {details.get('season', 'Unknown')}")
    st.sidebar.info(f"**Event**: {selected_event}")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🏁 Qualifying H2H", "🏆 Race Winner"])
    
    with tab1:
        st.subheader("🏁 Qualifying Head-to-Head Predictions")
        
        try:
            # Load predictions
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
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🤖 ML Model Accuracy", f"{metrics['model_accuracy']:.1%}")
            with col2:
                st.metric("📈 H2H Record Accuracy", f"{metrics['baseline_a_accuracy']:.1%}")
            
            # Display track map (simplified for now)
            st.subheader("🗺️ Circuit Map")
            st.info(f"Track map for {selected_event} - SVG display coming soon!")
            
            # Display predictions table
            st.subheader("🏁 Team-by-Team Predictions")
            
            # Create display table
            display_df = create_display_table(results_df)
            
            # Quick summary table
            st.markdown("**📊 Quick Summary**")
            summary_cols = st.columns(4)
            
            with summary_cols[0]:
                st.metric("Teams", len(display_df))
            with summary_cols[1]:
                correct_predictions = display_df['✅ Model Correct'].sum()
                st.metric("Model Correct", f"{correct_predictions}/{len(display_df)}")
            with summary_cols[2]:
                # Extract numeric confidence values
                confidence_values = []
                for conf_str in display_df['🎯 Model Confidence']:
                    try:
                        conf_val = float(conf_str.replace('%', '')) / 100
                        confidence_values.append(conf_val)
                    except:
                        continue
                
                avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            with summary_cols[3]:
                h2h_correct = display_df['✅ H2H Correct'].dropna().sum()
                h2h_total = display_df['✅ H2H Correct'].notna().sum()
                st.metric("H2H Correct", f"{h2h_correct}/{h2h_total}" if h2h_total > 0 else "N/A")
            
            # Display the table
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Download option
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"f1_predictions_{selected_event}.csv",
                mime="text/csv"
            )
            
            # Explanations
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
    
    with tab2:
        st.subheader("🏆 Race Winner Predictions")
        
        st.info("""
        **Race Winner Predictions**: This section predicts which driver will win the entire race, 
        not just beat their teammate. The model considers grid position, qualifying performance, 
        recent form, and track history.
        """)
        
        # Check if race winner model exists
        race_model_path = Path("models/simple_race_winner.joblib")
        if not race_model_path.exists():
            st.warning("🏆 **Race Winner Model Not Found**")
            st.info("""
            To enable race winner predictions, you need to:
            1. Collect race data: `python src/race_data.py --season 2024 --events "Bahrain Grand Prix" "Australian Grand Prix"`
            2. Train the model: `python simple_race_winner.py train --data data/interim/race_data.parquet`
            """)
            st.code("""
# Example commands:
python src/race_data.py --season 2024 --events "Bahrain Grand Prix" "Australian Grand Prix" --output data/interim/race_data.parquet
python simple_race_winner.py train --data data/interim/race_data.parquet --model models/simple_race_winner.joblib
            """)
        else:
            try:
                # Load race data
                race_data_path = Path("data/interim/race_data.parquet")
                if not race_data_path.exists():
                    st.warning("🏆 **Race Data Not Found**")
                    st.info("Please collect race data first using the commands above.")
                else:
                    # Load race data and model
                    import joblib
                    
                    race_df = pd.read_parquet(race_data_path)
                    race_model = joblib.load(race_model_path)
                    
                    # Filter for current event
                    event_race_data = race_df[(race_df['season'] == details['season']) & 
                                            (race_df['event_key'] == selected_event)]
                    
                    if event_race_data.empty:
                        st.info(f"🏆 No race data available for {selected_event}")
                    else:
                        # Make predictions using our simple model
                        feature_columns = ['grid_position', 'best_qual_pos', 'final_position']
                        X = event_race_data[feature_columns].fillna(event_race_data[feature_columns].median()).values
                        
                        if len(X) > 0:
                            probabilities = race_model.predict_proba(X)[:, 1]
                            
                            # Create results dataframe
                            race_predictions = event_race_data[['driver_code', 'team', 'grid_position', 'best_qual_pos', 'final_position']].copy()
                            race_predictions['probability'] = probabilities
                            
                            # Sort by probability (highest first)
                            race_predictions = race_predictions.sort_values('probability', ascending=False)
                            
                            # Normalize probabilities to sum to 1
                            total_prob = race_predictions['probability'].sum()
                            if total_prob > 0:
                                race_predictions['probability'] = race_predictions['probability'] / total_prob
                            
                            st.success(f"🏆 Race Winner Predictions for {selected_event}")
                            
                            # Display top 5 predictions
                            st.subheader("🥇 Top 5 Race Winners")
                            
                            for i, (_, row) in enumerate(race_predictions.head(5).iterrows()):
                                col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                                
                                with col1:
                                    if i == 0:
                                        st.markdown("🥇")
                                    elif i == 1:
                                        st.markdown("🥈")
                                    elif i == 2:
                                        st.markdown("🥉")
                                    else:
                                        st.markdown(f"**{i+1}**")
                                
                                with col2:
                                    st.markdown(f"**{row['driver_code']}**")
                                    st.markdown(f"*{row['team']}*")
                                
                                with col3:
                                    st.markdown(f"**{row['probability']:.1%}**")
                                    st.markdown(f"*Grid: {int(row['grid_position']) if pd.notna(row['grid_position']) else 'N/A'}*")
                                
                                with col4:
                                    if pd.notna(row['best_qual_pos']):
                                        st.markdown(f"Quali: **{int(row['best_qual_pos'])}**")
                                    else:
                                        st.markdown("Quali: **N/A**")
                                
                                if i < 4:  # Don't add separator after last row
                                    st.markdown("---")
                            
                            # Show full table
                            st.subheader("📊 Complete Race Predictions")
                            st.dataframe(
                                race_predictions[['driver_code', 'team', 'probability', 'grid_position', 'best_qual_pos']],
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Download race predictions
                            csv = race_predictions.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Race Predictions as CSV",
                                data=csv,
                                file_name=f"race_predictions_{selected_event}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No race predictions generated for this event.")
                            
            except Exception as e:
                st.error(f"Error loading race winner predictions: {e}")
                st.info("Please ensure the race winner model and data are properly set up.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        <p>🏎️ F1 Teammate Qualifying Predictor | Built with Streamlit & Machine Learning</p>
        <p>Data source: FastF1 | Model: XGBoost with CalibratedClassifierCV</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
