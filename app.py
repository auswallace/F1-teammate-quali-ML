#!/usr/bin/env python3
"""
F1 Teammate Qualifying Predictor - Streamlit App
Predicts which driver will beat their teammate in qualifying using ML models.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

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
    """Load available events from event codes configuration."""
    try:
        # Try to load from event codes configuration first
        event_codes = load_event_codes()
        
        if not event_codes or 'seasons' not in event_codes:
            return {}, {}
        
        # Build season -> events mapping
        season_events = {}
        event_details = {}
        
        for season_data in event_codes['seasons']:
            season = season_data['season']
            events = []
            
            for event in season_data['events']:
                round_num = event['round']
                event_key = f"{season}_R{round_num:02d}"
                events.append(event_key)
                
                # Build event details
                event_details[event_key] = {
                    'season': season,
                    'round': round_num,
                    'track_name': event['name'].replace(' Grand Prix', ' GP'),
                    'location': event['location'],
                    'event_date': event['date'],
                    'circuit_code': event['code'],
                    'is_sprint_weekend': event.get('format') == 'sprint'
                }
            
            season_events[season] = sorted(events)
        
        return season_events, event_details
        
    except Exception as e:
        st.error(f"Error loading events from configuration: {e}")
        return {}, {}

def format_event_name(event_key):
    """Format event key to readable name."""
    if '_R' in event_key:
        season, round_num = event_key.split('_R')
        # Get event details to show track name
        try:
            event_codes = load_event_codes()
            for season_data in event_codes.get('seasons', []):
                if season_data['season'] == int(season):
                    for event in season_data['events']:
                        if event['round'] == int(round_num):
                            track_name = event['name'].replace(' Grand Prix', ' GP')
                            return f"{season} {track_name}"
        except:
            pass
        # Fallback to round format
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
            # Default to first available event instead of last
            selected_event = st.selectbox("Event", options=events, index=0, format_func=format_event_name)
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Season", details.get('season', 'Unknown'))
    with col2:
        st.metric("Round", details.get('round', 'Unknown'))
    with col3:
        st.metric("Track", details.get('track_name', 'Unknown Track'))
    with col4:
        # Format date as MM DD
        event_date = details.get('event_date', '')
        if event_date:
            try:
                date_obj = datetime.strptime(event_date, '%Y-%m-%d %H:%M:%S')
                formatted_date = date_obj.strftime('%b %d').upper()
                st.metric("Date", formatted_date)
            except:
                st.metric("Date", "TBD")
        else:
            st.metric("Date", "TBD")
    
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
            
            # Try to load actual results for evaluation
            results_df = predictions_df.copy()
            try:
                labeled_path = Path("data/interim/qual_labeled.parquet")
                if labeled_path.exists():
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
                else:
                    st.info("📝 **Note:** Actual results not available for evaluation. Showing predictions only.")
                    results_df['model_correct'] = None
            except Exception as e:
                st.warning(f"⚠️ Could not load actual results: {e}")
                results_df['model_correct'] = None
            
            # Compute Baseline A: Prior head-to-head leader
            results_df = compute_baseline_a(results_df, selected_event, details['season'])
            
            # Calculate accuracy metrics (only if we have actual results)
            if results_df['model_correct'].notna().any():
                metrics = calculate_accuracy_metrics(results_df)
            else:
                metrics = {'model_accuracy': 0.0, 'baseline_a_accuracy': 0.0}
            
            # Display summary metrics
            st.subheader("📊 Event Performance Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                if metrics['model_accuracy'] > 0:
                    st.metric("🤖 ML Model Accuracy", f"{metrics['model_accuracy']:.1%}")
                else:
                    st.metric("🤖 ML Model Accuracy", "N/A (No actual results)")
            with col2:
                if metrics['baseline_a_accuracy'] > 0:
                    st.metric("📈 H2H Record Accuracy", f"{metrics['baseline_a_accuracy']:.1%}")
                else:
                    st.metric("📈 H2H Record Accuracy", "N/A (No actual results)")
            
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
                if '✅ Model Correct' in display_df.columns:
                    correct_predictions = display_df['✅ Model Correct'].sum()
                    st.metric("Model Correct", f"{correct_predictions}/{len(display_df)}")
                else:
                    st.metric("Model Correct", "N/A")
            with summary_cols[2]:
                # Extract numeric confidence values
                confidence_values = []
                if '🎯 Model Confidence' in display_df.columns:
                    for conf_str in display_df['🎯 Model Confidence']:
                        try:
                            conf_val = float(conf_str.replace('%', '')) / 100
                            confidence_values.append(conf_val)
                        except:
                            continue
                
                avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            with summary_cols[3]:
                if '✅ H2H Correct' in display_df.columns:
                    h2h_correct = display_df['✅ H2H Correct'].dropna().sum()
                    h2h_total = display_df['✅ H2H Correct'].notna().sum()
                    st.metric("H2H Correct", f"{h2h_correct}/{h2h_total}" if h2h_total > 0 else "N/A")
                else:
                    st.metric("H2H Correct", "N/A")
            
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
            st.info("""
            **Troubleshooting:**
            1. Make sure the pipeline is properly set up and run: `make build && make train`
            2. Check if the processed data exists: `data/processed/teammate_qual.parquet`
            3. Verify the event key format matches the data
            """)
    
    with tab2:
        st.subheader("🏆 Race Winner Predictions")
        
        st.info("""
        **Race Winner Predictions**: This section predicts which driver will win the entire race, 
        not just beat their teammate. The model considers grid position, qualifying performance, 
        recent form, and track history.
        """)
        
        # Cache fallback helper
        def load_cache(season, event_code):
            """Load cached predictions from JSON file."""
            import json
            import pathlib
            p = pathlib.Path("data/pred_cache") / str(season) / f"{event_code}.race.json"
            return json.loads(p.read_text()) if p.exists() else None
        
        try:
            # Try FastAPI endpoint first
            import requests
            import os
            
            api_base = os.getenv("F1_API_BASE", "http://localhost:8000")
            season = details['season']
            event_code = selected_event.split('_R')[1] if '_R' in selected_event else selected_event
            
            # Try API endpoint
            try:
                response = requests.get(f"{api_base}/api/predict/race_winner/{season}/{event_code}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    
                    if candidates:
                        st.success(f"🏆 Race Winner Predictions for {selected_event}")
                        
                        # Create dataframe for display
                        df = pd.DataFrame(candidates)
                        df.insert(0, "Rank", range(1, len(df) + 1))
                        df["Probability"] = (df["prob"] * 100).round(1).astype(str) + "%"
                        df["Confidence"] = (
                            (df["conf_low"] * 100).round(0).astype(int).astype(str)
                            + "% - "
                            + (df["conf_high"] * 100).round(0).astype(int).astype(str)
                            + "%"
                        )
                        
                        # Display top 5 with medals
                        st.subheader("🥇 Top 5 Race Winners")
                        for i, (_, row) in enumerate(df.head(5).iterrows()):
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
                                st.markdown(f"**{row['driver']}**")
                                st.markdown(f"*{row['team']}*")
                            
                            with col3:
                                st.markdown(f"**{row['Probability']}**")
                                st.markdown(f"*Confidence: {row['Confidence']}*")
                            
                            with col4:
                                st.markdown(f"Rank: **{row['Rank']}**")
                            
                            if i < 4:  # Don't add separator after last row
                                st.markdown("---")
                        
                        # Show complete table
                        st.subheader("📊 Complete Race Predictions")
                        
                        # Prepare columns for display
                        display_columns = ["Rank", "driver", "team", "Probability", "Confidence"]
                        
                        # Add actual results if available
                        if 'actual_winner' in df.columns and 'final_position' in df.columns:
                            display_columns.extend(["🏆 Actual", "✅ Winner"])
                            df["🏆 Actual"] = df.apply(
                                lambda row: f"P{row['final_position']}" if pd.notna(row['final_position']) else "DNF", 
                                axis=1
                            )
                            df["✅ Winner"] = df["actual_winner"].map({True: "🏆", False: "❌"})
                        
                        # Add grid position if available
                        if 'grid_position' in df.columns:
                            display_columns.insert(3, "grid_position")
                        
                        st.dataframe(
                            df[display_columns],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Download option
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Race Predictions as CSV",
                            data=csv,
                            file_name=f"race_predictions_{season}_{event_code}.csv",
                            mime="text/csv"
                        )
                        return
                        
            except requests.exceptions.RequestException as e:
                st.info("🌐 **API endpoint unavailable, trying cache...**")
            
            # Fallback to cache
            cache_data = load_cache(season, event_code)
            if cache_data:
                st.success(f"🏆 Race Winner Predictions for {selected_event} (from cache)")
                
                # Create dataframe for display
                df = pd.DataFrame(cache_data.get("candidates", []))
                if not df.empty:
                    df.insert(0, "Rank", range(1, len(df) + 1))
                    df["Probability"] = (df["prob"] * 100).round(1).astype(str) + "%"
                    df["Confidence"] = (
                        (df["conf_low"] * 100).round(0).astype(int).astype(str)
                        + "% - "
                        + (df["conf_high"] * 100).round(0).astype(int).astype(str)
                        + "%"
                    )
                    
                    # Display top 5 with medals
                    st.subheader("🥇 Top 5 Race Winners")
                    for i, (_, row) in enumerate(df.head(5).iterrows()):
                        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 1, 1])
                        
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
                            st.markdown(f"**{row['driver']}**")
                            st.markdown(f"*{row['team']}*")
                        
                        with col3:
                            st.markdown(f"**{row['Probability']}**")
                            st.markdown(f"*Confidence: {row['Confidence']}*")
                        
                        with col4:
                            if 'grid_position' in row and pd.notna(row['grid_position']):
                                st.markdown(f"Grid: **P{int(row['grid_position'])}**")
                            else:
                                st.markdown("Grid: **N/A**")
                        
                        with col5:
                            if 'actual_winner' in row and pd.notna(row['actual_winner']):
                                if row['actual_winner']:
                                    st.markdown("🏆 **Winner!**")
                                else:
                                    st.markdown("❌")
                            else:
                                st.markdown("**Prediction**")
                        
                        if i < 4:  # Don't add separator after last row
                            st.markdown("---")
                    
                    # Show complete table
                    st.subheader("📊 Complete Race Predictions")
                    
                    # Prepare columns for display
                    display_columns = ["Rank", "driver", "team", "Probability", "Confidence"]
                    
                    # Add actual results if available
                    if 'actual_winner' in df.columns and 'final_position' in df.columns:
                        display_columns.extend(["🏆 Actual", "✅ Winner"])
                        df["🏆 Actual"] = df.apply(
                            lambda row: f"P{row['final_position']}" if pd.notna(row['final_position']) else "DNF", 
                            axis=1
                        )
                        df["✅ Winner"] = df["actual_winner"].map({True: "🏆", False: "❌"})
                    
                    # Add grid position if available
                    if 'grid_position' in df.columns:
                        display_columns.insert(3, "grid_position")
                    
                    st.dataframe(
                        df[display_columns],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Race Predictions as CSV",
                        data=csv,
                        file_name=f"race_predictions_{season}_{event_code}.csv",
                        mime="text/csv"
                    )
                    return
            
            # Neither API nor cache available
            st.error("🏆 Race Winner Predictions Not Available")
            st.info("""
            **To enable race winner predictions, you need to:**
            1. Start the FastAPI backend: `uvicorn webapi.main:app --reload`
            2. Ensure the model exists: `webapi/ml/models/race_winner.joblib`
            3. Or precompute cache: `python tools/precompute_predictions.py --seasons 2024 2025`
            
            **For now, use the Qualifying H2H tab above!** 🏁
            """)
                        
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
