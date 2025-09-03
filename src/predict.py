"""
Prediction module for F1 teammate qualifying prediction pipeline.

Makes predictions for new events using trained models.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import yaml
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeammatePredictor:
    """Makes predictions for teammate qualifying outcomes."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the predictor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = Path(self.config['data']['models_dir'])
        self.reports_dir = Path(self.config['data']['reports_dir'])
        self.predictions_dir = self.reports_dir / "predictions"
        
        # Ensure directories exist
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize calibrator attribute
        self.calibrator = None
        
        # Load models
        self.models = self._load_models()
        
    def _load_models(self) -> Dict[str, Any]:
        """Load trained models and calibrator."""
        models = {}
        
        for model_name in ['logistic_regression', 'xgboost']:
            model_path = self.models_dir / f"{model_name}.joblib"
            if model_path.exists():
                models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model")
            else:
                logger.warning(f"Model not found: {model_path}")
        
        # Load calibrator if enabled
        if self.config['eval']['calibrate']:
            calibrator_path = self.models_dir / "xgb_calibrator.joblib"
            if calibrator_path.exists():
                self.calibrator = joblib.load(calibrator_path)
                logger.info(f"Loaded calibrator from {calibrator_path}")
            else:
                logger.warning(f"Calibrator not found at {calibrator_path}, predictions will be uncalibrated.")
        
        return models
    
    def predict_teammate(self, event_key: str, latest_data_paths: List[str]) -> pd.DataFrame:
        """Make predictions for a specific event."""
        logger.info(f"Making predictions for event: {event_key}")
        
        if not self.models:
            raise ValueError("No trained models available for prediction")
        
        # Load and prepare latest data
        event_data = self._load_event_data(latest_data_paths, event_key)
        
        if event_data is None or len(event_data) == 0:
            raise ValueError(f"No data found for event: {event_key}")
        
        # Prepare features
        X, driver_info = self._prepare_prediction_features(event_data)
        
        # Make predictions with all models
        predictions = {}
        for model_name, model in self.models.items():
            try:
                y_pred_proba = model.predict_proba(X)[:, 1]
                
                # Apply calibration if available for XGBoost
                if (model_name == 'xgboost' and 
                    self.calibrator is not None and 
                    'calibrator' in self.calibrator):
                    y_pred_proba = self.calibrator['calibrator'].predict(y_pred_proba)
                    logger.info(f"Applied calibration to {model_name}")
                
                predictions[model_name] = y_pred_proba
                logger.info(f"Generated predictions for {model_name}")
            except Exception as e:
                logger.error(f"Error making predictions with {model_name}: {e}")
                predictions[model_name] = None
        
        # Combine predictions
        results = self._combine_predictions(predictions, driver_info, event_key)
        
        # Save predictions
        self._save_predictions(results, event_key)
        
        return results
    
    def _load_event_data(self, data_paths: List[str], event_key: str) -> Optional[pd.DataFrame]:
        """Load data for a specific event."""
        logger.info(f"Loading data for event: {event_key}")
        
        event_data = []
        
        for data_path in data_paths:
            try:
                if Path(data_path).exists():
                    df = pd.read_parquet(data_path)
                    
                    # Filter to the specific event
                    if 'event_key' in df.columns:
                        event_df = df[df['event_key'] == event_key]
                        if len(event_df) > 0:
                            event_data.append(event_df)
                            logger.info(f"Found {len(event_df)} records for {event_key} in {data_path}")
                    
            except Exception as e:
                logger.error(f"Error loading data from {data_path}: {e}")
                continue
        
        if not event_data:
            logger.warning(f"No data found for event: {event_key}")
            return None
        
        # Combine all event data
        combined_data = pd.concat(event_data, ignore_index=True)
        logger.info(f"Combined event data: {combined_data.shape}")
        
        return combined_data
    
    def _prepare_prediction_features(self, event_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features for prediction."""
        logger.info("Preparing prediction features")
        
        # Get feature columns (exclude target and metadata)
        exclude_cols = {
            'beats_teammate_q', 'teammate_gap_ms', 'teammate_id', 'split',
            'season', 'event_key', 'driver_id', 'driver_name', 'constructor_id', 'constructor_name',
            'round', 'status', 'equal_time_tiebreak', 'single_teammate'
        }
        
        feature_cols = [col for col in event_data.columns if col not in exclude_cols]
        logger.info(f"Identified {len(feature_cols)} feature columns")
        
        # Prepare X
        X = event_data[feature_cols].copy()
        
        # Handle missing values in features
        X = self._handle_missing_features(X)
        
        # Convert categorical features to numeric
        X = self._encode_categorical_features(X)
        
        # Store driver information for results
        driver_info = event_data[['driver_id', 'driver_name', 'constructor_id', 'constructor_name']].copy()
        
        logger.info(f"Features prepared: X={X.shape}")
        return X, driver_info
    
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
        
        # For object columns, use label encoding
        object_cols = X.select_dtypes(include=['object']).columns
        for col in object_cols:
            if X[col].nunique() < 100:
                X[col] = pd.Categorical(X[col]).codes
                logger.info(f"Label encoded {col}")
        
        return X
    
    def _combine_predictions(self, predictions: Dict[str, np.ndarray], 
                            driver_info: pd.DataFrame, event_key: str) -> pd.DataFrame:
        """Combine predictions from all models."""
        logger.info("Combining predictions from all models")
        
        # Create results dataframe
        results = driver_info.copy()
        results['event_key'] = event_key
        
        # Add predictions from each model
        for model_name, pred_proba in predictions.items():
            if pred_proba is not None:
                results[f'{model_name}_prob'] = pred_proba
            else:
                results[f'{model_name}_prob'] = np.nan
        
        # Calculate ensemble prediction (simple average)
        model_probs = [col for col in results.columns if col.endswith('_prob')]
        if len(model_probs) > 1:
            results['ensemble_prob'] = results[model_probs].mean(axis=1)
        elif len(model_probs) == 1:
            results['ensemble_prob'] = results[model_probs[0]]
        else:
            results['ensemble_prob'] = np.nan
        
        # Add prediction metadata
        results['prediction_timestamp'] = pd.Timestamp.now()
        results['models_used'] = ', '.join([name for name, pred in predictions.items() if pred is not None])
        
        # Calculate rank within team
        results = self._calculate_team_rankings(results)
        
        logger.info(f"Combined predictions: {results.shape}")
        return results
    
    def _calculate_team_rankings(self, results: pd.DataFrame) -> pd.DataFrame:
        """Calculate rankings within each team."""
        logger.info("Calculating team rankings")
        
        # Group by constructor and sort by ensemble probability
        results['rank_within_team'] = results.groupby('constructor_id')['ensemble_prob'].rank(
            method='dense', ascending=False
        )
        
        # Add teammate information
        results['teammate_id'] = None
        results['teammate_name'] = None
        
        for constructor_id in results['constructor_id'].unique():
            team_drivers = results[results['constructor_id'] == constructor_id]
            if len(team_drivers) == 2:
                # Set teammate IDs
                driver1_idx = team_drivers.index[0]
                driver2_idx = team_drivers.index[1]
                
                results.loc[driver1_idx, 'teammate_id'] = team_drivers.loc[driver2_idx, 'driver_id']
                results.loc[driver1_idx, 'teammate_name'] = team_drivers.loc[driver2_idx, 'driver_name']
                
                results.loc[driver2_idx, 'teammate_id'] = team_drivers.loc[driver1_idx, 'driver_id']
                results.loc[driver2_idx, 'teammate_name'] = team_drivers.loc[driver1_idx, 'driver_name']
        
        return results
    
    def _save_predictions(self, results: pd.DataFrame, event_key: str) -> None:
        """Save predictions to file."""
        # Save as CSV
        csv_path = self.predictions_dir / f"predictions_{event_key}.csv"
        results.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")
        
        # Save as parquet
        parquet_path = self.predictions_dir / f"predictions_{event_key}.parquet"
        results.to_parquet(parquet_path, index=False)
        logger.info(f"Saved predictions to {parquet_path}")
        
        # Create summary
        self._create_prediction_summary(results, event_key)
    
    def _create_prediction_summary(self, results: pd.DataFrame, event_key: str) -> None:
        """Create a summary of predictions."""
        summary_path = self.predictions_dir / f"summary_{event_key}.md"
        
        with open(summary_path, 'w') as f:
            f.write(f"# Predictions Summary for {event_key}\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            f.write("## Model Information\n")
            f.write(f"- Models used: {results['models_used'].iloc[0]}\n")
            f.write(f"- Total drivers: {len(results)}\n")
            f.write(f"- Teams: {results['constructor_id'].nunique()}\n\n")
            
            f.write("## Team Predictions\n\n")
            
            for constructor_id in sorted(results['constructor_id'].unique()):
                team_results = results[results['constructor_id'] == constructor_id]
                f.write(f"### {constructor_id}\n\n")
                
                for _, driver in team_results.iterrows():
                    f.write(f"- **{driver['driver_name']}**\n")
                    f.write(f"  - Probability of beating teammate: {driver['ensemble_prob']:.3f}\n")
                    f.write(f"  - Rank within team: {driver['rank_within_team']:.0f}\n")
                    if pd.notna(driver['teammate_name']):
                        f.write(f"  - Teammate: {driver['teammate_name']}\n")
                    f.write("\n")
                
                f.write("---\n\n")
        
        logger.info(f"Saved prediction summary to {summary_path}")
    
    def predict_multiple_events(self, event_keys: List[str], 
                               latest_data_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """Make predictions for multiple events."""
        logger.info(f"Making predictions for {len(event_keys)} events")
        
        all_predictions = {}
        
        for event_key in event_keys:
            try:
                predictions = self.predict_teammate(event_key, latest_data_paths)
                all_predictions[event_key] = predictions
                logger.info(f"Completed predictions for {event_key}")
            except Exception as e:
                logger.error(f"Error predicting for {event_key}: {e}")
                all_predictions[event_key] = None
        
        return all_predictions

    def build_event_prediction_df(self, event_key: str, 
                                 include_actual: bool = False) -> pd.DataFrame:
        """Build a tidy prediction dataframe for a specific event.
        
        Args:
            event_key: Event key to predict
            include_actual: Whether to include actual results for evaluation
            
        Returns:
            DataFrame with predictions and optional actual results
        """
        try:
            # Load processed data for the event
            processed_path = Path(self.config['data']['processed_dir']) / "teammate_qual.parquet"
            if not processed_path.exists():
                raise FileNotFoundError(f"Processed data not found: {processed_path}")
            
            df_processed = pd.read_parquet(processed_path)
            event_data = df_processed[df_processed['event_key'] == event_key].copy()
            
            if len(event_data) == 0:
                raise ValueError(f"No data found for event: {event_key}")
            
            # Prepare features
            X, driver_info = self._prepare_prediction_features(event_data)
            
            # Make predictions with XGBoost (main model)
            if 'xgboost' not in self.models:
                raise ValueError("XGBoost model not found")
            
            model = self.models['xgboost']
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Apply calibration if available
            if (self.config['eval']['calibrate'] and 
                self.calibrator is not None and 
                'calibrator' in self.calibrator):
                y_pred_proba = self.calibrator['calibrator'].predict(y_pred_proba)
            
            # Create prediction dataframe
            results = []
            threshold = self.config['eval']['threshold']
            
            for i, (_, driver) in enumerate(driver_info.iterrows()):
                prob = y_pred_proba[i]
                pred_label = 1 if prob >= threshold else 0
                
                result = {
                    'driver_id': driver['driver_id'],
                    'driver_name': driver['driver_name'],
                    'constructor_id': driver['constructor_id'],
                    'constructor_name': driver['constructor_name'],
                    'model_prob': prob,
                    'model_pick': pred_label,
                    'model_confidence': prob if pred_label == 1 else (1 - prob)
                }
                
                if include_actual:
                    # Add actual results if available
                    actual_row = event_data[event_data['driver_id'] == driver['driver_id']]
                    if len(actual_row) > 0:
                        result['actual_beats_teammate'] = actual_row.iloc[0]['beats_teammate_q']
                        result['teammate_gap_ms'] = actual_row.iloc[0]['teammate_gap_ms']
                
                results.append(result)
            
            # Add teammate information
            df_results = pd.DataFrame(results)
            df_results = self._add_teammate_info(df_results)
            
            return df_results
            
        except Exception as e:
            logger.error(f"Error building prediction dataframe for {event_key}: {e}")
            raise

    def _add_teammate_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add teammate information to prediction dataframe."""
        # Group by constructor to find teammate pairs
        for constructor_id in df['constructor_id'].unique():
            team_drivers = df[df['constructor_id'] == constructor_id]
            if len(team_drivers) == 2:
                driver1, driver2 = team_drivers.iloc[0], team_drivers.iloc[1]
                
                # Add teammate info
                df.loc[df['driver_id'] == driver1['driver_id'], 'teammate_id'] = driver2['driver_id']
                df.loc[df['driver_id'] == driver1['driver_id'], 'teammate_name'] = driver2['driver_name']
                df.loc[df['driver_id'] == driver2['driver_id'], 'teammate_id'] = driver1['driver_id']
                df.loc[df['driver_id'] == driver2['driver_id'], 'teammate_name'] = driver1['driver_name']
        
        return df

    def build_event_feature_rows(self, event_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build feature rows for an event (for explanation purposes).
        
        Args:
            event_key: The event key to build features for
            
        Returns:
            Tuple of (feature_matrix, driver_info) where:
            - feature_matrix: DataFrame with features aligned to training schema
            - driver_info: DataFrame with driver metadata
        """
        try:
            # Load processed data
            processed_path = Path(self.config['data']['processed_dir']) / "teammate_qual.parquet"
            if not processed_path.exists():
                raise FileNotFoundError(f"Processed data not found: {processed_path}")
            
            # Load data for the specific event
            df_processed = pd.read_parquet(processed_path)
            event_data = df_processed[df_processed['event_key'] == event_key].copy()
            
            if len(event_data) == 0:
                raise ValueError(f"No data found for event: {event_key}")
            
            # Prepare features using existing method
            X, driver_info = self._prepare_prediction_features(event_data)
            
            return X, driver_info
            
        except Exception as e:
            logger.error(f"Error building feature rows for {event_key}: {e}")
            raise

    def build_future_event_pairs(self, event_key: str, season: int, 
                                lineups_path: str = "data/input/upcoming_lineups.csv",
                                events_path: str = "data/input/upcoming_events.yaml") -> pd.DataFrame:
        """
        Build future event pairs for pre-event predictions.
        
        Args:
            event_key: The event key to build pairs for
            season: The season year
            lineups_path: Path to upcoming lineups CSV
            events_path: Path to upcoming events YAML
            
        Returns:
            DataFrame with two rows per constructor (one per driver)
        """
        try:
            logger.info(f"Building future event pairs for {event_key} (season {season})")
            
            # Load upcoming lineups
            if not Path(lineups_path).exists():
                raise FileNotFoundError(f"Upcoming lineups file not found: {lineups_path}")
            
            lineups_df = pd.read_csv(lineups_path)
            event_lineups = lineups_df[
                (lineups_df['season'] == season) & 
                (lineups_df['event_key'] == event_key)
            ].copy()
            
            if len(event_lineups) == 0:
                raise ValueError(f"No lineups found for event {event_key} in season {season}")
            
            # Load upcoming events for track info
            track_info = {}
            if Path(events_path).exists():
                with open(events_path, 'r') as f:
                    upcoming_events = yaml.safe_load(f)
                    for event in upcoming_events:
                        if event['event_key'] == event_key:
                            track_info = event
                            break
            
            # Load historical data for backfilling and feature building
            processed_path = Path(self.config['data']['processed_dir']) / "teammate_qual.parquet"
            if not processed_path.exists():
                raise FileNotFoundError(f"Processed data not found: {processed_path}")
            
            df_processed = pd.read_parquet(processed_path)
            
            # Build future event pairs
            future_pairs = []
            
            for _, lineup in event_lineups.iterrows():
                constructor_id = lineup['constructor_id']
                driver_a = lineup['driver_id_a']
                driver_b = lineup['driver_id_b']
                
                # Backfill missing drivers from most recent pairing
                if pd.isna(driver_a) or driver_a == '':
                    driver_a = self._get_most_recent_driver(constructor_id, season, df_processed, 'driver_a')
                if pd.isna(driver_b) or driver_b == '':
                    driver_b = self._get_most_recent_driver(constructor_id, season, df_processed, 'driver_b')
                
                # Get driver names from historical data
                driver_a_name = self._get_driver_name(driver_a, df_processed)
                driver_b_name = self._get_driver_name(driver_b, df_processed)
                
                # Create two rows (one per driver)
                for driver_id, driver_name in [(driver_a, driver_a_name), (driver_b, driver_b_name)]:
                    if driver_id and driver_name:  # Skip if still missing
                        future_pairs.append({
                            'season': season,
                            'event_key': event_key,
                            'driver_id': driver_id,
                            'driver_name': driver_name,
                            'constructor_id': constructor_id,
                            'constructor_name': self._get_constructor_name(constructor_id, df_processed),
                            'track_id': track_info.get('track_id', 'UNKNOWN'),
                            'is_future_event': True
                        })
            
            if not future_pairs:
                raise ValueError(f"No valid driver pairs found for event {event_key}")
            
            logger.info(f"Built {len(future_pairs)} driver rows for {event_key}")
            return pd.DataFrame(future_pairs)
            
        except Exception as e:
            logger.error(f"Error building future event pairs for {event_key}: {e}")
            raise
    
    def _get_most_recent_driver(self, constructor_id: str, season: int, 
                                df_processed: pd.DataFrame, position: str) -> str:
        """Get the most recent driver for a constructor from historical data."""
        try:
            # First try same season, then previous season
            for search_season in [season, season - 1]:
                season_data = df_processed[df_processed['season'] == search_season]
                constructor_data = season_data[season_data['constructor_id'] == constructor_id]
                
                if len(constructor_data) >= 2:
                    # Get unique drivers for this constructor in this season
                    drivers = constructor_data['driver_id'].unique()
                    if len(drivers) >= 2:
                        # Return first or second driver based on position
                        if position == 'driver_a':
                            return drivers[0]
                        else:
                            return drivers[1] if len(drivers) > 1 else drivers[0]
            
            # Fallback: return None if no historical data found
            logger.warning(f"No historical driver data found for {constructor_id} in season {season}")
            return None
            
        except Exception as e:
            logger.warning(f"Error getting most recent driver for {constructor_id}: {e}")
            return None
    
    def _get_driver_name(self, driver_id: str, df_processed: pd.DataFrame) -> str:
        """Get driver name from historical data."""
        try:
            driver_data = df_processed[df_processed['driver_id'] == driver_id]
            if len(driver_data) > 0:
                return driver_data.iloc[0]['driver_name']
            return None
        except Exception as e:
            logger.warning(f"Error getting driver name for {driver_id}: {e}")
            return None
    
    def _get_constructor_name(self, constructor_id: str, df_processed: pd.DataFrame) -> str:
        """Get constructor name from historical data."""
        try:
            constructor_data = df_processed[df_processed['constructor_id'] == constructor_id]
            if len(constructor_data) > 0:
                return constructor_data.iloc[0]['constructor_name']
            return constructor_id.replace('_', ' ').title()  # Fallback
        except Exception as e:
            logger.warning(f"Error getting constructor name for {constructor_id}: {e}")
            return constructor_id.replace('_', ' ').title()  # Fallback

    def predict_future_event(self, event_key: str, season: int) -> pd.DataFrame:
        """
        Make predictions for a future event using upcoming lineups.
        
        Args:
            event_key: The event key to predict
            season: The season year
            
        Returns:
            DataFrame with predictions for each driver
        """
        try:
            logger.info(f"Making pre-event predictions for {event_key} (season {season})")
            
            # Build future event pairs
            future_pairs = self.build_future_event_pairs(event_key, season)
            
            if len(future_pairs) == 0:
                raise ValueError(f"No future pairs built for event {event_key}")
            
            # Prepare features for future event (pre-event mode)
            X, driver_info = self._prepare_future_event_features(future_pairs)
            
            # Make predictions with XGBoost (main model)
            if 'xgboost' not in self.models:
                raise ValueError("XGBoost model not found")
            
            model = self.models['xgboost']
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Apply calibration if available
            if (self.config['eval']['calibrate'] and 
                self.calibrator is not None and 
                'calibrator' in self.calibrator):
                y_pred_proba = self.calibrator['calibrator'].predict(y_pred_proba)
            
            # Create prediction dataframe
            results = []
            threshold = self.config['eval']['threshold']
            
            for i, (_, driver) in enumerate(driver_info.iterrows()):
                prob = y_pred_proba[i]
                pred_label = 1 if prob >= threshold else 0
                
                result = {
                    'driver_id': driver['driver_id'],
                    'driver_name': driver['driver_name'],
                    'constructor_id': driver['constructor_id'],
                    'constructor_name': driver['constructor_name'],
                    'model_prob': prob,
                    'model_pick': pred_label,
                    'model_confidence': prob if pred_label == 1 else (1 - prob),
                    'is_future_event': True,
                    'event_key': event_key,
                    'season': season
                }
                
                results.append(result)
            
            # Add teammate information
            df_results = pd.DataFrame(results)
            df_results = self._add_teammate_info(df_results)
            
            # Save predictions
            output_path = self.predictions_dir / f"future_{event_key}_predictions.csv"
            df_results.to_csv(output_path, index=False)
            logger.info(f"Saved future predictions to {output_path}")
            
            return df_results
            
        except Exception as e:
            logger.error(f"Error predicting future event {event_key}: {e}")
            raise
    
    def _prepare_future_event_features(self, future_pairs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for future events (pre-event mode).
        
        This method handles missing data that won't exist for future events:
        - Practice deltas: Set to NaN (will be imputed)
        - Weather: Set to missing (will be flagged)
        - Track features: Use track_id from upcoming_events.yaml
        """
        try:
            logger.info("Preparing features for future event (pre-event mode)")
            
            # Load processed data for feature engineering
            processed_path = Path(self.config['data']['processed_dir']) / "teammate_qual.parquet"
            df_processed = pd.read_parquet(processed_path)
            
            # Get feature columns (excluding targets and metadata)
            exclude_cols = ['beats_teammate_q', 'teammate_gap_ms', 'season', 'event_key', 
                          'driver_id', 'driver_name', 'constructor_id', 'constructor_name']
            feature_columns = [col for col in df_processed.columns if col not in exclude_cols]
            
            # Initialize feature matrix
            X = pd.DataFrame(index=future_pairs.index, columns=feature_columns)
            
            # Fill features based on historical data and upcoming event info
            for idx, (_, driver) in enumerate(future_pairs.iterrows()):
                driver_id = driver['driver_id']
                constructor_id = driver['constructor_id']
                track_id = driver['track_id']
                
                # Get historical features for this driver
                driver_history = df_processed[
                    (df_processed['driver_id'] == driver_id) & 
                    (df_processed['constructor_id'] == constructor_id)
                ].copy()
                
                if len(driver_history) > 0:
                    # Use most recent features (last event)
                    latest_features = driver_history.sort_values('event_key').iloc[-1]
                    
                    for col in feature_columns:
                        if col in latest_features:
                            X.loc[idx, col] = latest_features[col]
                        else:
                            X.loc[idx, col] = np.nan
                else:
                    # No history for this driver-constructor combination
                    for col in feature_columns:
                        X.loc[idx, col] = np.nan
                
                # Handle track-specific features
                if 'track_id' in feature_columns:
                    X.loc[idx, 'track_id'] = track_id
                
                # Set practice deltas to NaN (won't exist pre-event)
                practice_cols = [col for col in feature_columns if 'fp' in col.lower() and 'delta' in col.lower()]
                for col in practice_cols:
                    X.loc[idx, col] = np.nan
                
                # Set weather features to missing
                weather_cols = [col for col in feature_columns if any(w in col.lower() for w in ['temp', 'rain', 'wind', 'humidity'])]
                for col in weather_cols:
                    X.loc[idx, col] = np.nan
            
            # Handle missing values using existing imputation logic
            X = self._handle_missing_features(X)
            
            return X, future_pairs
            
        except Exception as e:
            logger.error(f"Error preparing future event features: {e}")
            raise


def main():
    """Main function to run prediction pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Make teammate qualifying predictions')
    parser.add_argument('--event', type=str, required=True, help='Event key to predict')
    parser.add_argument('--data-paths', nargs='+', required=True, help='Paths to data files')
    parser.add_argument('--future', action='store_true', help='Predict for future event using upcoming lineups')
    parser.add_argument('--season', type=int, help='Season year for future predictions')
    
    args = parser.parse_args()
    
    predictor = TeammatePredictor()
    
    try:
        if args.future:
            if not args.season:
                parser.error("--future requires --season")
            
            results = predictor.predict_future_event(args.event, args.season)
            logger.info(f"Future predictions completed for {args.event}")
            logger.info(f"Results shape: {results.shape}")
            
            # Print summary
            print(f"\nFuture Predictions for {args.event} (Season {args.season}):")
            print("=" * 60)
            print("📅 PRE-EVENT PREDICTIONS (using upcoming lineups)")
            print("=" * 60)
            
            # Group by constructor for better display
            for constructor_id in results['constructor_id'].unique():
                team_results = results[results['constructor_id'] == constructor_id]
                print(f"\n🏎️  {constructor_id.replace('_', ' ').title()}:")
                
                for _, row in team_results.iterrows():
                    print(f"   {row['driver_name']}: {row['model_prob']:.1%} probability of beating teammate")
                    
                    if row['model_pick'] == 1:
                        print(f"   🎯 Model Pick: {row['driver_name']} (confidence: {row['model_confidence']:.1%})")
            
            print(f"\n💾 Predictions saved to: future_{args.event}_predictions.csv")
            
        else:
            results = predictor.predict_teammate(args.event, args.data_paths)
            logger.info(f"Predictions completed for {args.event}")
            logger.info(f"Results shape: {results.shape}")
            
            # Print summary
            print(f"\nPredictions for {args.event}:")
            print("=" * 50)
            for _, row in results.iterrows():
                print(f"{row['driver_name']} ({row['constructor_id']}): "
                      f"{row['model_prob']:.3f} probability of beating teammate")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()
