"""
Race winner prediction model using XGBoost and calibration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any
import yaml
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import xgboost as xgb
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RaceWinnerModel:
    """XGBoost model for predicting race winners."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize model, optionally load from file."""
        self.model = None
        self.calibrator = None
        self.feature_columns = [
            'grid_position', 'best_qual_pos', 'rolling_driver_points_5',
            'rolling_team_points_5', 'track_driver_prev_best3', 'weather_simple'
        ]
        self.categorical_columns = ['driver_code', 'team', 'event_key']
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training."""
        if df.empty:
            return np.array([]), np.array([])
        
        # Handle missing values
        df_clean = df.copy()
        
        # Fill missing numeric values
        for col in self.feature_columns:
            if col in df_clean.columns:
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(0)
        
        # Create feature matrix
        X = df_clean[self.feature_columns].values
        
        # Create labels
        y = df_clean['label_winner'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, output_path: str, 
              test_seasons: Optional[List[int]] = None):
        """Train the race winner model with walk-forward validation."""
        logger.info("Starting race winner model training...")
        
        if df.empty:
            logger.error("No data provided for training")
            return
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        if len(X) == 0:
            logger.error("No features could be prepared")
            return
        
        # Create groups for GroupKFold (by season)
        groups = df['season'].values
        
        # Split data for walk-forward validation
        if test_seasons:
            # Use specified test seasons
            train_mask = ~df['season'].isin(test_seasons)
            test_mask = df['season'].isin(test_seasons)
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            groups_train = groups[train_mask]
        else:
            # Use last 2 seasons as test
            unique_seasons = sorted(df['season'].unique())
            if len(unique_seasons) >= 2:
                test_seasons = unique_seasons[-2:]
                train_mask = ~df['season'].isin(test_seasons)
                test_mask = df['season'].isin(test_seasons)
                
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
                groups_train = groups[train_mask]
            else:
                # Fallback to simple split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                groups_train = groups[:len(X_train)]
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Initialize XGBoost model
        base_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Use GroupKFold for cross-validation
        cv = GroupKFold(n_splits=3)
        
        # Train with calibration
        self.calibrator = CalibratedClassifierCV(
            base_model, 
            cv=cv, 
            method='isotonic',
            n_jobs=-1
        )
        
        # Fit the calibrator
        self.calibrator.fit(X_train, y_train, groups=groups_train)
        
        # Evaluate on test set
        y_pred_proba = self.calibrator.predict_proba(X_test)[:, 1]
        y_pred = self.calibrator.predict(X_test)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        
        # Top-1 accuracy (how often we predict the actual winner)
        test_df = df[test_mask].copy()
        test_df['pred_prob'] = y_pred_proba
        test_df['pred_winner'] = y_pred
        
        # Group by event and find predicted winner
        top1_correct = 0
        total_events = 0
        
        for (season, event_key), event_data in test_df.groupby(['season', 'event_key']):
            if len(event_data) > 0:
                total_events += 1
                # Find driver with highest probability
                predicted_winner = event_data.loc[event_data['pred_prob'].idxmax()]
                if predicted_winner['label_winner'] == 1:
                    top1_correct += 1
        
        top1_accuracy = top1_correct / total_events if total_events > 0 else 0
        
        logger.info(f"Model Performance:")
        logger.info(f"  ROC-AUC: {auc:.4f}")
        logger.info(f"  Log Loss: {logloss:.4f}")
        logger.info(f"  Brier Score: {brier:.4f}")
        logger.info(f"  Top-1 Accuracy: {top1_accuracy:.4f}")
        
        # Save model
        self.save_model(output_path)
        
        # Save evaluation results
        eval_results = {
            'training_date': datetime.now().isoformat(),
            'metrics': {
                'roc_auc': auc,
                'log_loss': logloss,
                'brier_score': brier,
                'top1_accuracy': top1_accuracy
            },
            'data_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'total_events': total_events,
                'correct_predictions': top1_correct
            }
        }
        
        eval_path = Path(output_path).parent / 'race_winner_eval.yaml'
        with open(eval_path, 'w') as f:
            yaml.dump(eval_results, f, default_flow_style=False)
        
        logger.info(f"Model saved to {output_path}")
        logger.info(f"Evaluation results saved to {eval_path}")
    
    def predict_event(self, df: pd.DataFrame, event_key: str) -> pd.DataFrame:
        """Predict race winner probabilities for a specific event."""
        if self.calibrator is None:
            logger.error("Model not trained or loaded")
            return pd.DataFrame()
        
        # Filter data for the event
        event_data = df[df['event_key'] == event_key].copy()
        
        if event_data.empty:
            logger.warning(f"No data found for event {event_key}")
            return pd.DataFrame()
        
        # Prepare features
        X, _ = self.prepare_features(event_data)
        
        if len(X) == 0:
            logger.error("Could not prepare features for prediction")
            return pd.DataFrame()
        
        # Get predictions
        probabilities = self.calibrator.predict_proba(X)[:, 1]
        
        # Create results dataframe
        results = event_data[['driver_code', 'team', 'grid_position', 'best_qual_pos']].copy()
        results['probability'] = probabilities
        
        # Sort by probability (highest first)
        results = results.sort_values('probability', ascending=False)
        
        # Add confidence intervals (simple ±5% for now)
        results['conf_low'] = np.maximum(0, results['probability'] - 0.05)
        results['conf_high'] = np.minimum(1, results['probability'] + 0.05)
        
        # Normalize probabilities to sum to 1
        total_prob = results['probability'].sum()
        if total_prob > 0:
            results['probability'] = results['probability'] / total_prob
            results['conf_low'] = results['conf_low'] / total_prob
            results['conf_high'] = results['conf_high'] / total_prob
        
        return results
    
    def save_model(self, output_path: str):
        """Save the trained model and calibrator."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_path = output_file.with_suffix('.joblib')
        joblib.dump(self.model, model_path)
        
        # Save calibrator
        calibrator_path = output_file.parent / f"{output_file.stem}_calibrator.joblib"
        joblib.dump(self.calibrator, calibrator_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Calibrator saved to {calibrator_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        try:
            self.calibrator = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.calibrator = None

def main():
    """CLI for race winner model training and evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Race Winner Model CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the race winner model')
    train_parser.add_argument('--data', required=True, help='Path to race data parquet file')
    train_parser.add_argument('--output', default='models/race_winner.pkl', help='Output model path')
    train_parser.add_argument('--test-seasons', nargs='+', type=int, help='Seasons to use for testing')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the race winner model')
    eval_parser.add_argument('--data', required=True, help='Path to race data parquet file')
    eval_parser.add_argument('--model', required=True, help='Path to trained model')
    eval_parser.add_argument('--test-seasons', nargs='+', type=int, help='Seasons to use for testing')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions for an event')
    predict_parser.add_argument('--data', required=True, help='Path to race data parquet file')
    predict_parser.add_argument('--model', required=True, help='Path to trained model')
    predict_parser.add_argument('--event', required=True, help='Event key to predict')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Load data
        df = pd.read_parquet(args.data)
        
        # Train model
        model = RaceWinnerModel()
        model.train(df, args.output, args.test_seasons)
        
    elif args.command == 'eval':
        # Load data and model
        df = pd.read_parquet(args.data)
        model = RaceWinnerModel(args.model)
        
        if model.calibrator is not None:
            # Re-run training to get evaluation metrics
            model.train(df, 'temp_model.pkl', args.test_seasons)
            # Clean up temp file
            Path('temp_model.pkl').unlink(missing_ok=True)
        else:
            logger.error("Could not load model for evaluation")
            
    elif args.command == 'predict':
        # Load data and model
        df = pd.read_parquet(args.data)
        model = RaceWinnerModel(args.model)
        
        if model.calibrator is not None:
            # Make predictions
            predictions = model.predict_event(df, args.event)
            if not predictions.empty:
                print(f"\nRace Winner Predictions for {args.event}:")
                print(predictions.to_string(index=False))
            else:
                logger.error("No predictions generated")
        else:
            logger.error("Could not load model for prediction")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
