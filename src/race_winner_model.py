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
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RaceWinnerModel:
    """XGBoost model for predicting race winners."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize model, optionally load from file."""
        self.model_obj: Optional[dict] = None  # holds {"model": calibrated_clf, "features": [...], "meta": {...}}
        self.calibrator = None
        self.feature_columns = [
            'grid_position', 'best_qual_pos', 'weather_simple'
        ]
        self.categorical_columns = ['driver_code', 'team', 'event_key']
        
        if model_path:
            self.load_model(model_path)
    
    def save_model(self, output_path: str):
        """Save model as single artifact file."""
        from pathlib import Path
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.model_obj is None:
            raise ValueError("No model object to save.")
        import joblib
        joblib.dump(self.model_obj, out)
        logger.info(f"Model saved to {output_path}")
    
    def load_model(self, model_path: str):
        """Load model from single artifact file."""
        import joblib
        self.model_obj = joblib.load(model_path)
        # convenience
        self.calibrator = self.model_obj.get("model", None)
        logger.info(f"Model loaded from {model_path}")
    
    def validate_features(self, df: pd.DataFrame):
        """Validate that no leakage columns are present in the data."""
        forbidden = {"finish_position", "final_classification", "result_position"}
        bad = forbidden.intersection(set(df.columns))
        if bad:
            raise ValueError(f"Leakage risks found in data: {sorted(bad)}")
    
    def _event_softmax(self, probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Convert calibrated P(win) to per-event softmax probabilities."""
        # probs are calibrated P(win) for each driver row in an event
        probs = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs) - np.log(1 - probs)
        exps = np.exp(logits - logits.max())  # stabilize
        return exps / exps.sum()
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Prepare features and labels for training."""
        if df.empty:
            return np.array([]), np.array([]), df
        
        # Validate features for leakage
        self.validate_features(df)
        
        # Handle missing values
        df_clean = df.copy()
        
        # Fill missing numeric values using pandas.api.types for dtypes
        import pandas.api.types as ptypes
        for col in self.feature_columns:
            if col in df_clean.columns:
                if ptypes.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    # Encode categorical features
                    df_clean[col] = pd.Categorical(df_clean[col]).codes
                    df_clean[col] = df_clean[col].fillna(0)
        
        # Create feature matrix
        X = df_clean[self.feature_columns].values
        
        # Create labels
        y = df_clean['label_winner'].values
        
        return X, y, df_clean
    
    def train(self, df: pd.DataFrame, output_path: str, 
              test_seasons: Optional[List[int]] = None):
        """Train the race winner model with walk-forward validation."""
        logger.info("Starting race winner model training...")
        
        if df.empty:
            logger.error("No data provided for training")
            return
        
        # Validate features for leakage
        self.validate_features(df)
        
        # Prepare features
        X, y, df_clean = self.prepare_features(df)
        
        if len(X) == 0:
            logger.error("No features could be prepared")
            return
        
        # Create groups for GroupKFold (by season)
        groups = df_clean['season'].values
        
        # Split data for walk-forward validation
        if test_seasons:
            # Use specified test seasons
            train_mask = ~df_clean['season'].isin(test_seasons)
            test_mask = df_clean['season'].isin(test_seasons)
            
            # If no training data available, use simple split instead
            if train_mask.sum() == 0:
                logger.warning(f"No training data available for specified test seasons {test_seasons}. Using simple train/test split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                groups_train = groups[:len(X_train)]
            else:
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
                groups_train = groups[train_mask]
        else:
            # Use last 2 seasons as test
            unique_seasons = sorted(df_clean['season'].unique())
            if len(unique_seasons) >= 2:
                test_seasons = unique_seasons[-2:]
                train_mask = ~df_clean['season'].isin(test_seasons)
                test_mask = df_clean['season'].isin(test_seasons)
                
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
                groups_train = groups[train_mask]
            else:
                # Fallback to simple split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
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
        
        # Use simpler CV strategy for small datasets
        if len(X_train) < 6:
            cv = 2  # Use 2-fold CV for very small datasets
        else:
            cv = 3  # Use 3-fold CV for larger datasets
        
        # Check if we have enough samples for each class
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            logger.error("Need at least 2 classes for classification")
            return
        
        # For very imbalanced datasets, use prefit calibration
        if len(unique_classes) == 2 and min(np.bincount(y_train)) < 3:
            logger.warning("Very imbalanced dataset detected. Using prefit calibration.")
            # Train base model first
            base_model.fit(X_train, y_train)
            # Then calibrate with prefit
            self.calibrator = CalibratedClassifierCV(
                base_model, 
                cv='prefit', 
                method='isotonic'
            )
            # Fit calibrator on training data
            self.calibrator.fit(X_train, y_train)
        else:
            # Train with regular calibration
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
        # Create a simple test dataframe for evaluation
        test_df = pd.DataFrame({
            'pred_prob': y_pred_proba,
            'pred_winner': y_pred,
            'label_winner': y_test
        })
        
        # Calculate top-1 accuracy (simplified for now)
        # Find the prediction with highest probability
        if len(test_df) > 0:
            predicted_winner_idx = test_df['pred_prob'].idxmax()
            top1_correct = 1 if test_df.loc[predicted_winner_idx, 'label_winner'] == 1 else 0
            total_events = 1
            top1_accuracy = top1_correct / total_events
        else:
            top1_correct = 0
            total_events = 0
            top1_accuracy = 0
        
        logger.info(f"Model Performance:")
        logger.info(f"  ROC-AUC: {auc:.4f}")
        logger.info(f"  Log Loss: {logloss:.4f}")
        logger.info(f"  Brier Score: {brier:.4f}")
        logger.info(f"  Top-1 Accuracy: {top1_accuracy:.4f}")
        
        # Create model object
        self.model_obj = {
            "model": self.calibrator, 
            "features": self.feature_columns, 
            "meta": {"trained_at": datetime.utcnow().isoformat()}
        }
        
        # Save model
        self.save_model(output_path)
        
        # Save evaluation results to <stem>_eval.yaml
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
        
        eval_path = Path(output_path).parent / f"{Path(output_path).stem}_eval.yaml"
        with open(eval_path, 'w') as f:
            yaml.dump(eval_results, f, default_flow_style=False)
        
        logger.info(f"Evaluation results saved to {eval_path}")
    
    def predict_event(self, df: pd.DataFrame, event_key: str) -> pd.DataFrame:
        """Predict race winner probabilities for a specific event."""
        if self.model_obj is None:
            raise ValueError("Model not loaded.")
        
        self.validate_features(df)
        event = df[df["event_key"] == event_key].copy()
        
        if event.empty:
            logger.warning(f"No data found for event: {event_key}")
            return pd.DataFrame()
        
        X, _, _ = self.prepare_features(event)
        p = self.model_obj["model"].predict_proba(X)[:, 1]
        
        # per-event softmax
        w = self._event_softmax(p)
        
        out = event[["driver_code", "team", "grid_position", "best_qual_pos"]].copy()
        out["probability"] = w
        
        # simple symmetric bands for display only
        out["conf_low"] = np.clip(w - 0.05, 0, 1)
        out["conf_high"] = np.clip(w + 0.05, 0, 1)
        
        return out.sort_values("probability", ascending=False)
    
    def evaluate(self, df: pd.DataFrame, test_seasons: List[int]) -> Dict[str, float]:
        """Evaluate model on specified test seasons without retraining."""
        if self.model_obj is None:
            raise ValueError("Model not loaded.")
        
        logger.info(f"Evaluating model on test seasons: {test_seasons}")
        
        # Validate features for leakage
        self.validate_features(df)
        
        # Filter to test seasons
        test_df = df[df['season'].isin(test_seasons)].copy()
        
        if test_df.empty:
            logger.error(f"No data found for test seasons: {test_seasons}")
            return {}
        
        # Prepare features
        X, y, _ = self.prepare_features(test_df)
        
        if len(X) == 0:
            logger.error("No features could be prepared for evaluation")
            return {}
        
        # Make predictions
        y_pred_proba = self.model_obj["model"].predict_proba(X)[:, 1]
        y_pred = self.model_obj["model"].predict(X)
        
        # Calculate metrics
        auc = roc_auc_score(y, y_pred_proba)
        logloss = log_loss(y, y_pred_proba)
        brier = brier_score_loss(y, y_pred_proba)
        
        # Top-1 accuracy
        test_df['pred_prob'] = y_pred_proba
        test_df['pred_winner'] = y_pred
        
        top1_correct = 0
        total_events = 0
        
        for (season, event_key), event_data in test_df.groupby(['season', 'event_key']):
            if len(event_data) > 0:
                total_events += 1
                predicted_winner = event_data.loc[event_data['pred_prob'].idxmax()]
                if predicted_winner['label_winner'] == 1:
                    top1_correct += 1
        
        top1_accuracy = top1_correct / total_events if total_events > 0 else 0
        
        metrics = {
            'roc_auc': auc,
            'log_loss': logloss,
            'brier_score': brier,
            'top1_accuracy': top1_accuracy,
            'total_events': total_events,
            'correct_predictions': top1_correct
        }
        
        logger.info(f"Evaluation Results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
        
        return metrics


def main():
    """CLI entry point for race winner model."""
    parser = argparse.ArgumentParser(description="Race Winner Prediction Model")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', required=True, help='Path to training data (parquet)')
    train_parser.add_argument('--output', required=True, help='Output path for model artifact')
    train_parser.add_argument('--test-seasons', nargs='+', type=int, help='Seasons to use for testing')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    eval_parser.add_argument('--data', required=True, help='Path to evaluation data (parquet)')
    eval_parser.add_argument('--model', required=True, help='Path to model artifact')
    eval_parser.add_argument('--test-seasons', nargs='+', type=int, required=True, help='Seasons to use for testing')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--data', required=True, help='Path to prediction data (parquet)')
    predict_parser.add_argument('--model', required=True, help='Path to model artifact')
    predict_parser.add_argument('--event', required=True, help='Event key to predict')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "train":
            df = pd.read_parquet(args.data)
            model = RaceWinnerModel()
            model.train(df, args.output, args.test_seasons)
            
        elif args.command == "eval":
            df = pd.read_parquet(args.data)
            model = RaceWinnerModel()
            model.load_model(args.model)
            metrics = model.evaluate(df, args.test_seasons)
            
            # Save evaluation results
            eval_path = Path(args.model).parent / f"{Path(args.model).stem}_eval.yaml"
            eval_results = {
                'evaluation_date': datetime.now().isoformat(),
                'test_seasons': args.test_seasons,
                'metrics': metrics
            }
            with open(eval_path, 'w') as f:
                yaml.dump(eval_results, f, default_flow_style=False)
            logger.info(f"Evaluation results saved to {eval_path}")
            
        elif args.command == "predict":
            df = pd.read_parquet(args.data)
            model = RaceWinnerModel()
            model.load_model(args.model)
            predictions = model.predict_event(df, args.event)
            
            if not predictions.empty:
                print(f"\nRace Winner Predictions for {args.event}:")
                print(predictions.to_string(index=False))
            else:
                logger.warning(f"No predictions generated for event: {args.event}")
                
    except Exception as e:
        logger.error(f"Error in {args.command} command: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
