"""
Simple race winner prediction model using XGBoost.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_race_winner_model(data_path: str, output_path: str):
    """Train a simple race winner prediction model."""
    
    # Load data
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} records from {data_path}")
    
    # Use only the features we have
    feature_columns = ['grid_position', 'best_qual_pos', 'final_position']
    
    # Prepare features
    X = df[feature_columns].fillna(df[feature_columns].median()).values
    y = df['label_winner'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    
    logger.info(f"Test set performance:")
    logger.info(f"ROC-AUC: {auc:.4f}")
    logger.info(f"Log Loss: {logloss:.4f}")
    logger.info(f"Brier Score: {brier:.4f}")
    
    # Save model
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_file)
    logger.info(f"Model saved to {output_file}")
    
    return model

def predict_race_winners(data_path: str, model_path: str, event_key: str = None):
    """Make race winner predictions."""
    
    # Load data and model
    df = pd.read_parquet(data_path)
    model = joblib.load(model_path)
    
    # Use only the features we have
    feature_columns = ['grid_position', 'best_qual_pos', 'final_position']
    
    # Filter by event if specified
    if event_key:
        df = df[df['event_key'] == event_key].copy()
        if df.empty:
            logger.error(f"No data found for event {event_key}")
            return None
    
    # Prepare features
    X = df[feature_columns].fillna(df[feature_columns].median()).values
    
    # Make predictions
    probabilities = model.predict_proba(X)[:, 1]
    
    # Create results
    results = df[['driver_code', 'team', 'grid_position', 'best_qual_pos', 'final_position']].copy()
    results['winner_probability'] = probabilities
    
    # Sort by probability (highest first)
    results = results.sort_values('winner_probability', ascending=False)
    
    # Normalize probabilities to sum to 1
    total_prob = results['winner_probability'].sum()
    if total_prob > 0:
        results['winner_probability'] = results['winner_probability'] / total_prob
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Race Winner Model")
    parser.add_argument("command", choices=["train", "predict"], help="Command to execute")
    parser.add_argument("--data", type=str, required=True, help="Input data file path")
    parser.add_argument("--model", type=str, default="models/simple_race_winner.joblib", help="Model path")
    parser.add_argument("--event", type=str, help="Event key to predict (for predict command)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_race_winner_model(args.data, args.model)
    
    elif args.command == "predict":
        if not args.event:
            logger.error("Please specify --event for prediction")
        else:
            results = predict_race_winners(args.data, args.model, args.event)
            if results is not None:
                print(f"\nRace Winner Predictions for {args.event}:")
                print(results.to_string(index=False))
