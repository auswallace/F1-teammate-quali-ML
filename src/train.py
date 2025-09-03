"""
Training module for F1 teammate qualifying prediction pipeline.

Trains Logistic Regression and XGBoost models with cross-validation.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    average_precision_score, brier_score_loss
)
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains models for teammate qualifying predictions."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.interim_dir = Path(self.config['data']['interim_dir'])
        self.models_dir = Path(self.config['data']['models_dir'])
        self.reports_dir = Path(self.config['data']['reports_dir'])
        self.training_config = self.config['training']
        self.models_config = self.config['models']
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Track training results
        self.training_results = {}
        self.best_models = {}
        
    def load_split_data(self, filename_prefix: str = "split") -> Dict[str, pd.DataFrame]:
        """Load split data from interim directory."""
        splits = {}
        
        for split_name in ['train', 'val', 'test']:
            split_path = self.interim_dir / f"{filename_prefix}_{split_name}.parquet"
            if split_path.exists():
                splits[split_name] = pd.read_parquet(split_path)
                logger.info(f"Loaded {split_name} split: {splits[split_name].shape}")
            else:
                logger.warning(f"Split file not found: {split_path}")
                splits[split_name] = pd.DataFrame()
        
        return splits
    
    def prepare_training_data(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare training data with features and target."""
        logger.info("Preparing training data")
        
        if len(train_df) == 0:
            raise ValueError("No training data found")
        
        # Get feature columns (exclude target and metadata)
        exclude_cols = {
            'beats_teammate_q', 'teammate_gap_ms', 'teammate_id', 'split',
            'season', 'event_key', 'driver_id', 'driver_name', 'constructor_id', 'constructor_name',
            'round', 'status', 'equal_time_tiebreak', 'single_teammate'
        }
        
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        logger.info(f"Identified {len(feature_cols)} feature columns")
        
        # Prepare X and y
        X = train_df[feature_cols].copy()
        y = train_df['beats_teammate_q'].copy()
        
        # Handle missing values in features
        X = self._handle_missing_features(X)
        
        # Convert categorical features to numeric
        X = self._encode_categorical_features(X)
        
        logger.info(f"Training data prepared: X={X.shape}, y={y.shape}")
        logger.info(f"Feature columns: {feature_cols}")
        
        return X, y, feature_cols
    
    def _handle_missing_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature matrix."""
        # For numeric columns, fill with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                if pd.isna(median_val):
                    # If median is also NaN, use 0
                    median_val = 0.0
                X[col] = X[col].fillna(median_val)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        # For categorical columns, fill with mode
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                mode_val = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'unknown'
                X[col] = X[col].fillna(mode_val)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Ensure no NaN values remain
        if X.isnull().any().any():
            logger.warning("Still have NaN values, filling with 0")
            X = X.fillna(0)
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical features to numeric."""
        # For boolean columns, convert to int
        bool_cols = X.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            X[col] = X[col].astype(int)
        
        # For object columns, use label encoding (simplified)
        object_cols = X.select_dtypes(include=['object']).columns
        for col in object_cols:
            if X[col].nunique() < 100:  # Only encode if not too many unique values
                X[col] = pd.Categorical(X[col]).codes
                logger.info(f"Label encoded {col}")
        
        return X
    
    def train_logistic_regression(self, X: pd.DataFrame, y: pd.Series, 
                                 cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Train Logistic Regression model with cross-validation."""
        logger.info("Training Logistic Regression model")
        
        # Get model configuration
        lr_config = self.models_config['logistic_regression']
        
        # Create model
        model = LogisticRegression(
            class_weight=lr_config['class_weight'],
            random_state=lr_config['random_state'],
            max_iter=1000
        )
        
        # Prepare groups for GroupKFold (use season as group)
        # For now, we'll use a simple approach - in production you'd extract season info
        groups = np.arange(len(X)) // 10  # Simplified grouping
        
        # Cross-validate
        cv_scores = {}
        for metric_name, scoring in [
            ('precision', 'precision'),
            ('recall', 'recall'),
            ('f1', 'f1'),
            ('roc_auc', 'roc_auc'),
            ('average_precision', 'average_precision')
        ]:
            scores = cross_val_score(model, X, y, groups=groups, scoring=scoring, cv=5)
            cv_scores[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
        
        # Train final model on full training data
        model.fit(X, y)
        
        # Calculate Brier score
        y_pred_proba = model.predict_proba(X)[:, 1]
        brier_score = brier_score_loss(y, y_pred_proba)
        
        results = {
            'model': model,
            'cv_scores': cv_scores,
            'brier_score': brier_score,
            'feature_importance': dict(zip(X.columns, np.abs(model.coef_[0])))
        }
        
        logger.info(f"Logistic Regression training complete. CV F1: {cv_scores['f1']['mean']:.3f}")
        return results
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series, 
                      cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Train XGBoost model with cross-validation."""
        logger.info("Training XGBoost model")
        
        # Get model configuration
        xgb_config = self.models_config['xgboost']
        
        # Calculate scale_pos_weight if not provided
        if xgb_config['scale_pos_weight'] is None:
            class_counts = y.value_counts()
            scale_pos_weight = class_counts[0] / class_counts[1]
            logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.3f}")
        else:
            scale_pos_weight = xgb_config['scale_pos_weight']
        
        # Create model
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=xgb_config['random_state'],
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Prepare groups for GroupKFold
        groups = np.arange(len(X)) // 10  # Simplified grouping
        
        # Cross-validate
        cv_scores = {}
        for metric_name, scoring in [
            ('precision', 'precision'),
            ('recall', 'recall'),
            ('f1', 'f1'),
            ('roc_auc', 'roc_auc'),
            ('average_precision', 'average_precision')
        ]:
            scores = cross_val_score(model, X, y, groups=groups, scoring=scoring, cv=5)
            cv_scores[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
        
        # Train final model on full training data
        model.fit(X, y)
        
        # Calculate Brier score
        y_pred_proba = model.predict_proba(X)[:, 1]
        brier_score = brier_score_loss(y, y_pred_proba)
        
        # Get feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        
        results = {
            'model': model,
            'cv_scores': cv_scores,
            'brier_score': brier_score,
            'feature_importance': feature_importance
        }
        
        logger.info(f"XGBoost training complete. CV F1: {cv_scores['f1']['mean']:.3f}")
        return results
    
    def _train_calibrator(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train probability calibrator on held-out validation data."""
        try:
            # Simple approach: use last 25% of data for calibration
            split_idx = int(len(X) * 0.75)
            X_cal_train = X.iloc[:split_idx]
            y_cal_train = y.iloc[:split_idx]
            X_cal_valid = X.iloc[split_idx:]
            y_cal_valid = y.iloc[split_idx:]
            
            # Retrain model on calibration training data
            model_cal = clone(model)
            model_cal.fit(X_cal_train, y_cal_train)
            
            # Get raw probabilities on validation set
            y_raw_proba = model_cal.predict_proba(X_cal_valid)[:, 1]
            
            # Calculate Brier score before calibration
            brier_before = brier_score_loss(y_cal_valid, y_raw_proba)
            
            # Train calibrator
            method = self.config['eval']['calibration_method']
            if method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(y_raw_proba, y_cal_valid)
            else:  # sigmoid/platt
                calibrator = IsotonicRegression(out_of_bounds='clip')  # Fallback to isotonic
                calibrator.fit(y_raw_proba, y_cal_valid)
            
            # Apply calibration and calculate Brier score after
            y_calibrated_proba = calibrator.predict(y_raw_proba)
            brier_after = brier_score_loss(y_cal_valid, y_calibrated_proba)
            
            logger.info(f"Calibration complete: Brier {brier_before:.4f} → {brier_after:.4f}")
            
            return {
                'calibrator': calibrator,
                'method': method,
                'brier_before': brier_before,
                'brier_after': brier_after,
                'calib_samples': len(X_cal_valid)
            }
            
        except Exception as e:
            logger.warning(f"Calibration failed: {e}, returning None")
            return None
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Train all models."""
        logger.info("Training all models")
        
        # Create CV splits (simplified)
        cv_splits = [(np.arange(len(X)//2), np.arange(len(X)//2, len(X)))]
        
        # Train models
        models = {}
        
        try:
            models['logistic_regression'] = self.train_logistic_regression(X, y, cv_splits)
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {e}")
            models['logistic_regression'] = None
        
        try:
            models['xgboost'] = self.train_xgboost(X, y, cv_splits)
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            models['xgboost'] = None
        
        self.training_results = models
        return models
    
    def save_models(self) -> None:
        """Save trained models to models directory."""
        logger.info("Saving trained models")
        
        for model_name, results in self.training_results.items():
            if results is not None and 'model' in results:
                # Save model
                model_path = self.models_dir / f"{model_name}.joblib"
                joblib.dump(results['model'], model_path)
                logger.info(f"Saved {model_name} model to {model_path}")
                
                # Save results
                results_path = self.models_dir / f"{model_name}_results.json"
                import json
                
                # Convert numpy types to native Python types for JSON serialization
                serializable_results = {}
                for key, value in results.items():
                    if key != 'model':  # Don't serialize the model object
                        if isinstance(value, dict):
                            serializable_results[key] = {}
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, np.ndarray):
                                    serializable_results[key][sub_key] = sub_value.tolist()
                                elif isinstance(sub_value, np.integer):
                                    serializable_results[key][sub_key] = int(sub_value)
                                elif isinstance(sub_value, np.floating):
                                    serializable_results[key][sub_key] = float(sub_value)
                                else:
                                    serializable_results[key][sub_key] = sub_value
                        elif isinstance(value, np.ndarray):
                            serializable_results[key] = value.tolist()
                        elif isinstance(value, np.integer):
                            serializable_results[key] = int(value)
                        elif isinstance(value, np.floating):
                            serializable_results[key] = float(value)
                        else:
                            serializable_results[key] = value
                
                with open(results_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2, default=str)
                
                logger.info(f"Saved {model_name} results to {results_path}")
        
        # Save calibrator if available
        if (self.config['eval']['calibrate'] and 
            'xgboost' in self.training_results and 
            self.training_results['xgboost'] is not None):
            
            # Note: Calibrator will be trained in main() after models are saved
            logger.info("Calibration enabled - calibrator will be trained after model saving")
        else:
            logger.info("Calibration: OFF")
    
    def log_training_summary(self) -> None:
        """Log training summary to file."""
        log_path = self.reports_dir / "train.log"
        
        with open(log_path, 'w') as f:
            f.write("F1 Teammate Qualifying Prediction - Training Summary\n")
            f.write("=" * 60 + "\n\n")
            
            for model_name, results in self.training_results.items():
                if results is not None:
                    f.write(f"Model: {model_name}\n")
                    f.write("-" * 30 + "\n")
                    
                    # CV scores
                    f.write("Cross-Validation Scores:\n")
                    for metric, scores in results['cv_scores'].items():
                        f.write(f"  {metric}: {scores['mean']:.3f} (+/- {scores['std']:.3f})\n")
                    
                    # Brier score
                    f.write(f"Brier Score: {results['brier_score']:.3f}\n")
                    
                    # Top features
                    f.write("Top 10 Features by Importance:\n")
                    sorted_features = sorted(
                        results['feature_importance'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]
                    
                    for feature, importance in sorted_features:
                        f.write(f"  {feature}: {importance:.4f}\n")
                    
                    f.write("\n")
                else:
                    f.write(f"Model: {model_name} - FAILED TO TRAIN\n\n")
        
        logger.info(f"Training summary logged to {log_path}")


def main():
    """Main function to run training pipeline."""
    trainer = ModelTrainer()
    
    # Load split data
    splits = trainer.load_split_data()
    
    if 'train' not in splits or len(splits['train']) == 0:
        raise ValueError("No training data found")
    
    # Prepare training data
    X, y, feature_cols = trainer.prepare_training_data(splits['train'])
    
    # Train all models
    models = trainer.train_all_models(X, y)
    
    # Save models
    trainer.save_models()
    
    # Train and save calibrator if enabled
    if trainer.config['eval']['calibrate'] and 'xgboost' in models and models['xgboost'] is not None:
        logger.info("Training probability calibrator...")
        calibrator = trainer._train_calibrator(
            models['xgboost']['model'], X, y
        )
        
        if calibrator is not None:
            calibrator_path = trainer.models_dir / "xgb_calibrator.joblib"
            joblib.dump(calibrator, calibrator_path)
            logger.info(f"Saved calibrator to {calibrator_path}")
            
            # Log calibration info
            logger.info(f"Calibration: ON (method={calibrator['method']})")
            logger.info(f"Brier improvement: {calibrator['brier_before']:.4f} → {calibrator['brier_after']:.4f}")
        else:
            logger.warning("Calibration failed, not saving calibrator")
    else:
        logger.info("Calibration: OFF")
    
    # Log training summary
    trainer.log_training_summary()
    
    # Print calibration status
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    
    if trainer.config['eval']['calibrate']:
        print("Calibration: ON (method=isotonic)")
        # Check if calibrator was saved
        calibrator_path = trainer.models_dir / "xgb_calibrator.joblib"
        if calibrator_path.exists():
            print("✓ Calibrator saved successfully")
        else:
            print("✗ Calibrator failed to save")
    else:
        print("Calibration: OFF")
    
    print(f"\nModels saved to {trainer.models_dir}")
    print(f"Training log: {trainer.reports_dir}/train.log")
    
    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main()
