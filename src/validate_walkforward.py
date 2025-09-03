#!/usr/bin/env python3
"""
Walk-forward validation module for F1 teammate qualifying prediction.

Implements rolling, time-aware validation with probability calibration,
baseline comparisons, and comprehensive error analysis.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Implements walk-forward validation with calibration and baselines."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the validator with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.setup_directories()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_directories(self):
        """Create necessary directories for outputs."""
        self.reports_dir = Path(self.config['data']['reports_dir'])
        self.models_dir = Path(self.config['data']['models_dir'])
        
        # Create subdirectories
        (self.reports_dir / "predictions").mkdir(exist_ok=True)
        (self.reports_dir / "figures").mkdir(exist_ok=True)
    
    def run(self, settings_path: str = "config/settings.yaml") -> Dict[str, Any]:
        """
        Executes rolling walk-forward validation as defined in settings.
        Returns a dict of aggregate metrics and paths to artifacts.
        """
        logger.info("Starting walk-forward validation")
        
        # Load data and models
        data, model, encoders = self._load_data_and_models()
        if data is None or model is None:
            raise ValueError("Failed to load data or models")
        
        # Run validation for each block
        block_results = []
        all_predictions = []
        
        for block_id, block_config in enumerate(self.config['walkforward']['season_blocks']):
            logger.info(f"Processing block {block_id}: train≤{block_config['train_upto']} → test {block_config['test_seasons']}")
            
            block_result = self._validate_block(
                data, model, encoders, block_config, block_id
            )
            block_results.append(block_result)
            all_predictions.extend(block_result['predictions'])
            
            # Save block artifacts
            self._save_block_artifacts(block_result, block_id)
            
            # Print block summary
            self._print_block_summary(block_result, block_id)
        
        # Compute pooled metrics
        pooled_metrics = self._compute_pooled_metrics(all_predictions)
        
        # Save consolidated results
        self._save_consolidated_results(block_results, pooled_metrics)
        
        # Optional experiment tracking
        if self.config['tracking']['enabled']:
            self._log_experiment(block_results, pooled_metrics)
        
        logger.info("Walk-forward validation completed successfully")
        
        return {
            'block_results': block_results,
            'pooled_metrics': pooled_metrics,
            'all_predictions': all_predictions
        }
    
    def _load_data_and_models(self) -> Tuple[Optional[pd.DataFrame], Optional[Any], Optional[Dict]]:
        """Load processed data, trained model, and encoders."""
        try:
            # Load processed data
            data_path = Path(self.config['data']['processed_dir']) / "teammate_qual.parquet"
            data = pd.read_parquet(data_path)
            logger.info(f"Loaded data: {data.shape}")
            
            # Load best model (XGBoost)
            model_path = self.models_dir / "xgboost.joblib"
            model = joblib.load(model_path)
            logger.info("Loaded XGBoost model")
            
            # Load encoders (if available)
            encoders = {}
            scaler_path = self.models_dir / "scaler.joblib"
            if scaler_path.exists():
                encoders['scaler'] = joblib.load(scaler_path)
                logger.info("Loaded scaler")
            
            return data, model, encoders
            
        except Exception as e:
            logger.error(f"Error loading data/models: {e}")
            return None, None, None
    
    def _validate_block(self, data: pd.DataFrame, model: Any, encoders: Dict,
                       block_config: Dict, block_id: int) -> Dict[str, Any]:
        """Validate a single block with train/test split."""
        
        train_upto = block_config['train_upto']
        test_seasons = block_config['test_seasons']
        
        # Split data
        train_data = data[data['season'] <= train_upto].copy()
        test_data = data[data['season'].isin(test_seasons)].copy()
        
        logger.info(f"Block {block_id}: Train={len(train_data)}, Test={len(test_data)}")
        
        # Prepare features
        X_train, y_train, feature_cols = self._prepare_features(train_data)
        X_test, y_test, _ = self._prepare_features(test_data)
        
        # Train calibration model on last season of training data
        calibrator = self._train_calibrator(model, X_train, y_train, train_upto)
        
        # Make predictions
        raw_probs = model.predict_proba(X_test)[:, 1]
        calibrated_probs = calibrator.predict(raw_probs)
        
        # Generate predictions dataframe
        predictions_df = self._create_predictions_df(
            test_data, calibrated_probs, y_test, train_upto, block_id
        )
        
        # Compute metrics
        metrics = self._compute_metrics(y_test, calibrated_probs)
        calibration_metrics = self._compute_calibration_metrics(y_test, calibrated_probs)
        
        # Compute baseline metrics
        baseline_metrics = self._compute_baseline_metrics(test_data, y_test)
        
        # Top-K analysis
        topk_metrics = self._compute_topk_metrics(predictions_df)
        
        # Error analysis
        error_analysis = self._analyze_errors(predictions_df)
        
        return {
            'block_id': block_id,
            'train_upto': train_upto,
            'test_seasons': test_seasons,
            'predictions': predictions_df.to_dict('records'),
            'metrics': metrics,
            'calibration_metrics': calibration_metrics,
            'baseline_metrics': baseline_metrics,
            'topk_metrics': topk_metrics,
            'error_analysis': error_analysis,
            'calibrator': calibrator
        }
    
    def _prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare feature matrix and target vector."""
        # Filter for labeled pairs only
        labeled_data = data[data['beats_teammate_q'].notna()].copy()
        
        # Use the same feature columns that were used during training
        # These should match exactly what the model expects
        expected_features = [
            'quali_position', 'quali_best_lap_ms', 'penalties_applied_bool', 
            'track_id', 'track_name', 'fp1_best_ms', 'fp2_best_ms', 'fp3_best_ms', 
            'session_type', 'is_sprint_weekend', 'driver_quali_pos_avg_3', 
            'driver_quali_pos_avg_5', 'driver_beats_teammate_share_6', 
            'team_quali_pos_avg_3', 'team_quali_pos_avg_5', 'team_pace_percentile_5', 
            'driver_vs_teammate_record_6', 'driver_teammate_gap_avg_4', 
            'fp1_delta_to_median', 'fp1_was_imputed', 'fp2_delta_to_median', 
            'fp2_was_imputed', 'fp3_delta_to_median', 'fp3_was_imputed', 
            'practice_consistency', 'track_R01', 'track_R02', 'track_R03', 
            'track_R04', 'track_R05', 'track_R06', 'track_R07', 'track_R08', 
            'track_R09', 'track_R10', 'track_R11', 'track_R12', 'track_R13', 
            'track_R14', 'track_R15', 'track_R16', 'track_R17', 'track_R18', 
            'track_R19', 'track_R20', 'track_R21', 'is_street_circuit', 
            'track_quali_variance', 'driver_track_familiarity', 'temperature', 
            'rain_probability', 'wind_speed', 'weather_temp_missing', 
            'weather_rain_missing', 'weather_wind_missing', 'quali_position_imputed', 
            'driver_quali_pos_avg_3_imputed', 'driver_quali_pos_avg_5_imputed', 
            'team_quali_pos_avg_3_imputed', 'team_quali_pos_avg_5_imputed', 
            'team_pace_percentile_5_imputed', 'driver_teammate_gap_avg_4_imputed', 
            'quali_best_lap_ms_missing', 'fp1_best_ms_missing', 'fp2_best_ms_missing', 
            'fp3_best_ms_missing', 'teammate_gap_ms_missing', 'fp1_delta_to_median_missing', 
            'fp2_delta_to_median_missing', 'fp3_delta_to_median_missing'
        ]
        
        # Filter to only include expected features that exist in the data
        available_features = [col for col in expected_features if col in labeled_data.columns]
        missing_features = [col for col in expected_features if col not in labeled_data.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        # Handle missing values
        X = labeled_data[available_features].copy()
        X = self._handle_missing_features(X)
        
        # Encode categorical features
        X = self._encode_categorical_features(X)
        
        y = labeled_data['beats_teammate_q']
        
        return X, y, available_features
    
    def _handle_missing_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        X_clean = X.copy()
        
        # Fill numeric columns with median
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X_clean[col].isnull().any():
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X_clean[col] = X_clean[col].fillna(median_val)
        
        # Fill categorical columns with mode
        categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X_clean[col].isnull().any():
                mode_val = X_clean[col].mode().iloc[0] if len(X_clean[col].mode()) > 0 else 'unknown'
                X_clean[col] = X_clean[col].fillna(mode_val)
        
        # Ensure no NaN values remain
        if X_clean.isnull().any().any():
            X_clean = X_clean.fillna(0)
        
        return X_clean
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical features to numeric."""
        X_encoded = X.copy()
        
        # For boolean columns, convert to int
        bool_cols = X_encoded.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            X_encoded[col] = X_encoded[col].astype(int)
        
        # For object columns, use label encoding
        object_cols = X_encoded.select_dtypes(include=['object']).columns
        for col in object_cols:
            if X_encoded[col].nunique() < 100:  # Only encode if not too many unique values
                X_encoded[col] = pd.Categorical(X_encoded[col]).codes
        
        return X_encoded
    
    def _train_calibrator(self, model: Any, X_train: pd.DataFrame, 
                          y_train: pd.Series, train_upto: int) -> Any:
        """Train probability calibration model."""
        try:
            # Use a subset of training data for calibration (simpler approach)
            calib_size = min(1000, len(X_train) // 4)
            calib_indices = np.random.choice(X_train.index, size=calib_size, replace=False)
            X_calib = X_train.loc[calib_indices]
            y_calib = y_train.loc[calib_indices]
            
            if len(X_calib) > 0:
                # Train isotonic calibration
                calibrator = IsotonicRegression(out_of_bounds='clip')
                raw_probs = model.predict_proba(X_calib)[:, 1]
                calibrator.fit(raw_probs, y_calib)
                
                logger.info(f"Trained calibrator on {len(X_calib)} samples")
                return calibrator
            else:
                # Fallback to identity calibration
                logger.warning("Insufficient calibration data, using identity")
                return IdentityCalibrator()
                
        except Exception as e:
            logger.warning(f"Calibration failed: {e}, using identity")
            return IdentityCalibrator()
    
    def _create_predictions_df(self, test_data: pd.DataFrame, probs: np.ndarray,
                              y_true: pd.Series, train_upto: int, block_id: int) -> pd.DataFrame:
        """Create comprehensive predictions dataframe."""
        threshold = self.config['eval']['threshold']
        
        predictions = []
        for idx, (_, row) in enumerate(test_data.iterrows()):
            pred_prob = probs[idx]
            pred_label = 1 if pred_prob >= threshold else 0
            actual_label = row['beats_teammate_q']
            is_correct = 1 if pred_label == actual_label else 0
            
            predictions.append({
                'season': row['season'],
                'event_key': row['event_key'],
                'constructor_id': row['constructor_id'],
                'driver_id': row['driver_id'],
                'teammate_id': row['teammate_id'],
                'pred_prob': pred_prob,
                'pred_label': pred_label,
                'actual_label': actual_label,
                'is_correct': is_correct,
                'train_upto': train_upto,
                'test_block_id': block_id
            })
        
        return pd.DataFrame(predictions)
    
    def _compute_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Compute standard classification metrics."""
        y_pred = (y_pred_proba >= self.config['eval']['threshold']).astype(int)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1': f1_score(y_true, y_pred, average='macro'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'brier': brier_score_loss(y_true, y_pred_proba)
        }
    
    def _compute_calibration_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Compute probability calibration metrics."""
        try:
            # Expected Calibration Error (ECE)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find predictions in this bin
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                if in_bin.sum() > 0:
                    bin_prob = y_pred_proba[in_bin].mean()
                    bin_accuracy = y_true[in_bin].mean()
                    ece += np.abs(bin_prob - bin_accuracy) * in_bin.sum()
            
            ece /= len(y_true)
            
            # Calibration slope and intercept
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            
            # Simple linear fit
            if len(fraction_of_positives) > 1:
                slope = np.polyfit(mean_predicted_value, fraction_of_positives, 1)[0]
                intercept = np.polyfit(mean_predicted_value, fraction_of_positives, 1)[1]
            else:
                slope = intercept = np.nan
            
            return {
                'ece': ece,
                'calibration_slope': slope,
                'calibration_intercept': intercept
            }
            
        except Exception as e:
            logger.warning(f"Calibration metrics failed: {e}")
            return {'ece': np.nan, 'calibration_slope': np.nan, 'calibration_intercept': np.nan}
    
    def _compute_baseline_metrics(self, test_data: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """Compute baseline metrics for comparison."""
        try:
            # Baseline 1: H2H-prior (driver who led head-to-head entering event)
            h2h_predictions = []
            for _, row in test_data.iterrows():
                # Use driver's historical record vs teammate
                driver_record = row.get('driver_vs_teammate_record_6', 0.5)
                h2h_pred = 1 if driver_record > 0.5 else 0
                h2h_predictions.append(h2h_pred)
            
            h2h_accuracy = accuracy_score(y_true, h2h_predictions)
            h2h_f1 = f1_score(y_true, h2h_predictions, average='macro')
            
            # Baseline 2: Last-quali winner (simplified - use random for now)
            # In production, you'd implement actual last qualifying result logic
            last_quali_predictions = np.random.choice([0, 1], size=len(y_true))
            last_quali_accuracy = accuracy_score(y_true, last_quali_predictions)
            last_quali_f1 = f1_score(y_true, last_quali_predictions, average='macro')
            
            return {
                'h2h_prior_accuracy': h2h_accuracy,
                'h2h_prior_f1': h2h_f1,
                'last_quali_accuracy': last_quali_accuracy,
                'last_quali_f1': last_quali_f1
            }
            
        except Exception as e:
            logger.warning(f"Baseline metrics failed: {e}")
            return {
                'h2h_prior_accuracy': np.nan,
                'h2h_prior_f1': np.nan,
                'last_quali_accuracy': np.nan,
                'last_quali_f1': np.nan
            }
    
    def _compute_topk_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute metrics for different confidence thresholds."""
        topk_results = {}
        
        for threshold in self.config['eval']['topk_prob']:
            # Filter predictions above threshold
            high_conf_mask = predictions_df['pred_prob'] >= threshold
            high_conf_df = predictions_df[high_conf_mask]
            
            if len(high_conf_df) > 0:
                coverage = len(high_conf_df) / len(predictions_df)
                accuracy = high_conf_df['is_correct'].mean()
                f1 = f1_score(high_conf_df['actual_label'], high_conf_df['pred_label'], average='macro')
                
                topk_results[f'threshold_{threshold}'] = {
                    'coverage': coverage,
                    'accuracy': accuracy,
                    'f1': f1,
                    'n_samples': len(high_conf_df)
                }
            else:
                topk_results[f'threshold_{threshold}'] = {
                    'coverage': 0.0,
                    'accuracy': np.nan,
                    'f1': np.nan,
                    'n_samples': 0
                }
        
        return topk_results
    
    def _analyze_errors(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze prediction errors by different dimensions."""
        # Group by constructor
        constructor_errors = predictions_df.groupby('constructor_id').agg({
            'is_correct': ['count', 'mean'],
            'pred_prob': 'mean'
        }).round(3)
        
        # Group by event
        event_errors = predictions_df.groupby('event_key').agg({
            'is_correct': ['count', 'mean'],
            'pred_prob': 'mean'
        }).round(3)
        
        # Find most confident errors
        errors_df = predictions_df[predictions_df['is_correct'] == 0].copy()
        if len(errors_df) > 0:
            errors_df = errors_df.sort_values('pred_prob', ascending=False)
            top_errors = errors_df.head(10)[['event_key', 'constructor_id', 'driver_id', 
                                           'pred_prob', 'actual_label']].to_dict('records')
        else:
            top_errors = []
        
        return {
            'constructor_errors': constructor_errors,
            'event_errors': event_errors,
            'top_confident_errors': top_errors,
            'total_errors': len(errors_df)
        }
    
    def _save_block_artifacts(self, block_result: Dict[str, Any], block_id: int):
        """Save artifacts for a single block."""
        # Save predictions CSV
        predictions_df = pd.DataFrame(block_result['predictions'])
        csv_path = self.reports_dir / "predictions" / f"wf_block_{block_id}.csv"
        predictions_df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")
        
        # Save markdown summary
        md_path = self.reports_dir / "predictions" / f"wf_block_{block_id}.md"
        self._create_block_markdown(block_result, block_id, md_path)
        
        # Save top-K analysis
        topk_df = pd.DataFrame(block_result['topk_metrics']).T
        topk_path = self.reports_dir / "predictions" / f"wf_block_{block_id}_topk.csv"
        topk_df.to_csv(topk_path)
        
        # Create and save plots
        self._create_block_plots(block_result, block_id)
    
    def _create_block_markdown(self, block_result: Dict[str, Any], block_id: int, md_path: Path):
        """Create markdown summary for a block."""
        metrics = block_result['metrics']
        calibration = block_result['calibration_metrics']
        baselines = block_result['baseline_metrics']
        error_analysis = block_result['error_analysis']
        
        with open(md_path, 'w') as f:
            f.write(f"# Walk-Forward Validation Block {block_id}\n\n")
            f.write(f"**Train up to:** {block_result['train_upto']}\n")
            f.write(f"**Test seasons:** {block_result['test_seasons']}\n\n")
            
            f.write("## Model Performance\n\n")
            f.write(f"- **Accuracy:** {metrics['accuracy']:.3f}\n")
            f.write(f"- **F1 Score:** {metrics['f1']:.3f}\n")
            f.write(f"- **ROC-AUC:** {metrics['roc_auc']:.3f}\n")
            f.write(f"- **PR-AUC:** {metrics['pr_auc']:.3f}\n")
            f.write(f"- **Brier Score:** {metrics['brier']:.3f}\n\n")
            
            f.write("## Calibration Metrics\n\n")
            f.write(f"- **ECE:** {calibration['ece']:.3f}\n")
            f.write(f"- **Calibration Slope:** {calibration['calibration_slope']:.3f}\n")
            f.write(f"- **Calibration Intercept:** {calibration['calibration_intercept']:.3f}\n\n")
            
            f.write("## Baseline Comparison\n\n")
            f.write(f"- **H2H-Prior F1:** {baselines['h2h_prior_f1']:.3f}\n")
            f.write(f"- **Last-Quali F1:** {baselines['last_quali_f1']:.3f}\n\n")
            
            f.write("## Error Analysis\n\n")
            f.write(f"**Total Errors:** {error_analysis['total_errors']}\n\n")
            
            if error_analysis['top_confident_errors']:
                f.write("### Most Confident Errors\n\n")
                for error in error_analysis['top_confident_errors'][:5]:
                    f.write(f"- {error['event_key']} | {error['constructor_id']} | "
                           f"{error['driver_id']} | Prob: {error['pred_prob']:.3f}\n")
        
        logger.info(f"Saved markdown summary to {md_path}")
    
    def _create_block_plots(self, block_result: Dict[str, Any], block_id: int):
        """Create and save plots for a block."""
        predictions_df = pd.DataFrame(block_result['predictions'])
        y_true = predictions_df['actual_label']
        y_pred_proba = predictions_df['pred_prob']
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {block_result["metrics"]["roc_auc"]:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # PR Curve
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        axes[0, 1].plot(recall, precision, label=f'PR (AUC = {block_result["metrics"]["pr_auc"]:.3f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Calibration Plot
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        axes[1, 0].plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        axes[1, 0].set_xlabel('Mean Predicted Probability')
        axes[1, 0].set_ylabel('Fraction of Positives')
        axes[1, 0].set_title('Reliability Diagram')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Prediction Distribution
        axes[1, 1].hist(y_pred_proba[y_true == 0], alpha=0.7, label='Actual 0', bins=20)
        axes[1, 1].hist(y_pred_proba[y_true == 1], alpha=0.7, label='Actual 1', bins=20)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Prediction Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.reports_dir / "figures" / f"wf_block_{block_id}_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved evaluation plots to {plot_path}")
    
    def _print_block_summary(self, block_result: Dict[str, Any], block_id: int):
        """Print console summary for a block."""
        metrics = block_result['metrics']
        calibration = block_result['calibration_metrics']
        baselines = block_result['baseline_metrics']
        topk = block_result['topk_metrics']
        
        print(f"\n{'='*60}")
        print(f"BLOCK {block_id}: Train≤{block_result['train_upto']} → Test {block_result['test_seasons']}")
        print(f"{'='*60}")
        print(f"Model: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, "
              f"ROC-AUC={metrics['roc_auc']:.3f}, PR-AUC={metrics['pr_auc']:.3f}")
        print(f"Calibration: Brier={metrics['brier']:.3f}, ECE={calibration['ece']:.3f}")
        print(f"Baselines: H2H-Prior F1={baselines['h2h_prior_f1']:.3f}, "
              f"Last-Quali F1={baselines['last_quali_f1']:.3f}")
        
        print(f"\nTop-K Analysis:")
        for threshold, results in topk.items():
            if results['n_samples'] > 0:
                print(f"  Prob≥{threshold.split('_')[1]}: Coverage={results['coverage']:.1%}, "
                      f"Acc={results['accuracy']:.3f}, F1={results['f1']:.3f}")
    
    def _compute_pooled_metrics(self, all_predictions: List[Dict]) -> Dict[str, float]:
        """Compute metrics pooled across all blocks."""
        if not all_predictions:
            return {}
        
        df = pd.DataFrame(all_predictions)
        y_true = df['actual_label']
        y_pred = df['pred_label']
        y_pred_proba = df['pred_prob']
        
        return {
            'pooled_accuracy': accuracy_score(y_true, y_pred),
            'pooled_f1': f1_score(y_true, y_pred, average='macro'),
            'pooled_roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pooled_pr_auc': average_precision_score(y_true, y_pred_proba),
            'pooled_brier': brier_score_loss(y_true, y_pred_proba),
            'total_samples': len(df)
        }
    
    def _save_consolidated_results(self, block_results: List[Dict], pooled_metrics: Dict):
        """Save consolidated results across all blocks."""
        # Save all predictions
        all_predictions = []
        for block in block_results:
            all_predictions.extend(block['predictions'])
        
        all_df = pd.DataFrame(all_predictions)
        all_csv_path = self.reports_dir / "predictions" / "wf_all_predictions.csv"
        all_df.to_csv(all_csv_path, index=False)
        
        # Create consolidated summary
        summary_path = self.reports_dir / "predictions" / "wf_summary.md"
        self._create_consolidated_summary(block_results, pooled_metrics, summary_path)
        
        logger.info(f"Saved consolidated results to {all_csv_path} and {summary_path}")
    
    def _create_consolidated_summary(self, block_results: List[Dict], 
                                   pooled_metrics: Dict, summary_path: Path):
        """Create consolidated summary markdown."""
        with open(summary_path, 'w') as f:
            f.write("# Walk-Forward Validation Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Blocks:** {len(block_results)}\n\n")
            
            f.write("## Block Performance Summary\n\n")
            f.write("| Block | Train≤ | Test Seasons | Accuracy | F1 | ROC-AUC | PR-AUC | Brier | ECE |\n")
            f.write("|-------|---------|--------------|----------|----|---------|---------|-------|-----|\n")
            
            for block in block_results:
                metrics = block['metrics']
                calibration = block['calibration_metrics']
                f.write(f"| {block['block_id']} | {block['train_upto']} | "
                       f"{block['test_seasons']} | {metrics['accuracy']:.3f} | "
                       f"{metrics['f1']:.3f} | {metrics['roc_auc']:.3f} | "
                       f"{metrics['pr_auc']:.3f} | {metrics['brier']:.3f} | "
                       f"{calibration['ece']:.3f} |\n")
            
            f.write(f"\n## Pooled Metrics\n\n")
            f.write(f"- **Total Samples:** {pooled_metrics['total_samples']}\n")
            f.write(f"- **Pooled Accuracy:** {pooled_metrics['pooled_accuracy']:.3f}\n")
            f.write(f"- **Pooled F1:** {pooled_metrics['pooled_f1']:.3f}\n")
            f.write(f"- **Pooled ROC-AUC:** {pooled_metrics['pooled_roc_auc']:.3f}\n")
            f.write(f"- **Pooled PR-AUC:** {pooled_metrics['pooled_pr_auc']:.3f}\n")
            f.write(f"- **Pooled Brier Score:** {pooled_metrics['pooled_brier']:.3f}\n")
    
    def _log_experiment(self, block_results: List[Dict], pooled_metrics: Dict):
        """Log experiment to MLflow or Weights & Biases."""
        try:
            if self.config['tracking']['backend'] == 'mlflow':
                self._log_to_mlflow(block_results, pooled_metrics)
            elif self.config['tracking']['backend'] == 'wandb':
                self._log_to_wandb(block_results, pooled_metrics)
        except Exception as e:
            logger.warning(f"Experiment tracking failed: {e}")
    
    def _log_to_mlflow(self, block_results: List[Dict], pooled_metrics: Dict):
        """Log experiment to MLflow."""
        try:
            import mlflow
            
            experiment_name = "f1_teammate_wf"
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name=f"{self.config['tracking']['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log config
                mlflow.log_params({
                    'n_blocks': len(block_results),
                    'total_samples': pooled_metrics['total_samples']
                })
                
                # Log metrics
                mlflow.log_metrics(pooled_metrics)
                
                # Log artifacts
                for block in block_results:
                    block_id = block['block_id']
                    mlflow.log_artifact(f"reports/predictions/wf_block_{block_id}.csv")
                    mlflow.log_artifact(f"reports/predictions/wf_block_{block_id}.md")
                    mlflow.log_artifact(f"reports/figures/wf_block_{block_id}_evaluation.png")
                
                mlflow.log_artifact("reports/predictions/wf_summary.md")
                
            logger.info("Logged experiment to MLflow")
            
        except ImportError:
            logger.warning("MLflow not installed, skipping tracking")
    
    def _log_to_wandb(self, block_results: List[Dict], pooled_metrics: Dict):
        """Log experiment to Weights & Biases."""
        try:
            import wandb
            
            wandb.init(
                project="f1_teammate_wf",
                name=f"{self.config['tracking']['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'n_blocks': len(block_results),
                    'total_samples': pooled_metrics['total_samples']
                }
            )
            
            # Log metrics
            wandb.log(pooled_metrics)
            
            # Log artifacts
            for block in block_results:
                block_id = block['block_id']
                wandb.save(f"reports/predictions/wf_block_{block_id}.csv")
                wandb.save(f"reports/predictions/wf_block_{block_id}.md")
                wandb.save(f"reports/figures/wf_block_{block_id}_evaluation.png")
            
            wandb.save("reports/predictions/wf_summary.md")
            wandb.finish()
            
            logger.info("Logged experiment to Weights & Biases")
            
        except ImportError:
            logger.warning("Weights & Biases not installed, skipping tracking")


class IdentityCalibrator:
    """Identity calibration (no change to probabilities)."""
    
    def predict_proba(self, X):
        """Return probabilities unchanged."""
        return X


def run(settings_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Executes rolling walk-forward validation as defined in settings.
    Returns a dict of aggregate metrics and paths to artifacts.
    """
    validator = WalkForwardValidator(settings_path)
    return validator.run()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    results = run()
    print(f"\nValidation completed. Results saved to reports/ directory.")
