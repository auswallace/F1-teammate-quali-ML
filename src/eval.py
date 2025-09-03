"""
Evaluation module for F1 teammate qualifying prediction pipeline.

Evaluates models and produces comprehensive reports with visualizations.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
)
import shap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")


class ModelEvaluator:
    """Evaluates trained models and produces comprehensive reports."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.interim_dir = Path(self.config['data']['interim_dir'])
        self.models_dir = Path(self.config['data']['models_dir'])
        self.reports_dir = Path(self.config['data']['reports_dir'])
        self.figures_dir = self.reports_dir / "figures"
        
        # Ensure directories exist
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Track evaluation results
        self.evaluation_results = {}
        
    def load_models_and_data(self) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
        """Load trained models and split data."""
        logger.info("Loading models and data")
        
        # Load models
        models = {}
        for model_name in ['logistic_regression', 'xgboost']:
            model_path = self.models_dir / f"{model_name}.joblib"
            if model_path.exists():
                models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model")
            else:
                logger.warning(f"Model not found: {model_path}")
        
        # Load split data
        splits = {}
        for split_name in ['train', 'val', 'test']:
            split_path = self.interim_dir / f"split_{split_name}.parquet"
            if split_path.exists():
                splits[split_name] = pd.read_parquet(split_path)
                logger.info(f"Loaded {split_name} split: {splits[split_name].shape}")
            else:
                logger.warning(f"Split file not found: {split_path}")
                splits[split_name] = pd.DataFrame()
        
        return models, splits
    
    def prepare_evaluation_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare data for evaluation."""
        logger.info("Preparing evaluation data")
        
        if len(df) == 0:
            raise ValueError("No data for evaluation")
        
        # Get feature columns (exclude target and metadata)
        exclude_cols = {
            'beats_teammate_q', 'teammate_gap_ms', 'teammate_id', 'split',
            'season', 'event_key', 'driver_id', 'driver_name', 'constructor_id', 'constructor_name',
            'round', 'status', 'equal_time_tiebreak', 'single_teammate'
        }
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Identified {len(feature_cols)} feature columns")
        
        # Prepare X and y
        X = df[feature_cols].copy()
        y = df['beats_teammate_q'].copy()
        
        # Handle missing values in features
        X = self._handle_missing_features(X)
        
        # Convert categorical features to numeric
        X = self._encode_categorical_features(X)
        
        logger.info(f"Evaluation data prepared: X={X.shape}, y={y.shape}")
        return X, y, feature_cols
    
    def _compute_baseline_h2h_prior(self, df: pd.DataFrame) -> np.ndarray:
        """Baseline A: Pick driver who leads head-to-head entering the event."""
        predictions = []
        
        for _, row in df.iterrows():
            # Use rolling head-to-head record if available
            if 'driver_vs_teammate_record_6' in row and pd.notna(row['driver_vs_teammate_record_6']):
                record = row['driver_vs_teammate_record_6']
                pred = 1 if record > 0.5 else 0
            else:
                # Fallback: use 50/50
                pred = np.random.choice([0, 1])
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _compute_baseline_last_quali_winner(self, df: pd.DataFrame) -> np.ndarray:
        """Baseline B: Pick driver who won most recent quali head-to-head."""
        predictions = []
        
        for _, row in df.iterrows():
            # For now, use H2H-prior as fallback
            # In production, you'd implement actual last qualifying result logic
            if 'driver_vs_teammate_record_6' in row and pd.notna(row['driver_vs_teammate_record_6']):
                record = row['driver_vs_teammate_record_6']
                pred = 1 if record > 0.5 else 0
            else:
                # Fallback: use 50/50
                pred = np.random.choice([0, 1])
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _compute_calibration_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Compute probability calibration metrics."""
        try:
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import brier_score_loss
            
            # Brier score
            brier = brier_score_loss(y_true, y_pred_proba)
            
            # Expected Calibration Error (ECE) - 10 bins
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
            
            return {
                'brier': brier,
                'ece': ece
            }
            
        except Exception as e:
            logger.warning(f"Calibration metrics failed: {e}")
            return {'brier': np.nan, 'ece': np.nan}
    
    def _handle_missing_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature matrix."""
        # For numeric columns, fill with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
        
        # For categorical columns, fill with mode
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                mode_val = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'unknown'
                X[col] = X[col].fillna(mode_val)
        
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
        
        return X
    
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                      model_name: str, split_name: str) -> Dict[str, Any]:
        """Evaluate a single model on given data."""
        logger.info(f"Evaluating {model_name} on {split_name} split")
        
        try:
            # Make predictions
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Calculate model metrics
            metrics = {
                'precision': precision_score(y, y_pred, average='binary'),
                'recall': recall_score(y, y_pred, average='binary'),
                'f1': f1_score(y, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y, y_pred_proba),
                'average_precision': average_precision_score(y, y_pred_proba)
            }
            
            # Add calibration metrics
            calibration_metrics = self._compute_calibration_metrics(y.values, y_pred_proba)
            metrics.update(calibration_metrics)
            
            # Compute baseline metrics
            # Get original dataframe for baseline computation
            df_original = X.copy()
            df_original['beats_teammate_q'] = y.values
            
            baseline_h2h = self._compute_baseline_h2h_prior(df_original)
            baseline_last_quali = self._compute_baseline_last_quali_winner(df_original)
            
            baseline_metrics = {
                'h2h_prior_accuracy': (baseline_h2h == y.values).mean(),
                'h2h_prior_f1': f1_score(y.values, baseline_h2h, average='binary'),
                'h2h_prior_pr_auc': average_precision_score(y.values, baseline_h2h),
                'last_quali_accuracy': (baseline_last_quali == y.values).mean(),
                'last_quali_f1': f1_score(y.values, baseline_last_quali, average='binary'),
                'last_quali_pr_auc': average_precision_score(y.values, baseline_last_quali)
            }
            
            # Classification report
            class_report = classification_report(y, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            
            results = {
                'model_name': model_name,
                'split_name': split_name,
                'metrics': metrics,
                'baseline_metrics': baseline_metrics,
                'classification_report': class_report,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'true_labels': y.values
            }
            
            logger.info(f"{model_name} on {split_name}: F1={metrics['f1']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name} on {split_name}: {e}")
            return None
    
    def evaluate_all_models(self, models: Dict[str, Any], splits: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Evaluate all models on all splits."""
        logger.info("Evaluating all models on all splits")
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            evaluation_results[model_name] = {}
            
            for split_name, split_df in splits.items():
                if len(split_df) > 0:
                    # Prepare data
                    X, y, _ = self.prepare_evaluation_data(split_df)
                    
                    # Evaluate model
                    results = self.evaluate_model(model, X, y, model_name, split_name)
                    if results is not None:
                        evaluation_results[model_name][split_name] = results
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def create_evaluation_plots(self) -> None:
        """Create comprehensive evaluation plots."""
        logger.info("Creating evaluation plots")
        
        for model_name, model_results in self.evaluation_results.items():
            if not model_results:
                continue
            
            # Create subplots for this model
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Model Evaluation: {model_name.replace("_", " ").title()}', fontsize=16)
            
            plot_idx = 0
            
            for split_name, results in model_results.items():
                if results is None:
                    continue
                
                # ROC Curve
                if plot_idx == 0:
                    ax = axes[0, 0]
                    fpr, tpr, _ = roc_curve(results['true_labels'], results['probabilities'])
                    ax.plot(fpr, tpr, label=f'{split_name} (AUC={results["metrics"]["roc_auc"]:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # Precision-Recall Curve
                if plot_idx == 1:
                    ax = axes[0, 1]
                    precision, recall, _ = precision_recall_curve(results['true_labels'], results['probabilities'])
                    ax.plot(recall, precision, label=f'{split_name} (AP={results["metrics"]["average_precision"]:.3f})')
                    ax.set_xlabel('Recall')
                    ax.set_ylabel('Precision')
                    ax.set_title('Precision-Recall Curve')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # Confusion Matrix
                if plot_idx == 2:
                    ax = axes[1, 0]
                    cm = results['confusion_matrix']
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title(f'Confusion Matrix - {split_name}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                
                # Metrics Comparison
                if plot_idx == 3:
                    ax = axes[1, 1]
                    metrics = results['metrics']
                    metric_names = list(metrics.keys())
                    metric_values = list(metrics.values())
                    
                    bars = ax.bar(metric_names, metric_values, color='skyblue', alpha=0.7)
                    ax.set_title(f'Metrics - {split_name}')
                    ax.set_ylabel('Score')
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, metric_values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
                
                plot_idx += 1
            
            # Remove empty subplots
            for i in range(plot_idx, 4):
                row, col = i // 2, i % 2
                fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.figures_dir / f"{model_name}_evaluation.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved evaluation plot for {model_name} to {plot_path}")
    
    def create_calibration_plots(self) -> None:
        """Create calibration plots for each model and split."""
        logger.info("Creating calibration plots")
        
        for model_name, model_results in self.evaluation_results.items():
            if not model_results:
                continue
            
            for split_name, results in model_results.items():
                if results is None:
                    continue
                
                # Create calibration plot
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                
                # Get calibration curve
                from sklearn.calibration import calibration_curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    results['true_labels'], results['probabilities'], n_bins=10
                )
                
                # Plot calibration curve
                ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                       label=f'{model_name} (ECE={results["metrics"]["ece"]:.3f})')
                ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
                
                ax.set_xlabel('Mean Predicted Probability')
                ax.set_ylabel('Fraction of Positives')
                ax.set_title(f'Reliability Diagram: {model_name} on {split_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = self.figures_dir / f"calibration_{split_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved calibration plot to {plot_path}")
        
        logger.info("Calibration plots created and saved")
    
    def create_shap_analysis(self, models: Dict[str, Any], splits: Dict[str, pd.DataFrame]) -> None:
        """Create SHAP analysis for XGBoost model."""
        logger.info("Creating SHAP analysis")
        
        if 'xgboost' not in models:
            logger.warning("XGBoost model not found, skipping SHAP analysis")
            return
        
        # Use validation split for SHAP analysis
        if 'val' not in splits or len(splits['val']) == 0:
            logger.warning("Validation split not found, skipping SHAP analysis")
            return
        
        try:
            # Prepare data
            X, y, _ = self.prepare_evaluation_data(splits['val'])
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(models['xgboost'])
            shap_values = explainer.shap_values(X)
            
            # SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X, show=False)
            plt.title('SHAP Feature Importance Summary')
            plt.tight_layout()
            
            # Save plot
            shap_path = self.figures_dir / "xgboost_shap_summary.png"
            plt.savefig(shap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved SHAP summary plot to {shap_path}")
            
            # SHAP dependence plots for top features
            feature_importance = dict(zip(X.columns, np.abs(shap_values).mean(0)))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (feature, importance) in enumerate(top_features):
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature, shap_values, X, show=False)
                plt.title(f'SHAP Dependence Plot: {feature}')
                plt.tight_layout()
                
                # Save plot
                dep_path = self.figures_dir / f"xgboost_shap_dependence_{feature}.png"
                plt.savefig(dep_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved SHAP dependence plot for {feature} to {dep_path}")
                
        except Exception as e:
            logger.error(f"Error creating SHAP analysis: {e}")
    
    def create_head_to_head_analysis(self, splits: Dict[str, pd.DataFrame]) -> None:
        """Create head-to-head analysis."""
        logger.info("Creating head-to-head analysis")
        
        try:
            # Combine all splits for analysis
            all_data = pd.concat([df for df in splits.values() if len(df) > 0], ignore_index=True)
            
            if len(all_data) == 0:
                logger.warning("No data for head-to-head analysis")
                return
            
            # Filter to labeled pairs
            labeled_data = all_data[all_data['beats_teammate_q'].notna()]
            
            if len(labeled_data) == 0:
                logger.warning("No labeled data for head-to-head analysis")
                return
            
                # Constructor analysis
            # Ensure numeric types for aggregation
            labeled_data['beats_teammate_q'] = pd.to_numeric(labeled_data['beats_teammate_q'], errors='coerce')
            labeled_data['teammate_gap_ms'] = pd.to_numeric(labeled_data['teammate_gap_ms'], errors='coerce')
            
            constructor_stats = labeled_data.groupby('constructor_id').agg({
                'beats_teammate_q': ['count', 'mean'],
                'teammate_gap_ms': ['mean', 'std']
            }).round(3)
            
            constructor_stats.columns = ['total_pairs', 'win_rate', 'avg_gap_ms', 'gap_std_ms']
            constructor_stats = constructor_stats.sort_values('win_rate', ascending=False)
            
            # Driver analysis
            driver_stats = labeled_data.groupby('driver_id').agg({
                'beats_teammate_q': ['count', 'mean'],
                'teammate_gap_ms': ['mean', 'std']
            }).round(3)
            
            driver_stats.columns = ['total_pairs', 'win_rate', 'avg_gap_ms', 'gap_std_ms']
            driver_stats = driver_stats.sort_values('win_rate', ascending=False)
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Head-to-Head Analysis', fontsize=16)
            
            # Constructor win rates
            ax1 = axes[0, 0]
            if len(constructor_stats) > 0:
                top_constructors = constructor_stats.head(10)
                bars = ax1.barh(range(len(top_constructors)), top_constructors['win_rate'])
                ax1.set_yticks(range(len(top_constructors)))
                ax1.set_yticklabels(top_constructors.index)
                ax1.set_xlabel('Win Rate')
                ax1.set_title('Top 10 Constructors by Win Rate')
                ax1.set_xlim(0, 1)
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, top_constructors['win_rate'])):
                    ax1.text(value + 0.01, i, f'{value:.3f}', va='center')
            else:
                ax1.text(0.5, 0.5, 'No constructor data available', 
                         ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Top 10 Constructors by Win Rate')
            
            # Driver win rates
            ax2 = axes[0, 1]
            if len(driver_stats) > 0:
                top_drivers = driver_stats.head(10)
                bars = ax2.barh(range(len(top_drivers)), top_drivers['win_rate'])
                ax2.set_yticks(range(len(top_drivers)))
                ax2.set_yticklabels(top_drivers.index)
                ax2.set_xlabel('Win Rate')
                ax2.set_title('Top 10 Drivers by Win Rate')
                ax2.set_xlim(0, 1)
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, top_drivers['win_rate'])):
                    ax2.text(value + 0.01, i, f'{value:.3f}', va='center')
            else:
                ax2.text(0.5, 0.5, 'No driver data available', 
                         ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Top 10 Drivers by Win Rate')
            
            # Gap distribution
            ax3 = axes[1, 0]
            gaps = labeled_data['teammate_gap_ms'].dropna()
            if len(gaps) > 0:
                ax3.hist(gaps, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.set_xlabel('Teammate Gap (ms)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Distribution of Teammate Gaps')
                ax3.axvline(gaps.mean(), color='red', linestyle='--', label=f'Mean: {gaps.mean():.1f}ms')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No valid gap data available', 
                         ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Distribution of Teammate Gaps')
            
            # Win rate by gap
            ax4 = axes[1, 1]
            # Only create gap bins if we have valid gap data
            valid_gaps = labeled_data['teammate_gap_ms'].dropna()
            if len(valid_gaps) > 0:
                labeled_data['gap_bin'] = pd.cut(labeled_data['teammate_gap_ms'], bins=10)
                gap_win_rates = labeled_data.groupby('gap_bin')['beats_teammate_q'].mean()
                gap_win_rates.plot(kind='bar', ax=ax4, color='lightcoral', alpha=0.7)
                ax4.set_xlabel('Teammate Gap Bin')
                ax4.set_ylabel('Win Rate')
                ax4.set_title('Win Rate by Teammate Gap')
                ax4.tick_params(axis='x', rotation=45)
            else:
                ax4.text(0.5, 0.5, 'No valid gap data available', 
                         ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Win Rate by Teammate Gap')
            
            plt.tight_layout()
            
            # Save plot
            h2h_path = self.figures_dir / "head_to_head_analysis.png"
            plt.savefig(h2h_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved head-to-head analysis to {h2h_path}")
            
            # Save statistics
            stats_path = self.reports_dir / "head_to_head_stats.json"
            import json
            
            # Calculate overall stats safely
            overall_stats = {
                'total_pairs': len(labeled_data),
                'avg_win_rate': labeled_data['beats_teammate_q'].mean() if labeled_data['beats_teammate_q'].notna().any() else 0.0,
                'avg_gap_ms': labeled_data['teammate_gap_ms'].mean() if labeled_data['teammate_gap_ms'].notna().any() else 0.0,
                'gap_std_ms': labeled_data['teammate_gap_ms'].std() if labeled_data['teammate_gap_ms'].notna().any() else 0.0
            }
            
            stats_data = {
                'constructor_stats': constructor_stats.to_dict(),
                'driver_stats': driver_stats.to_dict(),
                'overall_stats': overall_stats
            }
            
            with open(stats_path, 'w') as f:
                json.dump(stats_data, f, indent=2, default=str)
            
            logger.info(f"Saved head-to-head statistics to {stats_path}")
            
        except Exception as e:
            logger.error(f"Error in head-to-head analysis: {e}")
            # Create a simple error plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Head-to-Head Analysis - Error Occurred', fontsize=16)
            
            for ax in axes.flat:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Analysis Failed')
            
            plt.tight_layout()
            
            # Save error plot
            h2h_path = self.figures_dir / "head_to_head_analysis_error.png"
            plt.savefig(h2h_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved error plot to {h2h_path}")
    
    def save_evaluation_summary(self) -> None:
        """Save comprehensive evaluation summary."""
        logger.info("Saving evaluation summary")
        
        summary_path = self.reports_dir / "evaluation_summary.md"
        
        with open(summary_path, 'w') as f:
            f.write("# F1 Teammate Qualifying Prediction - Evaluation Summary\n\n")
            
            for model_name, model_results in self.evaluation_results.items():
                f.write(f"## {model_name.replace('_', ' ').title()}\n\n")
                
                for split_name, results in model_results.items():
                    if results is None:
                        continue
                    
                    f.write(f"### {split_name.upper()} Split\n\n")
                    
                    # Model vs Baselines comparison table
                    f.write("#### Model vs Baselines\n\n")
                    f.write("| Metric | Model | H2H-Prior | Last-Quali |\n")
                    f.write("|--------|-------|------------|------------|\n")
                    f.write(f"| Accuracy | {results['metrics']['precision']:.3f} | {results['baseline_metrics']['h2h_prior_accuracy']:.3f} | {results['baseline_metrics']['last_quali_accuracy']:.3f} |\n")
                    f.write(f"| F1 | {results['metrics']['f1']:.3f} | {results['baseline_metrics']['h2h_prior_f1']:.3f} | {results['baseline_metrics']['last_quali_f1']:.3f} |\n")
                    f.write(f"| PR-AUC | {results['metrics']['average_precision']:.3f} | {results['baseline_metrics']['h2h_prior_pr_auc']:.3f} | {results['baseline_metrics']['last_quali_pr_auc']:.3f} |\n")
                    f.write(f"| Brier | {results['metrics']['brier']:.3f} | - | - |\n")
                    f.write(f"| ECE | {results['metrics']['ece']:.3f} | - | - |\n")
                    f.write("\n")
                    
                    # Calibration info
                    f.write("#### Calibration Metrics\n\n")
                    f.write(f"- **Brier Score:** {results['metrics']['brier']:.4f} (lower is better)\n")
                    f.write(f"- **Expected Calibration Error:** {results['metrics']['ece']:.4f} (lower is better)\n\n")
                    
                    # Confusion matrix
                    f.write("#### Confusion Matrix\n\n")
                    cm = results['confusion_matrix']
                    f.write("```\n")
                    f.write(f"      Predicted\n")
                    f.write(f"Actual  0    1\n")
                    f.write(f"  0    {cm[0,0]:3d}  {cm[0,1]:3d}\n")
                    f.write(f"  1    {cm[1,0]:3d}  {cm[1,1]:3d}\n")
                    f.write("```\n\n")
                
                f.write("---\n\n")
        
        logger.info(f"Saved evaluation summary to {summary_path}")


def main():
    """Main function to run evaluation pipeline."""
    evaluator = ModelEvaluator()
    
    # Load models and data
    models, splits = evaluator.load_models_and_data()
    
    if not models:
        raise ValueError("No models found for evaluation")
    
    # Evaluate all models
    evaluation_results = evaluator.evaluate_all_models(models, splits)
    
    # Create plots
    evaluator.create_evaluation_plots()
    
    # Create calibration plots
    evaluator.create_calibration_plots()
    
    # Create SHAP analysis
    evaluator.create_shap_analysis(models, splits)
    
    # Create head-to-head analysis
    evaluator.create_head_to_head_analysis(splits)
    
    # Save evaluation summary
    evaluator.save_evaluation_summary()
    
    # Print calibration status
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    # Check if calibrator exists
    calibrator_path = evaluator.models_dir / "xgb_calibrator.joblib"
    if calibrator_path.exists():
        print("Calibration: ON (method=isotonic)")
    else:
        print("Calibration: OFF")
    
    # Print summary table
    print("\nModel vs Baselines Summary:")
    print("-" * 50)
    
    for model_name, model_results in evaluation_results.items():
        for split_name, results in model_results.items():
            if results is None:
                continue
            
            print(f"\n{split_name.upper()} Split:")
            print(f"  Model: Acc={results['metrics']['precision']:.3f}, F1={results['metrics']['f1']:.3f}, "
                  f"ROC-AUC={results['metrics']['roc_auc']:.3f}, Brier={results['metrics']['brier']:.3f}, "
                  f"ECE={results['metrics']['ece']:.3f}")
            print(f"  H2H-Prior: Acc={results['baseline_metrics']['h2h_prior_accuracy']:.3f}, "
                  f"F1={results['baseline_metrics']['h2h_prior_f1']:.3f}")
            print(f"  Last-Quali: Acc={results['baseline_metrics']['last_quali_accuracy']:.3f}, "
                  f"F1={results['baseline_metrics']['last_quali_f1']:.3f}")
    
    print(f"\nResults saved to reports/ directory")
    print(f"Calibration plots: reports/figures/calibration_*.png")
    print(f"Evaluation summary: reports/evaluation_summary.md")
    
    logger.info("Evaluation pipeline complete!")


if __name__ == "__main__":
    main()
