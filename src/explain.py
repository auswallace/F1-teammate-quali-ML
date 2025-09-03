"""
Local explanations for F1 teammate qualifying predictions.

This module provides interpretability for the ML model by computing SHAP values
or permutation importance for individual driver predictions.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import SHAP, fallback to permutation importance if it fails
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


class TeammateExplainer:
    """Generate local explanations for teammate qualifying predictions."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the explainer with configuration and models."""
        self.config = self._load_config(config_path)
        self.models_dir = Path(self.config.get('models', {}).get('models_dir', 'models'))
        self.reports_dir = Path(self.config.get('reports', {}).get('reports_dir', 'reports'))
        self.data_dir = Path(self.config.get('data', {}).get('processed_dir', 'data/processed'))
        
        # Create output directories
        (self.reports_dir / "explanations").mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "figures").mkdir(parents=True, exist_ok=True)
        
        # Load models and data
        self.model = None
        self.calibrator = None
        self.feature_columns = None
        self.schema = None
        self._load_models_and_data()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def _load_models_and_data(self):
        """Load trained models, calibrator, and feature schema."""
        try:
            # Load XGBoost model
            xgb_path = self.models_dir / "xgboost.joblib"
            if xgb_path.exists():
                self.model = joblib.load(xgb_path)
                logger.info("Loaded XGBoost model")
            else:
                raise FileNotFoundError("XGBoost model not found")
            
            # Load calibrator if available
            calibrator_path = self.models_dir / "xgb_calibrator.joblib"
            if calibrator_path.exists():
                self.calibrator = joblib.load(calibrator_path)
                logger.info("Loaded probability calibrator")
            
            # Load feature schema
            schema_path = self.data_dir / "schema.json"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    import json
                    self.schema = json.load(f)
                    self.feature_columns = self.schema.get('feature_columns', [])
                    logger.info(f"Loaded schema with {len(self.feature_columns)} features")
            else:
                # Fallback: try to infer from processed data
                processed_path = self.data_dir / "teammate_qual.parquet"
                if processed_path.exists():
                    df = pd.read_parquet(processed_path)
                    # Exclude target and metadata columns
                    exclude_cols = ['beats_teammate_q', 'teammate_gap_ms', 'season', 'event_key', 
                                  'driver_id', 'driver_name', 'constructor_id', 'constructor_name']
                    self.feature_columns = [col for col in df.columns if col not in exclude_cols]
                    logger.info(f"Inferred {len(self.feature_columns)} features from processed data")
                else:
                    raise FileNotFoundError("Neither schema.json nor processed data found")
                    
        except Exception as e:
            logger.error(f"Failed to load models and data: {e}")
            raise
    
    def explain_event_team(self, event_key: str, team: Optional[str] = None) -> Dict[str, Any]:
        """Generate explanations for a specific event and team."""
        logger.info(f"Generating explanations for event {event_key}, team {team or 'all'}")
        
        try:
            # Load processed data for the event
            processed_path = self.data_dir / "teammate_qual.parquet"
            if not processed_path.exists():
                raise FileNotFoundError(f"Processed data not found: {processed_path}")
            
            df_processed = pd.read_parquet(processed_path)
            event_data = df_processed[df_processed['event_key'] == event_key].copy()
            
            if len(event_data) == 0:
                raise ValueError(f"No data found for event: {event_key}")
            
            # Filter by team if specified
            if team and team != 'auto':
                event_data = event_data[event_data['constructor_id'] == team].copy()
                if len(event_data) == 0:
                    raise ValueError(f"No data found for team {team} in event {event_key}")
            
            # Get unique teams in the event
            teams = event_data['constructor_id'].unique()
            logger.info(f"Found {len(teams)} teams in event {event_key}")
            
            results = {}
            
            for constructor_id in teams:
                team_data = event_data[event_data['constructor_id'] == constructor_id].copy()
                if len(team_data) != 2:
                    logger.warning(f"Team {constructor_id} has {len(team_data)} drivers, skipping")
                    continue
                
                team_results = self._explain_team_pair(team_data, event_key, constructor_id)
                results[constructor_id] = team_results
            
            # Generate outputs
            self._save_explanations(event_key, team, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to explain event {event_key}: {e}")
            raise
    
    def _explain_team_pair(self, team_data: pd.DataFrame, event_key: str, constructor_id: str) -> Dict[str, Any]:
        """Generate explanations for a specific team pair."""
        logger.info(f"Explaining team {constructor_id}")
        
        # Prepare features
        X = team_data[self.feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Get predictions
        if self.calibrator and 'calibrator' in self.calibrator:
            raw_probs = self.model.predict_proba(X)[:, 1]
            calibrated_probs = self.calibrator['calibrator'].predict(raw_probs)
            logger.info("Applied calibrated probabilities")
        else:
            calibrated_probs = self.model.predict_proba(X)[:, 1]
            logger.info("Used raw model probabilities")
        
        # Get actual results if available
        actual_results = {}
        if 'beats_teammate_q' in team_data.columns:
            for _, row in team_data.iterrows():
                actual_results[row['driver_id']] = {
                    'beats_teammate': row['beats_teammate_q'],
                    'gap_ms': row.get('teammate_gap_ms', np.nan)
                }
        
        # Generate explanations for each driver
        driver_explanations = {}
        
        for i, (_, driver) in enumerate(team_data.iterrows()):
            driver_id = driver['driver_id']
            driver_name = driver['driver_name']
            prob = calibrated_probs[i]
            
            logger.info(f"Explaining {driver_name} (prob: {prob:.3f})")
            
            # Generate SHAP or permutation importance
            if SHAP_AVAILABLE:
                explanation = self._generate_shap_explanation(X.iloc[i:i+1], driver_name, prob)
            else:
                explanation = self._generate_permutation_explanation(X.iloc[i:i+1], driver_name, prob)
            
            # Add metadata
            explanation.update({
                'driver_id': driver_id,
                'driver_name': driver_name,
                'predicted_probability': prob,
                'actual_beats_teammate': actual_results.get(driver_id, {}).get('beats_teammate', None),
                'actual_gap_ms': actual_results.get(driver_id, {}).get('gap_ms', None)
            })
            
            driver_explanations[driver_id] = explanation
        
        return {
            'constructor_id': constructor_id,
            'constructor_name': team_data.iloc[0]['constructor_name'],
            'drivers': driver_explanations,
            'event_key': event_key
        }
    
    def _generate_shap_explanation(self, X: pd.DataFrame, driver_name: str, prob: float) -> Dict[str, Any]:
        """Generate SHAP explanation for a driver."""
        try:
            # Use TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            
            # Get feature contributions
            feature_contribs = list(zip(self.feature_columns, shap_values[0]))
            
            # Sort by absolute contribution
            feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Get top positive and negative factors
            top_positive = [f for f, v in feature_contribs if v > 0][:6]
            top_negative = [f for f, v in feature_contribs if v < 0][:6]
            
            return {
                'method': 'shap',
                'feature_contributions': feature_contribs,
                'top_positive_factors': top_positive,
                'top_negative_factors': top_negative,
                'shap_values': shap_values[0].tolist()
            }
            
        except Exception as e:
            logger.warning(f"SHAP failed for {driver_name}: {e}, falling back to permutation importance")
            return self._generate_permutation_explanation(X, driver_name, prob)
    
    def _generate_permutation_explanation(self, X: pd.DataFrame, driver_name: str, prob: float) -> Dict[str, Any]:
        """Generate permutation importance explanation for a driver."""
        try:
            # Use permutation importance
            from sklearn.inspection import permutation_importance
            
            # Create a simple scoring function
            def score_func(X):
                return self.model.predict_proba(X)[:, 1]
            
            # Compute permutation importance
            result = permutation_importance(
                self.model, X, None, 
                n_repeats=10, 
                random_state=42,
                scoring=score_func
            )
            
            # Get feature importances
            feature_importances = list(zip(self.feature_columns, result.importances_mean))
            feature_importances.sort(key=lambda x: x[1], reverse=True)
            
            # Get top features
            top_features = [f for f, v in feature_importances[:12]]
            
            return {
                'method': 'permutation_importance',
                'feature_importances': feature_importances,
                'top_features': top_features,
                'permutation_scores': result.importances_mean.tolist()
            }
            
        except Exception as e:
            logger.error(f"Permutation importance failed for {driver_name}: {e}")
            return {
                'method': 'failed',
                'error': str(e),
                'feature_contributions': [],
                'top_positive_factors': [],
                'top_negative_factors': []
            }
    
    def _save_explanations(self, event_key: str, team: Optional[str], results: Dict[str, Any]):
        """Save explanation outputs to files."""
        team_suffix = f"_{team}" if team and team != 'auto' else ""
        
        # Save CSV table
        csv_data = []
        for constructor_id, team_result in results.items():
            for driver_id, driver_expl in team_result['drivers'].items():
                if driver_expl['method'] == 'shap':
                    for feature, contrib in driver_expl['feature_contributions']:
                        csv_data.append({
                            'team': constructor_id,
                            'driver': driver_expl['driver_name'],
                            'feature': feature,
                            'value': driver_expl.get('feature_values', {}).get(feature, 'N/A'),
                            'shap_contrib': contrib,
                            'method': 'shap'
                        })
                elif driver_expl['method'] == 'permutation_importance':
                    for feature, importance in driver_expl['feature_importances']:
                        csv_data.append({
                            'team': constructor_id,
                            'driver': driver_expl['driver_name'],
                            'feature': feature,
                            'value': driver_expl.get('feature_values', {}).get(feature, 'N/A'),
                            'shap_contrib': importance,
                            'method': 'permutation_importance'
                        })
        
        if csv_data:
            csv_df = pd.DataFrame(csv_data)
            csv_path = self.reports_dir / "explanations" / f"{event_key}{team_suffix}_table.csv"
            csv_df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV table to {csv_path}")
        
        # Save waterfall plots
        for constructor_id, team_result in results.items():
            for driver_id, driver_expl in team_result['drivers'].items():
                if driver_expl['method'] == 'shap':
                    self._save_waterfall_plot(driver_expl, event_key, constructor_id, driver_expl['driver_name'])
        
        # Save markdown summary
        self._save_markdown_summary(event_key, team, results)
    
    def _save_waterfall_plot(self, driver_expl: Dict, event_key: str, team: str, driver_name: str):
        """Save waterfall plot for SHAP values."""
        try:
            if driver_expl['method'] != 'shap':
                return
            
            # Get top features (positive and negative)
            feature_contribs = driver_expl['feature_contributions']
            
            # Take top 12 features by absolute contribution
            top_features = feature_contribs[:12]
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            features = [f for f, _ in top_features]
            values = [v for _, v in top_features]
            colors = ['red' if v < 0 else 'blue' for v in values]
            
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('SHAP Value')
            ax.set_title(f'SHAP Values for {driver_name} ({team}) - {event_key}')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                       va='center', ha='left' if value >= 0 else 'right')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.reports_dir / "figures" / f"{event_key}_{team}_{driver_name}_waterfall.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved waterfall plot to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save waterfall plot for {driver_name}: {e}")
    
    def _save_markdown_summary(self, event_key: str, team: Optional[str], results: Dict[str, Any]):
        """Save human-readable markdown summary."""
        team_suffix = f"_{team}" if team and team != 'auto' else ""
        
        md_content = f"# Explanation Summary for {event_key}{team_suffix}\n\n"
        
        for constructor_id, team_result in results.items():
            md_content += f"## {team_result['constructor_name']} ({constructor_id})\n\n"
            
            for driver_id, driver_expl in team_result['drivers'].items():
                driver_name = driver_expl['driver_name']
                prob = driver_expl['predicted_probability']
                
                md_content += f"### {driver_name}\n\n"
                md_content += f"- **Model probability wins**: {prob:.3f} ({prob:.1%})\n"
                
                if driver_expl['method'] == 'shap':
                    md_content += f"- **Explanation method**: SHAP values\n"
                    
                    if driver_expl['top_positive_factors']:
                        md_content += f"- **Top positive factors**: {', '.join(driver_expl['top_positive_factors'][:5])}\n"
                    
                    if driver_expl['top_negative_factors']:
                        md_content += f"- **Top negative factors**: {', '.join(driver_expl['top_negative_factors'][:5])}\n"
                        
                elif driver_expl['method'] == 'permutation_importance':
                    md_content += f"- **Explanation method**: Permutation importance\n"
                    if driver_expl['top_features']:
                        md_content += f"- **Most important features**: {', '.join(driver_expl['top_features'][:5])}\n"
                
                # Add actual results if available
                if driver_expl['actual_beats_teammate'] is not None:
                    actual_result = "won" if driver_expl['actual_beats_teammate'] == 1 else "lost"
                    md_content += f"- **Actual result**: {actual_result}\n"
                    
                    if not pd.isna(driver_expl['actual_gap_ms']):
                        gap_sec = driver_expl['actual_gap_ms'] / 1000
                        md_content += f"- **Gap to teammate**: {gap_sec:.3f}s\n"
                
                md_content += "\n"
        
        # Save markdown file
        md_path = self.reports_dir / "explanations" / f"{event_key}{team_suffix}.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        logger.info(f"Saved markdown summary to {md_path}")


def run(event_key: str, team: Optional[str] = None) -> Dict[str, Any]:
    """Main function to run explanations for an event and team."""
    try:
        explainer = TeammateExplainer()
        results = explainer.explain_event_team(event_key, team)
        
        # Return paths to generated artifacts
        team_suffix = f"_{team}" if team and team != 'auto' else ""
        
        artifacts = {
            'csv_table': f"reports/explanations/{event_key}{team_suffix}_table.csv",
            'markdown_summary': f"reports/explanations/{event_key}{team_suffix}.md",
            'figures_dir': f"reports/figures/{event_key}{team_suffix}_*_waterfall.png"
        }
        
        logger.info("Explanation generation completed successfully")
        return {'success': True, 'results': results, 'artifacts': artifacts}
        
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # Test the module
    import sys
    if len(sys.argv) < 2:
        print("Usage: python explain.py <event_key> [team]")
        sys.exit(1)
    
    event_key = sys.argv[1]
    team = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = run(event_key, team)
    if result['success']:
        print("✅ Explanation generation successful!")
        print(f"📊 CSV table: {result['artifacts']['csv_table']}")
        print(f"📝 Summary: {result['artifacts']['markdown_summary']}")
        print(f"🖼️  Figures: {result['artifacts']['figures_dir']}")
    else:
        print(f"❌ Failed: {result['error']}")
        sys.exit(1)
