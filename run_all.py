#!/usr/bin/env python3
"""
Main orchestration script for F1 teammate qualifying prediction pipeline.

Runs the complete pipeline: data loading, labeling, feature engineering,
training, evaluation, and prediction.
"""

import argparse
import logging
import sys
from pathlib import Path
import joblib
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_pipeline():
    """Run the data processing pipeline."""
    logger.info("Starting data processing pipeline")
    
    try:
        # Import modules
        from src.load_data import F1DataLoader
        from src.labeling import TeammateLabeler
        from src.features import FeatureEngineer
        from src.split import DataSplitter
        
        # Step 1: Load and normalize data
        logger.info("Step 1: Loading and normalizing data")
        loader = F1DataLoader()
        loader.load_all_seasons()
        loader.normalize_schema()
        loader.save_normalized_data()
        
        # Step 2: Create labels
        logger.info("Step 2: Creating teammate labels")
        labeler = TeammateLabeler()
        df = labeler.load_normalized_data()
        labeled_df = labeler.create_teammate_labels(df)
        labeled_df = labeler.handle_sprint_weekends(labeled_df)
        if labeler.validate_labels(labeled_df):
            labeler.save_labeled_data(labeled_df)
        else:
            raise ValueError("Label validation failed")
        
        # Step 3: Feature engineering
        logger.info("Step 3: Feature engineering")
        engineer = FeatureEngineer()
        df = engineer.load_labeled_data()
        featured_df = engineer.create_features(df)
        engineer.save_processed_data(featured_df)
        
        # Step 4: Data splitting
        logger.info("Step 4: Data splitting")
        splitter = DataSplitter()
        df = splitter.load_processed_data()
        splits = splitter.create_temporal_splits(df)
        cv_splits = splitter.create_cv_splits(splits['train'])
        splitter.save_split_data(splits)
        
        logger.info("Data processing pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Data processing pipeline failed: {e}")
        return False


def run_training():
    """Run the training pipeline."""
    logger.info("Starting training pipeline")
    
    try:
        from src.train import ModelTrainer
        
        trainer = ModelTrainer()
        splits = trainer.load_split_data()
        
        if 'train' not in splits or len(splits['train']) == 0:
            raise ValueError("No training data found")
        
        X, y, feature_cols = trainer.prepare_training_data(splits['train'])
        models = trainer.train_all_models(X, y)
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
        
        trainer.log_training_summary()
        
        logger.info("Training pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return False


def run_evaluation():
    """Run the evaluation pipeline."""
    logger.info("Starting evaluation pipeline")
    
    try:
        from src.eval import ModelEvaluator
        
        evaluator = ModelEvaluator()
        models, splits = evaluator.load_models_and_data()
        
        if not models:
            raise ValueError("No models found for evaluation")
        
        evaluation_results = evaluator.evaluate_all_models(models, splits)
        evaluator.create_evaluation_plots()
        evaluator.create_shap_analysis(models, splits)
        evaluator.create_head_to_head_analysis(splits)
        evaluator.save_evaluation_summary()
        
        logger.info("Evaluation pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        return False


def run_prediction(event_key: str):
    """Run the prediction pipeline."""
    logger.info(f"Starting prediction pipeline for event: {event_key}")
    
    try:
        from src.predict import TeammatePredictor
        
        # Get data paths
        config_path = Path("config/settings.yaml")
        if not config_path.exists():
            raise FileNotFoundError("Configuration file not found")
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        processed_dir = Path(config['data']['processed_dir'])
        data_paths = [
            str(processed_dir / "teammate_qual.parquet")
        ]
        
        predictor = TeammatePredictor()
        results = predictor.predict_teammate(event_key, data_paths)
        
        logger.info(f"Prediction pipeline completed successfully for {event_key}!")
        logger.info(f"Results shape: {results.shape}")
        
        # Print summary
        print(f"\nPredictions for {event_key}:")
        print("=" * 50)
        for _, row in results.iterrows():
            message = f"{row['driver_name']} ({row['constructor_id']}): {row['ensemble_prob']:.3f} probability of beating teammate"
            print(message)
        
        return True
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}")
        return False


def run_walkforward_validation():
    """Run walk-forward validation with calibration and baselines."""
    try:
        logger.info("Starting walk-forward validation pipeline")
        
        # Import and run validation
        from src.validate_walkforward import run as run_validation
        
        results = run_validation()
        
        logger.info("Walk-forward validation pipeline completed successfully!")
        logger.info(f"Processed {len(results['block_results'])} blocks")
        logger.info(f"Total predictions: {results['pooled_metrics']['total_samples']}")
        
        # Print final summary
        print(f"\n{'='*60}")
        print("WALK-FORWARD VALIDATION COMPLETED")
        print(f"{'='*60}")
        print(f"Total Blocks: {len(results['block_results'])}")
        print(f"Total Samples: {results['pooled_metrics']['total_samples']}")
        print(f"Pooled Accuracy: {results['pooled_metrics']['pooled_accuracy']:.3f}")
        print(f"Pooled F1: {results['pooled_metrics']['pooled_f1']:.3f}")
        print(f"Pooled ROC-AUC: {results['pooled_metrics']['pooled_roc_auc']:.3f}")
        print(f"Pooled Brier: {results['pooled_metrics']['pooled_brier']:.3f}")
        print(f"\nResults saved to reports/ directory")
        print(f"CSV files: reports/predictions/wf_block_*.csv")
        print(f"Markdown: reports/predictions/wf_block_*.md")
        print(f"Plots: reports/figures/wf_block_*_evaluation.png")
        print(f"Summary: reports/predictions/wf_summary.md")
        
        return True
        
    except Exception as e:
        logger.error(f"Walk-forward validation pipeline failed: {e}")
        return False


def run_explanation(event_key: str, team: str = None):
    """Run explanation pipeline for a specific event and team."""
    try:
        logger.info(f"Starting explanation generation for event {event_key}, team {team or 'all'}")
        
        # Import and run explanation
        from src.explain import run as explain_run
        
        results = explain_run(event_key, team)
        
        if results['success']:
            logger.info("Explanation generation completed successfully!")
            logger.info(f"📊 CSV table: {results['artifacts']['csv_table']}")
            logger.info(f"📝 Summary: {results['artifacts']['markdown_summary']}")
            logger.info(f"🖼️  Figures: {results['artifacts']['figures_dir']}")
            
            # Print helpful note
            print("\n" + "="*60)
            print("📖 HOW TO READ THESE EXPLANATIONS:")
            print("="*60)
            print("• CSV table: Feature-by-feature breakdown with SHAP/permutation values")
            print("• Summary: Human-readable explanation of top factors for each driver")
            print("• Waterfall plots: Visual SHAP values (blue=positive, red=negative)")
            print("• Positive factors push probability UP, negative factors push DOWN")
            print("• Higher absolute values = more important features")
            print("="*60)
            
            return True
        else:
            logger.error(f"Explanation generation failed: {results['error']}")
            return False
            
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        return False


def print_status():
    """Print project status and next steps."""
    print("Project: F1 Teammate Qualifying Predictor")
    print("Status:")
    
    # Check data/input
    input_dir = Path("data/input")
    if input_dir.exists() and input_dir.is_symlink():
        try:
            target = input_dir.resolve()
            if target.exists():
                # Recursively search for parquet files in nested structure
                parquet_files = list(target.rglob("*.parquet"))
                print(f"  [✓] data/input linked ({len(parquet_files)} files)")
            else:
                print("  [✗] data/input linked but target missing")
        except:
            print("  [✗] data/input linked but target invalid")
    elif input_dir.exists():
        # Recursively search for parquet files in nested structure
        parquet_files = list(input_dir.rglob("*.parquet"))
        print(f"  [✓] data/input present ({len(parquet_files)} files)")
    else:
        print("  [ ] data/input missing (run: make link)")
    
    # Check processed features
    processed_file = Path("data/processed/teammate_qual.parquet")
    if processed_file.exists():
        print(f"  [✓] features built ({processed_file})")
    else:
        print("  [ ] features missing (run: make build)")
    
    # Check trained models
    xgb_model = Path("models/xgboost.joblib")
    lr_model = Path("models/logistic_regression.joblib")
    if xgb_model.exists() and lr_model.exists():
        print("  [✓] models trained (models/*.joblib)")
        
        # Check calibrator
        calibrator = Path("models/xgb_calibrator.joblib")
        if calibrator.exists():
            print("  [✓] calibrator trained (models/xgb_calibrator.joblib)")
        else:
            print("  [ ] calibrator missing (run: make train)")
    else:
        print("  [ ] models missing (run: make train)")
    
    # Check evaluation summary
    eval_summary = Path("reports/evaluation_summary.md")
    if eval_summary.exists():
        print("  [✓] eval summary (reports/evaluation_summary.md)")
    else:
        print("  [ ] eval summary missing (run: make eval)")
    
    # Check walk-forward summary
    wf_summary = Path("reports/predictions/wf_summary.md")
    if wf_summary.exists():
        print("  [✓] walk-forward summary (reports/predictions/wf_summary.md)")
    else:
        print("  [ ] walk-forward summary missing (run: make wf)")
    
    # Determine next step
    if not input_dir.exists():
        next_step = "make link"
    elif not processed_file.exists():
        next_step = "make build"
    elif not xgb_model.exists():
        next_step = "make train"
    elif not eval_summary.exists():
        next_step = "make eval"
    elif not wf_summary.exists():
        next_step = "make wf"
    else:
        next_step = "make predict EVENT=2025_11"
    
    print(f"Next step suggestion: {next_step}")
    print("Common commands:")
    print("  make build | make train | make eval | make wf | make predict EVENT=YYYY_RR")


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(
        description='F1 Teammate Qualifying Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --build                    # Run data processing pipeline
  python run_all.py --train                   # Train models
  python run_all.py --eval                    # Evaluate models
  python run_all.py --predict --event 2025_01 # Make predictions for event
  python run_all.py --validate-walkforward    # Run walk-forward validation
  python run_all.py --explain --event 2025_11 --team MCLAREN  # Explain predictions
  python run_all.py --all                     # Run complete pipeline
        """
    )
    
    parser.add_argument('--build', action='store_true',
                       help='Run data processing pipeline (load, label, features, split)')
    parser.add_argument('--train', action='store_true',
                       help='Train models')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate models')
    parser.add_argument('--predict', action='store_true',
                       help='Make predictions')
    parser.add_argument('--event', type=str,
                       help='Event key for predictions (e.g., 2025_01)')
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--validate-walkforward', action='store_true',
                       help='Run walk-forward validation with calibration and baselines')
    parser.add_argument('--explain', action='store_true',
                       help='Generate local explanations for event predictions')
    parser.add_argument('--team', type=str,
                       help='Team/constructor ID for explanations (optional, defaults to all teams)')
    parser.add_argument('--status', action='store_true',
                       help='Show project status and next steps')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.predict and not args.event:
        parser.error("--predict requires --event")
    
    if args.explain and not args.event:
        parser.error("--explain requires --event")
    
    if not any([args.build, args.train, args.eval, args.predict, args.all, args.validate_walkforward, args.explain, args.status]):
        parser.error("Must specify at least one action")
    
    # Run pipeline
    success = True
    
    if args.all or args.build:
        logger.info("Running data processing pipeline...")
        if not run_data_pipeline():
            success = False
            logger.error("Data processing pipeline failed")
    
    if args.all or args.train:
        logger.info("Running training pipeline...")
        if not run_training():
            success = False
            logger.error("Training pipeline failed")
    
    if args.all or args.eval:
        logger.info("Running evaluation pipeline...")
        if not run_evaluation():
            success = False
            logger.error("Evaluation pipeline failed")
    
    if args.all or args.predict:
        logger.info("Running prediction pipeline...")
        if not run_prediction(args.event):
            success = False
            logger.error("Prediction pipeline failed")
    
    if args.validate_walkforward:
        logger.info("Running walk-forward validation...")
        if not run_walkforward_validation():
            success = False
            logger.error("Walk-forward validation failed")
    
    if args.explain:
        logger.info("Running explanation pipeline...")
        if not run_explanation(args.event, args.team):
            success = False
            logger.error("Explanation pipeline failed")
    
    if args.status:
        print_status()
        sys.exit(0)
    
    # Final status
    if success:
        logger.info("All pipeline steps completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline failed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
