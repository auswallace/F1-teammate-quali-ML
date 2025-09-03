# Model Card: F1 Teammate Qualifying Prediction

## Model Overview

**Model Name**: F1 Teammate Qualifying Prediction Pipeline  
**Version**: 1.0.0  
**Date**: [Current Date]  
**Type**: Binary Classification  
**Purpose**: Predict whether a Formula 1 driver will beat their teammate in qualifying sessions

## Problem Statement

**Target Variable**: `beats_teammate_q`  
- **1**: Driver qualifies ahead of teammate  
- **0**: Driver qualifies behind teammate  

**Business Context**: Understanding driver performance relative to teammates is crucial for:
- Driver evaluation and contract decisions
- Team strategy and car development
- Fan engagement and betting markets
- Historical performance analysis

## Data

### Training Data
- **Source**: FastF1 library exports (2010-2025)
- **Size**: [X] qualifying sessions, [Y] driver-constructor pairs
- **Coverage**: All F1 seasons from 2010 to 2025
- **Geographic Scope**: Global F1 calendar

### Features
- **Driver Form**: Rolling qualifying position averages, teammate beating share
- **Team Performance**: Constructor trends, pace percentiles
- **Head-to-Head**: Historical records vs current teammate
- **Track Context**: Circuit type, familiarity, historical variance
- **Practice Data**: Session deltas, consistency metrics
- **Data Quality**: Missing value flags, outlier indicators

### Data Splits
- **Train**: 2010-2022 (13 seasons)
- **Validation**: 2023 (1 season)  
- **Test**: 2024-2025 (2 seasons)

## Model Architecture

### Primary Models
1. **Logistic Regression**
   - Class balancing for imbalanced data
   - L1/L2 regularization
   - Cross-validated hyperparameters

2. **XGBoost**
   - Gradient boosting with early stopping
   - Automatic class weight balancing
   - Feature importance analysis

3. **Ensemble**
   - Simple average of model probabilities
   - Robust to individual model failures

### Training Process
- **Cross-Validation**: GroupKFold by season (prevents temporal leakage)
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Class Balancing**: Automatic adjustment for imbalanced targets
- **Feature Selection**: Based on importance and correlation analysis

## Performance Metrics

### Classification Metrics
- **Precision**: [X.XX] - Accuracy of positive predictions
- **Recall**: [X.XX] - Coverage of actual positive cases
- **F1-Score**: [X.XX] - Harmonic mean of precision and recall
- **ROC-AUC**: [X.XX] - Area under ROC curve
- **PR-AUC**: [X.XX] - Area under precision-recall curve

### Calibration Metrics
- **Brier Score**: [X.XX] - Probability calibration quality
- **Calibration Plot**: [See reports/figures/]

### Business Metrics
- **Head-to-Head Accuracy**: [X.XX] - Win rate prediction accuracy
- **Constructor Performance**: [See reports/head_to_head_stats.json]

## Model Limitations

### Data Assumptions
- Two drivers per constructor per event
- Qualifying session determines race grid
- Historical data available for feature engineering
- No data leakage across temporal splits

### Prediction Scope
- **Binary outcome only**: Cannot predict qualifying positions or lap times
- **Teammate context required**: Predictions only meaningful within team pairs
- **Historical dependency**: Requires sufficient driver history for features
- **No external factors**: Weather, car issues, penalties not modeled

### Edge Cases
- **Single drivers**: Dropped from training (no teammate comparison)
- **Equal times**: Official qualifying order used as tiebreaker
- **Missing data**: Seasonal median imputation with quality flags
- **Sprint weekends**: Configurable session selection

## Ethical Considerations

### Fairness
- **Driver bias**: Model may reflect historical biases in F1
- **Team bias**: Constructor performance influences predictions
- **Temporal bias**: Recent seasons may have different characteristics

### Transparency
- **Feature importance**: SHAP analysis available for XGBoost
- **Prediction confidence**: Probability scores provided
- **Model interpretability**: Logistic regression coefficients available

### Privacy
- **Public data only**: All data from public F1 sources
- **No personal information**: Only racing performance data
- **Aggregated analysis**: Individual driver privacy maintained

## Usage Guidelines

### Input Requirements
- **Event key**: Format "YYYY_RR" (e.g., "2025_01")
- **Driver data**: Qualifying positions, lap times, practice sessions
- **Team context**: Constructor information, teammate relationships
- **Historical context**: Previous events for feature engineering

### Output Interpretation
- **Probability scores**: 0.0-1.0 scale (higher = more likely to beat teammate)
- **Ranking**: Within-team performance ordering
- **Confidence**: Based on feature availability and model agreement
- **Uncertainty**: Flagged for edge cases and missing data

### Best Practices
- **Regular retraining**: Update models with new season data
- **Feature monitoring**: Track feature drift and data quality
- **Performance validation**: Validate on new seasons before deployment
- **A/B testing**: Compare predictions with actual outcomes

## Deployment

### Model Serving
- **Format**: Joblib serialized models
- **Dependencies**: Python 3.10+, scikit-learn, xgboost
- **Memory**: ~100MB for all models
- **Inference time**: <100ms per prediction

### Monitoring
- **Data drift**: Track feature distributions over time
- **Performance decay**: Monitor metrics on new data
- **Prediction quality**: Validate against actual outcomes
- **System health**: Model loading and inference errors

### Updates
- **Retraining schedule**: After each F1 season
- **Version control**: Model artifacts and configurations
- **Rollback capability**: Previous model versions available
- **Testing protocol**: Validation on holdout data

## Contact & Support

**Maintainer**: [Your Name/Team]  
**Repository**: [GitHub URL]  
**Documentation**: [Wiki/Docs URL]  
**Issues**: [GitHub Issues URL]

## Changelog

### Version 1.0.0
- Initial model release
- Logistic Regression and XGBoost models
- Complete feature engineering pipeline
- Temporal validation framework
- SHAP explainability analysis

---

*This model card follows the Model Card for Model Reporting framework. For questions or updates, please contact the maintainer.*
