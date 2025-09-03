# F1 Teammate Qualifying Prediction Pipeline

A production-ready machine learning pipeline for predicting whether a Formula 1 driver will beat their teammate in qualifying sessions.

## 🎯 Project Overview

This pipeline predicts the binary outcome `beats_teammate_q` (1 if driver qualifies ahead of teammate, 0 otherwise) using historical F1 data. It's designed to work with your existing FastF1 data exports and provides a complete ML workflow from data processing to model deployment.

### Quickstart
```bash
make venv && make install
make link
make build
make train
make eval
# optional:
make wf
make predict EVENT=2025_11
# at any time:
make status
```

## 🏗️ Architecture

```
f1_teammate_qual/
├── config/settings.yaml          # Configuration file
├── data/
│   ├── input/                   # Symlink to f1-ml/data_processed
│   ├── interim/                 # Intermediate processed data
│   └── processed/               # Final feature dataset
├── models/                      # Trained models and encoders
├── reports/                     # Evaluation reports and visualizations
│   ├── figures/                 # Plots and charts
│   └── predictions/             # Prediction outputs
├── src/                         # Core pipeline modules
├── tests/                       # Unit tests
├── run_all.py                   # Main orchestration script
└── requirements.txt             # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd f1_teammate_qual

# Install dependencies
pip install -r requirements.txt

# Create symlink to your F1 data
ln -s ../f1-ml/data_processed data/input
```

### 2. Run Complete Pipeline

```bash
# Run the entire pipeline end-to-end
python run_all.py --all
```

### 3. Run Individual Steps

```bash
# Data processing only
python run_all.py --build

# Train models
python run_all.py --train

# Evaluate models
python run_all.py --eval

# Make predictions for a specific event
python run_all.py --predict --event 2025_01
```

## 📊 Data Pipeline

### Step 1: Data Loading (`src/load_data.py`)
- Loads parquet files from `f1-ml/data_processed/raw`
- Normalizes schema across all seasons (2010-2025)
- Creates standardized dataframe with required columns
- Output: `data/interim/qual_base.parquet`

### Step 2: Labeling (`src/labeling.py`)
- Creates target variable `beats_teammate_q`
- Computes `teammate_gap_ms` (time delta to teammate)
- Handles edge cases (single drivers, equal times, penalties)
- Output: `data/interim/qual_labeled.parquet`

### Step 3: Feature Engineering (`src/features.py`)
- **Driver Form**: Rolling qualifying position averages, teammate beating share
- **Team Form**: Constructor performance trends, pace percentiles
- **Head-to-Head**: Historical records vs current teammate
- **Track Features**: One-hot encoding, street circuit flags, familiarity
- **Practice Data**: Session deltas, consistency metrics
- **Data Hygiene**: Outlier clipping, missing value imputation
- Output: `data/processed/teammate_qual.parquet`

### Step 4: Data Splitting (`src/split.py`)
- **Temporal Split**: Train (2010-2022), Val (2023), Test (2024-2025)
- **Cross-Validation**: GroupKFold by season to prevent leakage
- Ensures no event overlap between train/val/test sets

## 🤖 Models

### Logistic Regression
- Balanced class weights for imbalanced data
- L1/L2 regularization
- Cross-validated hyperparameters

### XGBoost
- Gradient boosting with early stopping
- Automatic class weight balancing
- Feature importance analysis

### Ensemble
- Simple average of model probabilities
- Robust to individual model failures

## 📈 Evaluation Metrics

- **Classification**: Precision, Recall, F1, ROC-AUC, PR-AUC
- **Calibration**: Brier Score
- **Business**: Head-to-head win rates by constructor/driver
- **Explainability**: SHAP analysis, feature importance

## 🔮 Making Predictions

### Command Line
```bash
python run_all.py --predict --event 2025_01
```

### Programmatic
```python
from src.predict import TeammatePredictor

predictor = TeammatePredictor()
results = predictor.predict_teammate(
    event_key="2025_01",
    latest_data_paths=["data/processed/teammate_qual.parquet"]
)

# Results include:
# - driver_id, driver_name, constructor_id
# - probability of beating teammate
# - rank within team
# - teammate information
```

## ⚙️ Configuration

Edit `config/settings.yaml` to customize:

```yaml
# Data paths
data:
  input_dir: "../f1-ml/data_processed/raw"
  interim_dir: "data/interim"
  processed_dir: "data/processed"

# Feature engineering
features:
  driver_form_windows: [3, 5]
  team_form_windows: [3, 5]
  h2h_windows: [4, 6]

# Model parameters
models:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

# Data splitting
splits:
  train_seasons: [2010, 2011, ..., 2022]
  val_seasons: [2023]
  test_seasons: [2024, 2025]
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_labeling.py
```

## 📁 Output Files

### Models
- `models/logistic_regression.joblib`
- `models/xgboost.joblib`
- `models/*_results.json`

### Reports
- `reports/evaluation_summary.md`
- `reports/head_to_head_stats.json`
- `reports/train.log`

### Visualizations
- `reports/figures/*_evaluation.png`
- `reports/figures/xgboost_shap_summary.png`
- `reports/figures/head_to_head_analysis.png`

### Predictions
- `reports/predictions/predictions_<event>.csv`
- `reports/predictions/summary_<event>.md`

## 🔍 Data Schema

### Input Requirements
- Season, event, driver, constructor identifiers
- Qualifying position and lap times
- Practice session data (optional)
- Track and weather information (optional)

### Feature Output
- 50+ engineered features
- Rolling statistics with configurable windows
- One-hot encoded categorical variables
- Missing value flags and imputation indicators

## 🚨 Assumptions & Limitations

### Data Assumptions
- Two drivers per constructor per event
- Qualifying session determines race grid
- Historical data available for feature engineering
- No data leakage across temporal splits

### Model Limitations
- Binary classification (beats/doesn't beat teammate)
- No probability calibration guarantees
- Assumes teammate relationships remain stable
- Limited to qualifying session predictions

### Edge Cases Handled
- Single drivers (dropped from training)
- Equal qualifying times (official order used)
- Missing practice data (seasonal median imputation)
- Sprint weekends (configurable session selection)

## 🛠️ Development

### Adding New Features
1. Implement in `src/features.py`
2. Add to feature configuration
3. Update data validation
4. Add unit tests

### Adding New Models
1. Implement training in `src/train.py`
2. Add evaluation in `src/eval.py`
3. Update prediction pipeline
4. Add model configuration

### Extending to Other Predictions
- Modify labeling logic in `src/labeling.py`
- Update feature engineering for new targets
- Adjust evaluation metrics
- Extend prediction interface

## 📚 Dependencies

- **Core ML**: scikit-learn, xgboost, pandas, numpy
- **Visualization**: matplotlib, seaborn, shap
- **Data**: pyarrow, joblib
- **Configuration**: pyyaml

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- FastF1 library for F1 data access
- Scikit-learn for ML framework
- XGBoost for gradient boosting
- SHAP for model explainability

## 📞 Support

For questions or issues:
- Create GitHub issue
- Check documentation in `reports/`
- Review training logs in `reports/train.log`
- Examine configuration in `config/settings.yaml`
