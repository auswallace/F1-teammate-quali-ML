# F1 Teammate Qualifying Predictor

A machine learning pipeline that predicts which Formula 1 driver will qualify ahead of their teammate. Built for F1 enthusiasts who want to understand the data behind teammate battles, with support for both historical analysis and future race predictions.

## 🎯 What This Does

This repository takes your existing FastF1 data exports and builds a complete ML pipeline to predict teammate qualifying outcomes. It's designed to be production-ready while remaining accessible to F1 fans interested in data science. The system learns from historical patterns (driver form, team performance, track familiarity) to predict who will beat their teammate in qualifying.

## ✨ Features

- **Predict teammate head-to-head qualifying outcomes** with calibrated probabilities
- **Walk-forward validation across seasons** with no data leakage
- **Calibrated probabilities** for trustworthy confidence scores
- **Baseline comparisons** (H2H prior record, last quali winner)
- **Ability to predict upcoming races** using a simple lineups CSV
- **Local explanations** with SHAP values and permutation importance
- **Interactive Streamlit UI** for exploring predictions and results
- **Professional evaluation** with comprehensive metrics and visualizations

## 🚀 Quickstart

```bash
git clone <repo-url>
cd f1-teammate-qual
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --build
python run_all.py --train
python run_all.py --eval
python run_all.py --validate-walkforward
```

## 🏁 Predicting Future Races

The pipeline can predict upcoming races using a simple lineup configuration:

1. **Edit lineups** in `data/input/upcoming_lineups.csv`:
   ```csv
   season,event_key,constructor_id,driver_id_a,driver_id_b
   2025,2025_12,MCLAREN,NOR,PIS
   2025,2025_12,RED_BULL,VER,LAW
   ```

2. **Run predictions**:
   ```bash
   python run_all.py --predict-upcoming --event 2025_12
   ```

The system automatically backfills missing drivers from recent pairings and generates predictions using historical form data.

## 🖥️ Using the Streamlit App

Launch the interactive interface:
```bash
streamlit run app.py
```

Select any season and event to see:
- Model predictions with confidence scores
- Baseline comparisons (H2H record)
- Actual results (for historical events)
- Team-by-team breakdowns

## 📊 What You Get

### Core Outputs
- **`reports/eval_summary.md`** - Model performance vs baselines
- **`reports/figures/*`** - ROC curves, PR curves, calibration plots
- **`reports/predictions/*`** - Per-event prediction CSVs
- **`reports/explanations/*`** - SHAP explanations and markdown summaries

### Example Results
Recent model performance:
- **Accuracy**: 68.2%
- **F1 Score**: 0.67
- **ROC-AUC**: 0.72
- **Baseline H2H**: 64.1% (4.1% lift)

## 🏗️ Architecture

```
f1_teammate_qual/
├── config/settings.yaml          # All configuration
├── data/
│   ├── input/                   # Your F1 data exports
│   ├── interim/                 # Processed data
│   └── processed/               # Final features
├── models/                      # Trained models + calibrators
├── reports/                     # Results and visualizations
├── src/                         # Pipeline modules
└── run_all.py                   # Main orchestration
```

## 🔧 Key Commands

```bash
# Data pipeline
python run_all.py --build

# Train models
python run_all.py --train

# Evaluate performance
python run_all.py --eval

# Walk-forward validation
python run_all.py --validate-walkforward

# Explain predictions
python run_all.py --explain --event 2025_11 --team MCLAREN

# Predict upcoming race
python run_all.py --predict-upcoming --event 2025_12

# Check status
python run_all.py --status
```

## 🧠 How It Works

1. **Data Loading**: Normalizes your FastF1 exports into consistent schema
2. **Labeling**: Creates target variable from qualifying results
3. **Features**: Engineers 50+ features (driver form, team performance, track familiarity)
4. **Training**: XGBoost + Logistic Regression with temporal validation
5. **Calibration**: Adjusts probabilities to be trustworthy
6. **Prediction**: Generates predictions for any event (historical or future)

## 📈 Model Features

- **Driver Form**: Rolling qualifying averages, teammate beating share
- **Team Performance**: Constructor pace trends, field position
- **Head-to-Head**: Historical records vs current teammate
- **Track Knowledge**: Circuit familiarity, street circuit flags
- **Practice Data**: Session deltas, consistency metrics
- **Weather**: Temperature, rain, wind (when available)

## 🔍 Explainability

Get detailed explanations for any prediction:
```bash
python run_all.py --explain --event 2025_11 --team MCLAREN
```

This generates:
- **SHAP waterfall plots** showing feature contributions
- **Markdown summaries** with top positive/negative factors
- **CSV breakdowns** of all feature values and contributions

## ⚠️ Disclaimer

This is an educational/hobby project for F1 and data science enthusiasts. It's not intended for betting or official F1 predictions. The models are trained on historical data and may not reflect current driver form or team dynamics.

## 🛠️ Dependencies

- **ML**: scikit-learn, xgboost, shap
- **Data**: pandas, numpy, pyarrow
- **Viz**: matplotlib, seaborn
- **UI**: streamlit
- **Utils**: joblib, pyyaml

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

- **Issues**: Create a GitHub issue
- **Documentation**: Check `reports/` directory
- **Configuration**: Review `config/settings.yaml`
- **Status**: Run `python run_all.py --status`

---

Built with ❤️ for the F1 community. May the best teammate win!
