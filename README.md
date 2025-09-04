# F1 Teammate Qualifying Predictor

Predict which Formula 1 driver will qualify ahead of their teammate using machine learning. Built for F1 fans who want to understand the data behind teammate battles—and for ML practitioners learning production pipelines.

## 🏆 Headline Results

As of the latest evaluation, this model correctly predicts teammate qualifying head-to-head **≈87.8%** of the time (pooled across seasons). **F1: 0.878, ROC-AUC: 0.957, Brier: 0.090** (WTF is this? F1 measures overall accuracy, ROC-AUC measures how well we rank predictions, Brier measures probability calibration—lower is better).

**NEW: Race Winner Predictions** 🏆 - Predict which driver will win entire races with **2024-2025 season coverage** and **actual race results integration**!

## 🎯 What This Repo Does

- **Predict teammate qualifying winner** per event, with calibrated probabilities (so 90% confidence ≈ 90% accuracy)
- **Predict race winners** 🆕 for entire races with grid position, qualifying performance, and recent form analysis
- **Time-aware evaluation** using walk-forward validation across seasons (no data leakage)
- **Baseline comparisons** against simple rules: H2H prior record, last qualifying winner
- **Predict upcoming events** via a simple lineup CSV (no need to wait for race weekend)
- **Interactive Streamlit UI** for click-to-select race exploration with **dual prediction modes**

## 📊 Data Source & Flow

**Input**: Your FastF1 parquet exports (2010-2025 seasons) → **Local storage**: Processed features, trained models, evaluation reports.

**NEW**: **2025 season data** fully integrated with practice, qualifying, and race sessions from your raw data folder!

```
Raw F1 Data → Base Table → Labels → Features → Train/Eval → Predict
     ↓              ↓         ↓        ↓         ↓         ↓
  Parquet      Normalized  Target  50+ Engineered  Models   Results
  Files        Schema      Variable  Features     + Calibration
```

**Key**: We only use information available *before* qualifying starts (practice times, historical form, track familiarity). No cheating with future data!

## 🤖 Models (Short & Honest)

- **Logistic Regression**: Solid baseline, handles class imbalance with balanced weights
- **XGBoost**: Our workhorse, captures complex driver-team-track interactions  
- **Calibration**: Isotonic regression makes "90% confidence" actually mean 90% accuracy. Why? Raw ML probabilities are often overconfident—calibration fixes this.

**NEW: Race Winner Model** 🏆
- **XGBoost + CalibratedClassifierCV** for race outcome prediction
- **Per-event softmax normalization** ensuring probabilities sum to 100%
- **Feature engineering**: Grid position, qualifying performance, recent form, track history, weather
- **Single artifact storage** (`race_winner.joblib`) with model, features, and metadata

## 🚀 How to Run (Quickstart)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Link or copy data exports into data/input/ if needed
python run_all.py --build
python run_all.py --train
python run_all.py --eval
python run_all.py --validate-walkforward
```

## 🏁 Predict an Event

**Historical/event already run:**
```bash
python run_all.py --predict --event <event_key>
```

**Upcoming (pre-event):** Edit `data/input/upcoming_lineups.csv`, then:
```bash
python run_all.py --predict-upcoming --event 2025_12
```

**Where results go:** `reports/predictions/future_<event>_predictions.csv` (two rows per team; probabilities within a team complement to ~1).

**NEW: Race Winner Predictions** 🏆
```bash
# Train the race winner model
python src/race_winner_model.py train \
  --data "data/race_features_combined.parquet" \
  --output "webapi/ml/models/race_winner.joblib" \
  --test-seasons 2024 2025

# Predict specific event
python src/race_winner_model.py predict \
  --data "data/race_features_combined.parquet" \
  --model "webapi/ml/models/race_winner.joblib" \
  --event AUS
```

**Where results go:** `data/pred_cache/{season}/{event_code}.race.json` with full driver rankings and actual race results!

## 📊 How to Interpret Results (Training Notes for Future-Me)

- **Accuracy / F1**: "How often we're right / balance of precision and recall."
- **ROC-AUC** (WTF is this?): "If you randomly pick one true winner and one loser, ROC-AUC is the chance the model scores the winner higher. 0.5 = coin flip, 1.0 = perfect ranking."
- **PR-AUC**: "Focuses on positive class; helpful if classes are imbalanced."
- **Brier Score**: "Measures probability error (lower is better). If we say 0.9, did it happen ~90% of the time?"
- **ECE (Expected Calibration Error)**: "Do predicted probabilities match reality on average? Lower is better; <0.05 is great."
- **Why a 99% can still be wrong**: calibration, weird weekends, practice deltas lying (rain, red flags).

**Check these files:** `reports/figures/*calibration*.png` and `reports/predictions/*.csv`.

## 🎯 Baselines (Reality Check)

- **H2H-prior**: pick whoever led the matchup entering the event
- **Last-quali winner**: pick whoever won the most recent quali H2H

**Latest comparison (TEST split):**
| Metric | Model | H2H-Prior | Last-Quali |
|--------|-------|------------|------------|
| Accuracy | 87.4% | 81.6% | 81.6% |
| F1 | 88.1% | 80.1% | 80.1% |
| PR-AUC | 96.4% | 77.4% | 77.4% |

**Model lift:** +5.8% accuracy over simple baselines. Not bad!

## ⏰ Walk-Forward Validation (No-Leakage)

**Why this matters:** Simulates real-world deployment where you retrain models as new data arrives.

**Block performance:**
- **Block 0** (train ≤2021 → test 2022): 93.0% accuracy, 0.991 ROC-AUC (excellent)
- **Block 1** (train ≤2022 → test 2023): 85.0% accuracy, 0.938 ROC-AUC (good)
- **Block 2** (train ≤2023 → test 2024-2025): 87.7% accuracy, 0.956 ROC-AUC (strong)

**Drift analysis:** 2023 was a challenging year (new regulations?), but we recovered in 2024-2025.

## 🔍 Explainability

**Get detailed explanations for any prediction:**
```bash
python run_all.py --explain --event 2025_11 --team MCLAREN
```

**What you get:**
- **SHAP waterfall plots** showing feature contributions
- **Markdown summaries** with top positive/negative factors
- **CSV breakdowns** of all feature values and contributions

**Top up/down features per team explain why a call was made.**

## 🖥️ Dashboards

### **Streamlit App (Classic)**
**Launch the interactive interface:**
```bash
make dashboard
```

**What you'll see:** Season/event dropdowns, model vs actual vs baselines per team, confidence scores, and prediction explanations.

**NEW: Dual Prediction Modes** 🆕
- **🏁 Qualifying H2H**: Teammate qualifying predictions with accuracy metrics
- **🏆 Race Winner**: Full race outcome predictions with actual results integration

### **React Dashboard (Modern + Track Maps) 🆕**
**Launch the modern React interface with neon track maps:**
```bash
# Backend (FastAPI)
cd webapi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (React)
cd ../dashboard
npm install
npm run dev
```

**What you'll see:** 
- **Neon track maps** generated from FastF1 telemetry
- **F1-style dark theme** with modern UI components
- **Real-time predictions** with circular driver/team icons
- **Responsive design** that works on all devices
- **Interactive track visualization** with driver comparison overlays

**Features:**
- 🗺️ **Track Maps**: SVG-based track visualization with neon glow effects
- 🏆 **Predictions Table**: Rich table with circular icons and status indicators
- 🎨 **F1 Design**: Consistent branding with shared design tokens
- 📱 **Mobile Ready**: Responsive layout for all screen sizes

## 🗺️ Track Maps (Static SVG)

**Circuit Visualization**: Beautiful neon track maps generated from static SVG files.

### **Where to Place Files**
- **Directory**: `data/assets/tracks/<CIRCUIT>.svg`
- **Naming**: Use circuit abbreviations (e.g., `AUS.svg`, `MON.svg`, `BHR.svg`)
- **Sources**: Wikimedia Commons (CC-BY-SA) or official F1 diagrams

### **How to Map Events**
- **Configuration**: `config/circuits_map.yaml`
- **Keys**: Circuit IDs that match SVG filenames
- **Events**: Optional list of specific event keys for mapping
- **Example**:
  ```yaml
  AUS:
    svg: AUS.svg
    name: "Albert Park Circuit"
    country_code: "au"
    events: ["2025_03", "2024_03", "2023_01"]
  ```

### **How It Works**
1. **API Endpoint**: `/api/trackmap/{season}/{event_key}`
2. **Circuit Resolution**: Maps event to circuit via YAML config
3. **SVG Processing**: Extracts path data and viewBox from SVG files
4. **Neon Rendering**: React renders with glowing stroke effects
5. **Fallback**: Shows "not available" card if no SVG found

### **Attribution**
See `data/assets/tracks/README.md` for proper attribution guidelines when using external SVG sources.

## 🏆 Race Winner Predictions (NEW!)

**Full Race Outcome Prediction**: Predict which driver will win entire races, not just beat their teammate.

### **Features & Capabilities**
- **Grid Position Analysis**: Starting position impact on win probability
- **Qualifying Performance**: Best qualifying position history
- **Recent Form**: Last 5 races performance metrics
- **Track History**: Driver performance at specific circuits
- **Weather Integration**: Dry/wet/intermediate conditions
- **Per-Event Normalization**: Probabilities sum to 100% per race

### **Data Coverage**
- **2024 Season**: 2 events (R01, R03) - 39 race records
- **2025 Season**: 12 events (R01-R15) - 239 race records
- **Total**: 278 race records across both seasons
- **Real Results**: Actual race outcomes integrated for model evaluation

### **Display Features**
- **🏆 Top 5 Predictions**: Medal-style ranking with confidence scores
- **📊 Complete Table**: All 20 drivers ranked by win probability
- **✅ Actual Results**: Shows who actually won vs. predictions
- **🎯 Confidence Intervals**: Uncertainty ranges for each prediction
- **📥 CSV Export**: Downloadable results for analysis

### **Cache System**
- **API-First**: FastAPI endpoint `/api/predict/race_winner/{season}/{event_code}`
- **Cache Fallback**: Local JSON files for offline access
- **Automatic Updates**: Cache regenerated when new data available

## ⚠️ Limitations & Gotchas

- **Practice/track/weather missing pre-event** → lower confidence
- **Driver swaps and rookie pairings** reduce history
- **Sprint weekends vs standard qualifying**: We use the session that sets the race grid (configurable in `config/settings.yaml`)
- **No official affiliation**; educational/hobby project

## 🗺️ Roadmap (Short, Concrete)

- **Better practice deltas** (consistency metrics)
- **Era flags** (e.g., 2022 ground effect) & interactions  
- **Optional hyperparameter tuning** (Optuna) later
- **Race winner model expansion** 🆕 to more seasons and features
- **Real-time prediction updates** 🆕 during race weekends

## 📚 Credits & License

**Data sources:** FastF1 library, F1 official timing data
**ML framework:** scikit-learn, XGBoost, SHAP
**License:** MIT (you created this repo)

## 🆕 Recent Updates (September 2024)

### **Race Winner Model Integration**
- **New ML model** for predicting entire race outcomes
- **2025 season data** fully integrated with practice, qualifying, and race sessions
- **Actual race results** displayed alongside predictions for model evaluation
- **Cache system** for offline access and performance optimization

### **Enhanced Streamlit Interface**
- **Dual prediction modes**: Qualifying H2H + Race Winner predictions
- **Rich data display**: Grid positions, final results, winner indicators
- **Interactive tables**: Sortable predictions with confidence intervals
- **Export functionality**: CSV downloads for further analysis

### **Data Pipeline Improvements**
- **Combined datasets**: 2024-2025 seasons in single parquet file
- **Feature engineering**: Grid position, qualifying performance, weather conditions
- **Model artifacts**: Single joblib file with model, features, and metadata
- **Validation system**: Leakage guards and feature validation


---

**Built with ❤️ for the F1 community. May the best teammate win!**

*Questions? Run `python run_all.py --status` to check your setup, or create a GitHub issue.*
