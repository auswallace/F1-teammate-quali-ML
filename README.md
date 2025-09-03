# F1 Teammate Qualifying Predictor

Predict which Formula 1 driver will qualify ahead of their teammate using machine learning. Built for F1 fans who want to understand the data behind teammate battles—and for ML practitioners learning production pipelines.

## 🏆 Headline Results

As of the latest evaluation, this model correctly predicts teammate qualifying head-to-head **≈87.8%** of the time (pooled across seasons). **F1: 0.878, ROC-AUC: 0.957, Brier: 0.090** (WTF is this? F1 measures overall accuracy, ROC-AUC measures how well we rank predictions, Brier measures probability calibration—lower is better).

## 🎯 What This Repo Does

- **Predict teammate qualifying winner** per event, with calibrated probabilities (so 90% confidence ≈ 90% accuracy)
- **Time-aware evaluation** using walk-forward validation across seasons (no data leakage)
- **Baseline comparisons** against simple rules: H2H prior record, last qualifying winner
- **Predict upcoming events** via a simple lineup CSV (no need to wait for race weekend)
- **Interactive Streamlit UI** for click-to-select race exploration

## 📊 Data Source & Flow

**Input**: Your FastF1 parquet exports (2010-2025 seasons) → **Local storage**: Processed features, trained models, evaluation reports.

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

## 🖥️ Streamlit App

**Launch the interactive interface:**
```bash
streamlit run app.py
```

**What you'll see:** Season/event dropdowns, model vs actual vs baselines per team, confidence scores, and prediction explanations.

## ⚠️ Limitations & Gotchas

- **Practice/track/weather missing pre-event** → lower confidence
- **Driver swaps and rookie pairings** reduce history
- **Sprint weekends vs standard qualifying**: We use the session that sets the race grid (configurable in `config/settings.yaml`)
- **No official affiliation**; educational/hobby project

## 🗺️ Roadmap (Short, Concrete)

- **Better practice deltas** (consistency metrics)
- **Era flags** (e.g., 2022 ground effect) & interactions  
- **Optional hyperparameter tuning** (Optuna) later

## 📚 Credits & License

**Data sources:** FastF1 library, F1 official timing data
**ML framework:** scikit-learn, XGBoost, SHAP
**License:** MIT (you created this repo)

---

**Built with ❤️ for the F1 community. May the best teammate win!**

*Questions? Run `python run_all.py --status` to check your setup, or create a GitHub issue.*
