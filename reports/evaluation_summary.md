# F1 Teammate Qualifying Prediction - Evaluation Summary

## Logistic Regression

---

## Xgboost

### TRAIN Split

#### Model vs Baselines

| Metric | Model | H2H-Prior | Last-Quali |
|--------|-------|------------|------------|
| Accuracy | 0.929 | 0.771 | 0.771 |
| F1 | 0.923 | 0.752 | 0.752 |
| PR-AUC | 0.984 | 0.723 | 0.723 |
| Brier | 0.057 | - | - |
| ECE | 0.057 | - | - |

#### Calibration Metrics

- **Brier Score:** 0.0573 (lower is better)
- **Expected Calibration Error:** 0.0567 (lower is better)

#### Confusion Matrix

```
      Predicted
Actual  0    1
  0    1552  117
  1    137  1532
```

### VAL Split

#### Model vs Baselines

| Metric | Model | H2H-Prior | Last-Quali |
|--------|-------|------------|------------|
| Accuracy | 0.875 | 0.789 | 0.789 |
| F1 | 0.865 | 0.768 | 0.768 |
| PR-AUC | 0.948 | 0.746 | 0.746 |
| Brier | 0.097 | - | - |
| ECE | 0.056 | - | - |

#### Calibration Metrics

- **Brier Score:** 0.0975 (lower is better)
- **Expected Calibration Error:** 0.0560 (lower is better)

#### Confusion Matrix

```
      Predicted
Actual  0    1
  0     79   11
  1     13   77
```

### TEST Split

#### Model vs Baselines

| Metric | Model | H2H-Prior | Last-Quali |
|--------|-------|------------|------------|
| Accuracy | 0.874 | 0.816 | 0.816 |
| F1 | 0.881 | 0.801 | 0.801 |
| PR-AUC | 0.964 | 0.774 | 0.774 |
| Brier | 0.082 | - | - |
| ECE | 0.039 | - | - |

#### Calibration Metrics

- **Brier Score:** 0.0816 (lower is better)
- **Expected Calibration Error:** 0.0388 (lower is better)

#### Confusion Matrix

```
      Predicted
Actual  0    1
  0    156   23
  1     20  159
```

---

