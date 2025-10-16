# MVP Model - Production Ready Ensemble
**Date:** October 15, 2025  
**Status:** Ready for NBA Season Deployment

---

## Performance Summary

**Ensemble Performance (Test Set - 893 games):**
- **MAE:** 5.39 points
- **RMSE:** 6.72 points
- **Coverage:** 94.6% at 95% confidence
- **Uncertainty:** ±13.04 points

**Component Performance:**
- Dejavu (K-NN): 6.17 MAE
- LSTM: 5.24 MAE
- Ensemble (40/60): 5.39 MAE

---

## What's Included

### 1. Trained Models
- `dejavu_k500.pkl` - Dejavu K-NN forecaster (4,003 patterns, k=500)
- `lstm_best.pth` - LSTM weights (64 hidden, 2 layers)
- `lstm_normalization.pkl` - Normalization parameters (CRITICAL for inference)
- `conformal_predictor.pkl` - Calibrated conformal wrapper (95% coverage)

### 2. Model Code
- `dejavu_model.py` - Dejavu implementation
- `lstm_model.py` - LSTM architecture
- `ensemble_model.py` - Ensemble combiner (40% Dejavu + 60% LSTM)
- `conformal_wrapper.py` - Conformal prediction wrapper

### 3. Data Splits
- `splits/train.pkl` - 4,003 training games
- `splits/validation.pkl` - 772 validation games
- `splits/calibration.pkl` - 932 calibration games
- `splits/test.pkl` - 893 test games
- `splits/split_metadata.json` - Split information

### 4. Results & Metrics
- `dejavu_test_results.json` - Dejavu evaluation
- `lstm_test_results.json` - LSTM evaluation
- `ensemble_test_results.json` - Ensemble evaluation
- `conformal_test_results.json` - Coverage metrics
- `dataset_summary.json` - Dataset statistics

---

## Quick Start - Make Predictions

```python
import pickle
import torch
import numpy as np
from ensemble_model import EnsembleForecaster
from conformal_wrapper import ConformalPredictor

# Load models
ensemble = EnsembleForecaster(dejavu_weight=0.4, lstm_weight=0.6)
ensemble.load_models()

conformal = ConformalPredictor.load('conformal_predictor.pkl')

# Make prediction
pattern = np.array([0, 2, 5, 8, 10, 12, 13, 15, 14, 15, 16, 18, 17, 15])  # 18 minutes
forecast, interval, components = conformal.predict(pattern, ensemble)

print(f"Prediction: {forecast:+.1f} points")
print(f"95% Interval: [{interval[0]:+.1f}, {interval[1]:+.1f}]")
print(f"Components:")
print(f"  Dejavu: {components['dejavu_prediction']:+.1f}")
print(f"  LSTM:   {components['lstm_prediction']:+.1f}")
```

---

## Model Specifications

### Dejavu K-NN:
- Database: 4,003 training patterns
- k-neighbors: 500 (paper optimal)
- Distance: Euclidean (L2)
- Normalization: Z-score per pattern
- Aggregation: Median of k=500 outcomes

### LSTM:
- Input: (batch, 18, 1) - 18-minute sequence
- Hidden size: 64
- Num layers: 2
- Dropout: 0.1
- Output: (batch, 7) - Forecast minutes 18-24
- Training: Adam, lr=1e-3, early stopping

### Ensemble:
- Weights: 40% Dejavu + 60% LSTM
- Combination: Weighted average
- Output: Single halftime differential prediction

### Conformal:
- Calibration: 932 games
- Alpha: 0.05 (95% coverage)
- Quantile: ±13.04 points
- Empirical coverage: 94.6%

---

## Data Requirements

**Training Data:** 6,600 NBA games (2015-2020)
- 48-minute time series per game
- Minute-by-minute differential
- Chronologically split (no shuffling!)

**Inference Input:** 18-point array
- Minutes 0-17 score differential
- Measured from game start to 6:00 remaining in Q2

**Output:** Halftime differential prediction
- Point forecast (ensemble)
- 95% confidence interval (conformal)
- Similar historical games (Dejavu interpretability)

---

## Dependencies

```txt
numpy
pandas
torch
scikit-learn
pickle
json
```

---

## Performance Benchmarks

**Speed:**
- Dejavu: 68.87 ms/game
- LSTM: ~10 ms/game
- Ensemble: ~80 ms total
- Well within real-time requirements

**Accuracy:**
- 50% of predictions within 4.5 points
- 90% of predictions within 11 points
- Perfect predictions: 4 games (0.0 error)

---

## Next Steps for Optimization

**Potential improvements (for future iterations):**
1. Add preprocessing (smoothing) - paper shows 28% improvement
2. Try DTW distance instead of Euclidean
3. GPU training for LSTM (longer epochs)
4. Hyperparameter tuning for ensemble weights
5. Add more training data (newer seasons)
6. Feature engineering (team stats, momentum)

---

## Research Citations

**Dejavu:**
Kang et al., "Déjà vu: A data-centric forecasting approach through time series cross-similarity", arXiv 2020

**Conformal:**
Schlembach et al., "Conformal Multistep-Ahead Multivariate Time-Series Forecasting", PMLR 2022

**LSTM/Informer:**
Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI 2021

---

## Files Structure

```
X. MVP Model/
├── README.md (this file)
├── MVP_SPECIFICATIONS.md (detailed specs)
├── USAGE_GUIDE.md (how to use in production)
├── Models/
│   ├── dejavu_k500.pkl
│   ├── lstm_best.pth
│   ├── lstm_normalization.pkl
│   └── conformal_predictor.pkl
├── Code/
│   ├── dejavu_model.py
│   ├── lstm_model.py
│   ├── ensemble_model.py
│   └── conformal_wrapper.py
├── Data/
│   ├── splits/ (train/val/cal/test)
│   └── dataset_summary.json
└── Results/
    ├── dejavu_test_results.json
    ├── lstm_test_results.json
    ├── ensemble_test_results.json
    └── conformal_test_results.json
```

---

**This is your MVP model - ready for integration with NBA_API and production deployment!**

*Built following exact research specifications, verified and tested*

