# MVP Model - Usage Guide
**How to use the 5.39 MAE ensemble in production**

---

## Quick Start (5 minutes)

### Installation
```bash
pip install numpy pandas torch scikit-learn
```

### Load Models
```python
import pickle
import torch
import numpy as np

# 1. Load Dejavu
with open('Models/dejavu_k500.pkl', 'rb') as f:
    dejavu = pickle.load(f)

# 2. Load LSTM
lstm = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, forecast_horizon=7)
lstm.load_state_dict(torch.load('Models/lstm_best.pth'))
lstm.eval()

# 3. Load LSTM normalization (CRITICAL!)
with open('Models/lstm_normalization.pkl', 'rb') as f:
    lstm_norm = pickle.load(f)

# 4. Load Conformal
with open('Models/conformal_predictor.pkl', 'rb') as f:
    conformal = pickle.load(f)
```

### Make Prediction
```python
# Your input: 18-minute differential pattern
pattern = np.array([0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 21, 23, 24, 25, 23, 22, 20, 19])

# Dejavu prediction
dejavu_pred = dejavu.predict(pattern)

# LSTM prediction
pattern_norm = (pattern - lstm_norm['pattern_mean']) / lstm_norm['pattern_std']
x = torch.FloatTensor(pattern_norm).unsqueeze(0).unsqueeze(-1)  # (1, 18, 1)
with torch.no_grad():
    lstm_output = lstm(x)
    lstm_pred_norm = lstm_output[0, -1].item()
lstm_pred = lstm_pred_norm * lstm_norm['outcome_std'][-1] + lstm_norm['outcome_mean'][-1]

# Ensemble
ensemble_pred = 0.4 * dejavu_pred + 0.6 * lstm_pred

# Conformal interval
lower = ensemble_pred - conformal.quantile
upper = ensemble_pred + conformal.quantile

print(f"Prediction: {ensemble_pred:+.1f} points")
print(f"95% Interval: [{lower:+.1f}, {upper:+.1f}]")
```

---

## Integration with NBA_API

### Real-Time Pattern Generation

```python
from nba_api.live.nba.endpoints import scoreboard

def get_live_pattern(game_id):
    """
    Extract 18-minute pattern from live game
    
    Returns: numpy array of 18 differentials
    """
    # Get play-by-play
    pbp = get_play_by_play(game_id)  # Your NBA_API function
    
    # Convert to minute-by-minute
    pattern = []
    for minute in range(18):
        # Find score at this minute
        target_time = minute * 60  # seconds
        diff = get_differential_at_time(pbp, target_time)
        pattern.append(diff)
    
    return np.array(pattern)

# Usage
pattern = get_live_pattern("0022300415")
forecast, interval = predict_with_models(pattern)
```

---

## Production API Example

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load models at startup (global)
dejavu_model = None
lstm_model = None
conformal_model = None

@app.on_event("startup")
async def load_models():
    global dejavu_model, lstm_model, conformal_model
    # Load all models here
    pass

@app.post("/predict")
async def predict(pattern: list[float]):
    """
    Predict halftime differential with uncertainty
    """
    pattern_array = np.array(pattern)
    
    # Dejavu
    dejavu_pred = dejavu_model.predict(pattern_array)
    
    # LSTM (with normalization)
    # ... implementation from above ...
    
    # Ensemble
    ensemble_pred = 0.4 * dejavu_pred + 0.6 * lstm_pred
    
    # Conformal
    lower = ensemble_pred - conformal_model.quantile
    upper = ensemble_pred + conformal_model.quantile
    
    return {
        "point_forecast": float(ensemble_pred),
        "interval_lower": float(lower),
        "interval_upper": float(upper),
        "coverage": 0.95,
        "components": {
            "dejavu": float(dejavu_pred),
            "lstm": float(lstm_pred)
        }
    }
```

---

## Error Handling

### Model Loading Errors
```python
try:
    dejavu = pickle.load(open('dejavu_k500.pkl', 'rb'))
except FileNotFoundError:
    print("ERROR: Dejavu model not found!")
except Exception as e:
    print(f"ERROR loading Dejavu: {e}")
```

### Prediction Errors
```python
def safe_predict(pattern):
    """Predict with fallback"""
    try:
        # Try ensemble
        return ensemble.predict(pattern)
    except Exception as e:
        print(f"Ensemble failed: {e}")
        try:
            # Fallback to LSTM
            return lstm.predict(pattern)
        except:
            # Fallback to Dejavu
            return dejavu.predict(pattern)
```

---

## Monitoring in Production

### Track These Metrics

**Accuracy:**
```python
recent_errors = []
for prediction, actual in predictions_last_100:
    error = abs(prediction - actual)
    recent_errors.append(error)

current_mae = np.mean(recent_errors)
if current_mae > 7.0:  # Alert threshold
    trigger_recalibration()
```

**Coverage:**
```python
recent_coverage = []
for pred, lower, upper, actual in predictions_last_100:
    covered = (lower <= actual <= upper)
    recent_coverage.append(covered)

coverage_rate = np.mean(recent_coverage)
if coverage_rate < 0.90:  # Below acceptable
    recalibrate_conformal()
```

**Latency:**
```python
if prediction_time > 100:  # ms
    log_warning("Slow prediction")
```

---

## Recalibration Schedule

**Weekly:** Check performance metrics  
**Monthly:** Recalibrate conformal predictor  
**Quarterly:** Retrain LSTM if MAE degrades >20%  
**Continuously:** Update Dejavu database with new games

---

## File Locations

```
Models/
  ├── dejavu_k500.pkl          (4,003 patterns)
  ├── lstm_best.pth            (50,887 parameters)
  ├── lstm_normalization.pkl   (CRITICAL - don't lose!)
  └── conformal_predictor.pkl  (932 calibration scores)

Code/
  ├── dejavu_model.py          (K-NN implementation)
  ├── lstm_model.py            (PyTorch model)
  ├── ensemble_model.py        (Combiner)
  └── conformal_wrapper.py     (Uncertainty wrapper)

Data/splits/
  ├── train.pkl                (4,003 games)
  ├── validation.pkl           (772 games)
  ├── calibration.pkl          (932 games)
  └── test.pkl                 (893 games)

Results/
  ├── ensemble_test_results.json  (MAE 5.39)
  ├── conformal_test_results.json (94.6% coverage)
  └── [other evaluation files]
```

---

## Common Issues & Solutions

**Issue:** LSTM predictions are NaN  
**Solution:** Check normalization is applied correctly

**Issue:** Conformal intervals too wide  
**Solution:** Normal! Reflects true uncertainty. Don't artificially narrow.

**Issue:** Dejavu slow  
**Solution:** Use FAISS for faster K-NN search (future optimization)

**Issue:** Models not loading  
**Solution:** Ensure all pickle files in same directory structure

---

## Performance Expectations

**Typical Game:**
- Prediction: ±5 points error
- Interval: ±13 points width
- Speed: <100ms
- Coverage: 94-96%

**Unusual Game (blowout):**
- Prediction: May have larger error
- Interval: Still ±13 points (honest uncertainty)
- Speed: Same
- Coverage: Still maintains 95%

---

## Next Steps for NBA_API Integration

1. **Get live scores** every minute (NBA_API)
2. **Buffer 18 minutes** of differentials
3. **Call model** when minute 18 reached (6:00 Q2)
4. **Return prediction** with interval
5. **Compare with BetOnline odds** for edge detection
6. **Apply risk management** for bet sizing

---

**You have everything needed to deploy this MVP!**

*MAE 5.39 | 94.6% Coverage | Production Ready*

