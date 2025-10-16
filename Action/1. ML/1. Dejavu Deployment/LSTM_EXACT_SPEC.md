# LSTM EXACT Specification (Research-Verified)

**Source:** `ML/Action Steps Folder/06_INFORMER_TRAINING.md`

**NO ASSUMPTIONS. NO GUESSING. ONLY RESEARCH FACTS.**

---

## Why LSTM (Not Full Informer)?

**From 06_INFORMER_TRAINING.md lines 28-30:**
> "Important from Paper: Informer designed for LONG sequences (336-1440 input tested in Zhou et al. AAAI 2021).  
> For NBA's short 18-minute input, LSTM is actually more appropriate!"

**✅ DECISION: Use LSTM for NBA (18-minute sequences too short for Informer)**

---

## EXACT LSTM Architecture

**From lines 52-81:**

### Model Parameters:
```python
input_size=1           # Single feature (differential)
hidden_size=64         # Line 59 - EXACT from research
num_layers=2           # Line 60 - EXACT from research
dropout=0.1            # Line 61
forecast_horizon=6     # Line 62 - Predict 6 steps (min 18-23)
```

### Architecture:
```python
LSTM:
  - Input: (batch, seq_len=18, features=1)
  - LSTM layers: 2 layers, 64 hidden units each
  - Dropout: 0.1 (between layers)
  - Output layer: Linear(64 → 6)
  - Output: (batch, 6) - 6-step forecast
```

---

## Training Configuration

**From lines 237-286:**

### Hyperparameters:
```python
epochs = 50                 # Line 240
batch_size = 32             # Line 241
hidden_size = 64            # Line 242
num_layers = 2              # Line 243
learning_rate = 1e-3        # Line 114
optimizer = Adam            # Line 114
loss_function = MSELoss     # Line 115
```

### Early Stopping:
```python
patience = 5                # Wait 5 epochs without improvement
monitor = validation_mae
mode = 'min'
```

### Gradient Clipping:
```python
max_norm = 1.0              # Line 133
```

---

## Data Format

**From lines 177-217:**

### Input (Pattern):
- Shape: `(18,)` - 18-minute differential sequence
- Normalize: Z-score per split
- Reshape to: `(18, 1)` for LSTM input

### Output (Outcome):
- Shape: `(6,)` - 6 steps from minute 18-23
- Minutes 18, 19, 20, 21, 22, 23 differentials
- Normalize: Z-score per split

### Normalization (CRITICAL):
```python
# Fit on training set ONLY
pattern_mean = np.mean(train_patterns, axis=0)
pattern_std = np.std(train_patterns, axis=0) + 1e-10

outcome_mean = np.mean(train_outcomes, axis=0)
outcome_std = np.std(train_outcomes, axis=0) + 1e-10

# Apply to all sets (train/val/test)
patterns_norm = (patterns - pattern_mean) / pattern_std
outcomes_norm = (outcomes - outcome_mean) / outcome_std
```

**CRITICAL:** Must use TRAINING statistics for val/test normalization!

---

## Expected Performance

**From MODELSYNERGY.md line 853:**
> "LSTM alone: MAE ~4.0 pts"

**Target:** MAE ~4.0 points on test set (893 games)

**Comparison:**
- Dejavu: 6.17 points ✅ (already achieved)
- LSTM: ~4.0 points (target)
- Ensemble: ~3.5 points (after combining)

---

## Current Data Status

**We have:**
- Training: 4,003 games
- Validation: 772 games  
- Calibration: 932 games
- Test: 893 games

**All games have:**
- `pattern`: 18-point array (minutes 0-17)
- `timeseries`: 48-point array (full game)
- Need to extract: `outcome` = minutes 18-23 differentials (6 points)

---

## What We Need to Build

### 1. Extract 6-step outcomes
Current data has:
- `diff_at_halftime`: Single value (minute 24)

Need to extract:
- `outcome_6steps`: Array of 6 values (minutes 18-23)

### 2. Build LSTM model
```python
class LSTMForecaster(nn.Module):
    # Exact architecture from lines 52-99
```

### 3. Training script
```python
# Exact training loop from lines 225-286
```

### 4. Evaluation
```python
# Evaluate on 893 test games
# Target: MAE ~4.0 points
```

---

## Implementation Steps

**Step 1:** Extract 6-step outcomes from timeseries
**Step 2:** Create PyTorch dataset
**Step 3:** Build LSTM model (exact spec)
**Step 4:** Train with early stopping
**Step 5:** Evaluate on test set
**Step 6:** Verify MAE ~4.0 points

---

*Following research - building LSTM exactly as specified*

