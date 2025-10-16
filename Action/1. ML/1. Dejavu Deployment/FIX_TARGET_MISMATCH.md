# CRITICAL ERROR FOUND: Target Mismatch

## The Problem

**Dejavu predicts:** `diff_at_halftime` (minute 24) - TRUE halftime  
**LSTM predicts:** `outcome_6step[-1]` (minute 23) - NOT halftime

**Example from test:**
```
Dejavu target: 10 points (minute 24)
LSTM target:   12 points (minute 23)
Difference:    2 points
```

**This is WRONG!** Both models must predict the SAME target!

---

## The Root Cause

**From our time series:**
- Minutes 0-17: Pattern (18 points)
- Minutes 18-23: 6-step sequence
- Minute 24: ACTUAL halftime (0:00 Q2)

**Current setup:**
- LSTM trained on 6 steps → last step is minute 23
- Dejavu trained on minute 24
- **Different targets = invalid comparison!**

---

## The Fix

**Option 1: LSTM predicts 7 steps (minutes 18-24)**
- Change forecast_horizon from 6 to 7
- Retrain LSTM
- Use step 7 (minute 24) for comparison

**Option 2: Both predict minute 23**
- Change Dejavu to use minute 23
- Keep LSTM as is

**Option 3: Interpolate/extend LSTM**
- Keep current training
- Extrapolate from minute 23 to 24

---

## Correct Solution: Option 1

**Both models should predict minute 24 (true halftime).**

Change LSTM to:
```python
forecast_horizon = 7  # Minutes 18-24 (include halftime!)
```

Then both models predict the same target and comparison is valid.

---

**STATUS: NEED TO RETRAIN LSTM WITH CORRECTED TARGET**

