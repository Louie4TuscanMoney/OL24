# Data Split Specification (Research-Based)

**Source:** `ML/Action Steps Folder/03_DATA_SPLITTING.md`  
**Date:** Following verified research methodology

---

## Research-Verified Split Strategy

### The 4-Way Split (CRITICAL!)

From `03_DATA_SPLITTING.md` lines 33-36:

```python
train_ratio=0.60        # 60% - Training
val_ratio=0.10          # 10% - Validation  
calibration_ratio=0.15  # 15% - Calibration
test_ratio=0.15         # 15% - Test
```

**Total:** 60% + 10% + 15% + 15% = 100% ✅

---

## What Each Split Does

| Split | % | Games (6,600 total) | Purpose | Used By |
|-------|---|---------------------|---------|---------|
| **Training** | 60% | ~3,960 games | Build Dejavu database, Train LSTM | Dejavu + LSTM |
| **Validation** | 10% | ~660 games | Tune hyperparameters, early stopping | LSTM |
| **Calibration** | 15% | ~990 games | Calculate Conformal quantiles (±3.8) | Conformal |
| **Test** | 15% | ~990 games | Final evaluation (never seen by models) | Evaluation only |

---

## Critical Requirements (From Research)

### 1. **CHRONOLOGICAL ONLY** (lines 20, 58-59)
```python
# CRITICAL: Chronological split only (no shuffling!)
df = df.sort_values('game_date').reset_index(drop=True)
```

**Why:** Time series forecasting requires temporal order. Shuffling = data leakage!

### 2. **NO TEMPORAL OVERLAP** (lines 74-77)
```python
# Validate no temporal overlap
assert train_df['game_date'].max() < val_df['game_date'].min()
assert val_df['game_date'].max() < calibration_df['game_date'].min()
assert calibration_df['game_date'].max() < test_df['game_date'].min()
```

**Why:** Future cannot leak into past. Each split must be fully before the next.

### 3. **MINIMUM SAMPLE SIZES** (lines 219-224)
```python
min_requirements = {
    'train': 500,
    'val': 100,
    'calibration': 100,
    'test': 100
}
```

**Why:** Conformal needs ≥100 calibration samples for reliable quantiles.

---

## Our Current Dataset

**Total games:** 6,600 games  
**Date range:** April 2016 - September 2020  
**Seasons:** 6 seasons (2015-16 through 2020-21)

### Expected Split (6,600 games):

| Split | Games | Approximate Date Range |
|-------|-------|------------------------|
| **Training** | 3,960 (60%) | April 2016 - ~March 2019 |
| **Validation** | 660 (10%) | ~March 2019 - ~July 2019 |
| **Calibration** | 990 (15%) | ~July 2019 - ~February 2020 |
| **Test** | 990 (15%) | ~February 2020 - September 2020 |

---

## What We Need to Implement

From `03_DATA_SPLITTING.md`, we need a `ChronologicalDataSplitter` class:

**Input:**
- DataFrame with `game_date` column (or `date` in our case)
- All 6,600 games from `complete_timeseries.pkl`

**Output:**
- `train.parquet` - 3,960 games
- `validation.parquet` - 660 games
- `calibration.parquet` - 990 games
- `test.parquet` - 990 games
- `split_metadata.json` - Summary information

**Key Features:**
1. Sort by date first
2. Split chronologically (earliest → latest)
3. Validate no overlap
4. Save each split separately
5. Store metadata

---

## Why 4 Splits (Not Just Train/Test)?

From MODELSYNERGY.md lines 520-607:

**Training (60%):**
- Dejavu: Build pattern database (~3,960 patterns)
- LSTM: Train neural network weights

**Validation (10%):**
- LSTM: Tune hyperparameters (learning rate, hidden size, etc.)
- LSTM: Early stopping to prevent overfitting

**Calibration (15%):**
- Conformal: Calculate the ±3.8 quantile
- Process: Run ensemble on calibration games → measure errors → pick 95th percentile
- **CRITICAL:** Must be independent from training (no data leakage)

**Test (15%):**
- Final evaluation only
- Never used during model development
- Unbiased performance assessment

---

## Validation Checklist

From lines 186-252, we must verify:

✅ **No game ID overlap** between any splits  
✅ **Temporal ordering:** Train < Val < Cal < Test  
✅ **Sufficient samples:** All splits meet minimum requirements  
✅ **No data leakage:** Each game appears in exactly one split  

---

## Next Step

**Build:** `test_06_data_split.py`

This script will:
1. Load `complete_timeseries.pkl` (6,600 games)
2. Sort by date chronologically
3. Split into 60/10/15/15 ratio
4. Validate no overlap
5. Save 4 parquet files
6. Print summary statistics

Then we'll be ready to build the Dejavu K-NN model with proper train/test separation!

---

*Following research methodology from 03_DATA_SPLITTING.md*

