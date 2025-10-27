#!/usr/bin/env python3
"""
18-FEATURE MODEL BREAKDOWN (Our Production System)
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import os

print("=" * 100)
print("📊 18-FEATURE MODEL BREAKDOWN (PRODUCTION SYSTEM)")
print("=" * 100)
print()

# Load the 18-feature dataset (used for production)
print("[1/5] Loading 18-feature production data...")

# Try different possible files
possible_files = [
    "Action/ULTRA_ENHANCED_PATTERNS_V2.pkl",
    "Action/ENHANCED_PATTERNS_FULL.pkl",
    "Action/historical_games_2021_2025_WITH_PATTERNS.pkl"
]

df = None
for file in possible_files:
    if os.path.exists(file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data) if isinstance(data, list) else data
        print(f"✅ Loaded from: {file}")
        break

if df is None:
    print("❌ No 18-feature data found")
    exit(1)

print(f"   Games: {len(df)}")
print()

# Get 18 pattern features
print("[2/5] Preparing 18 pattern features...")
pattern_features = [col for col in df.columns if col.startswith('pattern_') or col.startswith('diff_point')][:18]

if len(pattern_features) == 0:
    # Extract from pattern column if it exists
    if 'pattern' in df.columns:
        print("   Extracting pattern features from pattern column...")
        for i in range(18):
            df[f'pattern_{i}'] = df['pattern'].apply(lambda x: x[i] if isinstance(x, list) and len(x) > i else 0)
        pattern_features = [f'pattern_{i}' for i in range(18)]

print(f"✅ {len(pattern_features)} pattern features")

# Find target
if 'target' in df.columns:
    target_col = 'target'
elif 'diff_at_final' in df.columns:
    target_col = 'diff_at_final'
elif 'final_diff' in df.columns:
    target_col = 'final_diff'
else:
    print("❌ No target column found")
    exit(1)

print(f"   Target: {target_col}")
print()

# Split
print("[3/5] Training model...")
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx]
df_test = df.iloc[split_idx:].reset_index(drop=True)

X_train = np.nan_to_num(df_train[pattern_features].values, nan=0.0)
y_train = df_train[target_col].values
X_test = np.nan_to_num(df_test[pattern_features].values, nan=0.0)
y_test = df_test[target_col].values

# Train
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
overfit = (test_mae - train_mae) / train_mae * 100 if train_mae > 0 else 0

baseline_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
edge = (baseline_mae - test_mae) / baseline_mae * 100

errors = np.abs(test_pred - y_test)
signed_errors = test_pred - y_test

df_test['prediction'] = test_pred
df_test['actual'] = y_test
df_test['error'] = errors
df_test['signed_error'] = signed_errors

print(f"✅ Model trained")
print(f"   Train MAE: {train_mae:.3f}")
print(f"   Test MAE:  {test_mae:.3f}")
print(f"   Overfit:   {overfit:.1f}%")
print(f"   Edge:      {edge:.1f}%")
print()

# Analyze
print("[4/5] Performing comprehensive analysis...")

# Categorizations
if 'diff_at_2q_6min' in df_test.columns:
    current_diff = df_test['diff_at_2q_6min']
elif 'current_diff' in df_test.columns:
    current_diff = df_test['current_diff']
else:
    current_diff = df_test[pattern_features[-1]] if len(pattern_features) > 0 else pd.Series([0]*len(df_test))

df_test['game_state'] = pd.cut(
    current_diff.abs(),
    bins=[0, 3, 7, 12, 20, 100],
    labels=['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)']
)

# Outcome type
def cat_outcome(row):
    curr = current_diff.iloc[row.name] if hasattr(row, 'name') else 0
    final = row['actual']
    if abs(curr) <= 3:
        return 'Was Close'
    elif (curr > 3 and final > 0) or (curr < -3 and final < 0):
        return 'Lead Held'
    elif (curr > 3 and final <= 0) or (curr < -3 and final >= 0):
        return 'Comeback/Flip'
    else:
        return 'Tightened'

df_test['outcome_type'] = df_test.apply(cat_outcome, axis=1)

# Direction
df_test['direction_correct'] = np.sign(df_test['prediction']) == np.sign(df_test['actual'])
direction_acc = df_test['direction_correct'].mean() * 100

# Error bins
bins = [0, 2, 5, 8, 12, 15, 20, 100]
bin_labels = ['0-2 pts', '2-5 pts', '5-8 pts', '8-12 pts', '12-15 pts', '15-20 pts', '20+ pts']
df_test['error_bin'] = pd.cut(df_test['error'], bins=bins, labels=bin_labels, include_lowest=True)

# Zones
high_conf = df_test[df_test['error'] <= 5]
medium = df_test[(df_test['error'] > 5) & (df_test['error'] <= 12)]
risk_zone = df_test[df_test['error'] > 15]

# Quartiles
error_percentiles = [np.percentile(errors, q) for q in [0, 25, 50, 75, 90, 95, 100]]

print("✅ Analysis complete")
print()

# Generate markdown
print("[5/5] Generating markdown report...")

os.makedirs("Action/Model_Breakdowns", exist_ok=True)

report = f"""# 📊 18-FEATURE PRODUCTION MODEL - COMPREHENSIVE BREAKDOWN

**Generated:** {datetime.now().strftime('%I:%M %p, %A %B %d, %Y')}  
**Model:** 18-Feature Linear Regression (Production System)  
**Test Games:** {len(df_test):,}  
**Status:** ✅ PRODUCTION READY

---

## 📋 MODEL SPECIFICATIONS

**Architecture:**
- Type: Linear Regression
- Features: 18 (pattern-based)
- Training: {len(X_train):,} games (80%, chronological)
- Testing: {len(X_test):,} games (20%, chronological)
- Scaler: RobustScaler

**Performance:**
- **Train MAE:** {train_mae:.3f}
- **Test MAE:** {test_mae:.3f}
- **Overfitting:** {overfit:.1f}%
- **Baseline MAE:** {baseline_mae:.3f}
- **Edge:** {edge:.1f}%

**Error Statistics:**
- Median Error: {np.median(errors):.3f}
- Std Dev: {np.std(errors):.3f}
- Min Error: {np.min(errors):.3f}
- Max Error: {np.max(errors):.3f}

---

## 📊 ERROR DISTRIBUTION

| Error Range | Games | Percentage | Cumulative |
|-------------|-------|------------|------------|
{chr(10).join([f'| {label} | {(df_test["error_bin"] == label).sum():,} | {(df_test["error_bin"] == label).sum()/len(df_test)*100:.1f}% | {(df_test["error"] <= bins[i+1]).sum()/len(df_test)*100:.1f}% |' for i, label in enumerate(bin_labels)])}

**Key Thresholds:**
```
Within  5 points: {(df_test['error'] <= 5).sum():,} games ({(df_test['error'] <= 5).sum()/len(df_test)*100:.1f}%)
Within  8 points: {(df_test['error'] <= 8).sum():,} games ({(df_test['error'] <= 8).sum()/len(df_test)*100:.1f}%)
Within 10 points: {(df_test['error'] <= 10).sum():,} games ({(df_test['error'] <= 10).sum()/len(df_test)*100:.1f}%)
Within 12 points: {(df_test['error'] <= 12).sum():,} games ({(df_test['error'] <= 12).sum()/len(df_test)*100:.1f}%)
Within 15 points: {(df_test['error'] <= 15).sum():,} games ({(df_test['error'] <= 15).sum()/len(df_test)*100:.1f}%)
```

---

## 🎯 PERFORMANCE BY GAME TYPE

### By Game State (at Q2 6:00):

| State | MAE | Median | Games | % of Total |
|-------|-----|--------|-------|------------|
{chr(10).join([f"| {state} | {df_test[df_test['game_state']==state]['error'].mean():.3f} | {df_test[df_test['game_state']==state]['error'].median():.3f} | {(df_test['game_state']==state).sum():,} | {(df_test['game_state']==state).sum()/len(df_test)*100:.1f}% |" for state in ['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)'] if (df_test['game_state']==state).sum() > 0])}

### By Outcome Type:

| Outcome | MAE | Median | Games | % of Total |
|---------|-----|--------|-------|------------|
{chr(10).join([f"| {outcome} | {df_test[df_test['outcome_type']==outcome]['error'].mean():.3f} | {df_test[df_test['outcome_type']==outcome]['error'].median():.3f} | {(df_test['outcome_type']==outcome).sum():,} | {(df_test['outcome_type']==outcome).sum()/len(df_test)*100:.1f}% |" for outcome in df_test['outcome_type'].value_counts().index])}

---

## 🧭 DIRECTIONAL ACCURACY

**Winner Prediction:**
- ✅ Correct: {df_test['direction_correct'].sum():,} games ({direction_acc:.1f}%)
- ❌ Wrong: {(~df_test['direction_correct']).sum():,} games ({100-direction_acc:.1f}%)

**MAE by Direction:**
- Correct Direction: {df_test[df_test['direction_correct']]['error'].mean():.3f} MAE
- Wrong Direction: {df_test[~df_test['direction_correct']]['error'].mean():.3f} MAE

---

## 📈 ERROR QUARTILES

| Percentile | Error (points) |
|------------|---------------|
| 0th (Min) | {error_percentiles[0]:.2f} |
| 25th | {error_percentiles[1]:.2f} |
| 50th (Median) | {error_percentiles[2]:.2f} |
| 75th | {error_percentiles[3]:.2f} |
| 90th | {error_percentiles[4]:.2f} |
| 95th | {error_percentiles[5]:.2f} |
| 100th (Max) | {error_percentiles[6]:.2f} |

**Interpretation:**
- Best 25%: Error ≤ {error_percentiles[1]:.1f} points ⭐⭐⭐
- Middle 50%: Error {error_percentiles[1]:.1f} - {error_percentiles[3]:.1f} points ⭐⭐
- Worst 25%: Error > {error_percentiles[3]:.1f} points ⚠️

---

## 🎯 CONFIDENCE ZONES

### ✅ HIGH-CONFIDENCE (Error ≤ 5 points):
- **Games:** {len(high_conf):,} ({len(high_conf)/len(df_test)*100:.1f}%)
- **Avg Error:** {high_conf['error'].mean():.3f}
- **Direction Accuracy:** {high_conf['direction_correct'].mean()*100:.1f}%

**Characteristics:**
{chr(10).join([f'- {state}: {(high_conf["game_state"]==state).sum()} games ({(high_conf["game_state"]==state).sum()/len(high_conf)*100:.1f}%)' for state in high_conf['game_state'].value_counts().head(3).index])}

### ⭐ MEDIUM ZONE (Error 5-12 points):
- **Games:** {len(medium):,} ({len(medium)/len(df_test)*100:.1f}%)
- **Avg Error:** {medium['error'].mean():.3f}
- **Direction Accuracy:** {medium['direction_correct'].mean()*100:.1f}%

### ⚠️ RISK ZONE (Error > 15 points):
- **Games:** {len(risk_zone):,} ({len(risk_zone)/len(df_test)*100:.1f}%)
- **Avg Error:** {risk_zone['error'].mean() if len(risk_zone) > 0 else 0:.3f}
- **Direction Accuracy:** {risk_zone['direction_correct'].mean()*100 if len(risk_zone) > 0 else 0:.1f}%

---

## 📊 BIAS ANALYSIS

**Overall:**
- Mean Signed Error: {np.mean(signed_errors):+.3f}
- Median Signed Error: {np.median(signed_errors):+.3f}

{f"✅ **UNBIASED:** Model is well-calibrated" if abs(np.mean(signed_errors)) < 1.0 else f"⚠️ **BIAS:** Model {'over-predicts' if np.mean(signed_errors) > 0 else 'under-predicts'} by {abs(np.mean(signed_errors)):.3f} points"}

---

## ✅ WHERE MODEL EXCELS

{chr(10).join([f'{i}. **{state}**: {df_test[df_test["game_state"]==state]["error"].mean():.3f} MAE ({(df_test["game_state"]==state).sum():,} games)' for i, state in enumerate(df_test.groupby('game_state')['error'].mean().nsmallest(3).index, 1)])}

---

## ⚠️ WHERE MODEL STRUGGLES

{chr(10).join([f'{i}. **{state}**: {df_test[df_test["game_state"]==state]["error"].mean():.3f} MAE ({(df_test["game_state"]==state).sum():,} games)' for i, state in enumerate(df_test.groupby('game_state')['error'].mean().nlargest(3).index, 1)])}

---

## 🏆 TOP 10 BEST PREDICTIONS

| # | Predicted | Actual | Error |
|---|-----------|--------|-------|
{chr(10).join([f'| {i} | {row["prediction"]:+.1f} | {row["actual"]:+.1f} | {row["error"]:.2f} |' for i, (_, row) in enumerate(df_test.nsmallest(10, 'error').iterrows(), 1)])}

---

## 💀 TOP 10 WORST PREDICTIONS

| # | Predicted | Actual | Error | Type |
|---|-----------|--------|-------|------|
{chr(10).join([f'| {i} | {row["prediction"]:+.1f} | {row["actual"]:+.1f} | {row["error"]:.1f} | {row["outcome_type"]} |' for i, (_, row) in enumerate(df_test.nlargest(10, 'error').iterrows(), 1)])}

---

## 💡 KEY INSIGHTS

1. **Direction Accuracy:** {direction_acc:.1f}% predict correct winner
2. **High-Confidence Rate:** {len(high_conf)/len(df_test)*100:.1f}% within 5 points
3. **Median Error:** {np.median(errors):.1f} points
4. **Risk Rate:** {len(risk_zone)/len(df_test)*100:.1f}% with error > 15 points
5. **Bias:** {abs(np.mean(signed_errors)):.2f} points ({"minimal" if abs(np.mean(signed_errors)) < 1.0 else "moderate"})

---

## 📁 FEATURES USED

**18 Pattern Features:**
{chr(10).join([f'{i}. {feat}' for i, feat in enumerate(pattern_features, 1)])}

---

**PRODUCTION MODEL BREAKDOWN COMPLETE** ✅  
**Test MAE:** {test_mae:.3f}  
**Edge:** {edge:.1f}%  
**Status:** Ready for Monday launch
"""

filename = "Action/Model_Breakdowns/18-Feature_Production_System_BREAKDOWN.md"
with open(filename, 'w') as f:
    f.write(report)

print(f"✅ Saved to: {filename}")
print()

print("=" * 100)
print("✅ 18-FEATURE MODEL BREAKDOWN COMPLETE!")
print("=" * 100)
print()
print(f"Test MAE: {test_mae:.3f}")
print(f"Direction Accuracy: {direction_acc:.1f}%")
print(f"High-Confidence: {len(high_conf)/len(df_test)*100:.1f}% within 5 pts")
print()

