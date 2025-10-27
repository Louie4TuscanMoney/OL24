#!/usr/bin/env python3
"""
🔬 COMPLETE SYSTEM AUDIT
Check EVERYTHING from data to predictions
No assumptions. Verify all claims.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from datetime import datetime

print("="*80)
print("🔬 COMPLETE SYSTEM AUDIT - VERIFY EVERYTHING")
print("="*80)
print()
print("User request: Check for bugs, verify we're not hallucinating")
print()

# ============================================================================
# AUDIT 1: DATA INTEGRITY
# ============================================================================
print("[AUDIT 1/8] DATA INTEGRITY CHECK")
print("="*80)
print()

# Load V3 dataset
with open('ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Total games loaded: {len(data)}")
print()

# Check for duplicates
game_ids = [g.get('game_id') for g in data]
unique_ids = len(set(game_ids))
print(f"Unique game IDs: {unique_ids}")
print(f"Duplicates: {len(game_ids) - unique_ids}")

if len(game_ids) != unique_ids:
    print("❌ WARNING: DUPLICATE GAMES DETECTED")
else:
    print("✅ No duplicates")
print()

# Check date range
dates = [g.get('date', '') for g in data if g.get('date')]
if dates:
    dates_sorted = sorted(dates)
    print(f"Date range: {dates_sorted[0]} to {dates_sorted[-1]}")
    
    # Check if chronologically ordered
    is_chronological = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
    if is_chronological:
        print("✅ Data is chronologically ordered (good for time series)")
    else:
        print("⚠️  Data NOT chronologically ordered (could cause data leakage)")
print()

# Check for missing targets
missing_half = sum(1 for g in data if g.get('diff_at_halftime') is None)
missing_final = sum(1 for g in data if g.get('diff_at_final') is None)
print(f"Missing halftime targets: {missing_half}")
print(f"Missing final targets: {missing_final}")
print()

# Check target distributions
half_diffs = [g.get('diff_at_halftime', 0) for g in data]
final_diffs = [g.get('diff_at_final', 0) for g in data]

print(f"Halftime differentials:")
print(f"  Mean: {np.mean(half_diffs):.2f}")
print(f"  Std: {np.std(half_diffs):.2f}")
print(f"  Min: {np.min(half_diffs):.0f}")
print(f"  Max: {np.max(half_diffs):.0f}")
print()

print(f"Final differentials:")
print(f"  Mean: {np.mean(final_diffs):.2f}")
print(f"  Std: {np.std(final_diffs):.2f}")
print(f"  Min: {np.min(final_diffs):.0f}")
print(f"  Max: {np.max(final_diffs):.0f}")
print()

# ============================================================================
# AUDIT 2: TRAIN/TEST SPLIT VERIFICATION
# ============================================================================
print("[AUDIT 2/8] TRAIN/TEST SPLIT VERIFICATION")
print("="*80)
print()

split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"Split ratio: 80/20")
print(f"Train: {len(train_data)} games ({len(train_data)/len(data)*100:.1f}%)")
print(f"Test: {len(test_data)} games ({len(test_data)/len(data)*100:.1f}%)")
print()

# Check if test is actually chronologically AFTER train
train_dates = [g.get('date', '') for g in train_data if g.get('date')]
test_dates = [g.get('date', '') for g in test_data if g.get('date')]

if train_dates and test_dates:
    latest_train = max(train_dates)
    earliest_test = min(test_dates)
    
    print(f"Latest train date: {latest_train}")
    print(f"Earliest test date: {earliest_test}")
    
    if earliest_test >= latest_train:
        print("✅ Test set is chronologically AFTER train set (no leakage)")
    else:
        print("❌ WARNING: TEMPORAL LEAKAGE - test overlaps with train")
print()

# Check train/test distributions
train_half = [g.get('diff_at_halftime', 0) for g in train_data]
test_half = [g.get('diff_at_halftime', 0) for g in test_data]

print(f"Train halftime diff: mean={np.mean(train_half):.2f}, std={np.std(train_half):.2f}")
print(f"Test halftime diff:  mean={np.mean(test_half):.2f}, std={np.std(test_half):.2f}")
print()

if abs(np.mean(train_half) - np.mean(test_half)) > 2:
    print("⚠️  WARNING: Train/test distributions differ significantly")
else:
    print("✅ Train/test distributions similar")
print()

# ============================================================================
# AUDIT 3: FEATURE EXTRACTION VERIFICATION
# ============================================================================
print("[AUDIT 3/8] FEATURE EXTRACTION VERIFICATION")
print("="*80)
print()

exclude = ['game_id', 'date', 'season', 'home_team', 'away_team', 'pattern',
           'diff_at_final', 'diff_at_halftime', 'diff_at_2q_6min']
all_features = [k for k in data[0].keys() if k not in exclude]

print(f"Total features: {len(all_features)}")
print()

# Check for NaN values
X = []
for game in data:
    features = [game.get(f, 0) for f in all_features]
    X.append(features)

X = np.array(X)

nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()

print(f"NaN values: {nan_count}")
print(f"Inf values: {inf_count}")

if nan_count > 0 or inf_count > 0:
    print("⚠️  WARNING: Invalid values detected")
else:
    print("✅ No invalid values")
print()

# Check feature variance (zero variance = useless feature)
variances = np.var(X, axis=0)
zero_var_count = sum(1 for v in variances if v < 1e-10)
print(f"Zero-variance features: {zero_var_count}/{len(all_features)}")

if zero_var_count > 0:
    print("⚠️  WARNING: Some features have no variance")
else:
    print("✅ All features have variance")
print()

# ============================================================================
# AUDIT 4: MODEL TRAINING VERIFICATION
# ============================================================================
print("[AUDIT 4/8] MODEL TRAINING VERIFICATION")
print("="*80)
print()

with open('STRIVE_FOR_GREATNESS_SYSTEM.pkl', 'rb') as f:
    strive = pickle.load(f)

models_a = strive['branch_a_halftime']['models']
models_b = strive['branch_b_final']['models']

print(f"Branch A models: {len(models_a)}")
print(f"Branch B models: {len(models_b)}")
print()

# Check if models are actually trained
for name, model in list(models_a.items())[:3]:
    # Check if model has been fit (has attributes)
    if hasattr(model, 'n_features_in_'):
        print(f"✅ {name}: trained on {model.n_features_in_} features")
    elif hasattr(model, 'feature_importances_'):
        print(f"✅ {name}: trained (has feature importances)")
    else:
        print(f"⚠️  {name}: may not be trained")

print()

# ============================================================================
# AUDIT 5: MAE CALCULATION VERIFICATION
# ============================================================================
print("[AUDIT 5/8] MAE CALCULATION VERIFICATION")
print("="*80)
print()

# Extract test data
feature_names = strive['feature_names']
scaler_a = strive['branch_a_halftime']['scaler']
scaler_b = strive['branch_b_final']['scaler']

X_test = []
y_half_test = []
y_final_test = []

for game in test_data:
    features = [game.get(f, 0) for f in feature_names]
    X_test.append(features)
    y_half_test.append(game.get('diff_at_halftime', 0))
    y_final_test.append(game.get('diff_at_final', 0))

X_test = np.nan_to_num(np.array(X_test), nan=0.0)
y_half_test = np.array(y_half_test)
y_final_test = np.array(y_final_test)

print(f"Test set size: {len(X_test)}")
print(f"Test features shape: {X_test.shape}")
print()

# Scale
X_test_scaled_a = scaler_a.transform(X_test)
X_test_scaled_b = scaler_b.transform(X_test)

# Predict with each model
print("Branch A predictions (Halftime):")
preds_half_all = []
maes_half = {}

for name, model in models_a.items():
    pred = model.predict(X_test_scaled_a)
    preds_half_all.append(pred)
    mae = mean_absolute_error(y_half_test, pred)
    maes_half[name] = mae
    print(f"  {name:15s}: {mae:.3f} MAE")

print()

# Ensemble (simple average)
ensemble_pred_half = np.mean(preds_half_all, axis=0)
ensemble_mae_half = mean_absolute_error(y_half_test, ensemble_pred_half)

print(f"Ensemble (average): {ensemble_mae_half:.3f} MAE")
print(f"Claimed MAE:        {strive['branch_a_halftime']['champion_mae']:.3f} MAE")
print(f"Difference:         {abs(ensemble_mae_half - strive['branch_a_halftime']['champion_mae']):.3f}")
print()

if abs(ensemble_mae_half - strive['branch_a_halftime']['champion_mae']) < 0.5:
    print("✅ MAE matches claimed performance")
else:
    print("❌ WARNING: MAE doesn't match!")
print()

# Branch B
print("Branch B predictions (Final):")
preds_final_all = []
maes_final = {}

for name, model in models_b.items():
    pred = model.predict(X_test_scaled_b)
    preds_final_all.append(pred)
    mae = mean_absolute_error(y_final_test, pred)
    maes_final[name] = mae
    print(f"  {name:15s}: {mae:.3f} MAE")

print()

ensemble_pred_final = np.mean(preds_final_all, axis=0)
ensemble_mae_final = mean_absolute_error(y_final_test, ensemble_pred_final)

print(f"Ensemble (average): {ensemble_mae_final:.3f} MAE")
print(f"Claimed MAE:        {strive['branch_b_final']['champion_mae']:.3f} MAE")
print(f"Difference:         {abs(ensemble_mae_final - strive['branch_b_final']['champion_mae']):.3f}")
print()

if abs(ensemble_mae_final - strive['branch_b_final']['champion_mae']) < 0.5:
    print("✅ MAE matches claimed performance")
else:
    print("❌ WARNING: MAE doesn't match!")
print()

# ============================================================================
# AUDIT 6: OVERFITTING CHECK
# ============================================================================
print("[AUDIT 6/8] OVERFITTING CHECK")
print("="*80)
print()

# Calculate MAE on TRAIN set
X_train = []
y_half_train = []
y_final_train = []

for game in train_data:
    features = [game.get(f, 0) for f in feature_names]
    X_train.append(features)
    y_half_train.append(game.get('diff_at_halftime', 0))
    y_final_train.append(game.get('diff_at_final', 0))

X_train = np.nan_to_num(np.array(X_train), nan=0.0)
y_half_train = np.array(y_half_train)
y_final_train = np.array(y_final_train)

# Scale with FITTED scaler (already fit on train during training)
X_train_scaled_a = scaler_a.transform(X_train)
X_train_scaled_b = scaler_b.transform(X_train)

# Predict on train
train_preds_half = []
for model in models_a.values():
    train_preds_half.append(model.predict(X_train_scaled_a))

train_preds_final = []
for model in models_b.values():
    train_preds_final.append(model.predict(X_train_scaled_b))

train_pred_half = np.mean(train_preds_half, axis=0)
train_pred_final = np.mean(train_preds_final, axis=0)

train_mae_half = mean_absolute_error(y_half_train, train_pred_half)
train_mae_final = mean_absolute_error(y_final_train, train_pred_final)

print(f"TRAIN MAE:")
print(f"  Halftime: {train_mae_half:.3f}")
print(f"  Final:    {train_mae_final:.3f}")
print()

print(f"TEST MAE:")
print(f"  Halftime: {ensemble_mae_half:.3f}")
print(f"  Final:    {ensemble_mae_final:.3f}")
print()

print(f"OVERFITTING GAP:")
gap_half = ensemble_mae_half - train_mae_half
gap_final = ensemble_mae_final - train_mae_final
gap_pct_half = (gap_half / train_mae_half) * 100
gap_pct_final = (gap_final / train_mae_final) * 100

print(f"  Halftime: {gap_half:+.3f} MAE ({gap_pct_half:+.1f}%)")
print(f"  Final:    {gap_final:+.3f} MAE ({gap_pct_final:+.1f}%)")
print()

if gap_pct_half < 10 and gap_pct_final < 10:
    print("✅ Overfitting gap < 10% (acceptable)")
elif gap_pct_half < 20 and gap_pct_final < 20:
    print("⚠️  Overfitting gap 10-20% (moderate)")
else:
    print("❌ WARNING: Overfitting gap > 20% (concerning)")
print()

# ============================================================================
# AUDIT 7: CROSS-VALIDATION (Multiple Calibrations)
# ============================================================================
print("[AUDIT 7/8] CROSS-VALIDATION (MULTIPLE CALIBRATIONS)")
print("="*80)
print()

# Time series cross-validation with 5 folds
tscv = TimeSeriesSplit(n_splits=5)

cv_maes_half = []
cv_maes_final = []

from xgboost import XGBRegressor

print("Running 5-fold time series cross-validation...")
print("(Using XGBoost as representative model)")
print()

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_cv_train = X[train_idx]
    X_cv_val = X[val_idx]
    y_cv_train_half = np.array([data[i].get('diff_at_halftime', 0) for i in train_idx])
    y_cv_val_half = np.array([data[i].get('diff_at_halftime', 0) for i in val_idx])
    y_cv_train_final = np.array([data[i].get('diff_at_final', 0) for i in train_idx])
    y_cv_val_final = np.array([data[i].get('diff_at_final', 0) for i in val_idx])
    
    # Handle NaN
    X_cv_train = np.nan_to_num(X_cv_train, nan=0.0)
    X_cv_val = np.nan_to_num(X_cv_val, nan=0.0)
    
    # Scale
    cv_scaler = StandardScaler()
    X_cv_train_scaled = cv_scaler.fit_transform(X_cv_train)
    X_cv_val_scaled = cv_scaler.transform(X_cv_val)
    
    # Train and test
    model_half = XGBRegressor(n_estimators=200, max_depth=8, random_state=42)
    model_half.fit(X_cv_train_scaled, y_cv_train_half)
    pred_half = model_half.predict(X_cv_val_scaled)
    mae_half = mean_absolute_error(y_cv_val_half, pred_half)
    cv_maes_half.append(mae_half)
    
    model_final = XGBRegressor(n_estimators=200, max_depth=8, random_state=42)
    model_final.fit(X_cv_train_scaled, y_cv_train_final)
    pred_final = model_final.predict(X_cv_val_scaled)
    mae_final = mean_absolute_error(y_cv_val_final, pred_final)
    cv_maes_final.append(mae_final)
    
    print(f"  Fold {fold}: Halftime {mae_half:.3f} | Final {mae_final:.3f}")

print()
print(f"Cross-validation average:")
print(f"  Halftime: {np.mean(cv_maes_half):.3f} ± {np.std(cv_maes_half):.3f} MAE")
print(f"  Final:    {np.mean(cv_maes_final):.3f} ± {np.std(cv_maes_final):.3f} MAE")
print()

print(f"Claimed MAE:")
print(f"  Halftime: {ensemble_mae_half:.3f}")
print(f"  Final:    {ensemble_mae_final:.3f}")
print()

# Check if claimed MAE is within CV range
cv_half_min = np.mean(cv_maes_half) - 2*np.std(cv_maes_half)
cv_half_max = np.mean(cv_maes_half) + 2*np.std(cv_maes_half)

if cv_half_min <= ensemble_mae_half <= cv_half_max:
    print("✅ Claimed halftime MAE is within CV confidence interval")
else:
    print(f"⚠️  WARNING: Claimed MAE {ensemble_mae_half:.3f} outside CV range [{cv_half_min:.3f}, {cv_half_max:.3f}]")

print()

# ============================================================================
# AUDIT 8: SANITY CHECK - BASELINE COMPARISON
# ============================================================================
print("[AUDIT 8/8] BASELINE COMPARISON")
print("="*80)
print()

# Baseline 1: Predict median
median_half = np.median(y_half_train)
median_final = np.median(y_final_train)

baseline_median_mae_half = mean_absolute_error(y_half_test, [median_half] * len(y_half_test))
baseline_median_mae_final = mean_absolute_error(y_final_test, [median_final] * len(y_final_test))

print(f"Baseline (predict median for all):")
print(f"  Halftime: {baseline_median_mae_half:.3f} MAE")
print(f"  Final:    {baseline_median_mae_final:.3f} MAE")
print()

# Baseline 2: Predict 0 (no differential)
baseline_zero_mae_half = mean_absolute_error(y_half_test, [0] * len(y_half_test))
baseline_zero_mae_final = mean_absolute_error(y_final_test, [0] * len(y_final_test))

print(f"Baseline (predict 0 for all):")
print(f"  Halftime: {baseline_zero_mae_half:.3f} MAE")
print(f"  Final:    {baseline_zero_mae_final:.3f} MAE")
print()

# Our system
print(f"Our system:")
print(f"  Halftime: {ensemble_mae_half:.3f} MAE")
print(f"  Final:    {ensemble_mae_final:.3f} MAE")
print()

improvement_half = baseline_zero_mae_half - ensemble_mae_half
improvement_final = baseline_zero_mae_final - ensemble_mae_final

print(f"Improvement vs baseline (predict 0):")
print(f"  Halftime: {improvement_half:.3f} MAE better ({improvement_half/baseline_zero_mae_half*100:.1f}%)")
print(f"  Final:    {improvement_final:.3f} MAE better ({improvement_final/baseline_zero_mae_final*100:.1f}%)")
print()

if improvement_half > 2 and improvement_final > 2:
    print("✅ System significantly beats baseline")
elif improvement_half > 0 and improvement_final > 0:
    print("⚠️  System beats baseline but margin is small")
else:
    print("❌ WARNING: System doesn't beat baseline!")
print()

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("="*80)
print("🔬 AUDIT COMPLETE - FINAL VERDICT")
print("="*80)
print()

print("DATA INTEGRITY:")
print(f"  Total games: {len(data)}")
print(f"  Unique IDs: {unique_ids}")
print(f"  Date range: {dates_sorted[0]} to {dates_sorted[-1]}")
print(f"  Duplicates: {len(game_ids) - unique_ids}")
print()

print("TRAIN/TEST SPLIT:")
print(f"  Train: {len(train_data)} ({len(train_data)/len(data)*100:.0f}%)")
print(f"  Test: {len(test_data)} ({len(test_data)/len(data)*100:.0f}%)")
print(f"  Temporal order: {'✅ Correct' if earliest_test >= latest_train else '❌ Leakage'}")
print()

print("PERFORMANCE:")
print(f"  Train MAE:  {train_mae_half:.3f} / {train_mae_final:.3f}")
print(f"  Test MAE:   {ensemble_mae_half:.3f} / {ensemble_mae_final:.3f}")
print(f"  CV MAE:     {np.mean(cv_maes_half):.3f} / {np.mean(cv_maes_final):.3f}")
print(f"  Baseline:   {baseline_zero_mae_half:.3f} / {baseline_zero_mae_final:.3f}")
print()

print("OVERFITTING:")
print(f"  Gap: {gap_pct_half:.1f}% / {gap_pct_final:.1f}%")
print(f"  Status: {'✅ Acceptable' if gap_pct_half < 15 and gap_pct_final < 15 else '⚠️ Check'}")
print()

print("IMPROVEMENT VS BASELINE:")
print(f"  Halftime: {improvement_half/baseline_zero_mae_half*100:.1f}% better")
print(f"  Final:    {improvement_final/baseline_zero_mae_final*100:.1f}% better")
print()

# Final decision
issues = []
if len(game_ids) != unique_ids:
    issues.append("Duplicate games")
if earliest_test < latest_train:
    issues.append("Temporal leakage")
if gap_pct_half > 20 or gap_pct_final > 20:
    issues.append("High overfitting")
if improvement_half < 1 or improvement_final < 1:
    issues.append("Doesn't beat baseline")
if abs(ensemble_mae_half - strive['branch_a_halftime']['champion_mae']) > 0.5:
    issues.append("MAE mismatch")

print("="*80)
if len(issues) == 0:
    print("✅ ALL AUDITS PASSED - SYSTEM IS LEGIT")
else:
    print(f"⚠️  ISSUES FOUND: {len(issues)}")
    for issue in issues:
        print(f"   - {issue}")
print("="*80)

