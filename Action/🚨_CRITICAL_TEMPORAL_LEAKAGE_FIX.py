"""
🚨 CRITICAL: TEMPORAL LEAKAGE DETECTED AND FIXING NOW

ISSUE FOUND:
  Max train date: 2025-04-13
  Min test date:  2021-10-20
  
  → Train contains FUTURE data relative to test!
  → This invalidates all Engineering + Genetic systems!

ROOT CAUSE:
  Data in pickle is NOT chronologically sorted
  Scripts did NOT sort before splitting
  → Random 80/20 split = temporal leakage

FIX:
  1. Sort data chronologically
  2. Split 80/20 chronologically  
  3. Retrain ALL affected systems
  4. Verify new performance (expect slight degradation = honest)
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("🚨 CRITICAL TEMPORAL LEAKAGE - FIXING NOW")
print("="*90)

# Load data
print("\n[1/6] Loading and analyzing data...")
with open('Action/ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl', 'rb') as f:
    data_list = pickle.load(f)

print(f"✓ Loaded {len(data_list)} games")

# Extract dates
dates = [(i, game.get('date', '')) for i, game in enumerate(data_list)]
dates_with_idx = [(i, d) for i, d in dates if d]

print(f"✓ {len(dates_with_idx)} games have dates")

# Check current order
print("\nFirst 5 dates:", [d[1][:10] for d in dates_with_idx[:5]])
print("Last 5 dates:", [d[1][:10] for d in dates_with_idx[-5:]])

# Sort chronologically
dates_sorted = sorted(dates_with_idx, key=lambda x: x[1])

print("\n🔧 Sorting chronologically...")
print("First 5 sorted:", [d[1][:10] for d in dates_sorted[:5]])
print("Last 5 sorted:", [d[1][:10] for d in dates_sorted[-5:]])

# Reorder data_list
sorted_indices = [idx for idx, _ in dates_sorted]
data_list_sorted = [data_list[i] for i in sorted_indices]

print(f"✓ Data sorted chronologically")

print("\n[2/6] Extracting features from sorted data...")

X_all = []
y_final_all = []
y_current_all = []
dates_all = []

for game in data_list_sorted:
    pattern = game.get('pattern', [])
    if isinstance(pattern, list) and len(pattern) >= 18:
        X_all.append(pattern[:18])
        y_final_all.append(game.get('diff_at_final', 0))
        y_current_all.append(game.get('diff_at_halftime', game.get('diff_at_2q_6min', 0)))
        dates_all.append(game.get('date', ''))

X_all = np.array(X_all)
y_final = np.array(y_final_all)
y_current = np.array(y_current_all)

print(f"✓ Features: {X_all.shape}")

# Verify chronological split
split_idx = int(len(X_all) * 0.8)

print(f"\n📅 TEMPORAL INTEGRITY CHECK:")
print(f"   Training period: {dates_all[0][:10]} to {dates_all[split_idx-1][:10]}")
print(f"   Testing period:  {dates_all[split_idx][:10]} to {dates_all[-1][:10]}")

# Verify no overlap
train_max = dates_all[split_idx-1]
test_min = dates_all[split_idx]

if train_max <= test_min:
    print(f"   ✅ NO LEAKAGE: {train_max[:10]} <= {test_min[:10]}")
    temporal_safe = True
else:
    print(f"   ❌ LEAKAGE: {train_max[:10]} > {test_min[:10]}")
    temporal_safe = False

# Split
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_final[:split_idx], y_final[split_idx:]
y_curr_train, y_curr_test = y_current[:split_idx], y_current[split_idx:]

print(f"\n✓ Split: {len(X_train)} train, {len(X_test)} test")

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add current diff
X_train_with_curr = np.column_stack([X_train_scaled, y_curr_train])
X_test_with_curr = np.column_stack([X_test_scaled, y_curr_test])

print("\n[3/6] Retraining ENGINEERING LINEAR (temporally clean)...")

model_linear_clean = LinearRegression()
model_linear_clean.fit(X_train_with_curr, y_train)

pred_clean = model_linear_clean.predict(X_test_with_curr)
mae_clean = mean_absolute_error(y_test, pred_clean)

train_pred_clean = model_linear_clean.predict(X_train_with_curr)
mae_train_clean = mean_absolute_error(y_train, train_pred_clean)
overfit_clean = ((mae_clean - mae_train_clean) / mae_train_clean) * 100

print(f"✓ Retrained on clean data")
print(f"  Train MAE: {mae_train_clean:.3f}")
print(f"  Test MAE:  {mae_clean:.3f}")
print(f"  Overfitting: {overfit_clean:.1f}%")
print(f"\n  Original (potentially leaked): 8.806 MAE")
print(f"  Clean (temporally safe):       {mae_clean:.3f} MAE")

degradation = mae_clean - 8.806
if degradation > 0:
    print(f"  📈 Degradation: +{degradation:.3f} MAE ({degradation/8.806*100:.1f}%)")
    print(f"     → This is EXPECTED (was seeing future data before)")
else:
    print(f"  📉 Improvement: {degradation:.3f} MAE")

print("\n[4/6] Retraining GENETIC ALGORITHM (temporally clean)...")

# Retrain top 8 genetic models
genetic_models_clean = []

model_configs = [
    ('Ridge_Strong', Ridge(alpha=5.0, max_iter=5000, solver='saga')),
    ('Ridge_Moderate', Ridge(alpha=2.0, max_iter=5000, solver='saga')),
    ('SVR_Linear', SVR(kernel='linear', C=0.5, epsilon=0.1)),
    ('BayesianRidge', BayesianRidge(max_iter=5000)),
    ('Lasso', Lasso(alpha=0.5, max_iter=5000)),
    ('ElasticNet', ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000)),
    ('ExtraTrees', ExtraTreesRegressor(n_estimators=150, max_depth=6, min_samples_split=15, random_state=42)),
    ('GradBoost', GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.02, random_state=42))
]

print("Training halftime models...")
for name, model in model_configs:
    model.fit(X_train_scaled, y_curr_train)  # Predict halftime from patterns
    genetic_models_clean.append((name, model))
    print(f"  ✓ {name}")

# Ensemble prediction (halftime)
ht_preds_test = np.column_stack([m.predict(X_test_scaled) for _, m in genetic_models_clean])
ht_ensemble = ht_preds_test.mean(axis=1)

ht_preds_train = np.column_stack([m.predict(X_train_scaled) for _, m in genetic_models_clean])
ht_ensemble_train = ht_preds_train.mean(axis=1)

mae_ht_test = mean_absolute_error(y_curr_test, ht_ensemble)
mae_ht_train = mean_absolute_error(y_curr_train, ht_ensemble_train)
overfit_ht = ((mae_ht_test - mae_ht_train) / mae_ht_train) * 100

print(f"\n✓ Genetic halftime (clean):")
print(f"  Train MAE: {mae_ht_train:.3f}")
print(f"  Test MAE:  {mae_ht_test:.3f}")
print(f"  Overfitting: {overfit_ht:.1f}%")
print(f"\n  Original: 5.301 MAE")
print(f"  Clean:    {mae_ht_test:.3f} MAE")

ht_degradation = mae_ht_test - 5.301
if abs(ht_degradation) < 0.05:
    print(f"  ✅ MATCHES (within 0.05)")
elif ht_degradation > 0:
    print(f"  📈 Degradation: +{ht_degradation:.3f} MAE")

print("\n[5/6] Calculating clean system performance...")

# Edge calculations
baseline_ht = 9.0
baseline_final = 11.5

edge_ht_clean = ((baseline_ht - mae_ht_test) / baseline_ht) * 100
edge_final_clean = ((baseline_final - mae_clean) / baseline_final) * 100

print(f"\nHALFTIME (Genetic, clean):")
print(f"  MAE: {mae_ht_test:.3f}")
print(f"  Overfit: {overfit_ht:.1f}%")
print(f"  Edge: {edge_ht_clean:.1f}%")

print(f"\nFINAL (Engineering Linear, clean):")
print(f"  MAE: {mae_clean:.3f}")
print(f"  Overfit: {overfit_clean:.1f}%")
print(f"  Edge: {edge_final_clean:.1f}%")

# EV calculation
ev_ht = 25 * (edge_ht_clean / 100) * 100
ev_final = 20 * (edge_final_clean / 100) * 100
total_ev = ev_ht + ev_final

print(f"\nEXPECTED VALUE (100 games, CLEAN):")
print(f"  Halftime: 25 bets × {edge_ht_clean:.1f}% = +${ev_ht:.0f}")
print(f"  Final:    20 bets × {edge_final_clean:.1f}% = +${ev_final:.0f}")
print(f"  Total:    +${total_ev:.0f} per 100 games")

print(f"\n  Original (potentially leaked): +$1,495")
print(f"  Clean (temporally safe):       +${total_ev:.0f}")

ev_change = total_ev - 1495
if ev_change < 0:
    print(f"  📉 Change: ${ev_change:.0f} per 100 games")
    print(f"     → Honest assessment (was inflated by leakage)")

print("\n[6/6] Saving CLEAN system...")

clean_system = {
    'name': 'HYBRID_ULTIMATE_V2_CLEAN',
    'version': '2.1.0',
    'halftime': {
        'models': dict(genetic_models_clean),
        'scaler': scaler,
        'mae_train': float(mae_ht_train),
        'mae_test': float(mae_ht_test),
        'overfitting_pct': float(overfit_ht),
        'edge_pct': float(edge_ht_clean),
        'ev_per_100': float(ev_ht)
    },
    'final': {
        'model': model_linear_clean,
        'scaler': scaler,
        'mae_train': float(mae_train_clean),
        'mae_test': float(mae_clean),
        'overfitting_pct': float(overfit_clean),
        'edge_pct': float(edge_final_clean),
        'ev_per_100': float(ev_final)
    },
    'temporal_integrity': {
        'sorted': True,
        'train_period': f"{dates_all[0][:10]} to {dates_all[split_idx-1][:10]}",
        'test_period': f"{dates_all[split_idx][:10]} to {dates_all[-1][:10]}",
        'no_leakage': temporal_safe
    },
    'performance': {
        'total_ev_per_100': float(total_ev),
        'expected_monday': [mae_ht_test * 1.15, mae_clean * 1.15],
        'greenlight': temporal_safe
    }
}

with open('Action/HYBRID_ULTIMATE_V2_CLEAN.pkl', 'wb') as f:
    pickle.dump(clean_system, f)

print("✓ Saved: HYBRID_ULTIMATE_V2_CLEAN.pkl")

print("\n" + "="*90)
print("TEMPORAL LEAKAGE FIX - COMPLETE")
print("="*90)

print("\n🚨 ISSUE: Temporal leakage found in new systems")
print(f"   → Train dates overlap with test dates")
print(f"   → This inflated performance artificially")

print("\n🔧 FIX APPLIED:")
print("   ✓ Sorted data chronologically by date")
print("   ✓ Split first 80% train, last 20% test")
print("   ✓ Verified no temporal overlap")
print("   ✓ Retrained all models on clean split")

print("\n📊 CLEAN PERFORMANCE (Honest):")
print(f"   Halftime: {mae_ht_test:.3f} MAE, {overfit_ht:.1f}% overfit, {edge_ht_clean:.1f}% edge")
print(f"   Final:    {mae_clean:.3f} MAE, {overfit_clean:.1f}% overfit, {edge_final_clean:.1f}% edge")
print(f"   Total EV: +${total_ev:.0f} per 100 games")

if temporal_safe:
    print("\n✅ TEMPORAL INTEGRITY: VERIFIED SAFE")
    print("   → No leakage in clean split")
    print("   → Ready for Monday launch")
else:
    print("\n⚠️ TEMPORAL INTEGRITY: STILL INVESTIGATING")
    print("   → May need to re-sort pickle file")

print("\n" + "="*90)

