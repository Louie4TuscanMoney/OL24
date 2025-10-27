#!/usr/bin/env python3
"""
🔥 TRAIN OPTIMIZED ENSEMBLE
5 models with BEST hyperparameters
Target: Get MAE to 5-6 range
"""

import pickle
import numpy as np
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sys
sys.path.insert(0, '1. ML/1. Dejavu Deployment')
from dejavu_model import DejavuForecaster

print("="*80)
print("🔥 TRAINING OPTIMIZED ENSEMBLE - 5 DIVERSE MODELS")
print("="*80)
print()

# Load data
print("[1/6] Loading enhanced patterns...")
with open('ULTRA_ENHANCED_PATTERNS_V2.pkl', 'rb') as f:
    patterns = pickle.load(f)

# Load best hyperparameters
with open('BEST_HYPERPARAMETERS.pkl', 'rb') as f:
    best_params = pickle.load(f)

print(f"✅ Loaded {len(patterns)} games")
print(f"✅ Loaded optimized hyperparameters")
print()

# Build matrices
print("[2/6] Building feature matrices...")
X = []
y = []
pattern_only = []  # For Dejavu

for p in patterns:
    if p.get('diff_at_final') is None:
        continue
    
    # Full feature vector (51+ features)
    features = []
    features.extend(p.get('pattern', [0]*18))
    features.extend([
        p.get('mean_diff', 0), p.get('std_diff', 0), p.get('trend', 0), p.get('volatility', 0),
        p.get('team_diff_lag1', 0), p.get('team_mean_lag1', 0),
        p.get('team_diff_rolling3', 0), p.get('team_volatility_rolling3', 2.0),
        p.get('team_form_10games', 0), p.get('team_consistency', 10.0),
        p.get('spectral_energy', 0), p.get('low_freq_power', 0), p.get('mid_freq_power', 0),
        p.get('high_freq_power', 0), p.get('dominant_freq', 0), p.get('spectral_entropy', 0),
        p.get('velocity', 0), p.get('acceleration', 0), p.get('recent_momentum', 0),
        p.get('lead_changes', 0), p.get('max_swing', 0), p.get('comeback_potential', 0),
        p.get('autocorr_lag1', 0), p.get('autocorr_lag3', 0), p.get('autocorr_lag5', 0),
        p.get('efg_proxy', 0), p.get('ts_proxy', 0), p.get('netrtg_proxy', 0),
        p.get('pie_proxy', 0), p.get('pm_proxy', 0), p.get('usg_proxy', 0),
        p.get('pace_proxy', 0), p.get('four_factors_proxy', 0)
    ])
    
    X.append(features)
    y.append(p['diff_at_final'])
    pattern_only.append(p.get('pattern', [0]*18))

X = np.array(X)
y = np.array(y)
pattern_only = np.array(pattern_only)

print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")
print()

# Time-based split (80/20, last 20% as test)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {len(X_train)} games")
print(f"Test: {len(X_test)} games (most recent 20%)")
print()

# TRAIN MODEL 1: XGBoost (optimized)
print("[3/6] Training XGBoost with OPTIMAL hyperparameters...")
xgb_params = best_params['xgboost']['params']
xgb_params['random_state'] = 42
xgb_params['n_jobs'] = -1

model_xgb = xgb.XGBRegressor(**xgb_params)
model_xgb.fit(X_train, y_train)

xgb_pred = model_xgb.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
print(f"✅ XGBoost MAE: {xgb_mae:.3f}")
model_xgb.save_model('model_xgboost_optimized.json')
print()

# TRAIN MODEL 2: ExtraTrees (best in research)
print("[4/6] Training ExtraTrees with OPTIMAL hyperparameters...")
et_params = best_params['extratrees']['params']
et_params['random_state'] = 42
et_params['n_jobs'] = -1

model_et = ExtraTreesRegressor(**et_params)
model_et.fit(X_train, y_train)

et_pred = model_et.predict(X_test)
et_mae = mean_absolute_error(y_test, et_pred)
print(f"✅ ExtraTrees MAE: {et_mae:.3f}")
with open('model_extratrees_optimized.pkl', 'wb') as f:
    pickle.dump(model_et, f)
print()

# TRAIN MODEL 3: RandomForest (robust baseline)
print("Training RandomForest (high n_estimators)...")
model_rf = RandomForestRegressor(
    n_estimators=800,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)
model_rf.fit(X_train, y_train)

rf_pred = model_rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
print(f"✅ RandomForest MAE: {rf_mae:.3f}")
with open('model_randomforest.pkl', 'wb') as f:
    pickle.dump(model_rf, f)
print()

# TRAIN MODEL 4: GradientBoosting (different from XGBoost)
print("Training HistGradientBoosting (handles NaN)...")
from sklearn.ensemble import HistGradientBoostingRegressor
model_gb = HistGradientBoostingRegressor(
    max_iter=500,
    max_depth=6,
    learning_rate=0.05,
    min_samples_leaf=5,
    random_state=42
)
model_gb.fit(X_train, y_train)

gb_pred = model_gb.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_pred)
print(f"✅ GradientBoosting MAE: {gb_mae:.3f}")
with open('model_gradientboosting.pkl', 'wb') as f:
    pickle.dump(model_gb, f)
print()

# MODEL 5: Dejavu (already trained, just test)
print("[5/6] Testing Dejavu on new features...")
dejavu = DejavuForecaster.load('1. ML/1. Dejavu Deployment/dejavu_FINAL_k500.pkl')

dejavu_predictions = []
for pattern in pattern_only[split_idx:]:
    if len(pattern) == 18:
        pred = dejavu.predict(pattern)
        dejavu_predictions.append(pred)
    else:
        dejavu_predictions.append(0)

dejavu_mae = mean_absolute_error(y_test, dejavu_predictions)
print(f"✅ Dejavu MAE: {dejavu_mae:.3f}")
print()

# COMPARE ALL MODELS
print("[6/6] Model comparison...")
print("="*80)
results = {
    'XGBoost (optimized)': xgb_mae,
    'ExtraTrees (optimized)': et_mae,
    'RandomForest': rf_mae,
    'GradientBoosting': gb_mae,
    'Dejavu (KNN)': dejavu_mae
}

sorted_results = sorted(results.items(), key=lambda x: x[1])

print("MODEL RANKING (by MAE):")
for rank, (name, mae) in enumerate(sorted_results, 1):
    print(f"  {rank}. {name:25s} MAE: {mae:.3f}")

print()
best_model = sorted_results[0][0]
best_mae = sorted_results[0][1]

print(f"🏆 BEST MODEL: {best_model}")
print(f"📊 BEST MAE: {best_mae:.3f}")
print()

if best_mae < 6.0:
    print("✅ EXCELLENT - Under 6.0 MAE!")
    print("   Next: Build stacked ensemble to get under 5.0")
elif best_mae < 7.0:
    print("✅ GOOD - Under 7.0 MAE")
    print("   Next: Stacking should get us under 5.5")
else:
    print("⚠️  Still above 7.0 - need stacking + LSTM")

print()

# Save all results
ensemble_results = {
    'models': results,
    'best_model': best_model,
    'best_mae': best_mae,
    'test_size': len(X_test),
    'feature_count': X.shape[1]
}

with open('ENSEMBLE_RESULTS.pkl', 'wb') as f:
    pickle.dump(ensemble_results, f)

print("✅ Results saved to: ENSEMBLE_RESULTS.pkl")
print()
print("="*80)
print("🔥 ENSEMBLE TRAINING COMPLETE")
print(f"   Best single model: {best_mae:.3f} MAE")
print(f"   Next: python3 🔥_4_STACK_ENSEMBLE.py")
print("="*80)

