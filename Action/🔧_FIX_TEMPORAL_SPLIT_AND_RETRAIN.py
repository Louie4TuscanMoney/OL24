#!/usr/bin/env python3
"""
🔧 FIX TEMPORAL SPLIT AND RETRAIN
Sort data chronologically, split properly, retrain, get TRUE MAE
"""

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                               HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

print("="*80)
print("🔧 FIXING TEMPORAL SPLIT AND RETRAINING")
print("="*80)
print()

# Load data
print("[1/5] Loading data...")
with open('ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"✅ Loaded {len(data)} games")
print()

# Sort chronologically
print("[2/5] Sorting chronologically...")
data_sorted = sorted(data, key=lambda x: x.get('date', ''))

print(f"✅ Data sorted by date")
print(f"   First date: {data_sorted[0].get('date')}")
print(f"   Last date: {data_sorted[-1].get('date')}")
print()

# Verify chronological order
dates = [g.get('date', '') for g in data_sorted]
is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
print(f"✅ Chronologically ordered: {is_sorted}")
print()

# Split chronologically (80/20)
split_idx = int(len(data_sorted) * 0.8)
train_data = data_sorted[:split_idx]
test_data = data_sorted[split_idx:]

latest_train_date = max(g.get('date', '') for g in train_data)
earliest_test_date = min(g.get('date', '') for g in test_data)

print(f"CHRONOLOGICAL SPLIT:")
print(f"  Train: {len(train_data)} games ({len(train_data)/len(data_sorted)*100:.0f}%)")
print(f"  Test:  {len(test_data)} games ({len(test_data)/len(data_sorted)*100:.0f}%)")
print()
print(f"  Train dates: {train_data[0].get('date')} to {latest_train_date}")
print(f"  Test dates:  {earliest_test_date} to {test_data[-1].get('date')}")
print()

if earliest_test_date >= latest_train_date:
    print("✅ NO TEMPORAL LEAKAGE - test is after train")
else:
    print("❌ STILL HAVE LEAKAGE - something wrong")
print()

# Extract features
exclude = ['game_id', 'date', 'season', 'home_team', 'away_team', 'pattern',
           'diff_at_final', 'diff_at_halftime', 'diff_at_2q_6min']
feature_names = [k for k in data_sorted[0].keys() if k not in exclude]

X_train = []
y_half_train = []
y_final_train = []

for game in train_data:
    features = [game.get(f, 0) for f in feature_names]
    X_train.append(features)
    y_half_train.append(game.get('diff_at_halftime', 0))
    y_final_train.append(game.get('diff_at_final', 0))

X_test = []
y_half_test = []
y_final_test = []

for game in test_data:
    features = [game.get(f, 0) for f in feature_names]
    X_test.append(features)
    y_half_test.append(game.get('diff_at_halftime', 0))
    y_final_test.append(game.get('diff_at_final', 0))

X_train = np.nan_to_num(np.array(X_train), nan=0.0)
X_test = np.nan_to_num(np.array(X_test), nan=0.0)
y_half_train = np.array(y_half_train)
y_half_test = np.array(y_half_test)
y_final_train = np.array(y_final_train)
y_final_test = np.array(y_final_test)

print(f"✅ Train shape: {X_train.shape}")
print(f"✅ Test shape: {X_test.shape}")
print()

# Save chronologically sorted data
with open('ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl', 'wb') as f:
    pickle.dump(data_sorted, f)

print("✅ Saved chronologically sorted data")
print()

# Scale
print("[3/5] Scaling features...")
scaler_a = StandardScaler()
scaler_b = StandardScaler()

X_train_scaled_a = scaler_a.fit_transform(X_train)
X_test_scaled_a = scaler_a.transform(X_test)

X_train_scaled_b = scaler_b.fit_transform(X_train)
X_test_scaled_b = scaler_b.transform(X_test)

print("✅ Features scaled")
print()

# Train models quickly
print("[4/5] Training models with CLEAN temporal split...")
print()

print("Branch A (Halftime):")
models_a = {}

print("  [1/10] XGBoost...")
models_a['xgboost'] = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42)
models_a['xgboost'].fit(X_train_scaled_a, y_half_train)

print("  [2/10] LightGBM...")
models_a['lightgbm'] = LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1)
models_a['lightgbm'].fit(X_train_scaled_a, y_half_train)

print("  [3/10] ExtraTrees...")
models_a['extratrees'] = ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
models_a['extratrees'].fit(X_train_scaled_a, y_half_train)

print("  [4/10] RandomForest...")
models_a['randomforest'] = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
models_a['randomforest'].fit(X_train_scaled_a, y_half_train)

print("  [5/10] HistGradient...")
models_a['histgradient'] = HistGradientBoostingRegressor(max_iter=200, random_state=42)
models_a['histgradient'].fit(X_train_scaled_a, y_half_train)

print("  [6/10] Ridge...")
models_a['ridge'] = Ridge(alpha=1.0)
models_a['ridge'].fit(X_train_scaled_a, y_half_train)

print("  [7/10] ElasticNet...")
models_a['elasticnet'] = ElasticNet(alpha=0.1, max_iter=2000)
models_a['elasticnet'].fit(X_train_scaled_a, y_half_train)

print("  [8/10] SVR...")
models_a['svr'] = SVR(C=1.0, epsilon=0.1)
models_a['svr'].fit(X_train_scaled_a, y_half_train)

print("  [9/10] MLP...")
models_a['mlp'] = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
models_a['mlp'].fit(X_train_scaled_a, y_half_train)

print("  [10/10] GradientBoost...")
models_a['gradboost'] = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
models_a['gradboost'].fit(X_train_scaled_a, y_half_train)

print()
print("Branch B (Final):")
models_b = {}

print("  [1/10] XGBoost...")
models_b['xgboost'] = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42)
models_b['xgboost'].fit(X_train_scaled_b, y_final_train)

print("  [2/10] LightGBM...")
models_b['lightgbm'] = LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1)
models_b['lightgbm'].fit(X_train_scaled_b, y_final_train)

print("  [3/10] ExtraTrees...")
models_b['extratrees'] = ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
models_b['extratrees'].fit(X_train_scaled_b, y_final_train)

print("  [4/10] RandomForest...")
models_b['randomforest'] = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
models_b['randomforest'].fit(X_train_scaled_b, y_final_train)

print("  [5/10] HistGradient...")
models_b['histgradient'] = HistGradientBoostingRegressor(max_iter=200, random_state=42)
models_b['histgradient'].fit(X_train_scaled_b, y_final_train)

print("  [6/10] Ridge...")
models_b['ridge'] = Ridge(alpha=1.0)
models_b['ridge'].fit(X_train_scaled_b, y_final_train)

print("  [7/10] ElasticNet...")
models_b['elasticnet'] = ElasticNet(alpha=0.1, max_iter=2000)
models_b['elasticnet'].fit(X_train_scaled_b, y_final_train)

print("  [8/10] SVR...")
models_b['svr'] = SVR(C=1.0, epsilon=0.1)
models_b['svr'].fit(X_train_scaled_b, y_final_train)

print("  [9/10] MLP...")
models_b['mlp'] = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
models_b['mlp'].fit(X_train_scaled_b, y_final_train)

print("  [10/10] GradientBoost...")
models_b['gradboost'] = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
models_b['gradboost'].fit(X_train_scaled_b, y_final_train)

print()
print("✅ All 20 models retrained on CLEAN chronological split")
print()

# Test
print("[5/5] Testing on CLEAN holdout set...")
print()

# Train MAE
train_preds_half = []
for model in models_a.values():
    train_preds_half.append(model.predict(X_train_scaled_a))

train_preds_final = []
for model in models_b.values():
    train_preds_final.append(model.predict(X_train_scaled_b))

train_mae_half = mean_absolute_error(y_half_train, np.mean(train_preds_half, axis=0))
train_mae_final = mean_absolute_error(y_final_train, np.mean(train_preds_final, axis=0))

# Test MAE
test_preds_half = []
for model in models_a.values():
    test_preds_half.append(model.predict(X_test_scaled_a))

test_preds_final = []
for model in models_b.values():
    test_preds_final.append(model.predict(X_test_scaled_b))

test_mae_half = mean_absolute_error(y_half_test, np.mean(test_preds_half, axis=0))
test_mae_final = mean_absolute_error(y_final_test, np.mean(test_preds_final, axis=0))

print(f"CLEAN RESULTS:")
print(f"  Train MAE: {train_mae_half:.3f} / {train_mae_final:.3f}")
print(f"  Test MAE:  {test_mae_half:.3f} / {test_mae_final:.3f}")
print()

gap_half = (test_mae_half - train_mae_half) / train_mae_half * 100
gap_final = (test_mae_final - train_mae_final) / train_mae_final * 100

print(f"  Overfitting gap: {gap_half:.1f}% / {gap_final:.1f}%")
print()

# Save cleaned system
strive_clean = {
    'branch_a_halftime': {
        'models': models_a,
        'scaler': scaler_a,
        'champion_mae': test_mae_half,
        'train_mae': train_mae_half,
        'champion_strategy': 'Simple Average (retrained clean)'
    },
    'branch_b_final': {
        'models': models_b,
        'scaler': scaler_b,
        'champion_mae': test_mae_final,
        'train_mae': train_mae_final,
        'champion_strategy': 'Simple Average (retrained clean)'
    },
    'metadata': {
        'total_games': len(data_sorted),
        'train_games': len(train_data),
        'test_games': len(test_data),
        'feature_count': len(feature_names),
        'models_trained': 20,
        'build_date': '2025-10-19',
        'philosophy': 'Strive for Greatness - LeBron James',
        'data_quality': 'CLEAN - No temporal leakage',
        'train_date_range': f"{train_data[0].get('date')} to {latest_train_date}",
        'test_date_range': f"{earliest_test_date} to {test_data[-1].get('date')}",
    },
    'feature_names': feature_names,
}

with open('STRIVE_FOR_GREATNESS_CLEAN.pkl', 'wb') as f:
    pickle.dump(strive_clean, f)

print("✅ Saved to: STRIVE_FOR_GREATNESS_CLEAN.pkl")
print()

# Summary
print("="*80)
print("🎯 CLEAN SYSTEM - TRUE PERFORMANCE")
print("="*80)
print()
print(f"Data split:")
print(f"  Train: {len(train_data)} games (2021-{latest_train_date[:4]})")
print(f"  Test:  {len(test_data)} games ({earliest_test_date[:4]}-2025)")
print()
print(f"TRUE MAE (no leakage):")
print(f"  Halftime: {test_mae_half:.3f}")
print(f"  Final:    {test_mae_final:.3f}")
print()
print(f"Overfitting:")
print(f"  Halftime: {gap_half:.1f}%")
print(f"  Final:    {gap_final:.1f}%")
print()
print(f"vs CONTAMINATED results:")
print(f"  Old (leakage): 5.398 / 9.965")
print(f"  New (clean):   {test_mae_half:.3f} / {test_mae_final:.3f}")
print(f"  Degradation:   {test_mae_half - 5.398:+.3f} / {test_mae_final - 9.965:+.3f}")
print()
print("="*80)

