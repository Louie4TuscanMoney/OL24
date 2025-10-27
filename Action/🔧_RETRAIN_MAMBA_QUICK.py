#!/usr/bin/env python3
"""
🔧 QUICK RETRAIN MAMBA WITH CORRECT FEATURES
Use V3 dataset with first 67 features (same data as Strive, different feature count)
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

print("="*80)
print("🔧 RETRAINING MAMBA WITH CORRECT FEATURES")
print("="*80)
print()

# Load V3 dataset
with open('ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"✅ Loaded {len(data)} games")
print()

# Extract features
exclude = ['game_id', 'date', 'season', 'home_team', 'away_team', 'pattern',
           'diff_at_final', 'diff_at_halftime', 'diff_at_2q_6min']
all_features = [k for k in data[0].keys() if k not in exclude]

# Mamba uses first 67 features
mamba_features = all_features[:67]

print(f"✅ Using first {len(mamba_features)} features for Mamba")
print()

# Prepare data
X = []
y_half = []
y_final = []

for game in data:
    features = [game.get(f, 0) for f in mamba_features]
    X.append(features)
    y_half.append(game.get('diff_at_halftime', 0))
    y_final.append(game.get('diff_at_final', 0))

X = np.nan_to_num(np.array(X), nan=0.0)
y_half = np.array(y_half)
y_final = np.array(y_final)

# Split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_half_train, y_half_test = y_half[:split_idx], y_half[split_idx:]
y_final_train, y_final_test = y_final[:split_idx], y_final[split_idx:]

print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")
print()

# Load existing Mamba system (use existing models, just update scaler)
with open('MAMBA_MENTALITY_SYSTEM.pkl', 'rb') as f:
    mamba = pickle.load(f)

print("Retraining scalers with correct features...")

# Retrain scalers
scaler_a = StandardScaler()
scaler_b = StandardScaler()

X_train_scaled_a = scaler_a.fit_transform(X_train)
X_test_scaled_a = scaler_a.transform(X_test)

X_train_scaled_b = scaler_b.fit_transform(X_train)
X_test_scaled_b = scaler_b.transform(X_test)

# Retrain models with correct data
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR

print("Retraining 10 models (Branch A - Halftime)...")

models_a = {}

# Quick training with default params
models_a['xgboost'] = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42)
models_a['xgboost'].fit(X_train_scaled_a, y_half_train)

models_a['lightgbm'] = LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1)
models_a['lightgbm'].fit(X_train_scaled_a, y_half_train)

models_a['extratrees'] = ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42)
models_a['extratrees'].fit(X_train_scaled_a, y_half_train)

models_a['randomforest'] = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
models_a['randomforest'].fit(X_train_scaled_a, y_half_train)

models_a['histgradient'] = HistGradientBoostingRegressor(max_iter=200, random_state=42)
models_a['histgradient'].fit(X_train_scaled_a, y_half_train)

models_a['ridge'] = Ridge(alpha=1.0)
models_a['ridge'].fit(X_train_scaled_a, y_half_train)

models_a['elasticnet'] = ElasticNet(alpha=0.1, max_iter=2000)
models_a['elasticnet'].fit(X_train_scaled_a, y_half_train)

models_a['svr'] = SVR(C=1.0, epsilon=0.1)
models_a['svr'].fit(X_train_scaled_a, y_half_train)

from sklearn.neural_network import MLPRegressor
models_a['mlp'] = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
models_a['mlp'].fit(X_train_scaled_a, y_half_train)

from sklearn.ensemble import GradientBoostingRegressor
models_a['gradboost'] = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
models_a['gradboost'].fit(X_train_scaled_a, y_half_train)

# Test
preds_a = []
for model in models_a.values():
    preds_a.append(model.predict(X_test_scaled_a))

pred_a = np.mean(preds_a, axis=0)
mae_a = mean_absolute_error(y_half_test, pred_a)

print(f"✅ Branch A MAE: {mae_a:.3f}")
print()

print("Retraining 10 models (Branch B - Final)...")

models_b = {}

models_b['xgboost'] = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42)
models_b['xgboost'].fit(X_train_scaled_b, y_final_train)

models_b['lightgbm'] = LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1)
models_b['lightgbm'].fit(X_train_scaled_b, y_final_train)

models_b['extratrees'] = ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42)
models_b['extratrees'].fit(X_train_scaled_b, y_final_train)

models_b['randomforest'] = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
models_b['randomforest'].fit(X_train_scaled_b, y_final_train)

models_b['histgradient'] = HistGradientBoostingRegressor(max_iter=200, random_state=42)
models_b['histgradient'].fit(X_train_scaled_b, y_final_train)

models_b['ridge'] = Ridge(alpha=1.0)
models_b['ridge'].fit(X_train_scaled_b, y_final_train)

models_b['elasticnet'] = ElasticNet(alpha=0.1, max_iter=2000)
models_b['elasticnet'].fit(X_train_scaled_b, y_final_train)

models_b['svr'] = SVR(C=1.0, epsilon=0.1)
models_b['svr'].fit(X_train_scaled_b, y_final_train)

models_b['mlp'] = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
models_b['mlp'].fit(X_train_scaled_b, y_final_train)

models_b['gradboost'] = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
models_b['gradboost'].fit(X_train_scaled_b, y_final_train)

# Test
preds_b = []
for model in models_b.values():
    preds_b.append(model.predict(X_test_scaled_b))

pred_b = np.mean(preds_b, axis=0)
mae_b = mean_absolute_error(y_final_test, pred_b)

print(f"✅ Branch B MAE: {mae_b:.3f}")
print()

# Update Mamba system
mamba['branch_a_halftime']['models'] = models_a
mamba['branch_a_halftime']['scaler'] = scaler_a
mamba['branch_a_halftime']['champion_mae'] = mae_a

mamba['branch_b_final']['models'] = models_b
mamba['branch_b_final']['scaler'] = scaler_b
mamba['branch_b_final']['champion_mae'] = mae_b

mamba['feature_names'] = mamba_features
mamba['feature_count_actual'] = len(mamba_features)

# Save
with open('MAMBA_MENTALITY_SYSTEM.pkl', 'wb') as f:
    pickle.dump(mamba, f)

print("✅ Mamba Mentality system updated and saved")
print()

print("="*80)
print("🎉 MAMBA RETRAINED SUCCESSFULLY")
print("="*80)
print()
print(f"MAMBA MENTALITY (retrained):")
print(f"  Halftime: {mae_a:.3f} MAE")
print(f"  Final:    {mae_b:.3f} MAE")
print()
print(f"STRIVE FOR GREATNESS (unchanged):")
print(f"  Halftime: 5.296 MAE")
print(f"  Final:    9.882 MAE")
print()
print("✅ Both systems ready for Monday 1 AM")
print("="*80)

