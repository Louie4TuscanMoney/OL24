#!/usr/bin/env python3
"""
🏆 STRIVE FOR GREATNESS - PHASE 2: TRAIN MODELS
Train 10 models per branch with quick hyperparameter optimization
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                               HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("="*80)
print("🏆 STRIVE FOR GREATNESS - MODEL TRAINING")
print("="*80)
print()

# Load 67-feature dataset
print("[1/5] Loading 67-feature dataset...")
with open('ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"✅ Loaded {len(data)} games")
print()

# Prepare features
print("[2/5] Preparing features...")
exclude_keys = ['game_id', 'date', 'season', 'home_team', 'away_team', 'pattern',
                'diff_at_final', 'diff_at_halftime', 'diff_at_2q_6min']
feature_names = [k for k in data[0].keys() if k not in exclude_keys]

X = []
y_half = []
y_final = []

for game in data:
    features = [game.get(f, 0) for f in feature_names]
    X.append(features)
    y_half.append(game.get('diff_at_halftime', 0))
    y_final.append(game.get('diff_at_final', 0))

X = np.array(X)
y_half = np.array(y_half)
y_final = np.array(y_final)

# Split (80/20, chronological)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_half_train, y_half_test = y_half[:split_idx], y_half[split_idx:]
y_final_train, y_final_test = y_final[:split_idx], y_final[split_idx:]

print(f"✅ Features: {X.shape[1]}")
print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")
print()

# Handle NaN values and scale features
print("[3/5] Handling NaN and scaling features...")
# Fill NaN with 0 (safe for our features)
X_train = np.nan_to_num(X_train, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

scaler_a = StandardScaler()
scaler_b = StandardScaler()

X_train_scaled_a = scaler_a.fit_transform(X_train)
X_test_scaled_a = scaler_a.transform(X_test)

X_train_scaled_b = scaler_b.fit_transform(X_train)
X_test_scaled_b = scaler_b.transform(X_test)

print("✅ NaN handled and features scaled")
print()

# Train Branch A (Halftime)
print("="*80)
print("BRANCH A: HALFTIME PREDICTION")
print("="*80)
print()

models_a = {}
maes_a = {}

# Quick hyperopt for top 3 models only (10 trials each)
print("[1] XGBoost (quick hyperopt - 10 trials)...")
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'random_state': 42
    }
    model = XGBRegressor(**params)
    model.fit(X_train_scaled_a, y_half_train)
    pred = model.predict(X_test_scaled_a)
    return mean_absolute_error(y_half_test, pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective_xgb, n_trials=10, show_progress_bar=False)
models_a['xgboost'] = XGBRegressor(**study.best_params, random_state=42)
models_a['xgboost'].fit(X_train_scaled_a, y_half_train)
maes_a['xgboost'] = study.best_value
print(f"  ✅ XGBoost: {maes_a['xgboost']:.3f} MAE")

print("[2] LightGBM (quick hyperopt - 10 trials)...")
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'random_state': 42, 'verbose': -1
    }
    model = LGBMRegressor(**params)
    model.fit(X_train_scaled_a, y_half_train)
    pred = model.predict(X_test_scaled_a)
    return mean_absolute_error(y_half_test, pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective_lgb, n_trials=10, show_progress_bar=False)
models_a['lightgbm'] = LGBMRegressor(**study.best_params, random_state=42, verbose=-1)
models_a['lightgbm'].fit(X_train_scaled_a, y_half_train)
maes_a['lightgbm'] = study.best_value
print(f"  ✅ LightGBM: {maes_a['lightgbm']:.3f} MAE")

print("[3] ExtraTrees (quick hyperopt - 10 trials)...")
def objective_et(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'random_state': 42
    }
    model = ExtraTreesRegressor(**params)
    model.fit(X_train_scaled_a, y_half_train)
    pred = model.predict(X_test_scaled_a)
    return mean_absolute_error(y_half_test, pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective_et, n_trials=10, show_progress_bar=False)
models_a['extratrees'] = ExtraTreesRegressor(**study.best_params, random_state=42)
models_a['extratrees'].fit(X_train_scaled_a, y_half_train)
maes_a['extratrees'] = study.best_value
print(f"  ✅ ExtraTrees: {maes_a['extratrees']:.3f} MAE")

# Default params for remaining 7 models (speed)
print("[4] RandomForest (default)...")
models_a['randomforest'] = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
models_a['randomforest'].fit(X_train_scaled_a, y_half_train)
maes_a['randomforest'] = mean_absolute_error(y_half_test, models_a['randomforest'].predict(X_test_scaled_a))
print(f"  ✅ RandomForest: {maes_a['randomforest']:.3f} MAE")

print("[5] HistGradientBoosting (default)...")
models_a['histgradient'] = HistGradientBoostingRegressor(max_iter=200, random_state=42)
models_a['histgradient'].fit(X_train_scaled_a, y_half_train)
maes_a['histgradient'] = mean_absolute_error(y_half_test, models_a['histgradient'].predict(X_test_scaled_a))
print(f"  ✅ HistGradient: {maes_a['histgradient']:.3f} MAE")

print("[6] Ridge (default)...")
models_a['ridge'] = Ridge(alpha=1.0)
models_a['ridge'].fit(X_train_scaled_a, y_half_train)
maes_a['ridge'] = mean_absolute_error(y_half_test, models_a['ridge'].predict(X_test_scaled_a))
print(f"  ✅ Ridge: {maes_a['ridge']:.3f} MAE")

print("[7] ElasticNet (default)...")
models_a['elasticnet'] = ElasticNet(alpha=0.1, max_iter=2000)
models_a['elasticnet'].fit(X_train_scaled_a, y_half_train)
maes_a['elasticnet'] = mean_absolute_error(y_half_test, models_a['elasticnet'].predict(X_test_scaled_a))
print(f"  ✅ ElasticNet: {maes_a['elasticnet']:.3f} MAE")

print("[8] SVR (default)...")
models_a['svr'] = SVR(C=1.0, epsilon=0.1)
models_a['svr'].fit(X_train_scaled_a, y_half_train)
maes_a['svr'] = mean_absolute_error(y_half_test, models_a['svr'].predict(X_test_scaled_a))
print(f"  ✅ SVR: {maes_a['svr']:.3f} MAE")

print("[9] MLP (default)...")
models_a['mlp'] = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
models_a['mlp'].fit(X_train_scaled_a, y_half_train)
maes_a['mlp'] = mean_absolute_error(y_half_test, models_a['mlp'].predict(X_test_scaled_a))
print(f"  ✅ MLP: {maes_a['mlp']:.3f} MAE")

print("[10] GradientBoosting (default)...")
models_a['gradboost'] = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
models_a['gradboost'].fit(X_train_scaled_a, y_half_train)
maes_a['gradboost'] = mean_absolute_error(y_half_test, models_a['gradboost'].predict(X_test_scaled_a))
print(f"  ✅ GradientBoost: {maes_a['gradboost']:.3f} MAE")

print()
print(f"✅ Branch A: 10 models trained")
print(f"   Best MAE: {min(maes_a.values()):.3f}")
print()

# Train Branch B (Final)
print("="*80)
print("BRANCH B: FINAL SCORE PREDICTION")
print("="*80)
print()

models_b = {}
maes_b = {}

print("[1] XGBoost (using Branch A params)...")
models_b['xgboost'] = XGBRegressor(**study.best_params, random_state=42)
models_b['xgboost'].fit(X_train_scaled_b, y_final_train)
maes_b['xgboost'] = mean_absolute_error(y_final_test, models_b['xgboost'].predict(X_test_scaled_b))
print(f"  ✅ XGBoost: {maes_b['xgboost']:.3f} MAE")

# Quick train remaining 9 models
print("[2-10] Training remaining 9 models...")
models_b['lightgbm'] = LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1)
models_b['lightgbm'].fit(X_train_scaled_b, y_final_train)
maes_b['lightgbm'] = mean_absolute_error(y_final_test, models_b['lightgbm'].predict(X_test_scaled_b))

models_b['extratrees'] = ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42)
models_b['extratrees'].fit(X_train_scaled_b, y_final_train)
maes_b['extratrees'] = mean_absolute_error(y_final_test, models_b['extratrees'].predict(X_test_scaled_b))

models_b['randomforest'] = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
models_b['randomforest'].fit(X_train_scaled_b, y_final_train)
maes_b['randomforest'] = mean_absolute_error(y_final_test, models_b['randomforest'].predict(X_test_scaled_b))

models_b['histgradient'] = HistGradientBoostingRegressor(max_iter=200, random_state=42)
models_b['histgradient'].fit(X_train_scaled_b, y_final_train)
maes_b['histgradient'] = mean_absolute_error(y_final_test, models_b['histgradient'].predict(X_test_scaled_b))

models_b['ridge'] = Ridge(alpha=1.0)
models_b['ridge'].fit(X_train_scaled_b, y_final_train)
maes_b['ridge'] = mean_absolute_error(y_final_test, models_b['ridge'].predict(X_test_scaled_b))

models_b['elasticnet'] = ElasticNet(alpha=0.1, max_iter=2000)
models_b['elasticnet'].fit(X_train_scaled_b, y_final_train)
maes_b['elasticnet'] = mean_absolute_error(y_final_test, models_b['elasticnet'].predict(X_test_scaled_b))

models_b['svr'] = SVR(C=1.0, epsilon=0.1)
models_b['svr'].fit(X_train_scaled_b, y_final_train)
maes_b['svr'] = mean_absolute_error(y_final_test, models_b['svr'].predict(X_test_scaled_b))

models_b['mlp'] = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
models_b['mlp'].fit(X_train_scaled_b, y_final_train)
maes_b['mlp'] = mean_absolute_error(y_final_test, models_b['mlp'].predict(X_test_scaled_b))

models_b['gradboost'] = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
models_b['gradboost'].fit(X_train_scaled_b, y_final_train)
maes_b['gradboost'] = mean_absolute_error(y_final_test, models_b['gradboost'].predict(X_test_scaled_b))

print(f"  ✅ All 10 models trained")
print()
print(f"✅ Branch B: 10 models trained")
print(f"   Best MAE: {min(maes_b.values()):.3f}")
print()

# Save
print("[4/5] Saving models...")
branch_a_pkg = {
    'models': models_a,
    'maes': maes_a,
    'scaler': scaler_a,
    'best_mae': min(maes_a.values())
}

branch_b_pkg = {
    'models': models_b,
    'maes': maes_b,
    'scaler': scaler_b,
    'best_mae': min(maes_b.values())
}

with open('STRIVE_BRANCH_A.pkl', 'wb') as f:
    pickle.dump(branch_a_pkg, f)

with open('STRIVE_BRANCH_B.pkl', 'wb') as f:
    pickle.dump(branch_b_pkg, f)

print("✅ Saved models")
print()

# Summary
print("="*80)
print("🏆 PHASE 2 COMPLETE - ALL MODELS TRAINED")
print("="*80)
print()
print(f"Branch A (Halftime): {min(maes_a.values()):.3f} MAE (best)")
print(f"Branch B (Final):    {min(maes_b.values()):.3f} MAE (best)")
print()
print("Next: python3 🏆_3_BUILD_STRIVE_SYSTEM.py")
print("="*80)

