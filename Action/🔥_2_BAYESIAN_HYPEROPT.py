#!/usr/bin/env python3
"""
🔥 BAYESIAN HYPERPARAMETER OPTIMIZATION
Find OPTIMAL params for XGBoost, LightGBM, ExtraTrees
Target: Reduce MAE by 15-25%
"""

import pickle
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔥 BAYESIAN HYPERPARAMETER OPTIMIZATION - FINDING BEST PARAMS")
print("="*80)
print()

# Load enhanced patterns
print("[1/5] Loading enhanced patterns...")
with open('ULTRA_ENHANCED_PATTERNS_V2.pkl', 'rb') as f:
    patterns = pickle.load(f)
print(f"✅ Loaded {len(patterns)} games")
print()

# Build feature matrix
print("[2/5] Building feature matrix...")
X = []
y = []

for p in patterns:
    if p.get('diff_at_final') is None:
        continue
    
    # Extract all features
    features = []
    
    # Pattern (18)
    features.extend(p.get('pattern', [0]*18))
    
    # Statistical (4)
    features.extend([
        p.get('mean_diff', 0),
        p.get('std_diff', 0),
        p.get('trend', 0),
        p.get('volatility', 0)
    ])
    
    # Lag (6)
    features.extend([
        p.get('team_diff_lag1', 0),
        p.get('team_mean_lag1', 0),
        p.get('team_diff_rolling3', 0),
        p.get('team_volatility_rolling3', 2.0),
        p.get('team_form_10games', 0),
        p.get('team_consistency', 10.0)
    ])
    
    # Spectral (6)
    features.extend([
        p.get('spectral_energy', 0),
        p.get('low_freq_power', 0),
        p.get('mid_freq_power', 0),
        p.get('high_freq_power', 0),
        p.get('dominant_freq', 0),
        p.get('spectral_entropy', 0)
    ])
    
    # Momentum (6)
    features.extend([
        p.get('velocity', 0),
        p.get('acceleration', 0),
        p.get('recent_momentum', 0),
        p.get('lead_changes', 0),
        p.get('max_swing', 0),
        p.get('comeback_potential', 0)
    ])
    
    # Autocorrelation (3)
    features.extend([
        p.get('autocorr_lag1', 0),
        p.get('autocorr_lag3', 0),
        p.get('autocorr_lag5', 0)
    ])
    
    # Advanced stats (8)
    features.extend([
        p.get('efg_proxy', 0),
        p.get('ts_proxy', 0),
        p.get('netrtg_proxy', 0),
        p.get('pie_proxy', 0),
        p.get('pm_proxy', 0),
        p.get('usg_proxy', 0),
        p.get('pace_proxy', 0),
        p.get('four_factors_proxy', 0)
    ])
    
    X.append(features)
    y.append(p['diff_at_final'])

X = np.array(X)
y = np.array(y)

print(f"✅ Feature matrix: {X.shape[0]} games × {X.shape[1]} features")
print()

# TimeSeriesSplit (proper CV for time series)
print("[3/5] Setting up TimeSeriesSplit CV...")
tscv = TimeSeriesSplit(n_splits=5)
print("✅ Using 5-fold TimeSeriesSplit")
print()

# OPTIMIZE XGBOOST
print("[4/5] Optimizing XGBoost (this will take 30-60 min)...")
print("Running 100 Bayesian optimization trials...")
print()

def xgboost_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)
    mae = -scores.mean()
    
    # Print progress every 10 trials
    if trial.number % 10 == 0:
        print(f"  Trial {trial.number}: MAE = {mae:.3f}")
    
    return mae

# Run optimization
xgb_study = optuna.create_study(direction='minimize', study_name='xgboost_optimization')
xgb_study.optimize(xgboost_objective, n_trials=100, show_progress_bar=True)

best_xgb_params = xgb_study.best_params
best_xgb_mae = xgb_study.best_value

print()
print(f"✅ XGBoost optimization complete!")
print(f"   Best MAE: {best_xgb_mae:.3f}")
print(f"   Best params: {best_xgb_params}")
print()

# OPTIMIZE EXTRATREES (best performer in research)
print("Optimizing ExtraTrees (30-45 min)...")

def extratrees_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = ExtraTreesRegressor(**params)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)
    mae = -scores.mean()
    
    if trial.number % 10 == 0:
        print(f"  Trial {trial.number}: MAE = {mae:.3f}")
    
    return mae

et_study = optuna.create_study(direction='minimize', study_name='extratrees_optimization')
et_study.optimize(extratrees_objective, n_trials=100, show_progress_bar=True)

best_et_params = et_study.best_params
best_et_mae = et_study.best_value

print()
print(f"✅ ExtraTrees optimization complete!")
print(f"   Best MAE: {best_et_mae:.3f}")
print(f"   Best params: {best_et_params}")
print()

# Save best parameters
print("[5/5] Saving best parameters...")
best_params = {
    'xgboost': {
        'params': best_xgb_params,
        'mae': best_xgb_mae
    },
    'extratrees': {
        'params': best_et_params,
        'mae': best_et_mae
    }
}

with open('BEST_HYPERPARAMETERS.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print(f"✅ Saved to: BEST_HYPERPARAMETERS.pkl")
print()

print("="*80)
print("🔥 HYPERPARAMETER OPTIMIZATION COMPLETE")
print("="*80)
print()
print(f"XGBoost MAE: {best_xgb_mae:.3f} (baseline was 10.13)")
print(f"ExtraTrees MAE: {best_et_mae:.3f}")
print()
if min(best_xgb_mae, best_et_mae) < 7.0:
    improvement = (8.22 - min(best_xgb_mae, best_et_mae)) / 8.22 * 100
    print(f"✅ IMPROVEMENT: {improvement:.1f}% better than baseline!")
    print(f"   Next: Build stacked ensemble to get to 4-5 MAE")
else:
    print(f"⚠️  Still above 7 MAE - need more features or ensemble")
print()
print(f"Next command: python3 🔥_3_TRAIN_OPTIMIZED_ENSEMBLE.py")
print("="*80)

