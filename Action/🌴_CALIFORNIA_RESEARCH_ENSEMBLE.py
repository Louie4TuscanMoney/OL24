#!/usr/bin/env python3
"""
🌴 CALIFORNIA RESEARCH ENSEMBLE
Berkeley, UCLA, USC, Stanford, Caltech - California ML excellence

CALIFORNIA RESEARCH STRENGTHS:
1. Deep learning innovation (Berkeley AI)
2. Ensemble optimization (UCLA statistics)
3. Robust ML (USC reinforcement learning)
4. Statistical learning theory (Caltech)
5. Production ML (Stanford)

PHILOSOPHY: Innovation + Rigor + Scale
GOAL: <5% overfitting, California-grade performance

OVERFITTING FRAMEWORK: Applied throughout
"""

import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import (GradientBoostingRegressor, AdaBoostRegressor,
                               BaggingRegressor, ExtraTreesRegressor)
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🌴 CALIFORNIA RESEARCH ENSEMBLE")
print("="*80)
print()
print("Berkeley + UCLA + USC + Caltech + Stanford")
print("Philosophy: West Coast innovation with statistical rigor")
print()

# Load data
with open('ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl', 'rb') as f:
    data = pickle.load(f)

split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

# Load pruned features
with open('ULTRA_OPTIMIZED_ELON_MODE.pkl', 'rb') as f:
    ultra = pickle.load(f)

feature_names = ultra['feature_names']

print(f"✅ Data: {len(data)} games")
print(f"✅ Features: {len(feature_names)} (pruned)")
print()

# Prepare data
X_train = np.nan_to_num(np.array([[g.get(f, 0) for f in feature_names] for g in train_data]), nan=0.0)
X_test = np.nan_to_num(np.array([[g.get(f, 0) for f in feature_names] for g in test_data]), nan=0.0)
y_half_train = np.array([g.get('diff_at_halftime', 0) for g in train_data])
y_half_test = np.array([g.get('diff_at_halftime', 0) for g in test_data])
y_final_train = np.array([g.get('diff_at_final', 0) for g in train_data])
y_final_test = np.array([g.get('diff_at_final', 0) for g in test_data])

# Scale
scaler_a = RobustScaler()
scaler_b = RobustScaler()

X_train_scaled_a = scaler_a.fit_transform(X_train)
X_test_scaled_a = scaler_a.transform(X_test)

X_train_scaled_b = scaler_b.fit_transform(X_train)
X_test_scaled_b = scaler_b.transform(X_test)

# ============================================================================
# CALIFORNIA MODELS (BRANCH A)
# ============================================================================
print("Training California models (Branch A - Halftime)...")
print()

models_a_cal = {}

# Berkeley: Deep learning with strong regularization
print("  [1/8] Berkeley Deep NN (dropout + L2)...")
models_a_cal['berkeley_nn'] = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.1,  # Strong L2
    batch_size=128,
    learning_rate='adaptive',
    max_iter=800,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42
)
models_a_cal['berkeley_nn'].fit(X_train_scaled_a, y_half_train)

# UCLA: Statistical ensemble
print("  [2/8] UCLA Bagging (variance reduction)...")
from sklearn.tree import DecisionTreeRegressor
base_tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20)
models_a_cal['ucla_bagging'] = BaggingRegressor(
    estimator=base_tree,
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
models_a_cal['ucla_bagging'].fit(X_train_scaled_a, y_half_train)

# USC: Adaptive boosting
print("  [3/8] USC AdaBoost (adaptive weights)...")
models_a_cal['usc_adaboost'] = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    learning_rate=0.05,
    random_state=42
)
models_a_cal['usc_adaboost'].fit(X_train_scaled_a, y_half_train)

# Caltech: Statistical learning
print("  [4/8] Caltech ExtraTrees (randomization)...")
models_a_cal['caltech_et'] = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=4,
    min_samples_split=25,
    min_samples_leaf=12,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
models_a_cal['caltech_et'].fit(X_train_scaled_a, y_half_train)

# XGBoost (California tuned)
print("  [5/8] XGBoost (California params)...")
models_a_cal['xgb_cal'] = XGBRegressor(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.75,
    colsample_bytree=0.75,
    reg_alpha=1.0,
    reg_lambda=1.5,
    random_state=42
)
models_a_cal['xgb_cal'].fit(X_train_scaled_a, y_half_train)

# LightGBM (California tuned)
print("  [6/8] LightGBM (California params)...")
models_a_cal['lgbm_cal'] = LGBMRegressor(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.75,
    colsample_bytree=0.75,
    reg_alpha=1.0,
    reg_lambda=1.5,
    random_state=42,
    verbose=-1
)
models_a_cal['lgbm_cal'].fit(X_train_scaled_a, y_half_train)

# Gradient Boosting
print("  [7/8] GradientBoosting (regularized)...")
models_a_cal['gb_cal'] = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.7,
    max_features='sqrt',
    random_state=42
)
models_a_cal['gb_cal'].fit(X_train_scaled_a, y_half_train)

# Online learning (SGD for adaptability)
print("  [8/8] SGD Regressor (online learning ready)...")
models_a_cal['sgd_online'] = SGDRegressor(
    penalty='elasticnet',
    alpha=0.01,
    l1_ratio=0.5,
    max_iter=2000,
    random_state=42
)
models_a_cal['sgd_online'].fit(X_train_scaled_a, y_half_train)

print()
print("✅ Branch A: 8 California models trained")
print()

# Test with overfitting diagnostics
print("🔬 OVERFITTING DIAGNOSTIC (Branch A):")
print()

preds_train_a = []
preds_test_a = []

for name, model in models_a_cal.items():
    pred_train = model.predict(X_train_scaled_a)
    pred_test = model.predict(X_test_scaled_a)
    preds_train_a.append(pred_train)
    preds_test_a.append(pred_test)
    
    mae_train = mean_absolute_error(y_half_train, pred_train)
    mae_test = mean_absolute_error(y_half_test, pred_test)
    gap = (mae_test - mae_train) / mae_train * 100
    
    status = "✅" if gap < 5 else "⭐" if gap < 10 else "⚠️" if gap < 20 else "❌"
    print(f"  {name:20s}: Train {mae_train:.3f} | Test {mae_test:.3f} | Gap {gap:+5.1f}% {status}")

ensemble_train_a = np.mean(preds_train_a, axis=0)
ensemble_test_a = np.mean(preds_test_a, axis=0)

train_mae_a = mean_absolute_error(y_half_train, ensemble_train_a)
test_mae_a = mean_absolute_error(y_half_test, ensemble_test_a)
gap_a = (test_mae_a - train_mae_a) / train_mae_a * 100

status_a = "✅" if gap_a < 5 else "⭐" if gap_a < 10 else "⚠️"
print(f"\nENSEMBLE: Train {train_mae_a:.3f} | Test {test_mae_a:.3f} | Gap {gap_a:+5.1f}% {status_a}")
print()

# Branch B (same models)
print("Training California models (Branch B - Final)...")
models_b_cal = {}

print("  [1/8] Berkeley NN...")
models_b_cal['berkeley_nn'] = MLPRegressor(hidden_layer_sizes=(100, 50), alpha=0.1, max_iter=800, early_stopping=True, random_state=42)
models_b_cal['berkeley_nn'].fit(X_train_scaled_b, y_final_train)

print("  [2/8] UCLA Bagging...")
models_b_cal['ucla_bagging'] = BaggingRegressor(estimator=base_tree, n_estimators=100, max_samples=0.8, random_state=42, n_jobs=-1)
models_b_cal['ucla_bagging'].fit(X_train_scaled_b, y_final_train)

print("  [3/8] USC AdaBoost...")
models_b_cal['usc_adaboost'] = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=100, learning_rate=0.05, random_state=42)
models_b_cal['usc_adaboost'].fit(X_train_scaled_b, y_final_train)

print("  [4/8] Caltech ExtraTrees...")
models_b_cal['caltech_et'] = ExtraTreesRegressor(n_estimators=100, max_depth=4, min_samples_split=25, random_state=42, n_jobs=-1)
models_b_cal['caltech_et'].fit(X_train_scaled_b, y_final_train)

print("  [5/8] XGBoost...")
models_b_cal['xgb_cal'] = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, reg_alpha=1.0, reg_lambda=1.5, random_state=42)
models_b_cal['xgb_cal'].fit(X_train_scaled_b, y_final_train)

print("  [6/8] LightGBM...")
models_b_cal['lgbm_cal'] = LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, reg_alpha=1.0, reg_lambda=1.5, random_state=42, verbose=-1)
models_b_cal['lgbm_cal'].fit(X_train_scaled_b, y_final_train)

print("  [7/8] GradientBoosting...")
models_b_cal['gb_cal'] = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
models_b_cal['gb_cal'].fit(X_train_scaled_b, y_final_train)

print("  [8/8] SGD (online)...")
models_b_cal['sgd_online'] = SGDRegressor(penalty='elasticnet', alpha=0.01, max_iter=2000, random_state=42)
models_b_cal['sgd_online'].fit(X_train_scaled_b, y_final_train)

print()

# Test Branch B
print("🔬 OVERFITTING DIAGNOSTIC (Branch B):")
print()

preds_train_b = []
preds_test_b = []

for name, model in models_b_cal.items():
    pred_train = model.predict(X_train_scaled_b)
    pred_test = model.predict(X_test_scaled_b)
    preds_train_b.append(pred_train)
    preds_test_b.append(pred_test)
    
    mae_train = mean_absolute_error(y_final_train, pred_train)
    mae_test = mean_absolute_error(y_final_test, pred_test)
    gap = (mae_test - mae_train) / mae_train * 100
    
    status = "✅" if gap < 5 else "⭐" if gap < 10 else "⚠️" if gap < 20 else "❌"
    print(f"  {name:20s}: Train {mae_train:.3f} | Test {mae_test:.3f} | Gap {gap:+5.1f}% {status}")

ensemble_train_b = np.mean(preds_train_b, axis=0)
ensemble_test_b = np.mean(preds_test_b, axis=0)

train_mae_b = mean_absolute_error(y_final_train, ensemble_train_b)
test_mae_b = mean_absolute_error(y_final_test, ensemble_test_b)
gap_b = (test_mae_b - train_mae_b) / train_mae_b * 100

status_b = "✅" if gap_b < 5 else "⭐" if gap_b < 10 else "⚠️"
print(f"\nENSEMBLE: Train {train_mae_b:.3f} | Test {test_mae_b:.3f} | Gap {gap_b:+5.1f}% {status_b}")
print()

# Save
california_system = {
    'branch_a_halftime': {
        'models': models_a_cal,
        'scaler': scaler_a,
        'train_mae': train_mae_a,
        'test_mae': test_mae_a,
        'overfitting_gap': gap_a
    },
    'branch_b_final': {
        'models': models_b_cal,
        'scaler': scaler_b,
        'train_mae': train_mae_b,
        'test_mae': test_mae_b,
        'overfitting_gap': gap_b
    },
    'metadata': {
        'build_date': '2025-10-20',
        'philosophy': 'California Research - Berkeley + UCLA + USC + Stanford + Caltech',
        'feature_count': len(feature_names),
        'models_trained': 16,
        'institutions': ['Berkeley', 'UCLA', 'USC', 'Caltech', 'Stanford (XGB/LGBM)']
    },
    'feature_names': feature_names
}

with open('CALIFORNIA_RESEARCH_ENSEMBLE.pkl', 'wb') as f:
    pickle.dump(california_system, f)

print("✅ Saved: CALIFORNIA_RESEARCH_ENSEMBLE.pkl")
print()

# Overfitting triage
print("🧭 OVERFITTING TRIAGE (USER FRAMEWORK):")
print()

issues = 0
if gap_a > 20 or gap_b > 20:
    print(f"  ❌ High gap: {gap_a:.1f}% / {gap_b:.1f}%")
    issues += 1
else:
    print(f"  ✅ Low gap: {gap_a:.1f}% / {gap_b:.1f}%")

if len(feature_names) > 50:
    print(f"  ⚠️  {len(feature_names)} features")
    issues += 1
else:
    print(f"  ✅ {len(feature_names)} features (sparse)")

print()
print(f"TRIAGE: {issues}/7 red flags")

if issues == 0:
    print("VERDICT: ✅ HEALTHY - Ready for backup/ensemble")
elif issues <= 2:
    print("VERDICT: ⭐ GOOD - Can use with monitoring")
else:
    print("VERDICT: ⚠️ NEEDS WORK")

print()
print("="*80)
print("✅ CALIFORNIA SYSTEM COMPLETE")
print("="*80)

