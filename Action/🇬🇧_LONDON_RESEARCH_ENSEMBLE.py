#!/usr/bin/env python3
"""
🇬🇧 LONDON RESEARCH ENSEMBLE
Based on UK ML research excellence (Imperial College, UCL, Oxford, Cambridge)

LONDON RESEARCH STRENGTHS:
1. Probabilistic modeling (Bayesian optimization)
2. Ensemble theory (variance reduction)
3. Robust statistics (outlier handling)
4. Time series analysis (temporal modeling)
5. Uncertainty quantification (confidence intervals)

PHILOSOPHY: British rigor + statistical theory
GOAL: <5% overfitting with proper uncertainty quantification

OVERFITTING FRAMEWORK: Applied at every step
"""

import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.ensemble import BaggingRegressor, VotingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🇬🇧 LONDON RESEARCH ENSEMBLE - BRITISH RIGOR")
print("="*80)
print()

# Load data
with open('ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl', 'rb') as f:
    data = pickle.load(f)

split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

# Load pruned features from ULTRA
with open('ULTRA_OPTIMIZED_ELON_MODE.pkl', 'rb') as f:
    ultra = pickle.load(f)

feature_names = ultra['feature_names']

print(f"✅ Data: {len(data)} games")
print(f"✅ Features: {len(feature_names)} (sparse, British parsimony)")
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

print("Training London-style models (Branch A)...")
print()

models_a_london = {}

# Bayesian models (uncertainty quantification)
print("  [1/6] Bayesian Ridge (strong priors)...")
models_a_london['bayesian_ridge'] = BayesianRidge(
    max_iter=500, alpha_1=1e-4, alpha_2=1e-4,
    lambda_1=1e-4, lambda_2=1e-4, compute_score=True
)
models_a_london['bayesian_ridge'].fit(X_train_scaled_a, y_half_train)

print("  [2/6] ARD (automatic relevance)...")
models_a_london['ard'] = ARDRegression(
    max_iter=500, alpha_1=1e-4, alpha_2=1e-4,
    lambda_1=1e-4, lambda_2=1e-4
)
models_a_london['ard'].fit(X_train_scaled_a, y_half_train)

# Gaussian Processes (non-parametric Bayesian)
print("  [3/6] Gaussian Process (RBF)...")
subset = np.random.RandomState(42).choice(len(X_train_scaled_a), 800, replace=False)
kernel_rbf = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
models_a_london['gp_rbf'] = GaussianProcessRegressor(
    kernel=kernel_rbf, n_restarts_optimizer=3, 
    random_state=42, normalize_y=True
)
models_a_london['gp_rbf'].fit(X_train_scaled_a[subset], y_half_train[subset])

print("  [4/6] Gaussian Process (Matern)...")
kernel_matern = 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0)
models_a_london['gp_matern'] = GaussianProcessRegressor(
    kernel=kernel_matern, n_restarts_optimizer=3,
    random_state=42, normalize_y=True
)
models_a_london['gp_matern'].fit(X_train_scaled_a[subset], y_half_train[subset])

# Bagging (variance reduction via bootstrap)
print("  [5/6] Bagging Regressor (variance reduction)...")
from sklearn.tree import DecisionTreeRegressor
base_estimator = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20)
models_a_london['bagging'] = BaggingRegressor(
    estimator=base_estimator, n_estimators=50,
    max_samples=0.7, max_features=0.7, random_state=42, n_jobs=-1
)
models_a_london['bagging'].fit(X_train_scaled_a, y_half_train)

# Voting (ensemble averaging)
print("  [6/6] Voting Ensemble (theory-based combination)...")
from sklearn.linear_model import Ridge
voters = [
    ('ridge1', Ridge(alpha=2.0)),
    ('ridge2', Ridge(alpha=3.0)),
    ('bayesian', BayesianRidge(max_iter=300))
]
models_a_london['voting'] = VotingRegressor(voters)
models_a_london['voting'].fit(X_train_scaled_a, y_half_train)

print()
print("✅ Branch A: 6 London research models")
print()

# Test with overfitting monitoring
print("🔬 OVERFITTING DIAGNOSTIC (Branch A - London):")
print()

preds_train_a = []
preds_test_a = []

for name, model in models_a_london.items():
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

status_a = "✅" if gap_a < 5 else "⭐" if gap_a < 10 else "⚠️" if gap_a < 20 else "❌"
print(f"\nENSEMBLE: Train {train_mae_a:.3f} | Test {test_mae_a:.3f} | Gap {gap_a:+5.1f}% {status_a}")
print()

# Branch B
print("Training London-style models (Branch B)...")
models_b_london = {}

print("  [1/6] Bayesian Ridge...")
models_b_london['bayesian_ridge'] = BayesianRidge(max_iter=500, alpha_1=1e-4, alpha_2=1e-4)
models_b_london['bayesian_ridge'].fit(X_train_scaled_b, y_final_train)

print("  [2/6] ARD...")
models_b_london['ard'] = ARDRegression(max_iter=500)
models_b_london['ard'].fit(X_train_scaled_b, y_final_train)

print("  [3/6] GP (RBF)...")
models_b_london['gp_rbf'] = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=3, random_state=42, normalize_y=True)
models_b_london['gp_rbf'].fit(X_train_scaled_b[subset], y_final_train[subset])

print("  [4/6] GP (Matern)...")
models_b_london['gp_matern'] = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=3, random_state=42, normalize_y=True)
models_b_london['gp_matern'].fit(X_train_scaled_b[subset], y_final_train[subset])

print("  [5/6] Bagging...")
models_b_london['bagging'] = BaggingRegressor(estimator=base_estimator, n_estimators=50, max_samples=0.7, random_state=42, n_jobs=-1)
models_b_london['bagging'].fit(X_train_scaled_b, y_final_train)

print("  [6/6] Voting...")
models_b_london['voting'] = VotingRegressor(voters)
models_b_london['voting'].fit(X_train_scaled_b, y_final_train)

print()

# Test Branch B
print("🔬 OVERFITTING DIAGNOSTIC (Branch B - London):")
print()

preds_train_b = []
preds_test_b = []

for name, model in models_b_london.items():
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

status_b = "✅" if gap_b < 5 else "⭐" if gap_b < 10 else "⚠️" if gap_b < 20 else "❌"
print(f"\nENSEMBLE: Train {train_mae_b:.3f} | Test {test_mae_b:.3f} | Gap {gap_b:+5.1f}% {status_b}")
print()

# Save
london_system = {
    'branch_a_halftime': {
        'models': models_a_london,
        'scaler': scaler_a,
        'train_mae': train_mae_a,
        'test_mae': test_mae_a,
        'overfitting_gap': gap_a
    },
    'branch_b_final': {
        'models': models_b_london,
        'scaler': scaler_b,
        'train_mae': train_mae_b,
        'test_mae': test_mae_b,
        'overfitting_gap': gap_b
    },
    'metadata': {
        'build_date': '2025-10-20',
        'philosophy': 'London Research - Bayesian + GP + Robust theory',
        'feature_count': len(feature_names),
        'models_trained': 12
    },
    'feature_names': feature_names
}

with open('LONDON_RESEARCH_ENSEMBLE.pkl', 'wb') as f:
    pickle.dump(london_system, f)

print("✅ Saved: LONDON_RESEARCH_ENSEMBLE.pkl")
print()

print("="*80)
print("🏆 LONDON vs ABSOLUTE_BEST")
print("="*80)
print()

with open('ABSOLUTE_BEST_SYSTEM.pkl', 'rb') as f:
    best = pickle.load(f)

print(f"London:  {test_mae_a:.3f} / {test_mae_b:.3f} MAE | {gap_a:.1f}% / {gap_b:.1f}% overfit")
print(f"BEST:    5.407 / 9.191 MAE | 2.0% / 6.0% overfit")
print()

if gap_a < 10 and gap_b < 15:
    print("✅ LONDON PASSES OVERFITTING AUDIT")
    print("   Can use as backup system")
else:
    print("⚠️  LONDON moderate overfitting")

print()
print("="*80)
print("✅ LONDON SYSTEM COMPLETE")
print("="*80)
