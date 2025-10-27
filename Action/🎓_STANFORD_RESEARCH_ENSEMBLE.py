#!/usr/bin/env python3
"""
🎓 STANFORD RESEARCH ENSEMBLE
Advanced academic methods for NBA prediction (diverse from Mamba/Strive)

PHILOSOPHY:
- Mamba/Strive use: XGBoost, LightGBM, RandomForest, trees
- Stanford uses: Deep learning, Bayesian, Gaussian processes, advanced ensembles

GOAL: Maximize diversity to reduce overfitting
"""

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎓 STANFORD RESEARCH ENSEMBLE - ADVANCED METHODS")
print("="*80)
print()
print("Philosophy: Maximize model diversity to reduce overfitting")
print("Approach: Deep learning + Bayesian + Gaussian processes")
print()

# Load clean chronological data
print("[1/6] Loading clean data...")
with open('ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"✅ Loaded {len(data)} games (chronologically sorted)")
print()

# Split
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"✅ Train: {len(train_data)} games")
print(f"✅ Test:  {len(test_data)} games")
print()

# Extract features
exclude = ['game_id', 'date', 'season', 'home_team', 'away_team', 'pattern',
           'diff_at_final', 'diff_at_halftime', 'diff_at_2q_6min']
feature_names = [k for k in data[0].keys() if k not in exclude]

print(f"✅ Features: {len(feature_names)}")
print()

# Prepare data
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

# Scale
print("[2/6] Scaling features...")
scaler_a = StandardScaler()
scaler_b = StandardScaler()

X_train_scaled_a = scaler_a.fit_transform(X_train)
X_test_scaled_a = scaler_a.transform(X_test)

X_train_scaled_b = scaler_b.fit_transform(X_train)
X_test_scaled_b = scaler_b.transform(X_test)

print("✅ Features scaled")
print()

# ============================================================================
# BRANCH A: HALFTIME PREDICTIONS (RESEARCH MODELS)
# ============================================================================
print("[3/6] Training Branch A - RESEARCH MODELS (Halftime)...")
print()

models_a = {}

# Model 1: Deep Neural Network (Multi-layer)
print("  [1/8] Deep Neural Network (3 hidden layers)...")
models_a['deep_nn'] = MLPRegressor(
    hidden_layer_sizes=(200, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.01,
    batch_size='auto',
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
models_a['deep_nn'].fit(X_train_scaled_a, y_half_train)

# Model 2: Bayesian Ridge (uncertainty quantification)
print("  [2/8] Bayesian Ridge Regression...")
models_a['bayesian_ridge'] = BayesianRidge(
    max_iter=500,
    alpha_1=1e-6,
    alpha_2=1e-6,
    lambda_1=1e-6,
    lambda_2=1e-6,
    compute_score=True
)
models_a['bayesian_ridge'].fit(X_train_scaled_a, y_half_train)

# Model 3: Automatic Relevance Determination (feature selection)
print("  [3/8] ARD Regression (feature selection)...")
models_a['ard'] = ARDRegression(
    max_iter=500,
    alpha_1=1e-6,
    alpha_2=1e-6,
    lambda_1=1e-6,
    lambda_2=1e-6,
    compute_score=True
)
models_a['ard'].fit(X_train_scaled_a, y_half_train)

# Model 4: Gaussian Process (non-parametric, uses subset for speed)
print("  [4/8] Gaussian Process (RBF kernel, 1000 samples)...")
# Use subset for GP (expensive)
train_subset_idx = np.random.RandomState(42).choice(len(X_train_scaled_a), 1000, replace=False)
kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
models_a['gaussian_process'] = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    random_state=42,
    normalize_y=True
)
models_a['gaussian_process'].fit(X_train_scaled_a[train_subset_idx], y_half_train[train_subset_idx])

# Model 5: Deep NN with dropout (regularization)
print("  [5/8] Deep Neural Network (dropout regularization)...")
models_a['deep_nn_dropout'] = MLPRegressor(
    hidden_layer_sizes=(150, 75),
    activation='tanh',
    solver='adam',
    alpha=0.1,  # Strong L2 regularization
    batch_size=128,
    learning_rate='adaptive',
    max_iter=1000,
    random_state=43,
    early_stopping=True,
    validation_fraction=0.15
)
models_a['deep_nn_dropout'].fit(X_train_scaled_a, y_half_train)

# Model 6: Wide Neural Network (more neurons)
print("  [6/8] Wide Neural Network...")
models_a['wide_nn'] = MLPRegressor(
    hidden_layer_sizes=(300,),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=800,
    random_state=44,
    early_stopping=True
)
models_a['wide_nn'].fit(X_train_scaled_a, y_half_train)

# Model 7: GP with Matern kernel (different smoothness)
print("  [7/8] Gaussian Process (Matern kernel, 800 samples)...")
train_subset_idx2 = np.random.RandomState(43).choice(len(X_train_scaled_a), 800, replace=False)
kernel_matern = 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0)
models_a['gp_matern'] = GaussianProcessRegressor(
    kernel=kernel_matern,
    n_restarts_optimizer=3,
    random_state=43,
    normalize_y=True
)
models_a['gp_matern'].fit(X_train_scaled_a[train_subset_idx2], y_half_train[train_subset_idx2])

# Model 8: Ensemble NN (different initializations)
print("  [8/8] Ensemble Neural Network...")
models_a['ensemble_nn'] = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.05,
    max_iter=1000,
    random_state=45,
    early_stopping=True
)
models_a['ensemble_nn'].fit(X_train_scaled_a, y_half_train)

print()
print("✅ Branch A: 8 research models trained")
print()

# Test Branch A
print("Testing Branch A on holdout...")
preds_half_train = []
preds_half_test = []

for name, model in models_a.items():
    pred_train = model.predict(X_train_scaled_a)
    pred_test = model.predict(X_test_scaled_a)
    preds_half_train.append(pred_train)
    preds_half_test.append(pred_test)
    
    mae_train = mean_absolute_error(y_half_train, pred_train)
    mae_test = mean_absolute_error(y_half_test, pred_test)
    print(f"  {name:20s}: Train {mae_train:.3f} | Test {mae_test:.3f} MAE")

ensemble_pred_half_train = np.mean(preds_half_train, axis=0)
ensemble_pred_half_test = np.mean(preds_half_test, axis=0)

train_mae_half = mean_absolute_error(y_half_train, ensemble_pred_half_train)
test_mae_half = mean_absolute_error(y_half_test, ensemble_pred_half_test)

print()
print(f"ENSEMBLE: Train {train_mae_half:.3f} | Test {test_mae_half:.3f} MAE")
gap_half = (test_mae_half - train_mae_half) / train_mae_half * 100
print(f"Overfitting gap: {gap_half:.1f}%")
print()

# ============================================================================
# BRANCH B: FINAL PREDICTIONS (RESEARCH MODELS)
# ============================================================================
print("[4/6] Training Branch B - RESEARCH MODELS (Final)...")
print()

models_b = {}

print("  [1/8] Deep Neural Network (3 hidden layers)...")
models_b['deep_nn'] = MLPRegressor(
    hidden_layer_sizes=(200, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.01,
    max_iter=1000,
    random_state=42,
    early_stopping=True
)
models_b['deep_nn'].fit(X_train_scaled_b, y_final_train)

print("  [2/8] Bayesian Ridge Regression...")
models_b['bayesian_ridge'] = BayesianRidge(max_iter=500)
models_b['bayesian_ridge'].fit(X_train_scaled_b, y_final_train)

print("  [3/8] ARD Regression...")
models_b['ard'] = ARDRegression(max_iter=500)
models_b['ard'].fit(X_train_scaled_b, y_final_train)

print("  [4/8] Gaussian Process (RBF kernel)...")
models_b['gaussian_process'] = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    random_state=42,
    normalize_y=True
)
models_b['gaussian_process'].fit(X_train_scaled_b[train_subset_idx], y_final_train[train_subset_idx])

print("  [5/8] Deep NN with dropout...")
models_b['deep_nn_dropout'] = MLPRegressor(
    hidden_layer_sizes=(150, 75),
    activation='tanh',
    alpha=0.1,
    max_iter=1000,
    random_state=43,
    early_stopping=True
)
models_b['deep_nn_dropout'].fit(X_train_scaled_b, y_final_train)

print("  [6/8] Wide Neural Network...")
models_b['wide_nn'] = MLPRegressor(
    hidden_layer_sizes=(300,),
    alpha=0.001,
    max_iter=800,
    random_state=44,
    early_stopping=True
)
models_b['wide_nn'].fit(X_train_scaled_b, y_final_train)

print("  [7/8] Gaussian Process (Matern kernel)...")
models_b['gp_matern'] = GaussianProcessRegressor(
    kernel=kernel_matern,
    n_restarts_optimizer=3,
    random_state=43,
    normalize_y=True
)
models_b['gp_matern'].fit(X_train_scaled_b[train_subset_idx2], y_final_train[train_subset_idx2])

print("  [8/8] Ensemble Neural Network...")
models_b['ensemble_nn'] = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    alpha=0.05,
    max_iter=1000,
    random_state=45,
    early_stopping=True
)
models_b['ensemble_nn'].fit(X_train_scaled_b, y_final_train)

print()
print("✅ Branch B: 8 research models trained")
print()

# Test Branch B
print("Testing Branch B on holdout...")
preds_final_train = []
preds_final_test = []

for name, model in models_b.items():
    pred_train = model.predict(X_train_scaled_b)
    pred_test = model.predict(X_test_scaled_b)
    preds_final_train.append(pred_train)
    preds_final_test.append(pred_test)
    
    mae_train = mean_absolute_error(y_final_train, pred_train)
    mae_test = mean_absolute_error(y_final_test, pred_test)
    print(f"  {name:20s}: Train {mae_train:.3f} | Test {mae_test:.3f} MAE")

ensemble_pred_final_train = np.mean(preds_final_train, axis=0)
ensemble_pred_final_test = np.mean(preds_final_test, axis=0)

train_mae_final = mean_absolute_error(y_final_train, ensemble_pred_final_train)
test_mae_final = mean_absolute_error(y_final_test, ensemble_pred_final_test)

print()
print(f"ENSEMBLE: Train {train_mae_final:.3f} | Test {test_mae_final:.3f} MAE")
gap_final = (test_mae_final - train_mae_final) / train_mae_final * 100
print(f"Overfitting gap: {gap_final:.1f}%")
print()

# ============================================================================
# SAVE STANFORD RESEARCH SYSTEM
# ============================================================================
print("[5/6] Saving Stanford Research Ensemble...")

stanford_system = {
    'branch_a_halftime': {
        'models': models_a,
        'scaler': scaler_a,
        'champion_mae': test_mae_half,
        'train_mae': train_mae_half,
        'champion_strategy': 'Research Ensemble (Deep+Bayesian+GP)',
        'overfitting_gap': gap_half
    },
    'branch_b_final': {
        'models': models_b,
        'scaler': scaler_b,
        'champion_mae': test_mae_final,
        'train_mae': train_mae_final,
        'champion_strategy': 'Research Ensemble (Deep+Bayesian+GP)',
        'overfitting_gap': gap_final
    },
    'metadata': {
        'total_games': len(data),
        'train_games': len(train_data),
        'test_games': len(test_data),
        'feature_count': len(feature_names),
        'models_trained': 16,
        'build_date': '2025-10-19',
        'philosophy': 'Stanford Research - Academic diversity',
        'model_types': 'Deep NN, Bayesian Ridge, ARD, Gaussian Processes',
        'data_quality': 'CLEAN - Chronological split',
    },
    'feature_names': feature_names,
}

with open('STANFORD_RESEARCH_ENSEMBLE.pkl', 'wb') as f:
    pickle.dump(stanford_system, f)

print("✅ Saved to: STANFORD_RESEARCH_ENSEMBLE.pkl")
print()

# ============================================================================
# COMPARE TO MAMBA AND STRIVE
# ============================================================================
print("[6/6] Comparing to Mamba and Strive...")
print()

# Load Mamba
with open('MAMBA_MENTALITY_SYSTEM.pkl', 'rb') as f:
    mamba = pickle.load(f)

# Load Strive
with open('STRIVE_FOR_GREATNESS_CLEAN.pkl', 'rb') as f:
    strive = pickle.load(f)

print("="*80)
print("🎯 THREE-SYSTEM COMPARISON")
print("="*80)
print()

print("HALFTIME PREDICTIONS:")
print(f"  Mamba (67f, traditional):   {mamba['branch_a_halftime']['champion_mae']:.3f} MAE")
print(f"  Strive (73f, traditional):  {strive['branch_a_halftime']['champion_mae']:.3f} MAE")
print(f"  Stanford (73f, research):   {test_mae_half:.3f} MAE")
print()
print(f"  Overfitting gaps:")
print(f"    Mamba:    {((mamba['branch_a_halftime']['champion_mae'] - mamba['branch_a_halftime'].get('train_mae', 3.0)) / mamba['branch_a_halftime'].get('train_mae', 3.0) * 100):.1f}%")
print(f"    Strive:   {((strive['branch_a_halftime']['champion_mae'] - strive['branch_a_halftime'].get('train_mae', 3.0)) / strive['branch_a_halftime'].get('train_mae', 3.0) * 100):.1f}%")
print(f"    Stanford: {gap_half:.1f}%")
print()

print("FINAL PREDICTIONS:")
print(f"  Mamba (67f, traditional):   {mamba['branch_b_final']['champion_mae']:.3f} MAE")
print(f"  Strive (73f, traditional):  {strive['branch_b_final']['champion_mae']:.3f} MAE")
print(f"  Stanford (73f, research):   {test_mae_final:.3f} MAE")
print()
print(f"  Overfitting gaps:")
print(f"    Mamba:    {((mamba['branch_b_final']['champion_mae'] - mamba['branch_b_final'].get('train_mae', 5.0)) / mamba['branch_b_final'].get('train_mae', 5.0) * 100):.1f}%")
print(f"    Strive:   {((strive['branch_b_final']['champion_mae'] - strive['branch_b_final'].get('train_mae', 5.0)) / strive['branch_b_final'].get('train_mae', 5.0) * 100):.1f}%")
print(f"    Stanford: {gap_final:.1f}%")
print()

# Test intelligent routing (pick best per game)
print("="*80)
print("🧬 INTELLIGENT 3-WAY ROUTING")
print("="*80)
print()

# Get predictions from all 3 systems on test set
# (We already have Stanford's, need to get Mamba and Strive's)

# For now, show potential
print("POTENTIAL ENSEMBLE STRATEGIES:")
print()
print("1. SIMPLE AVERAGE")
avg_half = (mamba['branch_a_halftime']['champion_mae'] + strive['branch_a_halftime']['champion_mae'] + test_mae_half) / 3
avg_final = (mamba['branch_b_final']['champion_mae'] + strive['branch_b_final']['champion_mae'] + test_mae_final) / 3
print(f"   Halftime: {avg_half:.3f} MAE (average of 3)")
print(f"   Final:    {avg_final:.3f} MAE (average of 3)")
print()

print("2. WEIGHTED BY INVERSE MAE")
weights_half = [1/mamba['branch_a_halftime']['champion_mae'], 
                1/strive['branch_a_halftime']['champion_mae'],
                1/test_mae_half]
weights_half = [w / sum(weights_half) for w in weights_half]
print(f"   Weights: Mamba {weights_half[0]:.2f}, Strive {weights_half[1]:.2f}, Stanford {weights_half[2]:.2f}")
print()

print("3. PICK BEST PER BRANCH")
best_half = min(mamba['branch_a_halftime']['champion_mae'], 
                strive['branch_a_halftime']['champion_mae'],
                test_mae_half)
best_final = min(mamba['branch_b_final']['champion_mae'],
                 strive['branch_b_final']['champion_mae'],
                 test_mae_final)
print(f"   Halftime: {best_half:.3f} MAE (best single system)")
print(f"   Final:    {best_final:.3f} MAE (best single system)")
print()

print("="*80)
print("✅ STANFORD RESEARCH ENSEMBLE COMPLETE")
print("="*80)
print()
print("SUMMARY:")
print(f"  Models: 16 (8 per branch)")
print(f"  Types: Deep NN, Bayesian, ARD, Gaussian Processes")
print(f"  Halftime MAE: {test_mae_half:.3f}")
print(f"  Final MAE: {test_mae_final:.3f}")
print(f"  Diversity: HIGH (different from Mamba/Strive)")
print()
print("NEXT: Build intelligent router to combine all 3 systems")
print("="*80)

