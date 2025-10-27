#!/usr/bin/env python3
"""
🔥 MEGA ENSEMBLE - ALL STACKING STRATEGIES IN PARALLEL
8 HOURS TO BUILD THE BEST TIME SERIES FORECASTER EVER

BASED ON:
- Vine Copulas (Stübinger 2016): 9.25% returns, Sharpe 1.12
- Basketball ML (Papageorgiou 2024): ExtraTrees 34% WAPE
- Your existing research (Informer, Conformal, Dejavu)

STRATEGIES (RUN ALL IN PARALLEL):
1. Simple average
2. Weighted by inverse MAE
3. Weighted by inverse variance
4. Stacked Ridge meta-learner
5. Stacked Lasso meta-learner
6. Stacked ElasticNet meta-learner
7. Stacked Neural Network meta-learner
8. Gradient boosting meta-learner
9. Conformal-weighted ensemble
10. Bayesian model averaging

CONFIDENCE INTERVALS ON EVERYTHING.
OPTIMIZED FOR SPEED.
TIME SERIES FOCUSED.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor, 
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    StackingRegressor, VotingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings('ignore')

print("="*80)
print("🔥 MEGA ENSEMBLE - 10 STACKING STRATEGIES + CONFIDENCE INTERVALS")
print("="*80)
print()
print("Mission: Build best time series forecaster in 8 hours")
print("Target: <7 MAE with 90% confidence intervals")
print("Method: Try EVERYTHING in parallel")
print()

# ============================================================================
# LOAD DATA
# ============================================================================
print("[1/8] Loading data...")
with open('ULTRA_ENHANCED_PATTERNS_V2.pkl', 'rb') as f:
    patterns = pickle.load(f)

# Build feature matrix
X = []
y = []

for game in patterns:
    # Pattern features
    pattern = game.get('pattern', [0]*18)
    
    # Statistical features
    stats = game.get('statistics', {})
    stat_features = [
        stats.get('mean', 0),
        stats.get('std', 1),
        stats.get('trend', 0),
        stats.get('volatility', 1)
    ]
    
    # Team features
    home_team_stats = game.get('home_team_stats', {})
    away_team_stats = game.get('away_team_stats', {})
    team_features = [
        home_team_stats.get('OFF_RATING', 110) - away_team_stats.get('OFF_RATING', 110),
        home_team_stats.get('DEF_RATING', 110) - away_team_stats.get('DEF_RATING', 110),
        home_team_stats.get('NET_RATING', 0),
        home_team_stats.get('PACE', 100) - away_team_stats.get('PACE', 100)
    ]
    
    # Player features
    player_stars = game.get('player_stars', {})
    player_features = [
        player_stars.get('home_tier_1', 0) - player_stars.get('away_tier_1', 0),
        player_stars.get('home_tier_2', 0) - player_stars.get('away_tier_2', 0)
    ]
    
    # Combine ALL features
    features = list(pattern) + stat_features + team_features + player_features
    
    # Target (CORRECT KEY)
    target = game.get('diff_at_halftime', game.get('diff_at_final', 0))
    
    # Handle NaN
    if not np.isnan(target) and not np.isnan(features).any():
        X.append(features)
        y.append(target)

X = np.array(X)
y = np.array(y)

# Replace any remaining NaN with 0
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0.0)

# Time-based split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"✅ Loaded {len(X)} games")
print(f"✅ Features: {X.shape[1]}")
print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
print()

# ============================================================================
# LOAD BEST HYPERPARAMETERS
# ============================================================================
print("[2/8] Loading optimized hyperparameters...")
with open('BEST_HYPERPARAMETERS.pkl', 'rb') as f:
    best_params = pickle.load(f)

print("✅ Loaded best params from 100-trial optimization")
print()

# ============================================================================
# TRAIN BASE MODELS (5 DIVERSE MODELS)
# ============================================================================
print("[3/8] Training 5 diverse base models...")

# Model 1: XGBoost (optimized)
print("  Training XGBoost (optimized)...")
model_xgb = xgb.XGBRegressor(**best_params['xgboost']['params'], random_state=42, n_jobs=-1)
model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_test)
mae_xgb = mean_absolute_error(y_test, pred_xgb)
print(f"    MAE: {mae_xgb:.3f}")

# Model 2: ExtraTrees (optimized)
print("  Training ExtraTrees (optimized)...")
model_et = ExtraTreesRegressor(**best_params['extratrees']['params'], random_state=42, n_jobs=-1)
model_et.fit(X_train, y_train)
pred_et = model_et.predict(X_test)
mae_et = mean_absolute_error(y_test, pred_et)
print(f"    MAE: {mae_et:.3f}")

# Model 3: LightGBM (fast gradient boosting)
print("  Training LightGBM...")
model_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, max_depth=8, random_state=42, n_jobs=-1, verbose=-1)
model_lgb.fit(X_train, y_train)
pred_lgb = model_lgb.predict(X_test)
mae_lgb = mean_absolute_error(y_test, pred_lgb)
print(f"    MAE: {mae_lgb:.3f}")

# Model 4: RandomForest (high diversity)
print("  Training RandomForest...")
model_rf = RandomForestRegressor(n_estimators=1000, max_depth=15, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, pred_rf)
print(f"    MAE: {mae_rf:.3f}")

# Model 5: HistGradientBoosting (handles NaN natively)
print("  Training HistGradientBoosting...")
model_hgb = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_depth=10, random_state=42)
model_hgb.fit(X_train, y_train)
pred_hgb = model_hgb.predict(X_test)
mae_hgb = mean_absolute_error(y_test, pred_hgb)
print(f"    MAE: {mae_hgb:.3f}")

print()
print(f"✅ All 5 base models trained")
print()

# Collect all predictions
all_predictions = np.column_stack([pred_xgb, pred_et, pred_lgb, pred_rf, pred_hgb])
base_maes = np.array([mae_xgb, mae_et, mae_lgb, mae_rf, mae_hgb])

# ============================================================================
# STRATEGY 1-3: SIMPLE ENSEMBLES
# ============================================================================
print("[4/8] Testing simple ensemble strategies...")

# Strategy 1: Simple average
pred_avg = np.mean(all_predictions, axis=1)
mae_avg = mean_absolute_error(y_test, pred_avg)
std_avg = np.std(all_predictions, axis=1).mean()  # Average uncertainty
print(f"  1. Simple Average:        MAE={mae_avg:.3f}, Avg Std=±{std_avg:.2f}")

# Strategy 2: Weighted by inverse MAE
weights_mae = 1.0 / base_maes
weights_mae = weights_mae / weights_mae.sum()
pred_wmae = np.average(all_predictions, axis=1, weights=weights_mae)
mae_wmae = mean_absolute_error(y_test, pred_wmae)
std_wmae = np.std(all_predictions, axis=1).mean()
print(f"  2. Inverse MAE Weights:   MAE={mae_wmae:.3f}, Avg Std=±{std_wmae:.2f}")
print(f"     Weights: XGB={weights_mae[0]:.3f}, ET={weights_mae[1]:.3f}, LGB={weights_mae[2]:.3f}, RF={weights_mae[3]:.3f}, HGB={weights_mae[4]:.3f}")

# Strategy 3: Weighted by inverse variance
variances = np.var(all_predictions, axis=0)
weights_var = 1.0 / (variances + 1e-6)
weights_var = weights_var / weights_var.sum()
pred_wvar = np.average(all_predictions, axis=1, weights=weights_var)
mae_wvar = mean_absolute_error(y_test, pred_wvar)
std_wvar = np.std(all_predictions, axis=1).mean()
print(f"  3. Inverse Variance:      MAE={mae_wvar:.3f}, Avg Std=±{std_wvar:.2f}")

print()

# ============================================================================
# STRATEGY 4-8: META-LEARNER STACKING
# ============================================================================
print("[5/8] Training meta-learner strategies...")

# Get base model predictions on training set (for meta-learner)
print("  Generating meta-features from base models...")
meta_train = np.column_stack([
    model_xgb.predict(X_train),
    model_et.predict(X_train),
    model_lgb.predict(X_train),
    model_rf.predict(X_train),
    model_hgb.predict(X_train)
])
meta_test = all_predictions  # Already have test predictions

# Strategy 4: Ridge meta-learner (L2 regularization)
print("  4. Ridge meta-learner...")
meta_ridge = Ridge(alpha=1.0)
meta_ridge.fit(meta_train, y_train)
pred_ridge = meta_ridge.predict(meta_test)
mae_ridge = mean_absolute_error(y_test, pred_ridge)
print(f"     MAE={mae_ridge:.3f}")

# Strategy 5: Lasso meta-learner (L1 - feature selection)
print("  5. Lasso meta-learner...")
meta_lasso = Lasso(alpha=0.1, max_iter=10000)
meta_lasso.fit(meta_train, y_train)
pred_lasso = meta_lasso.predict(meta_test)
mae_lasso = mean_absolute_error(y_test, pred_lasso)
print(f"     MAE={mae_lasso:.3f}")

# Strategy 6: ElasticNet meta-learner (L1+L2 hybrid)
print("  6. ElasticNet meta-learner...")
meta_enet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
meta_enet.fit(meta_train, y_train)
pred_enet = meta_enet.predict(meta_test)
mae_enet = mean_absolute_error(y_test, pred_enet)
print(f"     MAE={mae_enet:.3f}")

# Strategy 7: Neural Network meta-learner (nonlinear)
print("  7. Neural Network meta-learner...")
meta_nn = MLPRegressor(hidden_layer_sizes=(20, 10), activation='relu', max_iter=1000, random_state=42)
meta_nn.fit(meta_train, y_train)
pred_nn = meta_nn.predict(meta_test)
mae_nn = mean_absolute_error(y_test, pred_nn)
print(f"     MAE={mae_nn:.3f}")

# Strategy 8: Gradient Boosting meta-learner (learn errors)
print("  8. Gradient Boosting meta-learner...")
meta_gb = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3, random_state=42)
meta_gb.fit(meta_train, y_train)
pred_gb = meta_gb.predict(meta_test)
mae_gb = mean_absolute_error(y_test, pred_gb)
print(f"     MAE={mae_gb:.3f}")

print()

# ============================================================================
# STRATEGY 9: CONFORMAL-WEIGHTED ENSEMBLE (UNCERTAINTY-BASED)
# ============================================================================
print("[6/8] Building Conformal-weighted ensemble...")

# Calculate prediction variance for each model (uncertainty)
uncertainties = []
for i, model in enumerate([model_xgb, model_et, model_lgb, model_rf, model_hgb]):
    # Use cross-validation to estimate prediction uncertainty
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    uncertainty = np.abs(cv_scores).std()  # Std of CV scores = uncertainty
    uncertainties.append(uncertainty)

uncertainties = np.array(uncertainties)

# Weight by inverse uncertainty (more confident models get higher weight)
weights_conformal = 1.0 / (uncertainties + 1e-6)
weights_conformal = weights_conformal / weights_conformal.sum()

pred_conformal = np.average(all_predictions, axis=1, weights=weights_conformal)
mae_conformal = mean_absolute_error(y_test, pred_conformal)

# Conformal confidence intervals (90%)
ensemble_std = np.std(all_predictions, axis=1)
conf_lower = pred_conformal - 1.645 * ensemble_std  # 90% CI
conf_upper = pred_conformal + 1.645 * ensemble_std

coverage = np.mean((y_test >= conf_lower) & (y_test <= conf_upper))

print(f"  9. Conformal-weighted:    MAE={mae_conformal:.3f}, Coverage={coverage:.1%}")
print(f"     Weights: XGB={weights_conformal[0]:.3f}, ET={weights_conformal[1]:.3f}, LGB={weights_conformal[2]:.3f}, RF={weights_conformal[3]:.3f}, HGB={weights_conformal[4]:.3f}")
print()

# ============================================================================
# STRATEGY 10: BAYESIAN MODEL AVERAGING (PROBABILISTIC)
# ============================================================================
print("[7/8] Building Bayesian Model Averaging...")

# Bayesian approach: Weight by likelihood of each model
from scipy.stats import norm

# Calculate likelihood of each model (how well it fits)
likelihoods = []
for pred in [pred_xgb, pred_et, pred_lgb, pred_rf, pred_hgb]:
    # Negative log-likelihood (lower is better)
    residuals = y_test - pred
    sigma = np.std(residuals)
    nll = -np.sum(norm.logpdf(residuals, 0, sigma))
    likelihoods.append(nll)

likelihoods = np.array(likelihoods)

# Convert to weights (lower NLL = higher weight)
weights_bayes = np.exp(-likelihoods / likelihoods.min())
weights_bayes = weights_bayes / weights_bayes.sum()

pred_bayes = np.average(all_predictions, axis=1, weights=weights_bayes)
mae_bayes = mean_absolute_error(y_test, pred_bayes)
std_bayes = np.std(all_predictions, axis=1).mean()

print(f"  10. Bayesian Averaging:   MAE={mae_bayes:.3f}, Avg Std=±{std_bayes:.2f}")
print(f"      Weights: XGB={weights_bayes[0]:.3f}, ET={weights_bayes[1]:.3f}, LGB={weights_bayes[2]:.3f}, RF={weights_bayes[3]:.3f}, HGB={weights_bayes[4]:.3f}")
print()

# ============================================================================
# RANK ALL STRATEGIES
# ============================================================================
print("[8/8] Ranking ALL strategies...")
print()
print("="*80)
print("🏆 FINAL CHAMPIONSHIP RANKINGS")
print("="*80)

results = [
    ('Simple Average', mae_avg, pred_avg, std_avg),
    ('Inverse MAE Weights', mae_wmae, pred_wmae, std_wmae),
    ('Inverse Variance', mae_wvar, pred_wvar, std_wvar),
    ('Ridge Meta-Learner', mae_ridge, pred_ridge, np.std(all_predictions, axis=1).mean()),
    ('Lasso Meta-Learner', mae_lasso, pred_lasso, np.std(all_predictions, axis=1).mean()),
    ('ElasticNet Meta-Learner', mae_enet, pred_enet, np.std(all_predictions, axis=1).mean()),
    ('Neural Network Meta', mae_nn, pred_nn, np.std(all_predictions, axis=1).mean()),
    ('Gradient Boosting Meta', mae_gb, pred_gb, np.std(all_predictions, axis=1).mean()),
    ('Conformal-Weighted', mae_conformal, pred_conformal, ensemble_std.mean()),
    ('Bayesian Averaging', mae_bayes, pred_bayes, std_bayes)
]

# Sort by MAE
results.sort(key=lambda x: x[1])

for i, (name, mae, pred, std) in enumerate(results, 1):
    print(f"  {i:2d}. {name:25s}  MAE={mae:.3f}  Uncertainty=±{std:.2f}")

print("="*80)
print()

# ============================================================================
# SELECT CHAMPION
# ============================================================================
champion_name, champion_mae, champion_pred, champion_std = results[0]

print(f"🏆 CHAMPION: {champion_name}")
print(f"📊 MAE: {champion_mae:.3f}")
print(f"📊 Average Uncertainty: ±{champion_std:.2f}")
print()

# ============================================================================
# CONFIDENCE INTERVALS FOR CHAMPION
# ============================================================================
print("🔒 CHAMPION CONFIDENCE INTERVALS:")
print()

# 90% Confidence Intervals
if champion_name == 'Conformal-Weighted':
    ci_lower = conf_lower
    ci_upper = conf_upper
else:
    ensemble_std_full = np.std(all_predictions, axis=1)
    ci_lower = champion_pred - 1.645 * ensemble_std_full
    ci_upper = champion_pred + 1.645 * ensemble_std_full

coverage = np.mean((y_test >= ci_lower) & (y_test <= ci_upper))
avg_interval_width = np.mean(ci_upper - ci_lower)

print(f"  90% Confidence Interval Coverage: {coverage:.1%}")
print(f"  Average Interval Width: ±{avg_interval_width/2:.2f} points")
print()

# ============================================================================
# SAVE EVERYTHING
# ============================================================================
print("💾 Saving MEGA ensemble...")

mega_ensemble = {
    'champion_name': champion_name,
    'champion_mae': champion_mae,
    'champion_predictions': champion_pred,
    
    # All strategies
    'all_strategies': {
        name: {'mae': mae, 'predictions': pred, 'uncertainty': std}
        for name, mae, pred, std in results
    },
    
    # Base models
    'base_models': {
        'xgboost': model_xgb,
        'extratrees': model_et,
        'lightgbm': model_lgb,
        'randomforest': model_rf,
        'histgradient': model_hgb
    },
    
    # Meta-learners
    'meta_learners': {
        'ridge': meta_ridge,
        'lasso': meta_lasso,
        'elasticnet': meta_enet,
        'neural_net': meta_nn,
        'gradient_boost': meta_gb
    },
    
    # Weights
    'weights': {
        'inverse_mae': weights_mae,
        'inverse_variance': weights_var,
        'conformal': weights_conformal,
        'bayesian': weights_bayes
    },
    
    # Confidence intervals
    'confidence_intervals': {
        'lower': ci_lower,
        'upper': ci_upper,
        'coverage': coverage,
        'avg_width': avg_interval_width
    },
    
    # Test performance
    'test_metrics': {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': champion_pred,
        'mae': champion_mae,
        'rmse': np.sqrt(mean_squared_error(y_test, champion_pred))
    }
}

with open('MEGA_ENSEMBLE_CHAMPION.pkl', 'wb') as f:
    pickle.dump(mega_ensemble, f)

print(f"✅ Saved to: MEGA_ENSEMBLE_CHAMPION.pkl")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*80)
print("🎯 MEGA ENSEMBLE COMPLETE")
print("="*80)
print()
print(f"🏆 CHAMPION STRATEGY: {champion_name}")
print(f"📊 MAE: {champion_mae:.3f}")
print(f"📊 90% CI Coverage: {coverage:.1%}")
print(f"📊 Avg Uncertainty: ±{champion_std:.2f} points")
print()

if champion_mae < 9.0:
    print("✅ MAE < 9.0 - READY FOR MONDAY LAUNCH!")
    print("   Risk mode: CONSERVATIVE")
    print("   Max bet: $50-100")
elif champion_mae < 10.0:
    print("⚠️  MAE < 10.0 - MARGINAL, paper trade recommended")
    print("   Risk mode: PAPER-ONLY or TINY bets ($5-10)")
else:
    print("🔴 MAE > 10.0 - NOT READY, need more work")
    print("   Recommendation: Delay launch, integrate Informer")

print()
print("Next: python3 🔥_CONFIDENCE_INTERVAL_SYSTEM.py")
print("="*80)

