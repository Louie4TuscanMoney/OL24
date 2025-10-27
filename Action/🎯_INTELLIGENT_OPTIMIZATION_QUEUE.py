#!/usr/bin/env python3
"""
🎯 INTELLIGENT OPTIMIZATION QUEUE
1. Run quick 100 trials NOW (get results in 45 min)
2. Auto-start Stanford 5000 trials (run overnight)
3. Optimize for LOW DELTA (variance) + HIGH +EV (expected value)
4. Forward-feed results into system continuously
"""

import pickle
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import time
import os

print("="*80)
print("🎯 INTELLIGENT OPTIMIZATION QUEUE")
print("="*80)
print()
print("Strategy:")
print("  Phase 1: Quick 100 trials (NOW - 45 min)")
print("  Phase 2: Stanford 5000 trials (AUTO-START after Phase 1)")
print()
print("Optimizing for: LOW DELTA + HIGH +EV (betting-focused)")
print("="*80)
print()

# Load data
with open('ULTRA_ENHANCED_PATTERNS_V2.pkl', 'rb') as f:
    patterns = pickle.load(f)

# Build matrices
X = []
y = []

for p in patterns:
    if p.get('diff_at_final') is None:
        continue
    
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

X = np.array(X)
y = np.array(y)

tscv = TimeSeriesSplit(n_splits=5)

# CUSTOM OBJECTIVE: Optimize for +EV betting (not just MAE)
def betting_focused_objective(model, X, y, cv):
    """
    Optimize for LOW DELTA (variance) + HIGH ACCURACY
    This is what matters for +EV betting
    """
    scores_mae = []
    scores_mse = []
    
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        
        mae = mean_absolute_error(y_val, pred)
        mse = mean_squared_error(y_val, pred)
        
        scores_mae.append(mae)
        scores_mse.append(mse)
    
    avg_mae = np.mean(scores_mae)
    avg_mse = np.mean(scores_mse)
    std_mae = np.std(scores_mae)  # Variance in MAE (want LOW)
    
    # BETTING SCORE: Penalize high variance (unstable predictions = risky)
    # Formula: MAE + 0.5 * variance_penalty
    # We want CONSISTENT predictions (low delta) for confident betting
    variance_penalty = std_mae / avg_mae  # Coefficient of variation
    
    betting_score = avg_mae * (1 + 0.3 * variance_penalty)
    
    return betting_score, avg_mae, std_mae

# ============================================================================
# PHASE 1: QUICK 100 TRIALS (NOW)
# ============================================================================

print("="*80)
print("PHASE 1: QUICK 100-TRIAL OPTIMIZATION (45 min)")
print("="*80)
print()

def quick_xgboost_objective(trial):
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
    betting_score, avg_mae, std_mae = betting_focused_objective(model, X, y, tscv)
    
    # Store additional metrics
    trial.set_user_attr('avg_mae', avg_mae)
    trial.set_user_attr('std_mae', std_mae)
    trial.set_user_attr('variance_penalty', std_mae / avg_mae)
    
    if trial.number % 10 == 0:
        print(f"  Trial {trial.number:3d}: MAE={avg_mae:.3f}, Delta={std_mae:.3f}, Score={betting_score:.3f}")
    
    return betting_score  # Minimize betting score (MAE + variance penalty)

# Run Phase 1
print("Starting XGBoost (100 trials)...")
xgb_quick = optuna.create_study(direction='minimize', study_name='xgb_quick_100')
xgb_quick.optimize(quick_xgboost_objective, n_trials=100, show_progress_bar=True)

best_mae_quick = xgb_quick.best_trial.user_attrs['avg_mae']
best_delta_quick = xgb_quick.best_trial.user_attrs['std_mae']

print()
print(f"✅ Phase 1 Complete!")
print(f"   Best MAE: {best_mae_quick:.3f}")
print(f"   Best Delta: {best_delta_quick:.3f}")
print(f"   Best Params: {xgb_quick.best_params}")
print()

# Save quick results
quick_results = {
    'xgboost': {
        'best_params': xgb_quick.best_params,
        'best_mae': best_mae_quick,
        'best_delta': best_delta_quick,
        'n_trials': 100,
        'timestamp': datetime.now().isoformat()
    }
}

with open('QUICK_100_RESULTS.pkl', 'wb') as f:
    pickle.dump(quick_results, f)

print("✅ Quick results saved to: QUICK_100_RESULTS.pkl")
print()

# ============================================================================
# PHASE 2: STANFORD 5000 TRIALS (AUTO-START)
# ============================================================================

print("="*80)
print("PHASE 2: STANFORD 5000-TRIAL OPTIMIZATION (AUTO-STARTING)")
print("="*80)
print()
print("This will run for ~30-40 hours")
print("Optimizing for: Betting edge (low delta + high accuracy)")
print()
print("Press Ctrl+C in next 5 seconds to skip Stanford optimization...")
time.sleep(5)
print()
print("Starting Stanford optimization...")
print()

# Callback for checkpointing
class ForwardFeedingCallback:
    """
    Forward-feeding callback:
    - Saves checkpoints every 100 trials
    - Updates system with best params so far
    - Allows early stopping if target reached
    """
    def __init__(self, target_mae=5.0):
        self.target_mae = target_mae
        self.start_time = datetime.now()
        self.best_so_far = float('inf')
    
    def __call__(self, study, trial):
        current_best = study.best_value
        
        # Save checkpoint every 100 trials
        if trial.number % 100 == 0 and trial.number > 0:
            elapsed_min = (datetime.now() - self.start_time).total_seconds() / 60
            remaining_trials = 5000 - trial.number
            est_remaining_min = (elapsed_min / trial.number) * remaining_trials
            
            checkpoint = {
                'study_name': study.study_name,
                'best_value': current_best,
                'best_params': study.best_params,
                'best_mae': study.best_trial.user_attrs.get('avg_mae', current_best),
                'best_delta': study.best_trial.user_attrs.get('std_mae', 0),
                'n_trials': trial.number,
                'elapsed_min': elapsed_min,
                'est_remaining_min': est_remaining_min,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save checkpoint
            checkpoint_file = f"{study.study_name}_checkpoint.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            # Forward-feed: Save current best params for immediate use
            forward_feed_file = f"{study.study_name}_CURRENT_BEST.pkl"
            with open(forward_feed_file, 'wb') as f:
                pickle.dump({
                    'params': study.best_params,
                    'mae': checkpoint['best_mae'],
                    'delta': checkpoint['best_delta']
                }, f)
            
            print(f"\n💾 Checkpoint {trial.number}/5000")
            print(f"   Best MAE: {checkpoint['best_mae']:.4f} (Delta: {checkpoint['best_delta']:.4f})")
            print(f"   Progress: {trial.number/50:.1f}%")
            print(f"   Elapsed: {elapsed_min:.0f} min | Remaining: {est_remaining_min:.0f} min")
            
            # Early stopping if target reached
            if checkpoint['best_mae'] < self.target_mae:
                print(f"\n🏆 TARGET REACHED! MAE < {self.target_mae}")
                print(f"   Stopping early at trial {trial.number}")
                study.stop()
            
            print()

def stanford_xgboost_objective(trial):
    """XGBoost with EXPANDED search space for 5000 trials"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    betting_score, avg_mae, std_mae = betting_focused_objective(model, X, y, tscv)
    
    # Store for analysis
    trial.set_user_attr('avg_mae', avg_mae)
    trial.set_user_attr('std_mae', std_mae)
    trial.set_user_attr('cv_variance', std_mae / avg_mae)
    
    # Print every 50 trials
    if trial.number % 50 == 0:
        print(f"  Trial {trial.number:4d}: MAE={avg_mae:.4f}, Delta={std_mae:.4f}")
    
    return betting_score

def stanford_lightgbm_objective(trial):
    """LightGBM optimized for betting"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    betting_score, avg_mae, std_mae = betting_focused_objective(model, X, y, tscv)
    
    trial.set_user_attr('avg_mae', avg_mae)
    trial.set_user_attr('std_mae', std_mae)
    trial.set_user_attr('cv_variance', std_mae / avg_mae)
    
    if trial.number % 50 == 0:
        print(f"  Trial {trial.number:4d}: MAE={avg_mae:.4f}, Delta={std_mae:.4f}")
    
    return betting_score

def stanford_extratrees_objective(trial):
    """ExtraTrees (best in research)"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = ExtraTreesRegressor(**params)
    betting_score, avg_mae, std_mae = betting_focused_objective(model, X, y, tscv)
    
    trial.set_user_attr('avg_mae', avg_mae)
    trial.set_user_attr('std_mae', std_mae)
    trial.set_user_attr('cv_variance', std_mae / avg_mae)
    
    if trial.number % 50 == 0:
        print(f"  Trial {trial.number:4d}: MAE={avg_mae:.4f}, Delta={std_mae:.4f}")
    
    return betting_score

# ============================================================================
# EXECUTE: STANFORD 5000 TRIALS
# ============================================================================

print("Starting Stanford 5000-trial optimization...")
print("Target: MAE < 5.0 (championship level)")
print()

# Create persistent storage (SQLite) - can resume if interrupted
storage_xgb = optuna.storages.RDBStorage(
    url="sqlite:///stanford_xgboost.db",
    engine_kwargs={"connect_args": {"timeout": 30}}
)

storage_lgb = optuna.storages.RDBStorage(
    url="sqlite:///stanford_lightgbm.db",
    engine_kwargs={"connect_args": {"timeout": 30}}
)

storage_et = optuna.storages.RDBStorage(
    url="sqlite:///stanford_extratrees.db",
    engine_kwargs={"connect_args": {"timeout": 30}}
)

# MODEL 1: XGBoost (5000 trials)
print("[1/3] XGBoost - 5000 trials (~12-15 hours)...")
xgb_stanford = optuna.create_study(
    study_name='stanford_xgboost_5000',
    direction='minimize',
    storage=storage_xgb,
    load_if_exists=True
)

callback_xgb = ForwardFeedingCallback(target_mae=5.0)
xgb_stanford.optimize(
    stanford_xgboost_objective,
    n_trials=5000,
    callbacks=[callback_xgb],
    show_progress_bar=True
)

print()
print(f"✅ XGBoost complete!")
print(f"   Best MAE: {xgb_stanford.best_trial.user_attrs['avg_mae']:.4f}")
print(f"   Best Delta: {xgb_stanford.best_trial.user_attrs['std_mae']:.4f}")
print()

# MODEL 2: LightGBM (5000 trials)
print("[2/3] LightGBM - 5000 trials (~12-15 hours)...")
lgb_stanford = optuna.create_study(
    study_name='stanford_lightgbm_5000',
    direction='minimize',
    storage=storage_lgb,
    load_if_exists=True
)

callback_lgb = ForwardFeedingCallback(target_mae=5.0)
lgb_stanford.optimize(
    stanford_lightgbm_objective,
    n_trials=5000,
    callbacks=[callback_lgb],
    show_progress_bar=True
)

print()
print(f"✅ LightGBM complete!")
print(f"   Best MAE: {lgb_stanford.best_trial.user_attrs['avg_mae']:.4f}")
print(f"   Best Delta: {lgb_stanford.best_trial.user_attrs['std_mae']:.4f}")
print()

# MODEL 3: ExtraTrees (5000 trials)
print("[3/3] ExtraTrees - 5000 trials (~10-12 hours)...")
et_stanford = optuna.create_study(
    study_name='stanford_extratrees_5000',
    direction='minimize',
    storage=storage_et,
    load_if_exists=True
)

callback_et = ForwardFeedingCallback(target_mae=5.0)
et_stanford.optimize(
    stanford_extratrees_objective,
    n_trials=5000,
    callbacks=[callback_et],
    show_progress_bar=True
)

print()
print(f"✅ ExtraTrees complete!")
print(f"   Best MAE: {et_stanford.best_trial.user_attrs['avg_mae']:.4f}")
print(f"   Best Delta: {et_stanford.best_trial.user_attrs['std_mae']:.4f}")
print()

# ============================================================================
# FINAL RESULTS
# ============================================================================

print("="*80)
print("🏆 STANFORD 5000-TRIAL OPTIMIZATION COMPLETE")
print("="*80)
print()

results = {
    'xgboost': {
        'mae': xgb_stanford.best_trial.user_attrs['avg_mae'],
        'delta': xgb_stanford.best_trial.user_attrs['std_mae'],
        'params': xgb_stanford.best_params,
        'trials': len(xgb_stanford.trials)
    },
    'lightgbm': {
        'mae': lgb_stanford.best_trial.user_attrs['avg_mae'],
        'delta': lgb_stanford.best_trial.user_attrs['std_mae'],
        'params': lgb_stanford.best_params,
        'trials': len(lgb_stanford.trials)
    },
    'extratrees': {
        'mae': et_stanford.best_trial.user_attrs['avg_mae'],
        'delta': et_stanford.best_trial.user_attrs['std_mae'],
        'params': et_stanford.best_params,
        'trials': len(et_stanford.trials)
    }
}

# Find best model (lowest MAE)
best_model = min(results.items(), key=lambda x: x[1]['mae'])

print("RESULTS (sorted by MAE):")
for model, data in sorted(results.items(), key=lambda x: x[1]['mae']):
    print(f"  {model:15s} MAE: {data['mae']:.4f} | Delta: {data['delta']:.4f} | Trials: {data['trials']}")

print()
print(f"🏆 CHAMPION: {best_model[0]}")
print(f"📊 MAE: {best_model[1]['mae']:.4f}")
print(f"📊 Delta (variance): {best_model[1]['delta']:.4f}")
print()

# Save final results
final_results = {
    'all_models': results,
    'best_model': best_model[0],
    'best_mae': best_model[1]['mae'],
    'best_delta': best_model[1]['delta'],
    'best_params': best_model[1]['params'],
    'total_trials': sum(r['trials'] for r in results.values()),
    'optimization_objective': 'Betting-focused (MAE + variance penalty)',
    'timestamp': datetime.now().isoformat()
}

with open('STANFORD_FINAL_RESULTS.pkl', 'wb') as f:
    pickle.dump(final_results, f)

print("✅ Final results saved to: STANFORD_FINAL_RESULTS.pkl")
print()

if best_model[1]['mae'] < 5.0:
    print("🏆🏆🏆 CHAMPIONSHIP LEVEL! MAE < 5.0")
    print("   READY TO DOMINATE")
elif best_model[1]['mae'] < 6.0:
    print("🏆 EXCELLENT! MAE < 6.0")
    print("   Production-ready")
elif best_model[1]['mae'] < 7.0:
    print("✅ GOOD! MAE < 7.0")
    print("   Ready for cautious launch")
else:
    print("⚠️  MAE still > 7.0")
    print("   Need more work or different approach")

print()
print("="*80)
print("Next: Train final ensemble with championship parameters")
print("Command: python3 🔥_3_TRAIN_OPTIMIZED_ENSEMBLE.py")
print("="*80)

