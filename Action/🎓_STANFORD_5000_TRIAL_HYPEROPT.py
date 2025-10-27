#!/usr/bin/env python3
"""
🎓 STANFORD-LEVEL HYPERPARAMETER OPTIMIZATION
5000 trials with checkpointing, parallelization, multi-model
Research-grade optimization (what top ML labs do)
"""

import pickle
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎓 STANFORD-LEVEL HYPERPARAMETER OPTIMIZATION")
print("5000 Trials Per Model - Research-Grade Search")
print("="*80)
print()
print("⚠️  WARNING: This will take 30-50 HOURS to complete")
print("   Designed to run overnight/over weekend")
print("   Will save checkpoints every 100 trials")
print("   Can resume if interrupted")
print()
print("Press Ctrl+C in next 10 seconds to cancel...")
import time
time.sleep(10)
print()

# Load data
print("[1/6] Loading enhanced patterns...")
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

print(f"✅ Feature matrix: {X.shape}")
print()

# TimeSeriesSplit CV
tscv = TimeSeriesSplit(n_splits=5)

# CALLBACK TO SAVE CHECKPOINTS
class CheckpointCallback:
    def __init__(self, checkpoint_file, save_frequency=100):
        self.checkpoint_file = checkpoint_file
        self.save_frequency = save_frequency
        self.start_time = datetime.now()
    
    def __call__(self, study, trial):
        # Save every N trials
        if trial.number % self.save_frequency == 0 and trial.number > 0:
            checkpoint = {
                'study_name': study.study_name,
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': trial.number,
                'elapsed_time': (datetime.now() - self.start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            elapsed = (datetime.now() - self.start_time).total_seconds() / 60
            print(f"\n💾 Checkpoint saved at trial {trial.number}")
            print(f"   Best MAE so far: {study.best_value:.4f}")
            print(f"   Elapsed: {elapsed:.1f} min")
            print(f"   Est. remaining: {(elapsed/trial.number)*(5000-trial.number):.1f} min\n")

# OPTIMIZE XGBOOST (5000 trials)
print("[2/6] Optimizing XGBoost (5000 trials, ~20-25 hours)...")
print("Starting...")
print()

def xgboost_objective(trial):
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
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)
    mae = -scores.mean()
    
    # Print every 50 trials
    if trial.number % 50 == 0:
        print(f"  Trial {trial.number:4d}: MAE = {mae:.4f}")
    
    return mae

# Create study with SQLite database (persistent across runs)
storage = optuna.storages.RDBStorage(
    url="sqlite:///xgboost_5000_trials.db",
    engine_kwargs={"connect_args": {"timeout": 10}}
)

xgb_study = optuna.create_study(
    study_name='xgboost_5000_stanford',
    direction='minimize',
    storage=storage,
    load_if_exists=True  # Resume if interrupted
)

# Run with checkpoint callback
callback = CheckpointCallback('xgboost_checkpoint.pkl', save_frequency=100)
xgb_study.optimize(
    xgboost_objective,
    n_trials=5000,
    callbacks=[callback],
    show_progress_bar=True,
    n_jobs=1  # Sequential (more stable)
)

print()
print(f"✅ XGBoost optimization complete!")
print(f"   Best MAE: {xgb_study.best_value:.4f}")
print(f"   Best params: {xgb_study.best_params}")
print()

# OPTIMIZE LIGHTGBM (5000 trials)
print("[3/6] Optimizing LightGBM (5000 trials, ~20-25 hours)...")

def lightgbm_objective(trial):
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
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)
    mae = -scores.mean()
    
    if trial.number % 50 == 0:
        print(f"  Trial {trial.number:4d}: MAE = {mae:.4f}")
    
    return mae

lgb_storage = optuna.storages.RDBStorage(
    url="sqlite:///lightgbm_5000_trials.db",
    engine_kwargs={"connect_args": {"timeout": 10}}
)

lgb_study = optuna.create_study(
    study_name='lightgbm_5000_stanford',
    direction='minimize',
    storage=lgb_storage,
    load_if_exists=True
)

lgb_callback = CheckpointCallback('lightgbm_checkpoint.pkl', save_frequency=100)
lgb_study.optimize(
    lightgbm_objective,
    n_trials=5000,
    callbacks=[lgb_callback],
    show_progress_bar=True
)

print()
print(f"✅ LightGBM optimization complete!")
print(f"   Best MAE: {lgb_study.best_value:.4f}")
print()

# OPTIMIZE EXTRATREES (5000 trials)
print("[4/6] Optimizing ExtraTrees (5000 trials, ~15-20 hours)...")

def extratrees_objective(trial):
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
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)
    mae = -scores.mean()
    
    if trial.number % 50 == 0:
        print(f"  Trial {trial.number:4d}: MAE = {mae:.4f}")
    
    return mae

et_storage = optuna.storages.RDBStorage(
    url="sqlite:///extratrees_5000_trials.db",
    engine_kwargs={"connect_args": {"timeout": 10}}
)

et_study = optuna.create_study(
    study_name='extratrees_5000_stanford',
    direction='minimize',
    storage=et_storage,
    load_if_exists=True
)

et_callback = CheckpointCallback('extratrees_checkpoint.pkl', save_frequency=100)
et_study.optimize(
    extratrees_objective,
    n_trials=5000,
    callbacks=[et_callback],
    show_progress_bar=True
)

print()
print(f"✅ ExtraTrees optimization complete!")
print(f"   Best MAE: {et_study.best_value:.4f}")
print()

# OPTIMIZE RANDOM FOREST (5000 trials)
print("[5/6] Optimizing RandomForest (5000 trials, ~15-20 hours)...")

def randomforest_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)
    mae = -scores.mean()
    
    if trial.number % 50 == 0:
        print(f"  Trial {trial.number:4d}: MAE = {mae:.4f}")
    
    return mae

rf_storage = optuna.storages.RDBStorage(
    url="sqlite:///randomforest_5000_trials.db",
    engine_kwargs={"connect_args": {"timeout": 10}}
)

rf_study = optuna.create_study(
    study_name='randomforest_5000_stanford',
    direction='minimize',
    storage=rf_storage,
    load_if_exists=True
)

rf_callback = CheckpointCallback('randomforest_checkpoint.pkl', save_frequency=100)
rf_study.optimize(
    randomforest_objective,
    n_trials=5000,
    callbacks=[rf_callback],
    show_progress_bar=True
)

print()
print(f"✅ RandomForest optimization complete!")
print(f"   Best MAE: {rf_study.best_value:.4f}")
print()

# SAVE ALL RESULTS
print("[6/6] Saving Stanford-level optimized parameters...")

stanford_results = {
    'xgboost': {
        'best_params': xgb_study.best_params,
        'best_mae': xgb_study.best_value,
        'n_trials': len(xgb_study.trials)
    },
    'lightgbm': {
        'best_params': lgb_study.best_params,
        'best_mae': lgb_study.best_value,
        'n_trials': len(lgb_study.trials)
    },
    'extratrees': {
        'best_params': et_study.best_params,
        'best_mae': et_study.best_value,
        'n_trials': len(et_study.trials)
    },
    'randomforest': {
        'best_params': rf_study.best_params,
        'best_mae': rf_study.best_value,
        'n_trials': len(rf_study.trials)
    },
    'optimization_method': 'Bayesian (Optuna TPE Sampler)',
    'total_trials': 20000,
    'cv_strategy': 'TimeSeriesSplit (5 folds)',
    'timestamp': datetime.now().isoformat()
}

with open('STANFORD_5000_TRIAL_RESULTS.pkl', 'wb') as f:
    pickle.dump(stanford_results, f)

print(f"✅ Results saved to: STANFORD_5000_TRIAL_RESULTS.pkl")
print()

# SUMMARY
print("="*80)
print("🎓 STANFORD-LEVEL OPTIMIZATION COMPLETE")
print("="*80)
print()
print("RESULTS:")
print(f"  XGBoost:      {stanford_results['xgboost']['best_mae']:.4f} MAE ({stanford_results['xgboost']['n_trials']} trials)")
print(f"  LightGBM:     {stanford_results['lightgbm']['best_mae']:.4f} MAE ({stanford_results['lightgbm']['n_trials']} trials)")
print(f"  ExtraTrees:   {stanford_results['extratrees']['best_mae']:.4f} MAE ({stanford_results['extratrees']['n_trials']} trials)")
print(f"  RandomForest: {stanford_results['randomforest']['best_mae']:.4f} MAE ({stanford_results['randomforest']['n_trials']} trials)")
print()

best_model = min(stanford_results.items(), key=lambda x: x[1]['best_mae'] if isinstance(x[1], dict) and 'best_mae' in x[1] else float('inf'))
print(f"🏆 BEST MODEL: {best_model[0]}")
print(f"📊 BEST MAE: {best_model[1]['best_mae']:.4f}")
print()

if best_model[1]['best_mae'] < 5.0:
    print("🏆🏆🏆 CHAMPIONSHIP LEVEL - Under 5.0 MAE!")
    print("   This is publishable research quality")
elif best_model[1]['best_mae'] < 6.0:
    print("🏆 EXCELLENT - Under 6.0 MAE")
    print("   Production-ready for aggressive betting")
elif best_model[1]['best_mae'] < 7.0:
    print("✅ GOOD - Under 7.0 MAE")
    print("   Ready for conservative betting")

print()
print("Next: Train final ensemble with these optimal parameters")
print("Command: python3 🔥_3_TRAIN_OPTIMIZED_ENSEMBLE.py")
print()
print("="*80)

