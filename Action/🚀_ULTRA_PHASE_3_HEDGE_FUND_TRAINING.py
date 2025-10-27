"""
🚀 ULTRA PHASE 3 - HEDGE FUND TRAINING ON 72 ELITE FEATURES
Elite signal processing with FFT, Wavelets, Spectral Analysis

INPUT: ULTRA_15K_SIGNAL_PROCESSING_V1.pkl (72 features!)
OUTPUT: ULTRA_HEDGE_FUND_SYSTEM_V1.pkl
TIME: 45-60 minutes
MODE: HEDGE FUND ELITE - Maximum performance extraction
"""

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, LassoCV, ElasticNetCV
from sklearn.cluster import KMeans
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("🚀 ULTRA PHASE 3 - HEDGE FUND TRAINING (72 ELITE FEATURES)")
print("="*90)
print(f"\nStart time: {datetime.now().strftime('%I:%M %p')}")
print("Mode: HEDGE FUND ELITE - FFT + Wavelets + Spectral")
print("\n" + "="*90)

print("\n[STEP 1] LOADING ULTRA DATASET")
print("="*90)

try:
    with open('Action/ULTRA_15K_SIGNAL_PROCESSING_V1.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    X_all = dataset['features']
    y_final = dataset['targets']['final_diff']
    y_current = dataset['targets']['current_diff']
    metadata = dataset['metadata']
    feature_names = dataset['feature_names']
    
    print(f"✓ Loaded ULTRA dataset:")
    print(f"  Games: {len(X_all)}")
    print(f"  Features: {X_all.shape[1]} ELITE!")
    print(f"  Version: {dataset['version']}")
    
except:
    print("✗ ULTRA dataset not found - run Phase 2 first!")
    exit(1)

print("\n[STEP 2] TEMPORAL TRAIN/VAL/TEST SPLIT")
print("="*90)

n_total = len(X_all)
split1 = int(n_total * 0.7)
split2 = int(n_total * 0.85)

X_train = X_all[:split1]
X_val = X_all[split1:split2]
X_test = X_all[split2:]

y_train = y_final[:split1]
y_val = y_final[split1:split2]
y_test = y_final[split2:]

print(f"✓ Temporal splits:")
print(f"  Train: {len(X_train):5d} games")
print(f"  Val:   {len(X_val):5d} games")
print(f"  Test:  {len(X_test):5d} games")

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled (NO LEAKAGE!)")

print("\n[STEP 3] LASSO FEATURE SELECTION (72 → Elite subset)")
print("="*90)

print("\nRunning LassoCV on 72 features...")
lasso = LassoCV(cv=5, alphas=np.logspace(-3, 1, 30), max_iter=10000, random_state=42, n_jobs=-1)
lasso.fit(X_train_scaled, y_train)

feature_importance = np.abs(lasso.coef_)
n_selected = np.sum(feature_importance > 0.01)

print(f"✓ LASSO selected {n_selected}/72 features (alpha={lasso.alpha_:.4f})")

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance,
    'selected': feature_importance > 0.01
}).sort_values('importance', ascending=False)

importance_df.to_csv('Action/ultra_feature_importances.csv', index=False)

print("\nTop 15 features:")
for i, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:<25} {row['importance']:>8.4f} {'✓' if row['selected'] else ''}")

print("\n[STEP 4] TRAINING ELITE MODEL POOL")
print("="*90)

training_log = []

models_to_train = [
    ('LASSO', lambda: lasso),
    ('Ridge', lambda: Ridge(alpha=2.0)),
    ('ElasticNet', lambda: ElasticNetCV(cv=3, l1_ratio=[0.1, 0.5, 0.9], random_state=42, n_jobs=-1)),
    ('LightGBM', lambda: LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       reg_alpha=3.0, reg_lambda=3.0, random_state=42, verbose=-1)),
    ('XGBoost', lambda: XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                     reg_alpha=3.0, reg_lambda=3.0, random_state=42)),
    ('RandomForest', lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
    ('GradientBoosting', lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42))
]

trained_models = {}

print("\nTraining 7 elite models on 72 features...")

for model_name, model_fn in models_to_train:
    print(f"\n  Training {model_name}...")
    start = datetime.now()
    
    if model_name == 'LASSO':
        model = model_fn()
    else:
        model = model_fn()
        model.fit(X_train_scaled, y_train)
    
    pred_train = model.predict(X_train_scaled)
    pred_val = model.predict(X_val_scaled)
    pred_test = model.predict(X_test_scaled)
    
    mae_train = mean_absolute_error(y_train, pred_train)
    mae_val = mean_absolute_error(y_val, pred_val)
    mae_test = mean_absolute_error(y_test, pred_test)
    
    overfit_pct = ((mae_val - mae_train) / mae_train) * 100
    elapsed = (datetime.now() - start).total_seconds()
    
    trained_models[model_name] = {
        'model': model,
        'mae_train': mae_train,
        'mae_val': mae_val,
        'mae_test': mae_test,
        'overfit_pct': overfit_pct,
        'predictions_val': pred_val,
        'predictions_test': pred_test
    }
    
    training_log.append({
        'model': model_name,
        'mae_train': mae_train,
        'mae_val': mae_val,
        'mae_test': mae_test,
        'overfit_pct': overfit_pct,
        'training_time_sec': elapsed
    })
    
    print(f"    ✓ Train: {mae_train:.3f}, Val: {mae_val:.3f}, Test: {mae_test:.3f}, Overfit: {overfit_pct:.1f}%")

pd.DataFrame(training_log).to_csv('Action/ultra_training_log.csv', index=False)
print(f"\n✓ Saved: ultra_training_log.csv")

print("\n[STEP 5] ROLLING WALK-FORWARD VALIDATION (15 Folds)")
print("="*90)

print("\nPerforming 15-fold rolling validation on ULTRA features...")

n_folds = 15
fold_size = len(X_all) // (n_folds + 1)
rolling_results = []

for fold in range(n_folds):
    train_end = (fold + 1) * fold_size
    test_start = train_end
    test_end = min(train_end + fold_size, len(X_all))
    
    if test_end - test_start < 50:
        break
    
    X_fold_train = X_all[:train_end]
    X_fold_test = X_all[test_start:test_end]
    y_fold_train = y_final[:train_end]
    y_fold_test = y_final[test_start:test_end]
    
    scaler_fold = RobustScaler()
    X_fold_train_sc = scaler_fold.fit_transform(X_fold_train)
    X_fold_test_sc = scaler_fold.transform(X_fold_test)
    
    # Train Ridge (fast, stable)
    model_fold = Ridge(alpha=2.0)
    model_fold.fit(X_fold_train_sc, y_fold_train)
    
    pred_fold = model_fold.predict(X_fold_test_sc)
    mae_fold = mean_absolute_error(y_fold_test, pred_fold)
    
    rolling_results.append({
        'fold': fold + 1,
        'train_size': len(X_fold_train),
        'test_size': len(X_fold_test),
        'mae': mae_fold
    })
    
    print(f"  Fold {fold+1:2d}: Train {len(X_fold_train):5d}, Test {len(X_fold_test):4d} → MAE: {mae_fold:.3f}")

pd.DataFrame(rolling_results).to_csv('Action/ultra_rolling_validation.csv', index=False)

rolling_maes = [r['mae'] for r in rolling_results]
rolling_mean = np.mean(rolling_maes)
rolling_std = np.std(rolling_maes)

print(f"\n✓ ULTRA Rolling validation: {rolling_mean:.3f} ± {rolling_std:.3f} MAE")
print(f"  Best fold: {min(rolling_maes):.3f}")
print(f"  Worst fold: {max(rolling_maes):.3f}")
print(f"✓ Saved: ultra_rolling_validation.csv")

print("\n[STEP 6] ELITE ENSEMBLE")
print("="*90)

# Inverse MAE weighted ensemble
val_maes = np.array([m['mae_val'] for m in trained_models.values()])
weights = 1.0 / val_maes
weights = weights / weights.sum()

all_val_preds = np.column_stack([m['predictions_val'] for m in trained_models.values()])
ensemble_pred_val = (all_val_preds * weights).sum(axis=1)
mae_ensemble_val = mean_absolute_error(y_val, ensemble_pred_val)

all_test_preds = np.column_stack([m['predictions_test'] for m in trained_models.values()])
ensemble_pred_test = (all_test_preds * weights).sum(axis=1)
mae_ensemble_test = mean_absolute_error(y_test, ensemble_pred_test)

print(f"✓ ULTRA Elite ensemble:")
print(f"  Validation MAE: {mae_ensemble_val:.3f}")
print(f"  Test MAE: {mae_ensemble_test:.3f}")

print("\n[STEP 7] DEPLOYMENT DECISION")
print("="*90)

baseline_rolling = 8.816

print(f"\n📊 COMPARISON:")
print(f"  Baseline (30 features): {baseline_rolling:.3f} ± 0.380 MAE")
print(f"  ULTRA (72 features): {rolling_mean:.3f} ± {rolling_std:.3f} MAE")

improvement = baseline_rolling - rolling_mean
improve_pct = (improvement / baseline_rolling) * 100

print(f"\n  Improvement: {improvement:.3f} MAE ({improve_pct:.1f}%)")

if rolling_mean < 8.3:
    decision = "BREAKTHROUGH"
    action = "Deploy ULTRA immediately!"
elif rolling_mean < 8.6:
    decision = "IMPROVEMENT"
    action = "Deploy ULTRA for Monday"
elif rolling_mean < 9.0:
    decision = "MARGINAL"
    action = "Keep simple, use ULTRA for analysis"
else:
    decision = "NO_CHANGE"
    action = "Keep simple system"

print(f"\n🎯 DEPLOYMENT DECISION: {decision}")
print(f"   Action: {action}")

print("\n[STEP 8] SAVING ULTRA HEDGE FUND SYSTEM")
print("="*90)

ultra_hedge_fund_system = {
    'name': 'ULTRA_HEDGE_FUND_SYSTEM',
    'version': 'ULTRA_1.0',
    'created': datetime.now().isoformat(),
    'n_games': len(X_all),
    'n_features': 72,
    'feature_categories': dataset['feature_categories'],
    'models': {k: v['model'] for k, v in trained_models.items()},
    'ensemble_weights': weights.tolist(),
    'scaler': scaler,
    'performance': {
        'rolling_mean': float(rolling_mean),
        'rolling_std': float(rolling_std),
        'test_mae': float(mae_ensemble_test),
        'val_mae': float(mae_ensemble_val),
        'improvement_vs_baseline': float(improvement)
    },
    'deployment': {
        'decision': decision,
        'action': action
    },
    'feature_importances': importance_df.to_dict('records')
}

with open('Action/ULTRA_HEDGE_FUND_SYSTEM_V1.pkl', 'wb') as f:
    pickle.dump(ultra_hedge_fund_system, f)

print(f"✓ Saved: ULTRA_HEDGE_FUND_SYSTEM_V1.pkl")

# Save config
config = {
    'version': 'ULTRA_1.0',
    'n_features': 72,
    'n_models': len(trained_models),
    'deployment_decision': decision,
    'rolling_mae': float(rolling_mean),
    'improvement': float(improvement),
    'created': datetime.now().isoformat()
}

with open('Action/ultra_system_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Saved: ultra_system_config.json")

print("\n" + "="*90)
print("🚀 ULTRA PHASE 3 COMPLETE!")
print("="*90)

print(f"\n📊 FINAL RESULTS:")
print(f"  Rolling MAE: {rolling_mean:.3f} ± {rolling_std:.3f}")
print(f"  vs Baseline: {baseline_rolling:.3f} ± 0.380")
print(f"  Improvement: {improvement:.3f} MAE ({improve_pct:.1f}%)")
print(f"  Decision: {decision}")

if decision in ["BREAKTHROUGH", "IMPROVEMENT"]:
    print(f"\n🔥 DEPLOY ULTRA HEDGE FUND SYSTEM!")
    print(f"   Expected Monday: {rolling_mean:.3f} ± {rolling_std:.3f} MAE")
    
    new_edge = ((11.5 - rolling_mean) / 11.5) * 100
    old_edge = 21.5
    edge_gain = new_edge - old_edge
    
    print(f"   Old edge: {old_edge:.1f}%")
    print(f"   New edge: {new_edge:.1f}%")
    print(f"   Gain: +{edge_gain:.1f} percentage points")
    
    new_ev = 20 * (new_edge / 100) * 100
    old_ev = 20 * (old_edge / 100) * 100
    ev_gain = new_ev - old_ev
    
    print(f"\n   Old EV: +${old_ev:.0f} per 100")
    print(f"   New EV: +${new_ev:.0f} per 100")
    print(f"   Gain: +${ev_gain:.0f} per 100 games")
    print(f"   Season gain: +${ev_gain * 50:.0f}")

print("\n✅ ULTRA TRAINING COMPLETE - HEDGE FUND READY!")
print("="*90)

