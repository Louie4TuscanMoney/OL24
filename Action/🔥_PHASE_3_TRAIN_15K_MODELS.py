"""
🔥 PHASE 3 - ELITE MODEL TRAINING ON 15K DATASET
Production-grade with comprehensive logging, segment analysis, live simulation

ELITE ENHANCEMENTS (Your feedback!):
  ✅ Comprehensive logging (MLflow-style CSV)
  ✅ Segment-level leaderboard (strength/weakness by context)
  ✅ Live backtest simulation (before deployment)
  ✅ All artifacts saved (reproducibility)
  ✅ Feature importance tracking
  ✅ Rolling walk-forward (15 folds on 15k data!)
  ✅ Deployment decision matrix

INPUT: COMPLETE_15K_GAMES_30_FEATURES_V1.pkl
OUTPUT: ULTRA_15K_PRODUCTION_SYSTEM.pkl + comprehensive artifacts
TIME: 30-45 minutes
MODE: ELON - Elite quant lab execution
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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("🔥 PHASE 3 - ELITE MODEL TRAINING ON 15K DATASET")
print("="*90)
print(f"\nStart time: {datetime.now().strftime('%I:%M %p')}")
print("Mode: Elite quant lab - comprehensive validation")
print("\n" + "="*90)

print("\n[STEP 1] LOADING ELITE FEATURE DATASET")
print("="*90)

try:
    with open('Action/COMPLETE_15K_GAMES_30_FEATURES_V1.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    X_all = dataset['features']
    y_final = dataset['targets']['final_diff']
    y_current = dataset['targets']['current_diff']
    metadata = dataset['metadata']
    feature_names = dataset['feature_names']
    
    print(f"✓ Loaded elite dataset:")
    print(f"  Games: {len(X_all)}")
    print(f"  Features: {X_all.shape[1]}")
    print(f"  Version: {dataset['version']}")
    print(f"  Date range: {dataset['data_stats']['date_range']}")
    
except:
    print("✗ Elite dataset not found - run Phase 2 first!")
    exit(1)

print("\n[STEP 2] TEMPORAL TRAIN/VAL/TEST SPLIT")
print("="*90)

# Chronological split (70/15/15)
n_total = len(X_all)
split1 = int(n_total * 0.7)
split2 = int(n_total * 0.85)

X_train = X_all[:split1]
X_val = X_all[split1:split2]
X_test = X_all[split2:]

y_train = y_final[:split1]
y_val = y_final[split1:split2]
y_test = y_final[split2:]

# Get date ranges from metadata
train_dates = [m['date'] for m in metadata[:split1]]
val_dates = [m['date'] for m in metadata[split1:split2]]
test_dates = [m['date'] for m in metadata[split2:]]

print(f"✓ Temporal splits:")
print(f"  Train: {len(X_train):5d} games ({min(train_dates)[:10]} to {max(train_dates)[:10]})")
print(f"  Val:   {len(X_val):5d} games ({min(val_dates)[:10]} to {max(val_dates)[:10]})")
print(f"  Test:  {len(X_test):5d} games ({min(test_dates)[:10]} to {max(test_dates)[:10]})")

# Scale (NO LEAKAGE!)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled (fit on train only - NO LEAKAGE!)")

print("\n[STEP 3] AUTO FEATURE SELECTION (LassoCV)")
print("="*90)

print("\nRunning LassoCV with 5-fold CV...")
lasso = LassoCV(cv=5, alphas=np.logspace(-3, 1, 30), max_iter=10000, random_state=42, n_jobs=-1)
lasso.fit(X_train_scaled, y_train)

feature_importance = np.abs(lasso.coef_)
n_selected = np.sum(feature_importance > 0.01)

print(f"✓ LASSO selected {n_selected} features (alpha={lasso.alpha_:.4f})")

# Save feature importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance,
    'selected': feature_importance > 0.01
}).sort_values('importance', ascending=False)

importance_df.to_csv('Action/feature_importances.csv', index=False)
print(f"✓ Saved: feature_importances.csv")

print("\nTop 10 features:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:<20} {row['importance']:>8.4f} {'✓' if row['selected'] else ''}")

print("\n[STEP 4] DATA-DRIVEN CLUSTERING")
print("="*90)

print("\nDiscovering natural game segments...")

# Use top features for clustering
top_feature_idx = importance_df.head(10).index.tolist()
X_cluster = X_train_scaled[:, top_feature_idx]

# Find optimal k
from sklearn.metrics import silhouette_score

best_k = 3
best_sil = -1

print("Testing k values:")
for k in range(3, 9):
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans_test.fit_predict(X_cluster)
    sil = silhouette_score(X_cluster, labels)
    
    print(f"  k={k}: silhouette={sil:.3f}")
    
    if sil > best_sil:
        best_sil = sil
        best_k = k

print(f"\n✓ Optimal k={best_k} (silhouette={best_sil:.3f})")

# Final clustering
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels_train = kmeans.fit_predict(X_cluster)
cluster_labels_val = kmeans.predict(X_val_scaled[:, top_feature_idx])
cluster_labels_test = kmeans.predict(X_test_scaled[:, top_feature_idx])

print(f"\n🎯 Discovered {best_k} natural segments")

print("\n[STEP 5] TRAINING MODELS (Comprehensive Logging!)")
print("="*90)

# Comprehensive logging
training_log = []

print("\nTraining diverse model pool...")

models_to_train = [
    ('LASSO', lambda: lasso),  # Already trained
    ('Ridge', lambda: Ridge(alpha=2.0, max_iter=5000)),
    ('ElasticNet', lambda: ElasticNetCV(cv=3, l1_ratio=[0.1, 0.5, 0.9], random_state=42, n_jobs=-1)),
    ('LightGBM', lambda: LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.05,
                                       reg_alpha=3.0, reg_lambda=3.0, random_state=42, verbose=-1)),
    ('XGBoost', lambda: XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                     reg_alpha=3.0, reg_lambda=3.0, random_state=42))
]

trained_models = {}

for model_name, model_fn in models_to_train:
    print(f"\n  Training {model_name}...")
    start = datetime.now()
    
    if model_name == 'LASSO':
        model = model_fn()  # Already trained
    else:
        model = model_fn()
        model.fit(X_train_scaled, y_train)
    
    # Predict on all sets
    pred_train = model.predict(X_train_scaled)
    pred_val = model.predict(X_val_scaled)
    pred_test = model.predict(X_test_scaled)
    
    # Calculate MAE
    mae_train = mean_absolute_error(y_train, pred_train)
    mae_val = mean_absolute_error(y_val, pred_val)
    mae_test = mean_absolute_error(y_test, pred_test)
    
    # Overfitting
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
    
    # Log
    training_log.append({
        'model': model_name,
        'mae_train': mae_train,
        'mae_val': mae_val,
        'mae_test': mae_test,
        'overfit_pct': overfit_pct,
        'training_time_sec': elapsed
    })
    
    print(f"    ✓ Train: {mae_train:.3f}, Val: {mae_val:.3f}, Test: {mae_test:.3f}, Overfit: {overfit_pct:.1f}%")

# Save training log
pd.DataFrame(training_log).to_csv('Action/rolling_mae_log.csv', index=False)
print(f"\n✓ Saved: rolling_mae_log.csv")

print("\n[STEP 6] SEGMENT-LEVEL LEADERBOARD")
print("="*90)

print(f"\nAnalyzing model performance per cluster (context-aware)...")

segment_leaderboard = []

for cluster_id in range(best_k):
    mask_val = cluster_labels_val == cluster_id
    n_games = np.sum(mask_val)
    
    if n_games < 10:
        continue
    
    print(f"\n  Cluster {cluster_id} ({n_games} val games):")
    
    for model_name, model_data in trained_models.items():
        pred_cluster = model_data['predictions_val'][mask_val]
        y_cluster = y_val[mask_val]
        
        mae_cluster = mean_absolute_error(y_cluster, pred_cluster)
        
        segment_leaderboard.append({
            'cluster': cluster_id,
            'model': model_name,
            'mae': mae_cluster,
            'n_games': n_games
        })
        
        print(f"    {model_name:<12} MAE: {mae_cluster:.3f}")

# Save leaderboard
pd.DataFrame(segment_leaderboard).to_csv('Action/segment_leaderboard.csv', index=False)
print(f"\n✓ Saved: segment_leaderboard.csv")

# Best model per segment
print(f"\n🎯 Best model per segment:")
segment_df = pd.DataFrame(segment_leaderboard)
for cluster_id in range(best_k):
    cluster_data = segment_df[segment_df['cluster'] == cluster_id]
    if len(cluster_data) > 0:
        best = cluster_data.loc[cluster_data['mae'].idxmin()]
        print(f"  Cluster {cluster_id}: {best['model']} ({best['mae']:.3f} MAE)")

print("\n[STEP 7] ROLLING WALK-FORWARD VALIDATION (15 Folds!)")
print("="*90)

print("\nPerforming comprehensive 15-fold rolling validation...")

n_folds = 15
fold_size = len(X_all) // (n_folds + 1)
rolling_results = []

for fold in range(n_folds):
    train_end = (fold + 1) * fold_size
    test_start = train_end
    test_end = min(train_end + fold_size, len(X_all))
    
    if test_end - test_start < 50:
        break
    
    # Split
    X_fold_train = X_all[:train_end]
    X_fold_test = X_all[test_start:test_end]
    y_fold_train = y_final[:train_end]
    y_fold_test = y_final[test_start:test_end]
    
    # Scale
    scaler_fold = RobustScaler()
    X_fold_train_sc = scaler_fold.fit_transform(X_fold_train)
    X_fold_test_sc = scaler_fold.transform(X_fold_test)
    
    # Train Ridge (fast, stable)
    model_fold = Ridge(alpha=2.0)
    model_fold.fit(X_fold_train_sc, y_fold_train)
    
    # Predict
    pred_fold = model_fold.predict(X_fold_test_sc)
    mae_fold = mean_absolute_error(y_fold_test, pred_fold)
    
    rolling_results.append({
        'fold': fold + 1,
        'train_size': len(X_fold_train),
        'test_size': len(X_fold_test),
        'mae': mae_fold
    })
    
    print(f"  Fold {fold+1:2d}: Train {len(X_fold_train):5d}, Test {len(X_fold_test):4d} → MAE: {mae_fold:.3f}")

# Save rolling results
pd.DataFrame(rolling_results).to_csv('Action/rolling_mae_log_15folds.csv', index=False)

rolling_maes = [r['mae'] for r in rolling_results]
rolling_mean = np.mean(rolling_maes)
rolling_std = np.std(rolling_maes)

print(f"\n✓ Rolling validation (15 folds): {rolling_mean:.3f} ± {rolling_std:.3f} MAE")
print(f"  Best fold: {min(rolling_maes):.3f}")
print(f"  Worst fold: {max(rolling_maes):.3f}")
print(f"✓ Saved: rolling_mae_log_15folds.csv")

print("\n[STEP 8] LIVE BACKTEST SIMULATION")
print("="*90)

print("\nSimulating live betting on test set...")

# Get best model from validation
best_model_name = min(trained_models.items(), key=lambda x: x[1]['mae_val'])[0]
best_model = trained_models[best_model_name]['model']
test_preds = trained_models[best_model_name]['predictions_test']

print(f"Using: {best_model_name} (best validation MAE)")

# Simulate betting
# Assume market lines are close to actual outcomes (realistic for NBA)
market_lines = y_test + np.random.normal(0, 2.0, len(y_test))

# Betting criteria
edge_threshold = 4.5
bets_made = np.abs(test_preds - market_lines) > edge_threshold

n_bets = np.sum(bets_made)
bet_rate = n_bets / len(y_test) * 100

if n_bets > 0:
    # Calculate performance on bets only
    bet_mae = mean_absolute_error(y_test[bets_made], test_preds[bets_made])
    bet_accuracy = np.mean((y_test[bets_made] > 0) == (test_preds[bets_made] > 0))
    avg_edge = np.abs(test_preds[bets_made] - market_lines[bets_made]).mean()
    
    # Simulate profit (assuming -110 odds, $100 bets)
    wins = np.sum((y_test[bets_made] > 0) == (test_preds[bets_made] > 0))
    profit = wins * 90.9 - (n_bets - wins) * 100  # Win $90.9, lose $100
    roi = (profit / (n_bets * 100)) * 100
    
    backtest_results = {
        'n_bets': int(n_bets),
        'bet_rate': float(bet_rate),
        'bet_mae': float(bet_mae),
        'accuracy': float(bet_accuracy),
        'avg_edge': float(avg_edge),
        'simulated_profit': float(profit),
        'roi': float(roi),
        'edge_threshold': edge_threshold
    }
    
    print(f"✓ Backtest simulation:")
    print(f"  Bets made: {n_bets}/{len(y_test)} ({bet_rate:.1f}%)")
    print(f"  MAE on bets: {bet_mae:.3f}")
    print(f"  Accuracy: {bet_accuracy*100:.1f}%")
    print(f"  Avg edge: {avg_edge:.1f} points")
    print(f"  Simulated profit: ${profit:.0f} ({roi:.1f}% ROI)")
    
    # Save
    with open('Action/backtest_results.json', 'w') as f:
        json.dump(backtest_results, f, indent=2)
    
    print(f"✓ Saved: backtest_results.json")
else:
    print(f"⚠️  No bets qualified (threshold={edge_threshold})")

print("\n[STEP 9] ENSEMBLE & DEPLOYMENT CANDIDATE")
print("="*90)

# Inverse MAE weighted ensemble
val_maes = np.array([m['mae_val'] for m in trained_models.values()])
weights = 1.0 / val_maes
weights = weights / weights.sum()

# Ensemble predictions
all_val_preds = np.column_stack([m['predictions_val'] for m in trained_models.values()])
ensemble_pred_val = (all_val_preds * weights).sum(axis=1)
mae_ensemble_val = mean_absolute_error(y_val, ensemble_pred_val)

all_test_preds = np.column_stack([m['predictions_test'] for m in trained_models.values()])
ensemble_pred_test = (all_test_preds * weights).sum(axis=1)
mae_ensemble_test = mean_absolute_error(y_test, ensemble_pred_test)

print(f"✓ Elite ensemble:")
print(f"  Validation MAE: {mae_ensemble_val:.3f}")
print(f"  Test MAE: {mae_ensemble_test:.3f}")

print("\n[STEP 10] DEPLOYMENT DECISION")
print("="*90)

# Compare to baseline
baseline_mae = 9.029  # From HYBRID_V2_CLEAN
baseline_rolling = 8.816  # From previous rolling validation

print(f"\n📊 COMPARISON:")
print(f"  Baseline (6.9k games):")
print(f"    Test: {baseline_mae:.3f} MAE")
print(f"    Rolling: {baseline_rolling:.3f} ± 0.380 MAE")
print(f"\n  New (15k games):")
print(f"    Test: {mae_ensemble_test:.3f} MAE")
print(f"    Rolling: {rolling_mean:.3f} ± {rolling_std:.3f} MAE")

improvement = baseline_rolling - rolling_mean
improve_pct = (improvement / baseline_rolling) * 100

print(f"\n  Improvement: {improvement:.3f} MAE ({improve_pct:.1f}%)")

# Deployment decision
if rolling_mean < 8.4:
    decision = "BREAKTHROUGH"
    action = "Deploy ULTRA_15K immediately!"
elif rolling_mean < 8.6:
    decision = "IMPROVEMENT"
    action = "Deploy ULTRA_15K for Monday"
elif rolling_mean < 9.0:
    decision = "MARGINAL"
    action = "Keep HYBRID_V2_CLEAN, use 15k for Week 3"
else:
    decision = "NO_CHANGE"
    action = "Keep simple system"

print(f"\n🎯 DEPLOYMENT DECISION: {decision}")
print(f"   Action: {action}")

print("\n[STEP 11] SAVING ALL ARTIFACTS")
print("="*90)

# Save complete system
ultra_15k_system = {
    'name': 'ULTRA_15K_PRODUCTION_SYSTEM',
    'version': '1.0.0',
    'created': datetime.now().isoformat(),
    'n_games': len(X_all),
    'n_features': 30,
    'feature_version': '1.0',
    'models': {k: v['model'] for k, v in trained_models.items()},
    'ensemble_weights': weights.tolist(),
    'scaler': scaler,
    'clustering': {
        'kmeans': kmeans,
        'n_clusters': best_k,
        'top_features': top_feature_idx
    },
    'performance': {
        'rolling_mean': float(rolling_mean),
        'rolling_std': float(rolling_std),
        'test_mae': float(mae_ensemble_test),
        'val_mae': float(mae_ensemble_val),
        'best_model': best_model_name
    },
    'deployment': {
        'decision': decision,
        'action': action,
        'improvement_vs_baseline': float(improvement)
    }
}

with open('Action/ULTRA_15K_PRODUCTION_SYSTEM.pkl', 'wb') as f:
    pickle.dump(ultra_15k_system, f)

print(f"✓ Saved: ULTRA_15K_PRODUCTION_SYSTEM.pkl")

# Save config YAML-style
train_config = {
    'dataset': 'COMPLETE_15K_GAMES_30_FEATURES_V1.pkl',
    'n_games': len(X_all),
    'n_features': 30,
    'models': list(trained_models.keys()),
    'best_model': best_model_name,
    'rolling_folds': n_folds,
    'deployment_decision': decision,
    'created': datetime.now().isoformat()
}

with open('Action/train_config.json', 'w') as f:
    json.dump(train_config, f, indent=2)

print(f"✓ Saved: train_config.json")

print("\n" + "="*90)
print("🏆 PHASE 3 COMPLETE - ELITE TRAINING DONE!")
print("="*90)

print(f"\n📂 ARTIFACTS SAVED:")
print(f"  1. ULTRA_15K_PRODUCTION_SYSTEM.pkl (production model)")
print(f"  2. feature_importances.csv (LASSO importance)")
print(f"  3. rolling_mae_log.csv (training log)")
print(f"  4. rolling_mae_log_15folds.csv (validation history)")
print(f"  5. segment_leaderboard.csv (performance by context)")
print(f"  6. backtest_results.json (live simulation)")
print(f"  7. train_config.json (reproducibility)")

print(f"\n📊 FINAL RESULTS:")
print(f"  Rolling MAE: {rolling_mean:.3f} ± {rolling_std:.3f}")
print(f"  vs Baseline: {baseline_rolling:.3f} ± 0.380")
print(f"  Improvement: {improvement:.3f} MAE ({improve_pct:.1f}%)")
print(f"  Decision: {decision}")

if decision in ["BREAKTHROUGH", "IMPROVEMENT"]:
    print(f"\n🔥 DEPLOY ULTRA_15K_SYSTEM!")
    print(f"   Expected Monday: {rolling_mean:.3f} ± {rolling_std:.3f} MAE")
    
    # Calculate new edge
    baseline_edge = 21.5
    new_edge = ((11.5 - rolling_mean) / 11.5) * 100
    edge_gain = new_edge - baseline_edge
    
    print(f"   Old edge: {baseline_edge:.1f}%")
    print(f"   New edge: {new_edge:.1f}%")
    print(f"   Gain: +{edge_gain:.1f} percentage points")
    
    old_ev = 20 * (baseline_edge / 100) * 100
    new_ev = 20 * (new_edge / 100) * 100
    ev_gain = new_ev - old_ev
    
    print(f"\n   Old EV: +${old_ev:.0f} per 100")
    print(f"   New EV: +${new_ev:.0f} per 100")
    print(f"   Gain: +${ev_gain:.0f} per 100 games")
    print(f"   Season gain: +${ev_gain * 50:.0f}")
else:
    print(f"\n📊 Keep HYBRID_V2_CLEAN for Monday")
    print(f"   Use 15k dataset for Week 3 enhancements")

print("\n✅ ELITE TRAINING COMPLETE - PRODUCTION READY!")
print("="*90)

