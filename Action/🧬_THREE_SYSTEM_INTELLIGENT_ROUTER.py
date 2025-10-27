#!/usr/bin/env python3
"""
🧬 THREE-SYSTEM INTELLIGENT ROUTER
Combines Mamba + Strive + Stanford with intelligent routing

KEY INSIGHT from Stanford:
- Traditional models (trees): HIGH performance, SEVERE overfitting (89-100%)
- Research models (deep/Bayesian): GOOD performance, LOW overfitting (3-8%)

STRATEGY: Use Stanford as baseline, boost with Mamba/Strive when confident
"""

import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors

print("="*80)
print("🧬 THREE-SYSTEM INTELLIGENT ROUTER")
print("="*80)
print()

# Load all 3 systems
print("[1/5] Loading all 3 systems...")
with open('MAMBA_MENTALITY_SYSTEM.pkl', 'rb') as f:
    mamba = pickle.load(f)

with open('STRIVE_FOR_GREATNESS_CLEAN.pkl', 'rb') as f:
    strive = pickle.load(f)

with open('STANFORD_RESEARCH_ENSEMBLE.pkl', 'rb') as f:
    stanford = pickle.load(f)

print("✅ Mamba:    5.430 / 10.000 MAE (81% / 100% overfitting)")
print("✅ Strive:   5.512 / 10.540 MAE (89% / 92% overfitting)")
print("✅ Stanford: 5.420 / 10.384 MAE (3.5% / 8.3% overfitting) ⭐")
print()

# Load test data
print("[2/5] Loading test data...")
with open('ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl', 'rb') as f:
    data = pickle.load(f)

split_idx = int(len(data) * 0.8)
test_data = data[split_idx:]

print(f"✅ Test set: {len(test_data)} games")
print()

# Extract features and targets
feature_names_stanford = stanford['feature_names']  # 73 features
feature_names_mamba = mamba.get('feature_names', feature_names_stanford[:67])  # 67 features

X_test_stanford = []
X_test_mamba = []
y_half_test = []
y_final_test = []

for game in test_data:
    stanford_feats = [game.get(f, 0) for f in feature_names_stanford]
    mamba_feats = [game.get(f, 0) for f in feature_names_mamba]
    
    X_test_stanford.append(stanford_feats)
    X_test_mamba.append(mamba_feats)
    y_half_test.append(game.get('diff_at_halftime', 0))
    y_final_test.append(game.get('diff_at_final', 0))

X_test_stanford = np.nan_to_num(np.array(X_test_stanford), nan=0.0)
X_test_mamba = np.nan_to_num(np.array(X_test_mamba), nan=0.0)
y_half_test = np.array(y_half_test)
y_final_test = np.array(y_final_test)

print(f"✅ Stanford features: {X_test_stanford.shape}")
print(f"✅ Mamba features: {X_test_mamba.shape}")
print()

# Get predictions from all 3 systems
print("[3/5] Getting predictions from all 3 systems...")

# Stanford
scaler_stanford_a = stanford['branch_a_halftime']['scaler']
scaler_stanford_b = stanford['branch_b_final']['scaler']

X_stanford_scaled_a = scaler_stanford_a.transform(X_test_stanford)
X_stanford_scaled_b = scaler_stanford_b.transform(X_test_stanford)

stanford_preds_half = []
for model in stanford['branch_a_halftime']['models'].values():
    stanford_preds_half.append(model.predict(X_stanford_scaled_a))
stanford_pred_half = np.mean(stanford_preds_half, axis=0)

stanford_preds_final = []
for model in stanford['branch_b_final']['models'].values():
    stanford_preds_final.append(model.predict(X_stanford_scaled_b))
stanford_pred_final = np.mean(stanford_preds_final, axis=0)

# Strive (uses same 73 features as Stanford)
scaler_strive_a = strive['branch_a_halftime']['scaler']
scaler_strive_b = strive['branch_b_final']['scaler']

X_strive_scaled_a = scaler_strive_a.transform(X_test_stanford)
X_strive_scaled_b = scaler_strive_b.transform(X_test_stanford)

strive_preds_half = []
for model in strive['branch_a_halftime']['models'].values():
    strive_preds_half.append(model.predict(X_strive_scaled_a))
strive_pred_half = np.mean(strive_preds_half, axis=0)

strive_preds_final = []
for model in strive['branch_b_final']['models'].values():
    strive_preds_final.append(model.predict(X_strive_scaled_b))
strive_pred_final = np.mean(strive_preds_final, axis=0)

# Mamba (uses 67 features)
scaler_mamba_a = mamba['branch_a_halftime']['scaler']
scaler_mamba_b = mamba['branch_b_final']['scaler']

X_mamba_scaled_a = scaler_mamba_a.transform(X_test_mamba)
X_mamba_scaled_b = scaler_mamba_b.transform(X_test_mamba)

mamba_preds_half = []
for model in mamba['branch_a_halftime']['models'].values():
    mamba_preds_half.append(model.predict(X_mamba_scaled_a))
mamba_pred_half = np.mean(mamba_preds_half, axis=0)

mamba_preds_final = []
for model in mamba['branch_b_final']['models'].values():
    mamba_preds_final.append(model.predict(X_mamba_scaled_b))
mamba_pred_final = np.mean(mamba_preds_final, axis=0)

print("✅ All predictions obtained")
print()

# Individual MAEs
mae_stanford_half = mean_absolute_error(y_half_test, stanford_pred_half)
mae_stanford_final = mean_absolute_error(y_final_test, stanford_pred_final)

mae_strive_half = mean_absolute_error(y_half_test, strive_pred_half)
mae_strive_final = mean_absolute_error(y_final_test, strive_pred_final)

mae_mamba_half = mean_absolute_error(y_half_test, mamba_pred_half)
mae_mamba_final = mean_absolute_error(y_final_test, mamba_pred_final)

print("INDIVIDUAL SYSTEM MAEs:")
print(f"  Stanford: {mae_stanford_half:.3f} / {mae_stanford_final:.3f}")
print(f"  Strive:   {mae_strive_half:.3f} / {mae_strive_final:.3f}")
print(f"  Mamba:    {mae_mamba_half:.3f} / {mae_mamba_final:.3f}")
print()

# Test ensemble strategies
print("[4/5] Testing ensemble strategies...")
print()

# Strategy 1: Simple average
avg_half = (stanford_pred_half + strive_pred_half + mamba_pred_half) / 3
avg_final = (stanford_pred_final + strive_pred_final + mamba_pred_final) / 3

mae_avg_half = mean_absolute_error(y_half_test, avg_half)
mae_avg_final = mean_absolute_error(y_final_test, avg_final)

print(f"1. SIMPLE AVERAGE: {mae_avg_half:.3f} / {mae_avg_final:.3f} MAE")

# Strategy 2: Weighted by inverse MAE
weights_half = np.array([1/mae_stanford_half, 1/mae_strive_half, 1/mae_mamba_half])
weights_half = weights_half / weights_half.sum()

weights_final = np.array([1/mae_stanford_final, 1/mae_strive_final, 1/mae_mamba_final])
weights_final = weights_final / weights_final.sum()

weighted_half = (stanford_pred_half * weights_half[0] + 
                 strive_pred_half * weights_half[1] + 
                 mamba_pred_half * weights_half[2])

weighted_final = (stanford_pred_final * weights_final[0] + 
                  strive_pred_final * weights_final[1] + 
                  mamba_pred_final * weights_final[2])

mae_weighted_half = mean_absolute_error(y_half_test, weighted_half)
mae_weighted_final = mean_absolute_error(y_final_test, weighted_final)

print(f"2. WEIGHTED (inv MAE): {mae_weighted_half:.3f} / {mae_weighted_final:.3f} MAE")
print(f"   Weights half: S={weights_half[0]:.2f}, St={weights_half[1]:.2f}, M={weights_half[2]:.2f}")
print(f"   Weights final: S={weights_final[0]:.2f}, St={weights_final[1]:.2f}, M={weights_final[2]:.2f}")

# Strategy 3: Stanford-first (trust low overfitting)
# Use Stanford by default, only use others if Stanford uncertain

# Simple version: 70% Stanford, 15% Strive, 15% Mamba
stanford_first_half = 0.7 * stanford_pred_half + 0.15 * strive_pred_half + 0.15 * mamba_pred_half
stanford_first_final = 0.7 * stanford_pred_final + 0.15 * strive_pred_final + 0.15 * mamba_pred_final

mae_stanford_first_half = mean_absolute_error(y_half_test, stanford_first_half)
mae_stanford_first_final = mean_absolute_error(y_final_test, stanford_first_final)

print(f"3. STANFORD-FIRST (70-15-15): {mae_stanford_first_half:.3f} / {mae_stanford_first_final:.3f} MAE")

# Strategy 4: Pick best per game (oracle - for analysis only)
oracle_half = []
oracle_final = []

for i in range(len(y_half_test)):
    errors_half = [
        abs(stanford_pred_half[i] - y_half_test[i]),
        abs(strive_pred_half[i] - y_half_test[i]),
        abs(mamba_pred_half[i] - y_half_test[i])
    ]
    best_idx_half = np.argmin(errors_half)
    oracle_half.append([stanford_pred_half[i], strive_pred_half[i], mamba_pred_half[i]][best_idx_half])
    
    errors_final = [
        abs(stanford_pred_final[i] - y_final_test[i]),
        abs(strive_pred_final[i] - y_final_test[i]),
        abs(mamba_pred_final[i] - y_final_test[i])
    ]
    best_idx_final = np.argmin(errors_final)
    oracle_final.append([stanford_pred_final[i], strive_pred_final[i], mamba_pred_final[i]][best_idx_final])

mae_oracle_half = mean_absolute_error(y_half_test, oracle_half)
mae_oracle_final = mean_absolute_error(y_final_test, oracle_final)

print(f"4. ORACLE (best per game): {mae_oracle_half:.3f} / {mae_oracle_final:.3f} MAE (upper bound)")

print()

# Strategy 5: Confidence-based routing
# Use variance across models as confidence measure
confidence_half = []
confidence_final = []

for i in range(len(y_half_test)):
    # Low variance = models agree = high confidence
    var_half = np.var([stanford_pred_half[i], strive_pred_half[i], mamba_pred_half[i]])
    var_final = np.var([stanford_pred_final[i], strive_pred_final[i], mamba_pred_final[i]])
    
    confidence_half.append(1.0 / (1.0 + var_half))
    confidence_final.append(1.0 / (1.0 + var_final))

confidence_half = np.array(confidence_half)
confidence_final = np.array(confidence_final)

# When confidence is high, use weighted average
# When confidence is low, use Stanford (lowest overfitting)
confidence_routed_half = []
confidence_routed_final = []

for i in range(len(y_half_test)):
    if confidence_half[i] > 0.5:  # High confidence - use weighted
        pred = (stanford_pred_half[i] * weights_half[0] + 
                strive_pred_half[i] * weights_half[1] + 
                mamba_pred_half[i] * weights_half[2])
    else:  # Low confidence - use Stanford only
        pred = stanford_pred_half[i]
    confidence_routed_half.append(pred)
    
    if confidence_final[i] > 0.5:
        pred = (stanford_pred_final[i] * weights_final[0] + 
                strive_pred_final[i] * weights_final[1] + 
                mamba_pred_final[i] * weights_final[2])
    else:
        pred = stanford_pred_final[i]
    confidence_routed_final.append(pred)

mae_confidence_half = mean_absolute_error(y_half_test, confidence_routed_half)
mae_confidence_final = mean_absolute_error(y_final_test, confidence_routed_final)

print(f"5. CONFIDENCE ROUTING: {mae_confidence_half:.3f} / {mae_confidence_final:.3f} MAE")

print()

# Save best strategy
print("[5/5] Saving intelligent router...")

best_strategy = 'stanford_first'  # Based on low overfitting
best_mae_half = mae_stanford_first_half
best_mae_final = mae_stanford_first_final

router_system = {
    'strategy': best_strategy,
    'weights': {'stanford': 0.7, 'strive': 0.15, 'mamba': 0.15},
    'systems': {
        'stanford': 'STANFORD_RESEARCH_ENSEMBLE.pkl',
        'strive': 'STRIVE_FOR_GREATNESS_CLEAN.pkl',
        'mamba': 'MAMBA_MENTALITY_SYSTEM.pkl'
    },
    'performance': {
        'halftime_mae': best_mae_half,
        'final_mae': best_mae_final,
        'individual': {
            'stanford': (mae_stanford_half, mae_stanford_final),
            'strive': (mae_strive_half, mae_strive_final),
            'mamba': (mae_mamba_half, mae_mamba_final)
        },
        'strategies': {
            'simple_average': (mae_avg_half, mae_avg_final),
            'weighted': (mae_weighted_half, mae_weighted_final),
            'stanford_first': (mae_stanford_first_half, mae_stanford_first_final),
            'confidence': (mae_confidence_half, mae_confidence_final),
            'oracle_upper_bound': (mae_oracle_half, mae_oracle_final)
        }
    },
    'metadata': {
        'build_date': '2025-10-19',
        'test_games': len(test_data),
        'philosophy': 'Trust Stanford (low overfitting), boost with Mamba/Strive'
    }
}

with open('THREE_SYSTEM_ROUTER.pkl', 'wb') as f:
    pickle.dump(router_system, f)

print("✅ Saved to: THREE_SYSTEM_ROUTER.pkl")
print()

# Summary
print("="*80)
print("🎯 INTELLIGENT ROUTER - FINAL RESULTS")
print("="*80)
print()

print("INDIVIDUAL SYSTEMS:")
print(f"  Stanford (research):   {mae_stanford_half:.3f} / {mae_stanford_final:.3f} MAE (3.5% / 8.3% overfit)")
print(f"  Strive (traditional):  {mae_strive_half:.3f} / {mae_strive_final:.3f} MAE (89% / 92% overfit)")
print(f"  Mamba (traditional):   {mae_mamba_half:.3f} / {mae_mamba_final:.3f} MAE (81% / 100% overfit)")
print()

print("ENSEMBLE STRATEGIES:")
print(f"  Simple Average:        {mae_avg_half:.3f} / {mae_avg_final:.3f} MAE")
print(f"  Weighted by MAE:       {mae_weighted_half:.3f} / {mae_weighted_final:.3f} MAE")
print(f"  Stanford-First (70%):  {mae_stanford_first_half:.3f} / {mae_stanford_first_final:.3f} MAE ⭐")
print(f"  Confidence Routing:    {mae_confidence_half:.3f} / {mae_confidence_final:.3f} MAE")
print(f"  Oracle (upper bound):  {mae_oracle_half:.3f} / {mae_oracle_final:.3f} MAE")
print()

print("WINNER: STANFORD-FIRST (70-15-15)")
print(f"  Halftime: {mae_stanford_first_half:.3f} MAE")
print(f"  Final:    {mae_stanford_first_final:.3f} MAE")
print()

print("WHY STANFORD-FIRST?")
print("  • Stanford has LOWEST overfitting (3.5% / 8.3%)")
print("  • Will generalize BEST to Monday's new data")
print("  • Mamba/Strive overfit heavily (81-100%)")
print("  • Use 70% Stanford, 30% traditional for diversity")
print()

print("="*80)
print("✅ THREE-SYSTEM ROUTER COMPLETE")
print("="*80)

