#!/usr/bin/env python3
"""
HELIOS FIX: Remove target leakage + Better LASSO + Retrain
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime

print("=" * 100)
print("🔥 HELIOS FIX: REMOVE LEAKAGE + BETTER LASSO + RETRAIN")
print("=" * 100)
print()

# Load data
print("[1/6] Loading data...")
with open("helios/data/checkpoints/collection_progress.pkl", 'rb') as f:
    progress = pickle.load(f)
    games_data = progress.get('collected_games', [])

print(f"✅ Loaded {len(games_data)} games")
print()

# Extract features PROPERLY (only Q2 6:00 info, not full game)
print("[2/6] Extracting Q2 6:00 features ONLY (no target leakage)...")
features_list = []

for game in games_data:
    try:
        # Only use pattern up to Q2 6:00 (first ~6 points of 18-point pattern)
        pattern = game.get('pattern', [0] * 18)
        pattern_q2 = pattern[:6]  # Only first 6 points (Q1 + early Q2)
        
        features = {}
        features['game_id'] = game.get('game_id', 'unknown')
        
        # Basic state at Q2 6:00
        features['home_score'] = game.get('home_score', 0)
        features['away_score'] = game.get('away_score', 0)
        features['current_diff'] = features['home_score'] - features['away_score']
        features['current_diff_abs'] = abs(features['current_diff'])
        features['total_score'] = features['home_score'] + features['away_score']
        
        # Pattern features (only up to Q2 6:00)
        for i, val in enumerate(pattern_q2):
            features[f'diff_point_{i+1}'] = val
        
        # Rolling stats on Q2 6:00 pattern
        pattern_array = np.array(pattern_q2)
        features['pattern_mean'] = np.mean(pattern_array)
        features['pattern_std'] = np.std(pattern_array)
        features['pattern_min'] = np.min(pattern_array)
        features['pattern_max'] = np.max(pattern_array)
        features['pattern_range'] = features['pattern_max'] - features['pattern_min']
        
        # Momentum
        if len(pattern_array) > 1:
            diffs = np.diff(pattern_array)
            features['momentum'] = np.mean(diffs)
            features['momentum_last'] = diffs[-1] if len(diffs) > 0 else 0
            features['acceleration'] = diffs[-1] - diffs[0] if len(diffs) > 1 else 0
        else:
            features['momentum'] = 0
            features['momentum_last'] = 0
            features['acceleration'] = 0
        
        # Volatility
        features['volatility'] = np.std(pattern_array)
        
        # Lead changes
        lead_changes = 0
        for i in range(len(pattern_array) - 1):
            if (pattern_array[i] >= 0) != (pattern_array[i+1] >= 0):
                lead_changes += 1
        features['lead_changes'] = lead_changes
        features['max_lead'] = np.max(np.abs(pattern_array))
        
        # Rolling windows
        if len(pattern_array) >= 3:
            features['roll_3_mean'] = np.mean(pattern_array[-3:])
            features['roll_3_std'] = np.std(pattern_array[-3:])
        else:
            features['roll_3_mean'] = features['pattern_mean']
            features['roll_3_std'] = 0
        
        # Interactions
        features['diff_momentum'] = features['current_diff'] * features['momentum']
        features['diff_volatility'] = features['current_diff'] * features['volatility']
        
        # Target (final differential)
        features['target'] = game.get('final_diff', features['current_diff'])
        
        features_list.append(features)
        
    except Exception as e:
        pass

print(f"✅ Extracted features from {len(features_list)} games")
print()

# Convert to DataFrame
print("[3/6] Creating DataFrame...")
df = pd.DataFrame(features_list)
df = df.replace([np.inf, -np.inf], [1e10, -1e10])
df = df.fillna(0)
print(f"✅ Shape: {df.shape}")
print(f"   Features: {df.shape[1] - 2}")
print()

# Prepare for LASSO
print("[4/6] Running BETTER LASSO (less aggressive)...")
feature_cols = [col for col in df.columns if col not in ['game_id', 'target']]
X = df[feature_cols].values
y = df['target'].values

# Chronological split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Better LASSO (less aggressive, more features selected)
alphas = np.logspace(-6, 0, 100)  # More lenient alpha range
lasso = LassoCV(alphas=alphas, cv=5, max_iter=50000, n_jobs=-1, random_state=42)
lasso.fit(X_train_scaled, y_train)

print(f"   Best alpha: {lasso.alpha_:.6f}")

# Get selected features
coefs = np.abs(lasso.coef_)
selected_features = [feature_cols[i] for i in range(len(coefs)) if coefs[i] > 1e-10]

print(f"   Selected features: {len(selected_features)} / {len(feature_cols)}")
print()

if len(selected_features) == 0:
    print("⚠️  LASSO still too aggressive, using top 15 by importance instead...")
    top_indices = np.argsort(coefs)[::-1][:15]
    selected_features = [feature_cols[i] for i in top_indices]
    print(f"   Using top 15 features")

print("Top features:")
feature_importance = sorted(zip(feature_cols, coefs), key=lambda x: x[1], reverse=True)
for i, (name, coef) in enumerate(feature_importance[:15], 1):
    print(f"   {i:2d}. {name:25s} {coef:10.6f}")
print()

# Train final model on selected features
print("[5/6] Training Ridge on selected features...")
selected_indices = [i for i, col in enumerate(feature_cols) if col in selected_features]
X_train_selected = X_train_scaled[:, selected_indices]
X_test_selected = X_test_scaled[:, selected_indices]

model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_selected, y_train)

train_pred = model.predict(X_train_selected)
test_pred = model.predict(X_test_selected)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
overfit = (test_mae - train_mae) / train_mae * 100 if train_mae > 0 else 0

baseline_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
edge = (baseline_mae - test_mae) / baseline_mae * 100

print(f"   Train MAE: {train_mae:.3f}")
print(f"   Test MAE:  {test_mae:.3f}")
print(f"   Overfit:   {overfit:.1f}%")
print(f"   Baseline:  {baseline_mae:.3f}")
print(f"   Edge:      {edge:.1f}%")
print()

# Save
print("[6/6] Saving...")
results = {
    'model': model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'selected_features': selected_features,
    'selected_indices': selected_indices,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'overfit_pct': overfit,
    'baseline_mae': baseline_mae,
    'edge_pct': edge,
    'lasso_model': lasso
}

with open("helios/data/HELIOS_FIXED_MODEL.pkl", 'wb') as f:
    pickle.dump(results, f)

print("✅ Saved to: helios/data/HELIOS_FIXED_MODEL.pkl")
print()

print("=" * 100)
print("✅ HELIOS FIXED & RETRAINED!")
print("=" * 100)
print()
print(f"Features: {len(selected_features)} selected from {len(feature_cols)} total")
print(f"Train: {len(X_train)} games | Test: {len(X_test)} games")
print()
print(f"📊 FINAL PERFORMANCE:")
print(f"   Test MAE: {test_mae:.3f}")
print(f"   Overfit:  {overfit:.1f}%")
print(f"   Edge:     {edge:.1f}%")
print()

# Compare to baseline
baseline_comparison = 9.029  # Our existing system
if test_mae < baseline_comparison:
    improvement = baseline_comparison - test_mae
    print(f"🎉 IMPROVEMENT: {improvement:.3f} MAE better than baseline ({baseline_comparison:.3f})")
else:
    print(f"⚠️  WORSE: {test_mae - baseline_comparison:.3f} MAE worse than baseline ({baseline_comparison:.3f})")

print()

