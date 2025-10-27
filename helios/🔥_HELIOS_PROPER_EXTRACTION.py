#!/usr/bin/env python3
"""
HELIOS PROPER EXTRACTION
Extract Q2 6:00 features from raw event data
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNetCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime

print("=" * 100)
print("🔥 HELIOS PROPER EXTRACTION - Q2 6:00 FEATURES")
print("=" * 100)
print()

# Load data
print("[1/7] Loading raw event data...")
with open("helios/data/checkpoints/collection_progress.pkl", 'rb') as f:
    progress = pickle.load(f)
    games = progress.get('collected_games', [])

print(f"✅ Loaded {len(games)} games")
print()

# Extract Q2 6:00 features
print("[2/7] Extracting Q2 6:00 snapshots...")
features_list = []
failed = 0

for idx, game in enumerate(games):
    try:
        score_timeline = np.array(game.get('score_timeline', []))
        targets = game.get('targets', {})
        
        if len(score_timeline) == 0 or 'diff_at_final' not in targets:
            failed += 1
            continue
        
        # Find Q2 6:00 position (~25% of game)
        # Assuming timeline has ~400 events for full game
        # Q2 6:00 = end of Q1 (12 min) + 6 min of Q2 = 18 min of 48 min = 37.5%
        q2_6min_idx = int(len(score_timeline) * 0.375)
        q2_6min_idx = min(q2_6min_idx, len(score_timeline) - 1)
        
        # Get pattern up to Q2 6:00
        pattern_q2 = score_timeline[:q2_6min_idx+1]
        
        # Current state at Q2 6:00
        current_diff = pattern_q2[-1] if len(pattern_q2) > 0 else 0
        
        features = {}
        features['game_id'] = game.get('game_id', f'game_{idx}')
        features['current_diff'] = float(current_diff)
        features['current_diff_abs'] = float(abs(current_diff))
        
        # Statistical features on pattern up to Q2 6:00
        if len(pattern_q2) > 1:
            features['pattern_mean'] = float(np.mean(pattern_q2))
            features['pattern_std'] = float(np.std(pattern_q2))
            features['pattern_min'] = float(np.min(pattern_q2))
            features['pattern_max'] = float(np.max(pattern_q2))
            features['pattern_range'] = float(features['pattern_max'] - features['pattern_min'])
            features['pattern_median'] = float(np.median(pattern_q2))
            
            # Momentum
            diffs = np.diff(pattern_q2)
            features['momentum'] = float(np.mean(diffs))
            features['momentum_std'] = float(np.std(diffs))
            features['momentum_last'] = float(diffs[-1]) if len(diffs) > 0 else 0.0
            
            # Volatility
            features['volatility'] = float(np.std(pattern_q2))
            features['mad'] = float(np.mean(np.abs(pattern_q2 - features['pattern_mean'])))
            
            # Lead changes
            lead_changes = 0
            for i in range(len(pattern_q2) - 1):
                if (pattern_q2[i] >= 0) != (pattern_q2[i+1] >= 0):
                    lead_changes += 1
            features['lead_changes'] = lead_changes
            features['max_lead'] = float(np.max(np.abs(pattern_q2)))
            
            # Rolling windows
            if len(pattern_q2) >= 3:
                features['roll_3'] = float(np.mean(pattern_q2[-3:]))
            else:
                features['roll_3'] = features['pattern_mean']
            
            if len(pattern_q2) >= 5:
                features['roll_5'] = float(np.mean(pattern_q2[-5:]))
            else:
                features['roll_5'] = features['pattern_mean']
            
            if len(pattern_q2) >= 10:
                features['roll_10'] = float(np.mean(pattern_q2[-10:]))
            else:
                features['roll_10'] = features['pattern_mean']
        else:
            features['pattern_mean'] = 0.0
            features['pattern_std'] = 0.0
            features['pattern_min'] = 0.0
            features['pattern_max'] = 0.0
            features['pattern_range'] = 0.0
            features['pattern_median'] = 0.0
            features['momentum'] = 0.0
            features['momentum_std'] = 0.0
            features['momentum_last'] = 0.0
            features['volatility'] = 0.0
            features['mad'] = 0.0
            features['lead_changes'] = 0
            features['max_lead'] = 0.0
            features['roll_3'] = 0.0
            features['roll_5'] = 0.0
            features['roll_10'] = 0.0
        
        # Target (final score differential)
        features['target'] = float(targets.get('diff_at_final', current_diff))
        
        features_list.append(features)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(games)} games")
    
    except Exception as e:
        failed += 1
        if failed < 5:
            print(f"  ⚠️  Failed game {idx}: {e}")

print()
print(f"✅ Extracted features from {len(features_list)} games")
print(f"   Failed: {failed} games")
print()

# Convert to DataFrame
print("[3/7] Creating DataFrame...")
df = pd.DataFrame(features_list)
df = df.replace([np.inf, -np.inf], [1e10, -1e10])
df = df.fillna(0)

print(f"✅ Shape: {df.shape}")
print(f"   Features: {df.shape[1] - 2}")
print()

# Sort chronologically (assuming game_id is chronological)
df = df.sort_values('game_id').reset_index(drop=True)
print("✅ Sorted chronologically")
print()

# Prepare data
print("[4/7] Preparing for training...")
feature_cols = [col for col in df.columns if col not in ['game_id', 'target']]
X = df[feature_cols].values
y = df['target'].values

# Chronological split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"   Features: {len(feature_cols)}")
print(f"   Train: {len(X_train)} games (80%)")
print(f"   Test: {len(X_test)} games (20%)")
print()

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try ElasticNet for better feature selection
print("[5/7] Running ElasticNetCV for optimal feature selection...")
alphas = np.logspace(-4, 0, 50)
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

elastic = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5, max_iter=10000, n_jobs=-1, random_state=42)
elastic.fit(X_train_scaled, y_train)

print(f"   Best alpha: {elastic.alpha_:.6f}")
print(f"   Best l1_ratio: {elastic.l1_ratio_:.3f}")

# Get selected features
coefs = np.abs(elastic.coef_)
selected_mask = coefs > 1e-6
selected_features = [feature_cols[i] for i in range(len(coefs)) if selected_mask[i]]

print(f"   Selected: {len(selected_features)} / {len(feature_cols)} features")
print()

if len(selected_features) > 0:
    print("Top selected features:")
    feature_importance = sorted(zip(feature_cols, coefs), key=lambda x: x[1], reverse=True)
    for i, (name, coef) in enumerate(feature_importance[:min(15, len(selected_features))], 1):
        if coef > 1e-6:
            print(f"   {i:2d}. {name:25s} {coef:10.6f}")
    print()
else:
    print("⚠️  No features selected, using all features")
    selected_features = feature_cols

# Train final Ridge model
print("[6/7] Training Ridge on selected features...")
selected_indices = [i for i, col in enumerate(feature_cols) if col in selected_features]
X_train_selected = X_train_scaled[:, selected_indices] if len(selected_indices) > 0 else X_train_scaled
X_test_selected = X_test_scaled[:, selected_indices] if len(selected_indices) > 0 else X_test_scaled

model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_selected, y_train)

train_pred = model.predict(X_train_selected)
test_pred = model.predict(X_test_selected)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
overfit = ((test_mae - train_mae) / train_mae * 100) if train_mae > 0 else 0

baseline_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
edge = ((baseline_mae - test_mae) / baseline_mae * 100) if baseline_mae > 0 else 0

print(f"   Train MAE: {train_mae:.3f}")
print(f"   Test MAE:  {test_mae:.3f}")
print(f"   Overfit:   {overfit:.1f}%")
print(f"   Baseline:  {baseline_mae:.3f}")
print(f"   Edge:      {edge:.1f}%")
print()

# Save
print("[7/7] Saving model...")
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
    'elastic_model': elastic,
    'num_games': len(df),
    'num_features': len(selected_features)
}

with open("helios/data/HELIOS_OPTIMAL_MODEL.pkl", 'wb') as f:
    pickle.dump(results, f)

print("✅ Saved to: helios/data/HELIOS_OPTIMAL_MODEL.pkl")
print()

print("=" * 100)
print("✅ HELIOS OPTIMAL MODEL COMPLETE!")
print("=" * 100)
print()
print(f"📊 FINAL PERFORMANCE:")
print(f"   Games: {len(df):,}")
print(f"   Features: {len(selected_features)}")
print(f"   Test MAE: {test_mae:.3f}")
print(f"   Overfit: {overfit:.1f}%")
print(f"   Edge: {edge:.1f}%")
print()

# Compare to baseline
baseline_comparison = 9.029
if test_mae < baseline_comparison:
    improvement = baseline_comparison - test_mae
    pct_improvement = improvement / baseline_comparison * 100
    print(f"🎉 IMPROVEMENT: {improvement:.3f} MAE better ({pct_improvement:.1f}% improvement)")
    print(f"   Baseline: {baseline_comparison:.3f}")
    print(f"   Helios:   {test_mae:.3f}")
elif test_mae > baseline_comparison + 0.5:
    print(f"⚠️  WORSE: {test_mae - baseline_comparison:.3f} MAE worse than baseline")
else:
    print(f"✅ COMPARABLE: Within 0.5 MAE of baseline ({baseline_comparison:.3f})")

print()
print("=" * 100)

