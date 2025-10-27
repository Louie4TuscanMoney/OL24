#!/usr/bin/env python3
"""
PHASE 5: TRAIN FINAL MODEL ON HELIOS DATA
Train model on 6,691 games with 75 features
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime

print("=" * 100)
print("🔥 PHASE 5: TRAIN FINAL MODEL")
print("=" * 100)
print()

# Load data
print("[1/5] Loading data...")
with open("helios/data/HELIOS_6291_GAMES_720_FEATURES.pkl", 'rb') as f:
    df = pickle.load(f)

print(f"✅ Loaded {len(df)} games with {df.shape[1]-2} features")
print()

# Prepare data
print("[2/5] Preparing data...")
feature_cols = [col for col in df.columns if col not in ['game_id', 'target']]
X = df[feature_cols].values
y = df['target'].values

# Chronological split (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"   Features: {len(feature_cols)}")
print(f"   Train: {len(X_train)} games")
print(f"   Test: {len(X_test)} games")
print()

# Scale
print("[3/5] Scaling features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Scaled")
print()

# Train
print("[4/5] Training Ridge model...")
print(f"Time: {datetime.now().strftime('%I:%M %p')}")
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_scaled, y_train)
print("✅ Trained")
print()

# Evaluate
print("[5/5] Evaluating...")
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"   Train MAE: {train_mae:.3f}")
print(f"   Test MAE:  {test_mae:.3f}")
print(f"   Overfit:   {(test_mae - train_mae) / train_mae * 100:.1f}%")
print()

# Compare to baseline (always predict 0)
baseline_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
edge = (baseline_mae - test_mae) / baseline_mae * 100
print(f"   Baseline MAE: {baseline_mae:.3f}")
print(f"   Edge: {edge:.1f}%")
print()

# Save
print("Saving model...")
results = {
    'model': model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'baseline_mae': baseline_mae,
    'edge_pct': edge
}

with open("helios/data/HELIOS_FINAL_MODEL.pkl", 'wb') as f:
    pickle.dump(results, f)

print("✅ Saved to: helios/data/HELIOS_FINAL_MODEL.pkl")
print()

print("=" * 100)
print("✅ PHASE 5 COMPLETE!")
print("=" * 100)
print()
print(f"Final model trained on 6,691 games")
print(f"Test MAE: {test_mae:.3f}")
print(f"Edge: {edge:.1f}%")
print()

