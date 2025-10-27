#!/usr/bin/env python3
"""
⚡ SIMPLIFIED XGBOOST TRAINING
Train on features that actually exist in our data
"""

import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

print("="*60)
print("⚡ SIMPLIFIED XGBOOST TRAINING")
print("="*60)
print()

# Load enhanced patterns
print("[1/5] Loading enhanced patterns...")
with open('ENHANCED_PATTERNS_FULL.pkl', 'rb') as f:
    patterns = pickle.load(f)
print(f"✅ Loaded {len(patterns)} games")
print()

# Extract features and targets
print("[2/5] Converting to training matrices...")
X = []
y = []

for p in patterns:
    if p.get('diff_at_final') is None:
        continue
    
    features = []
    
    # 1. PBP pattern (18 values)
    features.extend(p['pattern'])
    
    # 2. Statistical features (4)
    stat = p['pattern_statistical']
    features.extend([
        stat['mean'],
        stat['std'],
        stat['trend'],
        stat['volatility']
    ])
    
    # 3. Quality (1)
    qual = p['quality_metrics']
    quality_val = 1.0 if qual['quality_grade'] == 'A' else (0.5 if qual['quality_grade'] == 'B' else 0.0)
    features.append(quality_val)
    
    # 4. Team features (6 or more, use what we have)
    team = p.get('team_features', {})
    features.extend([
        team.get('home_off_rating', 110.0),
        team.get('home_def_rating', 110.0),
        team.get('away_off_rating', 110.0),
        team.get('away_def_rating', 110.0),
        team.get('home_win_pct', 0.5),
        team.get('away_win_pct', 0.5)
    ])
    
    # 5. Player features (6 or more, use what we have)
    player = p.get('player_features', {})
    features.extend([
        player.get('home_star_count', 0),
        player.get('away_star_count', 0),
        player.get('home_avg_tier', 3.0),
        player.get('away_avg_tier', 3.0),
        player.get('home_depth', 0),
        player.get('away_depth', 0)
    ])
    
    X.append(features)
    y.append(p['diff_at_final'])

X = np.array(X)
y = np.array(y)

print(f"✅ Training matrix: {X.shape[0]} games × {X.shape[1]} features")
print()

# Split train/test
print("[3/5] Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Train: {len(X_train)} games")
print(f"✅ Test: {len(X_test)} games")
print()

# Train XGBoost
print("[4/5] Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ Training complete")
print()

# Evaluate
print("[5/5] Evaluating...")
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"📊 Training MAE: {train_mae:.2f}")
print(f"📊 Test MAE: {test_mae:.2f}")
print()

if test_mae < train_mae * 1.3:
    print("✅ Model is not severely overfitting")
else:
    print("⚠️  Possible overfitting detected")
print()

# Save model
print("Saving model...")
model.save_model('xgboost_simple_v1.json')
print("✅ Saved to: xgboost_simple_v1.json")
print()

print("="*60)
print(f"✅ XGBOOST TRAINING COMPLETE")
print(f"   Test MAE: {test_mae:.2f}")
print(f"   Features: {X.shape[1]}")
print("="*60)

