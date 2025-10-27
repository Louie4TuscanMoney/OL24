#!/usr/bin/env python3
"""
⚡ TRAIN XGBOOST - RAPID VERSION
Train XGBoost on enhanced features
No hyperparameter tuning (fail forward, optimize later)
Time: 45 min
"""

import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("="*60)
print("⚡ RAPID XGBOOST TRAINING")
print("="*60)

# Load enhanced patterns
print("\n[1/5] Loading enhanced patterns...")
with open('ENHANCED_PATTERNS_FULL.pkl', 'rb') as f:
    patterns = pickle.load(f)

print(f"✅ Loaded {len(patterns)} games")

# Convert to training data
print("\n[2/5] Converting to training matrices...")

X = []
y = []
feature_names = []

for p in patterns:
    if p.get('diff_at_final') is None:
        continue
    
    features = []
    
    # PBP temporal (18)
    features.extend(p['pattern'])
    
    # PBP statistical (13)
    stat = p['pattern_statistical']
    features.extend([
        stat['mean'], stat['std'], stat['skewness'], stat['kurtosis'],
        stat['velocity'], stat['acceleration'], stat['trend'], stat['volatility'],
        stat['entropy'], stat['autocorr'], stat['variance_ratio'],
        stat['abs_max'], stat['range']
    ])
    
    # PBP spectral (4)
    spec = p['pattern_spectral']
    features.extend([
        spec['low_freq_power'], spec['mid_freq_power'],
        spec['high_freq_power'], spec['freq_concentration']
    ])
    
    # PBP betting (12)
    bet = p['pattern_betting']
    features.extend([
        bet['blowout_risk'], bet['comeback_potential'], bet['pattern_stability'],
        bet['momentum_strength'], bet['expected_range_low'], bet['expected_range_high'],
        bet['expected_range_width'], bet['pattern_quality'], bet['data_reliability'],
        bet['betting_confidence'], 1 if bet['variance_opportunity'] else 0,
        1 if bet['momentum_opportunity'] else 0
    ])
    
    # Team features (11)
    team = p['team_features']
    features.extend([
        team['home_off_rating'], team['home_def_rating'], team['home_net_rating'],
        team['home_pace'], team['away_off_rating'], team['away_def_rating'],
        team['away_net_rating'], team['away_pace'],
        team['net_rating_diff'], team['pace_avg'], team['off_vs_def']
    ])
    
    # Player features (6)
    player = p['player_features']
    features.extend([
        player['home_star_tier'], player['away_star_tier'], player['star_power_diff'],
        player['has_superstar_home'], player['has_superstar_away'], player['total_star_power']
    ])
    
    X.append(features)
    y.append(p['diff_at_final'])

X = np.array(X)
y = np.array(y)

print(f"✅ Created training matrix: {X.shape}")
print(f"   Samples: {len(X):,}")
print(f"   Features: {X.shape[1]}")

# Temporal split (no shuffle - preserve time order)
print("\n[3/5] Creating temporal train/val split...")

split_idx = int(len(X) * 0.8)

X_train = X[:split_idx]
X_val = X[split_idx:]
y_train = y[:split_idx]
y_val = y[split_idx:]

print(f"✅ Train: {len(X_train):,} games")
print(f"✅ Val: {len(X_val):,} games")

# Scale features
print("\n[4/5] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save scaler
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Features scaled")

# Train XGBoost
print("\n[5/5] Training XGBoost...")
print("   Using default hyperparameters (no tuning for speed)")

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0
)

model.fit(X_train_scaled, y_train, verbose=False)

print(f"✅ Training complete")

# Evaluate
print("\n📊 Evaluation:")

y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)

train_mae = np.mean(np.abs(y_train - y_train_pred))
val_mae = np.mean(np.abs(y_val - y_val_pred))

print(f"\n   Train MAE: {train_mae:.2f}")
print(f"   Val MAE: {val_mae:.2f}")
print(f"   Gap: {val_mae - train_mae:.2f} (overfitting indicator)")

# Compare to baseline
baseline_mae = 10.75
improvement = (baseline_mae - val_mae) / baseline_mae * 100

print(f"\n   Baseline (Dejavu): 10.75")
print(f"   XGBoost: {val_mae:.2f}")
print(f"   Improvement: {improvement:.1f}%")

if val_mae < baseline_mae:
    print(f"   ✅ XGBoost is better!")
elif val_mae < baseline_mae * 1.1:
    print(f"   ⚠️  XGBoost is similar (within 10%)")
else:
    print(f"   ❌ XGBoost is worse (overfitting?)")

# Feature importance
print(f"\n🔍 Top 10 Most Important Features:")
feature_names_list = (
    [f'temporal_{i}' for i in range(18)] +
    ['stat_mean', 'stat_std', 'stat_skew', 'stat_kurt', 'stat_vel', 'stat_acc',
     'stat_trend', 'stat_vol', 'stat_entropy', 'stat_autocorr', 'stat_var_ratio',
     'stat_abs_max', 'stat_range'] +
    ['spec_low', 'spec_mid', 'spec_high', 'spec_conc'] +
    ['bet_blowout', 'bet_comeback', 'bet_stability', 'bet_momentum',
     'bet_range_low', 'bet_range_high', 'bet_range_width', 'bet_quality',
     'bet_reliability', 'bet_confidence', 'bet_var_opp', 'bet_mom_opp'] +
    ['team_home_off', 'team_home_def', 'team_home_net', 'team_home_pace',
     'team_away_off', 'team_away_def', 'team_away_net', 'team_away_pace',
     'team_net_diff', 'team_pace_avg', 'team_off_vs_def'] +
    ['player_home_tier', 'player_away_tier', 'player_diff',
     'player_super_home', 'player_super_away', 'player_total']
)

importances = model.feature_importances_
top_10_idx = np.argsort(importances)[-10:]

for idx in reversed(top_10_idx):
    print(f"   {feature_names_list[idx]}: {importances[idx]:.4f}")

# Save model
model.save_model('xgboost_enhanced_v1.json')

print(f"\n✅ Model saved: xgboost_enhanced_v1.json")

print(f"\n🚀 Next: python3 ⚡_5_update_game_engine.py")

print("="*60)

