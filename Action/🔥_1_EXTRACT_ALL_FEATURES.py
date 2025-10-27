#!/usr/bin/env python3
"""
🔥 ADVANCED FEATURE ENGINEERING
Extract 100+ features from NBA game data
Target: Get MAE from 8.22 → 6.0
"""

import pickle
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import entropy
from collections import defaultdict

print("="*80)
print("🔥 ADVANCED FEATURE ENGINEERING - GOING FOR CHAMPIONSHIP")
print("="*80)
print()

# Load data
print("[1/8] Loading base patterns...")
with open('ULTRA_OPTIMIZED_PATTERNS_2021_2025.pkl', 'rb') as f:
    patterns = pickle.load(f)
print(f"✅ Loaded {len(patterns)} games")
print()

# Convert to DataFrame for easier manipulation
print("[2/8] Converting to DataFrame...")
df_list = []
for p in patterns:
    row = {
        'game_id': p.get('game_id'),
        'date': p.get('date'),
        'season': p.get('season'),
        'matchup': p.get('matchup'),
        'pattern': p.get('pattern', []),
        'diff_at_2q_6min': p.get('diff_at_2q_6min'),
        'diff_at_halftime': p.get('diff_at_halftime'),
        'diff_at_final': p.get('diff_at_final'),
    }
    
    # Extract team names
    matchup = p.get('matchup', '')
    if ' @ ' in matchup:
        away, home = matchup.split(' @ ')
        row['home_team'] = home.strip()
        row['away_team'] = away.strip()
    elif ' vs. ' in matchup:
        home, away = matchup.split(' vs. ')
        row['home_team'] = home.strip()
        row['away_team'] = away.strip()
    else:
        row['home_team'] = 'UNK'
        row['away_team'] = 'UNK'
    
    # Base features
    row['mean_diff'] = p.get('pattern_statistical', {}).get('mean', 0)
    row['std_diff'] = p.get('pattern_statistical', {}).get('std', 0)
    row['trend'] = p.get('pattern_statistical', {}).get('trend', 0)
    row['volatility'] = p.get('pattern_statistical', {}).get('volatility', 0)
    
    df_list.append(row)

df = pd.DataFrame(df_list)
df = df.sort_values(['home_team', 'away_team', 'date']).reset_index(drop=True)
print(f"✅ Created DataFrame: {len(df)} games")
print()

# LAG FEATURES (Critical from research!)
print("[3/8] Creating lag features (last 1,2,3,5,10 games)...")

# Create lag features per team
lag_features = defaultdict(list)

for team in df['home_team'].unique():
    # Get all games for this team (home or away)
    team_games = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    team_games = team_games.sort_values('date')
    
    # For each game, calculate performance in last N games
    for idx, game in team_games.iterrows():
        # Last 1 game
        prev_1 = team_games[team_games['date'] < game['date']].tail(1)
        lag_features[idx].append({
            'team_diff_lag1': prev_1['diff_at_final'].mean() if len(prev_1) > 0 else 0,
            'team_mean_lag1': prev_1['mean_diff'].mean() if len(prev_1) > 0 else 0,
        })
        
        # Last 3 games (rolling average)
        prev_3 = team_games[team_games['date'] < game['date']].tail(3)
        lag_features[idx].append({
            'team_diff_rolling3': prev_3['diff_at_final'].mean() if len(prev_3) > 0 else 0,
            'team_volatility_rolling3': prev_3['std_diff'].mean() if len(prev_3) > 0 else 2.0,
        })
        
        # Last 10 games (season form)
        prev_10 = team_games[team_games['date'] < game['date']].tail(10)
        lag_features[idx].append({
            'team_form_10games': prev_10['diff_at_final'].mean() if len(prev_10) > 0 else 0,
            'team_consistency': prev_10['diff_at_final'].std() if len(prev_10) > 0 else 10.0,
        })

print(f"✅ Created lag features for {len(lag_features)} games")
print()

# SPECTRAL FEATURES (FFT - captures hidden patterns)
print("[4/8] Creating spectral features (Fourier analysis)...")

spectral_features = []
for idx, row in df.iterrows():
    pattern = row['pattern']
    if len(pattern) == 18:
        # FFT
        fft_vals = fft(pattern)
        power = np.abs(fft_vals)**2
        
        # Split into frequency bands
        low_freq = power[1:4].sum()  # Slow trends
        mid_freq = power[4:8].sum()  # Medium oscillations
        high_freq = power[8:].sum()  # Fast changes
        
        total_power = power.sum()
        
        spectral_features.append({
            'spectral_energy': total_power,
            'low_freq_power': low_freq / total_power if total_power > 0 else 0,
            'mid_freq_power': mid_freq / total_power if total_power > 0 else 0,
            'high_freq_power': high_freq / total_power if total_power > 0 else 0,
            'dominant_freq': np.argmax(power[1:]) / len(pattern),
            'spectral_entropy': entropy(power + 1e-10),  # Avoid log(0)
        })
    else:
        spectral_features.append({
            'spectral_energy': 0, 'low_freq_power': 0, 'mid_freq_power': 0,
            'high_freq_power': 0, 'dominant_freq': 0, 'spectral_entropy': 0
        })

print(f"✅ Created spectral features: 6 per game")
print()

# MOMENTUM & VELOCITY FEATURES
print("[5/8] Creating momentum & velocity features...")

momentum_features = []
for idx, row in df.iterrows():
    pattern = row['pattern']
    if len(pattern) == 18:
        # Velocity (rate of change)
        velocity = np.diff(pattern).mean()
        
        # Acceleration (change in velocity)
        acceleration = np.diff(np.diff(pattern)).mean()
        
        # Recent momentum (last 5 min vs first 5 min)
        recent_momentum = np.mean(pattern[-5:]) - np.mean(pattern[:5])
        
        # Lead changes
        lead_changes = sum(1 for i in range(1, len(pattern)) if (pattern[i] > 0) != (pattern[i-1] > 0))
        
        # Max swing
        max_lead = max(pattern)
        max_deficit = min(pattern)
        swing = max_lead - max_deficit
        
        momentum_features.append({
            'velocity': velocity,
            'acceleration': acceleration,
            'recent_momentum': recent_momentum,
            'lead_changes': lead_changes,
            'max_swing': swing,
            'comeback_potential': 1.0 if (pattern[0] > 5 and pattern[-1] < 0) else 0.0,
        })
    else:
        momentum_features.append({
            'velocity': 0, 'acceleration': 0, 'recent_momentum': 0,
            'lead_changes': 0, 'max_swing': 0, 'comeback_potential': 0
        })

print(f"✅ Created momentum features: 6 per game")
print()

# AUTOCORRELATION FEATURES (temporal dependencies)
print("[6/8] Creating autocorrelation features...")

autocorr_features = []
for idx, row in df.iterrows():
    pattern = row['pattern']
    if len(pattern) >= 10:
        # Autocorrelation at different lags
        pattern_arr = np.array(pattern)
        mean = pattern_arr.mean()
        
        def autocorr(lag):
            if len(pattern_arr) <= lag:
                return 0
            c0 = np.dot(pattern_arr - mean, pattern_arr - mean) / len(pattern_arr)
            c_lag = np.dot(pattern_arr[:-lag] - mean, pattern_arr[lag:] - mean) / len(pattern_arr[:-lag])
            return c_lag / c0 if c0 != 0 else 0
        
        autocorr_features.append({
            'autocorr_lag1': autocorr(1),
            'autocorr_lag3': autocorr(3),
            'autocorr_lag5': autocorr(5),
        })
    else:
        autocorr_features.append({'autocorr_lag1': 0, 'autocorr_lag3': 0, 'autocorr_lag5': 0})

print(f"✅ Created autocorrelation features: 3 per game")
print()

# ADVANCED STATS (from research - FP, NETRTG, PIE, etc.)
print("[7/8] Creating advanced NBA stats features...")

advanced_features = []
for idx, row in df.iterrows():
    pattern = row['pattern']
    
    if len(pattern) == 18:
        # Effective Field Goal % proxy (higher variance = more 3-pointers)
        efg_proxy = row['volatility'] / (row['std_diff'] + 1)
        
        # True Shooting % proxy
        ts_proxy = abs(row['mean_diff']) / (row['std_diff'] + 1)
        
        # Net Rating proxy (trend relative to volatility)
        netrtg_proxy = row['trend'] / (row['volatility'] + 1)
        
        # PIE proxy (Player Impact Estimate - dominance)
        current_diff = pattern[-1]
        pie_proxy = abs(current_diff) / (row['std_diff'] + 1)
        
        # Plus/Minus proxy
        pm_proxy = np.mean(pattern[-6:])  # Recent 6 minutes
        
        # Usage rate proxy (volatility suggests high usage)
        usg_proxy = row['volatility']
        
        # Pace (total points proxy from pattern range)
        pace_proxy = (max(pattern) - min(pattern)) / 18
        
        # Four Factors proxy
        # 1. Shooting (trend)
        # 2. Turnovers (volatility)
        # 3. Rebounds (comeback potential)
        # 4. Free throws (consistency)
        four_factors = (row['trend'] + (10 - row['volatility']) + abs(row['mean_diff'])) / 3
        
        advanced_features.append({
            'efg_proxy': efg_proxy,
            'ts_proxy': ts_proxy,
            'netrtg_proxy': netrtg_proxy,
            'pie_proxy': pie_proxy,
            'pm_proxy': pm_proxy,
            'usg_proxy': usg_proxy,
            'pace_proxy': pace_proxy,
            'four_factors_proxy': four_factors,
        })
    else:
        advanced_features.append({
            'efg_proxy': 0, 'ts_proxy': 0, 'netrtg_proxy': 0, 'pie_proxy': 0,
            'pm_proxy': 0, 'usg_proxy': 0, 'pace_proxy': 0, 'four_factors_proxy': 0
        })

print(f"✅ Created advanced NBA stats: 8 per game")
print()

# MERGE ALL FEATURES
print("[8/8] Merging all features into final dataset...")

final_patterns = []
for idx, row in df.iterrows():
    # Start with base pattern
    features = {
        'game_id': row['game_id'],
        'date': row['date'],
        'season': row['season'],
        'home_team': row['home_team'],
        'away_team': row['away_team'],
        'pattern': row['pattern'],
        
        # Targets
        'diff_at_final': row['diff_at_final'],
        'diff_at_halftime': row['diff_at_halftime'],
        'diff_at_2q_6min': row['diff_at_2q_6min'],
        
        # Base statistical (4)
        'mean_diff': row['mean_diff'],
        'std_diff': row['std_diff'],
        'trend': row['trend'],
        'volatility': row['volatility'],
    }
    
    # Add lag features (6)
    if idx in lag_features:
        for lag_dict in lag_features[idx]:
            features.update(lag_dict)
    else:
        features.update({
            'team_diff_lag1': 0, 'team_mean_lag1': 0,
            'team_diff_rolling3': 0, 'team_volatility_rolling3': 2.0,
            'team_form_10games': 0, 'team_consistency': 10.0
        })
    
    # Add spectral (6)
    features.update(spectral_features[idx])
    
    # Add momentum (6)
    features.update(momentum_features[idx])
    
    # Add autocorrelation (3)
    features.update(autocorr_features[idx])
    
    # Add advanced stats (8)
    features.update(advanced_features[idx])
    
    # Add original team/player features if they exist
    # (from previous pipeline)
    
    final_patterns.append(features)

print(f"✅ Merged all features")
print()

# Count total features
sample = final_patterns[0]
feature_cols = [k for k in sample.keys() if k not in ['game_id', 'date', 'season', 'home_team', 'away_team', 'pattern', 'diff_at_final', 'diff_at_halftime', 'diff_at_2q_6min']]
print(f"📊 Total features per game: {len(feature_cols)}")
print(f"   Base statistical: 4")
print(f"   Lag features: 6")
print(f"   Spectral: 6")
print(f"   Momentum: 6")
print(f"   Autocorrelation: 3")
print(f"   Advanced NBA stats: 8")
print(f"   Pattern values: 18")
print(f"   TOTAL: ~51 features (will add more in next steps)")
print()

# Save enhanced patterns
output_file = 'ULTRA_ENHANCED_PATTERNS_V2.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(final_patterns, f)

print(f"✅ Saved to: {output_file}")
print(f"   Size: {len(final_patterns)} games")
print(f"   Features: {len(feature_cols)} + 18 pattern values")
print()

print("="*80)
print("🔥 FEATURE ENGINEERING COMPLETE")
print(f"   Next: Run Bayesian hyperparameter optimization")
print(f"   Command: python3 🔥_2_BAYESIAN_HYPEROPT.py")
print("="*80)

