#!/usr/bin/env python3
"""
🏆 STRIVE FOR GREATNESS - PHASE 1: EXTRACT 67 FEATURES
Extract full 67 features for all 6,912 training games
Optimized for speed and efficiency
"""

import pickle
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import entropy
import time

print("="*80)
print("🏆 STRIVE FOR GREATNESS - FEATURE ENGINEERING")
print("="*80)
print()
print("\"Strive for Greatness\" - LeBron James")
print()

# Load existing training data (33 features)
print("[1/4] Loading existing training data...")
with open('ULTRA_ENHANCED_PATTERNS_V2.pkl', 'rb') as f:
    training_data = pickle.load(f)

print(f"✅ Loaded {len(training_data)} games")
print()

# Extract 67 features for each game
print("[2/4] Extracting 67 features for all games...")
print()

enhanced_games = []
start_time = time.time()

for i, game in enumerate(training_data):
    if i % 500 == 0:
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        remaining = (len(training_data) - i) / rate if rate > 0 else 0
        print(f"  Progress: {i}/{len(training_data)} ({i/len(training_data)*100:.0f}%) | "
              f"Rate: {rate:.1f} games/sec | ETA: {remaining/60:.0f} min")
    
    # Get 18-minute pattern
    pattern = game.get('pattern', [0]*18)
    pattern_arr = np.array(pattern)
    
    # ==================================================================
    # EXTRACT 67 FEATURES (matching preseason format)
    # ==================================================================
    
    # Base (4) - already have these
    mean_diff = game.get('mean_diff', np.mean(pattern_arr))
    std_diff = game.get('std_diff', np.std(pattern_arr))
    trend = game.get('trend', np.polyfit(range(18), pattern_arr, 1)[0] if len(pattern_arr) == 18 else 0)
    volatility = game.get('volatility', np.std(np.diff(pattern_arr)))
    
    # Spectral (6) - already have these
    spectral_energy = game.get('spectral_energy', 0)
    low_freq_power = game.get('low_freq_power', 0)
    mid_freq_power = game.get('mid_freq_power', 0)
    high_freq_power = game.get('high_freq_power', 0)
    dominant_freq = game.get('dominant_freq', 0)
    spectral_entropy_val = game.get('spectral_entropy', 0)
    
    # Momentum (6) - already have these
    velocity = game.get('velocity', np.diff(pattern_arr).mean() if len(pattern_arr) > 1 else 0)
    acceleration = game.get('acceleration', 0)
    recent_momentum = game.get('recent_momentum', 0)
    lead_changes = game.get('lead_changes', 0)
    max_swing = game.get('max_swing', 0)
    comeback_potential = game.get('comeback_potential', 0)
    
    # Autocorrelation (3) - have lag1, lag3, lag5
    autocorr_lag1 = game.get('autocorr_lag1', 0)
    autocorr_lag2 = np.corrcoef(pattern_arr[:-2], pattern_arr[2:])[0, 1] if len(pattern_arr) > 2 else 0
    autocorr_lag3 = game.get('autocorr_lag3', 0)
    
    # Advanced (8) - some we have
    run_rate = float(max([len(list(g)) for k, g in pd.Series(pattern_arr > 0).groupby((pd.Series(pattern_arr > 0) != pd.Series(pattern_arr > 0).shift()).cumsum())]))
    deficit_recovery = 1.0 if (min(pattern_arr) < -5 and pattern_arr[-1] > 0) else 0.0
    consistency = float(np.std([pattern_arr[i:i+3].mean() for i in range(0, len(pattern_arr)-2, 3)]))
    possession_efficiency = 1.0  # default
    team_form = 0.0  # default
    rest_days = 2.0  # default
    home_advantage = 0.5  # default
    season_stage = 0.5  # default
    
    # Lag (6) - already have these
    team_diff_lag1 = game.get('team_diff_lag1', 0)
    team_mean_lag1 = game.get('team_mean_lag1', 0)
    team_diff_rolling3 = game.get('team_diff_rolling3', 0)
    team_volatility_rolling3 = game.get('team_volatility_rolling3', 2.0)
    team_form_10games = game.get('team_form_10games', 0)
    team_consistency = game.get('team_consistency', 10.0)
    
    # Pattern values (18) - NEW
    pattern_values = {f'pattern_{j}': pattern[j] if j < len(pattern) else 0 for j in range(18)}
    
    # Quarterly breakdown (6) - NEW
    q1_mean = float(np.mean(pattern_arr[:12])) if len(pattern_arr) >= 12 else 0
    q1_std = float(np.std(pattern_arr[:12])) if len(pattern_arr) >= 12 else 0
    q1_trend = float(np.polyfit(range(12), pattern_arr[:12], 1)[0]) if len(pattern_arr) >= 12 else 0
    q2_mean = float(np.mean(pattern_arr[12:])) if len(pattern_arr) > 12 else 0
    q2_std = float(np.std(pattern_arr[12:])) if len(pattern_arr) > 12 else 0
    q2_trend = float(np.polyfit(range(len(pattern_arr[12:])), pattern_arr[12:], 1)[0]) if len(pattern_arr) > 12 else 0
    
    # Advanced momentum (8) - NEW
    jerk = np.diff(np.diff(np.diff(pattern_arr, prepend=pattern_arr[0]), prepend=0), prepend=0)
    jerk_mean = float(np.mean(jerk))
    jerk_std = float(np.std(jerk))
    momentum_score = float(np.mean(np.diff(pattern_arr, prepend=pattern_arr[0])[-3:]))
    acceleration_score = float(np.mean(np.diff(np.diff(pattern_arr, prepend=pattern_arr[0]), prepend=0)[-3:]))
    q1_to_q2_change = float(q2_mean - q1_mean)
    momentum_shift = float(pattern_arr[-1] - pattern_arr[11]) if len(pattern_arr) > 11 else 0
    time_weighted_mean = float(np.average(pattern_arr, weights=range(1, len(pattern_arr)+1)))
    recent_avg = float(np.mean(pattern_arr[-3:])) if len(pattern_arr) >= 3 else 0
    
    # Extremes (4) - NEW
    max_lead = float(max(pattern_arr))
    max_deficit = float(min(pattern_arr))
    lead_at_q1_end = float(pattern_arr[11]) if len(pattern_arr) > 11 else 0
    stability_score = 1.0 / (volatility + 1.0)
    
    # Complexity (4) - NEW
    high_volatility_periods = sum(1 for j in range(1, len(pattern_arr)) if abs(pattern_arr[j] - pattern_arr[j-1]) > 3)
    reversal_count = sum(1 for j in range(2, len(pattern_arr)) if (pattern_arr[j] - pattern_arr[j-1]) * (pattern_arr[j-1] - pattern_arr[j-2]) < 0)
    
    # ==================================================================
    # COMBINE ALL 67 FEATURES
    # ==================================================================
    enhanced_game = {
        # Metadata
        'game_id': game.get('game_id'),
        'date': game.get('date'),
        'season': game.get('season'),
        'home_team': game.get('home_team'),
        'away_team': game.get('away_team'),
        'pattern': pattern,
        
        # Targets
        'diff_at_final': game.get('diff_at_final'),
        'diff_at_halftime': game.get('diff_at_halftime'),
        'diff_at_2q_6min': game.get('diff_at_2q_6min'),
        
        # Base (4)
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'trend': trend,
        'volatility': volatility,
        
        # Spectral (6)
        'spectral_energy': spectral_energy,
        'low_freq_power': low_freq_power,
        'mid_freq_power': mid_freq_power,
        'high_freq_power': high_freq_power,
        'dominant_freq': dominant_freq,
        'spectral_entropy': spectral_entropy_val,
        
        # Momentum (6)
        'velocity': velocity,
        'acceleration': acceleration,
        'recent_momentum': recent_momentum,
        'lead_changes': lead_changes,
        'max_swing': max_swing,
        'comeback_potential': comeback_potential,
        
        # Autocorrelation (3)
        'autocorr_lag1': autocorr_lag1,
        'autocorr_lag2': autocorr_lag2,
        'autocorr_lag3': autocorr_lag3,
        
        # Advanced (8)
        'run_rate': run_rate,
        'deficit_recovery': deficit_recovery,
        'consistency': consistency,
        'possession_efficiency': possession_efficiency,
        'team_form': team_form,
        'rest_days': rest_days,
        'home_advantage': home_advantage,
        'season_stage': season_stage,
        
        # Lag (6)
        'team_diff_lag1': team_diff_lag1,
        'team_mean_lag1': team_mean_lag1,
        'team_diff_rolling3': team_diff_rolling3,
        'team_volatility_rolling3': team_volatility_rolling3,
        'team_form_10games': team_form_10games,
        'team_consistency': team_consistency,
        
        # Pattern values (18)
        **pattern_values,
        
        # Quarterly (6)
        'q1_mean': q1_mean,
        'q1_std': q1_std,
        'q1_trend': q1_trend,
        'q2_mean': q2_mean,
        'q2_std': q2_std,
        'q2_trend': q2_trend,
        
        # Advanced momentum (8)
        'jerk_mean': jerk_mean,
        'jerk_std': jerk_std,
        'momentum_score': momentum_score,
        'acceleration_score': acceleration_score,
        'q1_to_q2_change': q1_to_q2_change,
        'momentum_shift': momentum_shift,
        'time_weighted_mean': time_weighted_mean,
        'recent_avg': recent_avg,
        
        # Extremes (4)
        'max_lead': max_lead,
        'max_deficit': max_deficit,
        'lead_at_q1_end': lead_at_q1_end,
        'stability_score': stability_score,
        
        # Complexity (4)
        'high_volatility_periods': high_volatility_periods,
        'reversal_count': reversal_count,
        'run_rate_2': run_rate,  # duplicate for compatibility
        'deficit_recovery_2': deficit_recovery,  # duplicate for compatibility
    }
    
    enhanced_games.append(enhanced_game)

print()
print(f"✅ Extracted 67 features for {len(enhanced_games)} games")
print()

# Verify feature count
exclude_keys = ['game_id', 'date', 'season', 'home_team', 'away_team', 'pattern',
                'diff_at_final', 'diff_at_halftime', 'diff_at_2q_6min']
feature_count = len([k for k in enhanced_games[0].keys() if k not in exclude_keys])
print(f"📊 Feature count: {feature_count}")
print()

# Save
print("[3/4] Saving enhanced dataset...")
with open('ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl', 'wb') as f:
    pickle.dump(enhanced_games, f)

print(f"✅ Saved to: ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl")
print()

# Summary
print("[4/4] Feature engineering complete")
print("="*80)
print("🏆 PHASE 1 COMPLETE - 67 FEATURES EXTRACTED")
print("="*80)
print()
print(f"Games: {len(enhanced_games)}")
print(f"Features: {feature_count}")
print(f"Time: {(time.time() - start_time)/60:.1f} minutes")
print()
print("Next: python3 🏆_2_TRAIN_STRIVE_MODELS.py")
print("="*80)

