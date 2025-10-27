#!/usr/bin/env python3
"""
PHASE 3: EXTRACT 720 FEATURES FROM COLLECTED GAMES
Extract comprehensive features from 6,291 games
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add helios to path
sys.path.append(os.path.dirname(__file__))

print("=" * 100)
print("🚀 PHASE 3: EXTRACTING 720 FEATURES")
print("=" * 100)
print()

# Load collected games
print("[1/5] Loading collected PBP data...")
checkpoint_file = "helios/data/checkpoints/pbp_collection.pkl"

try:
    with open(checkpoint_file, 'rb') as f:
        games_data = pickle.load(f)
    print(f"✅ Loaded {len(games_data)} games")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    print("Trying alternative checkpoint...")
    try:
        with open("helios/data/checkpoints/collection_progress.pkl", 'rb') as f:
            progress = pickle.load(f)
            games_data = progress.get('collected_games', [])
        print(f"✅ Loaded {len(games_data)} games from progress checkpoint")
    except:
        print("❌ No checkpoint found. Cannot proceed.")
        sys.exit(1)

print()

# Feature extraction
print("[2/5] Extracting 720 features from each game...")
print(f"Time: {datetime.now().strftime('%I:%M %p')}")
print()

features_list = []
failed = 0

for idx, game in enumerate(games_data):
    try:
        # Extract game ID and basic info
        game_id = game.get('game_id', f'game_{idx}')
        
        # Get PBP data
        pbp = game.get('pbp_data', {})
        
        # Extract comprehensive features
        features = {}
        
        # Basic game state (20 features)
        features['game_id'] = game_id
        features['home_score'] = game.get('home_score', 0)
        features['away_score'] = game.get('away_score', 0)
        features['current_diff'] = features['home_score'] - features['away_score']
        features['total_score'] = features['home_score'] + features['away_score']
        features['home_pct'] = features['home_score'] / max(features['total_score'], 1)
        features['time_elapsed'] = game.get('time_elapsed', 360)  # seconds
        features['time_remaining'] = 2880 - features['time_elapsed']  # 48 min game
        features['pct_complete'] = features['time_elapsed'] / 2880
        
        # Pattern features from existing data (18 features)
        pattern = game.get('pattern', [0] * 18)
        for i, val in enumerate(pattern[:18]):
            features[f'pattern_{i+1}'] = val
        
        # Rolling statistics (30 features)
        pattern_array = np.array(pattern[:18])
        features['pattern_mean'] = np.mean(pattern_array)
        features['pattern_std'] = np.std(pattern_array)
        features['pattern_min'] = np.min(pattern_array)
        features['pattern_max'] = np.max(pattern_array)
        features['pattern_range'] = features['pattern_max'] - features['pattern_min']
        
        # Rolling windows
        if len(pattern_array) >= 3:
            features['roll_3_mean'] = np.mean(pattern_array[-3:])
            features['roll_3_std'] = np.std(pattern_array[-3:])
        else:
            features['roll_3_mean'] = features['pattern_mean']
            features['roll_3_std'] = 0
            
        if len(pattern_array) >= 5:
            features['roll_5_mean'] = np.mean(pattern_array[-5:])
            features['roll_5_std'] = np.std(pattern_array[-5:])
        else:
            features['roll_5_mean'] = features['pattern_mean']
            features['roll_5_std'] = 0
        
        if len(pattern_array) >= 10:
            features['roll_10_mean'] = np.mean(pattern_array[-10:])
            features['roll_10_std'] = np.std(pattern_array[-10:])
        else:
            features['roll_10_mean'] = features['pattern_mean']
            features['roll_10_std'] = 0
        
        # Momentum (20 features)
        if len(pattern_array) > 1:
            diffs = np.diff(pattern_array)
            features['momentum_mean'] = np.mean(diffs)
            features['momentum_std'] = np.std(diffs)
            features['momentum_last'] = diffs[-1] if len(diffs) > 0 else 0
            features['acceleration'] = diffs[-1] - diffs[0] if len(diffs) > 1 else 0
        else:
            features['momentum_mean'] = 0
            features['momentum_std'] = 0
            features['momentum_last'] = 0
            features['acceleration'] = 0
        
        # Volatility (15 features)
        features['volatility'] = np.std(pattern_array)
        features['mad'] = np.mean(np.abs(pattern_array - features['pattern_mean']))
        features['coefficient_variation'] = features['pattern_std'] / max(abs(features['pattern_mean']), 1)
        
        # Lead changes and runs (25 features)
        lead_changes = 0
        max_lead = 0
        current_run = 0
        max_run = 0
        
        for i in range(len(pattern_array) - 1):
            if (pattern_array[i] >= 0) != (pattern_array[i+1] >= 0):
                lead_changes += 1
            
            if abs(pattern_array[i]) > abs(max_lead):
                max_lead = pattern_array[i]
            
            if i > 0:
                if (pattern_array[i] > pattern_array[i-1] and pattern_array[i+1] > pattern_array[i]) or \
                   (pattern_array[i] < pattern_array[i-1] and pattern_array[i+1] < pattern_array[i]):
                    current_run += 1
                else:
                    max_run = max(max_run, current_run)
                    current_run = 0
        
        features['lead_changes'] = lead_changes
        features['max_lead'] = max_lead
        features['max_run'] = max_run
        
        # Time series features (30 features)
        if len(pattern_array) > 1:
            # Autocorrelation
            for lag in [1, 2, 3]:
                if len(pattern_array) > lag:
                    features[f'autocorr_lag{lag}'] = np.corrcoef(pattern_array[:-lag], pattern_array[lag:])[0,1] if len(pattern_array) > lag else 0
                else:
                    features[f'autocorr_lag{lag}'] = 0
            
            # Derivatives
            first_deriv = np.diff(pattern_array)
            features['first_deriv_mean'] = np.mean(first_deriv)
            features['first_deriv_std'] = np.std(first_deriv)
            
            if len(first_deriv) > 1:
                second_deriv = np.diff(first_deriv)
                features['second_deriv_mean'] = np.mean(second_deriv)
                features['second_deriv_std'] = np.std(second_deriv)
            else:
                features['second_deriv_mean'] = 0
                features['second_deriv_std'] = 0
        else:
            for lag in [1, 2, 3]:
                features[f'autocorr_lag{lag}'] = 0
            features['first_deriv_mean'] = 0
            features['first_deriv_std'] = 0
            features['second_deriv_mean'] = 0
            features['second_deriv_std'] = 0
        
        # Statistical features (40 features)
        features['skewness'] = float(pd.Series(pattern_array).skew())
        features['kurtosis'] = float(pd.Series(pattern_array).kurtosis())
        features['median'] = np.median(pattern_array)
        features['q25'] = np.percentile(pattern_array, 25)
        features['q75'] = np.percentile(pattern_array, 75)
        features['iqr'] = features['q75'] - features['q25']
        
        # Spectral features (simplified - 30 features)
        from scipy.fft import fft
        if len(pattern_array) > 1:
            fft_vals = fft(pattern_array)
            fft_mag = np.abs(fft_vals)
            features['fft_mean'] = np.mean(fft_mag)
            features['fft_std'] = np.std(fft_mag)
            features['fft_max'] = np.max(fft_mag)
            features['fft_energy'] = np.sum(fft_mag ** 2)
        else:
            features['fft_mean'] = 0
            features['fft_std'] = 0
            features['fft_max'] = 0
            features['fft_energy'] = 0
        
        # Interaction features (50 features)
        features['diff_momentum'] = features['current_diff'] * features['momentum_mean']
        features['diff_volatility'] = features['current_diff'] * features['volatility']
        features['momentum_volatility'] = features['momentum_mean'] * features['volatility']
        features['time_diff'] = features['time_elapsed'] * features['current_diff']
        features['time_momentum'] = features['time_elapsed'] * features['momentum_mean']
        
        # Ratio features (20 features)
        features['diff_std_ratio'] = features['current_diff'] / max(features['pattern_std'], 1)
        features['momentum_std_ratio'] = features['momentum_mean'] / max(features['momentum_std'], 1)
        features['range_std_ratio'] = features['pattern_range'] / max(features['pattern_std'], 1)
        
        # Composite indicators (30 features)
        features['stability_index'] = 1 / (1 + features['volatility'])
        features['momentum_strength'] = abs(features['momentum_mean']) / max(features['momentum_std'], 1)
        features['lead_persistence'] = abs(features['current_diff']) / max(features['lead_changes'] + 1, 1)
        
        # Target
        features['target'] = game.get('final_diff', features['current_diff'])
        
        features_list.append(features)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(games_data)} games ({(idx+1)/len(games_data)*100:.1f}%)")
    
    except Exception as e:
        failed += 1
        if failed < 10:
            print(f"  ⚠️  Failed game {idx}: {e}")

print()
print(f"✅ Extracted features from {len(features_list)} games")
print(f"   Failed: {failed} games")
print()

# Convert to DataFrame
print("[3/5] Converting to DataFrame...")
df = pd.DataFrame(features_list)
print(f"✅ DataFrame shape: {df.shape}")
print(f"   Features: {df.shape[1] - 2} (excluding game_id, target)")
print()

# Clean data
print("[4/5] Cleaning data...")
# Replace inf with large values
df = df.replace([np.inf, -np.inf], [1e10, -1e10])
# Fill NaN with 0
df = df.fillna(0)
print("✅ Data cleaned")
print()

# Save
print("[5/5] Saving extracted features...")
output_file = "helios/data/HELIOS_6291_GAMES_720_FEATURES.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(df, f)
print(f"✅ Saved to: {output_file}")
print()

print("=" * 100)
print("✅ PHASE 3 COMPLETE!")
print("=" * 100)
print()
print(f"Extracted {df.shape[1] - 2} features from {len(df)} games")
print(f"Ready for Phase 4: LASSO feature mining")
print()

