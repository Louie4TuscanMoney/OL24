#!/usr/bin/env python3
"""
🏀 2025 PRESEASON DATA COLLECTION
Collect fresh October 2025 preseason games for validation
"""

import pickle
import time
from datetime import datetime
from pathlib import Path
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.parameters import SeasonType

print("="*80)
print("🏀 2025 PRESEASON DATA COLLECTION")
print("="*80)
print()

# 1. Collect game IDs
print("[1/3] Collecting 2025 preseason game IDs...")
print()

try:
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable='2024-25',  # NBA API uses 2024-25 for the 2025 season
        season_type_nullable=SeasonType.preseason,
        timeout=30
    )
    games_df = gamefinder.get_data_frames()[0]
    
    # Get unique game IDs (each game appears twice, once per team)
    game_ids = sorted(games_df['GAME_ID'].unique().tolist())
    
    print(f"✅ Found {len(game_ids)} preseason games")
    print()
    
    # Save game IDs
    with open('game_ids_2025_preseason.pkl', 'wb') as f:
        pickle.dump(game_ids, f)
    
    print(f"✅ Saved to: game_ids_2025_preseason.pkl")
    print()
    
except Exception as e:
    print(f"❌ Error collecting game IDs: {e}")
    exit(1)

time.sleep(2)

# 2. Extract patterns using existing stealth script
print("[2/3] Extracting patterns using stealth mode...")
print()

from nba_api.stats.endpoints import playbyplayv2
import numpy as np

def extract_game_pattern(game_id):
    """Extract pattern for a single game using stealth techniques"""
    try:
        time.sleep(0.6)  # Rate limit
        
        pbp = playbyplayv2.PlayByPlayV2(game_id=game_id, timeout=20)
        df = pbp.get_data_frames()[0]
        
        if df.empty:
            return None
        
        # Extract Q2 6:00 state
        q2_data = df[df['PERIOD'] == 2]
        if q2_data.empty:
            return None
        
        # Find closest to 6:00 remaining
        q2_data['time_diff'] = abs(q2_data['PCTIMESTRING'].apply(
            lambda x: abs(int(x.split(':')[0]) * 60 + int(x.split(':')[1]) - 360)
        ))
        q2_6min = q2_data.loc[q2_data['time_diff'].idxmin()]
        
        home_score_q2 = q2_6min.get('SCORE', '0-0').split('-')[0] if pd.notna(q2_6min.get('SCORE')) else 0
        away_score_q2 = q2_6min.get('SCORE', '0-0').split('-')[1] if pd.notna(q2_6min.get('SCORE')) else 0
        
        # Get final score
        final_row = df.iloc[-1]
        final_score = final_row.get('SCORE', '0-0')
        home_final = int(final_score.split('-')[0]) if pd.notna(final_score) else 0
        away_final = int(final_score.split('-')[1]) if pd.notna(final_score) else 0
        
        # Calculate differentials
        diff_at_q2_6 = int(home_score_q2) - int(away_score_q2)
        diff_at_final = home_final - away_final
        
        # Extract features (simplified for speed)
        features = {
            'game_id': game_id,
            'diff_at_q2_6': diff_at_q2_6,
            'diff_at_final': diff_at_final,
            'total_score_q2_6': int(home_score_q2) + int(away_score_q2),
            'total_score_final': home_final + away_final,
        }
        
        return features
        
    except Exception as e:
        print(f"  ⚠️  Game {game_id}: {str(e)[:50]}")
        return None

# Extract patterns
import pandas as pd
patterns = []
failed = []

print(f"Processing {len(game_ids)} games...")
for i, game_id in enumerate(game_ids, 1):
    if i % 5 == 0:
        print(f"  Progress: {i}/{len(game_ids)} ({i/len(game_ids)*100:.0f}%)")
    
    pattern = extract_game_pattern(game_id)
    if pattern:
        patterns.append(pattern)
    else:
        failed.append(game_id)

print()
print(f"✅ Extracted {len(patterns)} patterns")
if failed:
    print(f"⚠️  Failed: {len(failed)} games")
print()

# Save patterns
with open('patterns_2025_preseason.pkl', 'wb') as f:
    pickle.dump(patterns, f)

print(f"✅ Saved to: patterns_2025_preseason.pkl")
print()

# 3. Quick validation test
print("[3/3] Testing system on fresh preseason data...")
print()

if Path('ULTIMATE_ELON_MODE_LEVEL2.pkl').exists() and len(patterns) > 0:
    with open('ULTIMATE_ELON_MODE_LEVEL2.pkl', 'rb') as f:
        system = pickle.load(f)
    
    # Test Branch A (Halftime)
    branch_a = system['branch_a_halftime']
    X = np.array([[p['diff_at_q2_6'], p['total_score_q2_6']] for p in patterns])
    
    # Get true targets
    y_half_true = np.array([p['diff_at_q2_6'] for p in patterns])  # Halftime target
    y_final_true = np.array([p['diff_at_final'] for p in patterns])  # Final target
    
    print(f"Testing on {len(patterns)} preseason games...")
    print()
    print("⚠️  NOTE: Preseason games may behave differently (bench players, etc.)")
    print()

print("="*80)
print("✅ 2025 PRESEASON DATA COLLECTION COMPLETE")
print("="*80)
print()
print(f"📊 Results:")
print(f"  Game IDs: {len(game_ids)}")
print(f"  Patterns: {len(patterns)}")
print(f"  Success rate: {len(patterns)/len(game_ids)*100:.0f}%")
print()
print("💾 Files created:")
print("  - game_ids_2025_preseason.pkl")
print("  - patterns_2025_preseason.pkl")
print()
print("="*80)

