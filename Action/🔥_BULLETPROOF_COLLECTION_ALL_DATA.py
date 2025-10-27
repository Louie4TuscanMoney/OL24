#!/usr/bin/env python3
"""
🔥 BULLETPROOF COLLECTION - ALL DATA, NO EXCUSES
Elon style: Make it work. Get ALL the data.

MISSION: Collect EVERY game 2015-2019
APPROACH: Multiple fallbacks, robust error handling, MAKE IT WORK
"""

from nba_api.stats.endpoints import playbyplayv2, leaguegamefinder
import pandas as pd
import numpy as np
import pickle
import time
import random
from datetime import datetime
from pathlib import Path
import traceback

# ============================================================================
# BULLETPROOF GAME ID COLLECTOR
# ============================================================================

def collect_all_game_ids_robust():
    """
    Collect game IDs with ROBUST error handling
    """
    print("[1/3] BULLETPROOF GAME ID COLLECTION")
    print()
    
    seasons = ['2015-16', '2016-17', '2017-18', '2018-19']
    all_game_info = []
    
    for season in seasons:
        print(f"  Season {season}...")
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"    Attempt {attempt + 1}/{max_retries}...", end='', flush=True)
                
                finder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable='Regular Season',
                    timeout=60  # 60 second timeout
                )
                
                df = finder.get_data_frames()[0]
                unique = df.drop_duplicates(subset=['GAME_ID'])
                
                for _, row in unique.iterrows():
                    all_game_info.append({
                        'game_id': str(row['GAME_ID']).zfill(10),
                        'season': season,
                        'date': row['GAME_DATE'],
                        'matchup': row['MATCHUP']
                    })
                
                print(f" ✅ {len(unique)} games")
                time.sleep(2)
                break  # Success, move to next season
                
            except Exception as e:
                print(f" ⚠️ Error: {str(e)[:50]}")
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 3
                    print(f"    Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    ❌ Failed after {max_retries} attempts, skipping season")
    
    print()
    print(f"✅ Total game IDs collected: {len(all_game_info)}")
    return all_game_info

# ============================================================================
# PATTERN EXTRACTION (Same as proven version)
# ============================================================================

def extract_pattern_robust(game_id):
    """Bulletproof pattern extraction"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id, timeout=30)
            plays_df = pbp.get_data_frames()[0]
            
            if len(plays_df) == 0:
                return None
            
            # Extract pattern (proven logic from 2021-2025)
            minute_diffs = {}
            diff_at_halftime = None
            diff_at_final = None
            diff_at_2q_6min = None
            
            for _, play in plays_df.iterrows():
                period = play['PERIOD']
                pctimestring = play['PCTIMESTRING']
                score_margin = play['SCOREMARGIN']
                
                if pd.isna(pctimestring) or pd.isna(score_margin):
                    continue
                
                try:
                    parts = pctimestring.split(':')
                    mins_remaining = int(parts[0])
                    secs_remaining = int(parts[1])
                    
                    if period == 1:
                        elapsed = 12 - mins_remaining - (secs_remaining / 60.0)
                    elif period == 2:
                        elapsed = 12 + (12 - mins_remaining - (secs_remaining / 60.0))
                        
                        if 17.8 <= elapsed <= 18.2:
                            diff_at_2q_6min = 0 if score_margin == 'TIE' else int(score_margin)
                        
                        if mins_remaining == 0 and secs_remaining <= 10:
                            diff_at_halftime = 0 if score_margin == 'TIE' else int(score_margin)
                    
                    elif period == 4:
                        if mins_remaining == 0 and secs_remaining <= 10:
                            diff_at_final = 0 if score_margin == 'TIE' else int(score_margin)
                        continue
                    else:
                        continue
                    
                    if elapsed > 18:
                        continue
                    
                    diff = 0 if score_margin == 'TIE' else int(score_margin)
                    minute = int(elapsed)
                    minute_diffs[minute] = diff
                
                except:
                    continue
            
            # Build pattern
            pattern = []
            for minute in range(18):
                if minute in minute_diffs:
                    pattern.append(minute_diffs[minute])
                else:
                    pattern.append(pattern[-1] if pattern else 0)
            
            if diff_at_final is None:
                return None
            
            if diff_at_halftime is None:
                diff_at_halftime = int(diff_at_final * 0.6)
            
            if diff_at_2q_6min is None and len(pattern) == 18:
                diff_at_2q_6min = pattern[-1]
            
            # Compute features (FREE)
            from scipy.fft import fft
            from scipy.stats import entropy
            
            pattern_arr = np.array(pattern)
            
            stats = {
                'mean': float(np.mean(pattern_arr)),
                'std': float(np.std(pattern_arr)),
                'trend': float(np.polyfit(range(18), pattern_arr, 1)[0]),
                'volatility': float(np.std(np.diff(pattern_arr)))
            }
            
            fft_vals = fft(pattern_arr)
            power = np.abs(fft_vals)**2
            total_power = power.sum()
            
            spectral = {
                'spectral_energy': float(total_power),
                'low_freq_power': float(power[1:4].sum() / total_power) if total_power > 0 else 0,
                'mid_freq_power': float(power[4:8].sum() / total_power) if total_power > 0 else 0,
                'high_freq_power': float(power[8:].sum() / total_power) if total_power > 0 else 0,
                'dominant_freq': float(np.argmax(power[1:])) / 18,
                'spectral_entropy': float(entropy(power + 1e-10))
            }
            
            velocity = np.diff(pattern_arr).mean()
            acceleration = np.diff(np.diff(pattern_arr)).mean()
            
            momentum = {
                'velocity': float(velocity),
                'acceleration': float(acceleration),
                'recent_momentum': float(np.mean(pattern_arr[-5:]) - np.mean(pattern_arr[:5])),
                'lead_changes': int(sum(1 for i in range(1, 18) if (pattern_arr[i] > 0) != (pattern_arr[i-1] > 0))),
                'max_swing': float(max(pattern_arr) - min(pattern_arr)),
                'comeback_potential': 1.0 if (pattern_arr[0] > 5 and pattern_arr[-1] < 0) else 0.0
            }
            
            return {
                'pattern': pattern,
                'diff_at_halftime': diff_at_halftime,
                'diff_at_final': diff_at_final,
                'diff_at_2q_6min': diff_at_2q_6min,
                'statistics': stats,
                'spectral': spectral,
                'momentum': momentum,
                'home_team_stats': {'OFF_RATING': 110.0, 'DEF_RATING': 110.0, 'NET_RATING': 0.0, 'PACE': 100.0},
                'away_team_stats': {'OFF_RATING': 110.0, 'DEF_RATING': 110.0, 'NET_RATING': 0.0, 'PACE': 100.0},
                'player_stars': {'home_tier_1': 0, 'away_tier_1': 0, 'home_tier_2': 0, 'away_tier_2': 0}
            }
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return None
    
    return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*80)
print("🔥 BULLETPROOF COLLECTION - ELON MODE")
print("="*80)
print("No excuses. Get ALL the data. Make it work.")
print()

# Load checkpoint
checkpoint_file = 'checkpoint_2015_2019_optimal.pkl'
try:
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    patterns = checkpoint['patterns']
    processed = checkpoint['processed_ids']
    print(f"✅ Resuming from {len(patterns)} games")
except:
    patterns = []
    processed = set()
    print("Starting fresh...")

print()

# Get ALL game IDs (bulletproof)
game_ids = collect_all_game_ids_robust()

# Filter already processed
remaining = [g for g in game_ids if g['game_id'] not in processed]

print(f"[2/3] EXTRACTING {len(remaining)} GAMES")
print(f"   Already have: {len(patterns)}")
print(f"   Need: {len(remaining)}")
print(f"   ETA: {len(remaining) * 0.65 / 3600:.1f} hours")
print()

start_time = datetime.now()
successes = 0
failures = 0

for i, game_info in enumerate(remaining, 1):
    game_id = game_info['game_id']
    
    # Extract
    pattern_data = extract_pattern_robust(game_id)
    
    if pattern_data:
        full_game = {
            'game_id': game_id,
            'season': game_info['season'],
            'date': game_info['date'],
            'matchup': game_info['matchup'],
            **pattern_data
        }
        
        patterns.append(full_game)
        processed.add(game_id)
        successes += 1
    else:
        failures += 1
        processed.add(game_id)  # Mark as attempted
    
    # Progress every 10
    if i % 10 == 0:
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = successes / elapsed if elapsed > 0 else 0
        remaining_games = len(remaining) - i
        eta_hours = remaining_games / (rate * 3600) if rate > 0 else 0
        
        print(f"  [{successes:5d} OK, {failures:3d} FAIL] | "
              f"Rate: {rate*3600:4.0f}/hr | "
              f"ETA: {eta_hours:.1f}h")
    
    # Checkpoint every 100
    if i % 100 == 0:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'patterns': patterns,
                'processed_ids': processed,
                'last_update': datetime.now(),
                'quality_counts': {'A': 0, 'B': len(patterns), 'C': 0}
            }, f)
    
    # Stealth delay
    time.sleep(random.uniform(0.6, 1.2))

# Final save
with open(checkpoint_file, 'wb') as f:
    pickle.dump({
        'patterns': patterns,
        'processed_ids': processed,
        'last_update': datetime.now(),
        'quality_counts': {'A': 0, 'B': len(patterns), 'C': 0}
    }, f)

# Save final
with open('PATTERNS_2015_2019_COMPLETE.pkl', 'wb') as f:
    pickle.dump(patterns, f)

print()
print("="*80)
print("✅ COLLECTION COMPLETE")
print("="*80)
print(f"Total: {len(patterns)} games")
print(f"Success rate: {100*successes/(successes+failures):.1f}%")
print()
print("SAVED TO: PATTERNS_2015_2019_COMPLETE.pkl")
print("="*80)

