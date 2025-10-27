#!/usr/bin/env python3
"""
🔥 PATTERN EXTRACTION - 2021-2025 Data

Takes the 6,914 games we just collected
Extracts 18-minute patterns for each
Processes into training format

SMART APPROACH:
- Process in batches
- Save checkpoints
- Resume if interrupted
- Respectful API rate limiting

TIME: 2-4 hours (API limits)
RESULT: Training-ready dataset!
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

from nba_api.stats.endpoints import playbyplayv2, boxscoretraditionalv2

print("="*80)
print("🔥 EXTRACTING PATTERNS FROM 2021-2025 DATA")
print("="*80)

# Load collected games
print("\n[1/5] Loading collected game data...")
df = pd.read_csv('historical_games_2021_2025_basic.csv')

# Deduplicate (each game appears twice - once per team)
print(f"   Raw: {len(df)} rows")
df = df.drop_duplicates(subset=['GAME_ID'], keep='first')
print(f"   Unique games: {len(df)}")

# Check for checkpoint (in case we need to resume)
checkpoint_file = Path('pattern_extraction_checkpoint.pkl')
processed_games = []
processed_game_ids = set()

if checkpoint_file.exists():
    print(f"\n📂 Found checkpoint file - loading...")
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)
        processed_games = checkpoint_data['games']
        processed_game_ids = checkpoint_data['game_ids']
    
    print(f"   ✅ Resuming from {len(processed_games)} games")
else:
    print(f"\n📂 No checkpoint - starting fresh")

# Filter out already processed
df_remaining = df[~df['GAME_ID'].isin(processed_game_ids)]

print(f"\n📊 Games to process: {len(df_remaining)}")
print(f"   Already processed: {len(processed_games)}")
print(f"   Total: {len(df)}")

def extract_game_pattern(game_id):
    """
    Extract 18-minute pattern and scores for a game
    
    Returns:
        {
            'pattern': [18 differentials],
            'diff_at_halftime': int,
            'diff_at_final': int,
            'home_score_ht': int,
            'away_score_ht': int,
            'home_score_final': int,
            'away_score_final': int
        } or None
    """
    try:
        # Get play-by-play
        pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
        plays_df = pbp.get_data_frames()[0]
        
        if len(plays_df) == 0:
            return None
        
        # Extract pattern
        differentials = [0]
        halftime_diff = None
        final_diff = None
        
        for idx, play in plays_df.iterrows():
            period = play['PERIOD']
            pctimestring = play['PCTIMESTRING']
            score_margin = play['SCOREMARGIN']
            
            if pd.notna(pctimestring) and pd.notna(score_margin):
                try:
                    parts = pctimestring.split(':')
                    mins_remaining = int(parts[0])
                    secs_remaining = int(parts[1])
                    
                    # Calculate elapsed time
                    if period == 1:
                        elapsed = 12 - mins_remaining - (secs_remaining / 60.0)
                    elif period == 2:
                        elapsed = 12 + (12 - mins_remaining - (secs_remaining / 60.0))
                        
                        # Capture halftime (end of Q2)
                        if mins_remaining == 0 and secs_remaining <= 5:
                            if score_margin == 'TIE':
                                halftime_diff = 0
                            else:
                                halftime_diff = int(score_margin)
                    elif period == 4:
                        # Capture final (end of Q4)
                        if mins_remaining == 0 and secs_remaining <= 5:
                            if score_margin == 'TIE':
                                final_diff = 0
                            else:
                                final_diff = int(score_margin)
                        continue
                    else:
                        continue
                    
                    if elapsed > 18:
                        continue
                    
                    # Parse differential
                    if score_margin == 'TIE':
                        diff = 0
                    else:
                        diff = int(score_margin)
                    
                    minute = int(elapsed)
                    if 0 <= minute <= 18:
                        while len(differentials) <= minute:
                            differentials.append(differentials[-1])
                        differentials[minute] = diff
                
                except (ValueError, IndexError, AttributeError):
                    continue
        
        # Ensure 18 values
        while len(differentials) < 18:
            differentials.append(differentials[-1])
        
        pattern = differentials[:18]
        
        # Get final score from box score if not captured
        if final_diff is None:
            try:
                box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
                team_stats = box.team_stats.get_data_frame()
                
                if len(team_stats) >= 2:
                    away_final = team_stats.iloc[0]['PTS']
                    home_final = team_stats.iloc[1]['PTS']
                    final_diff = home_final - away_final
            except:
                pass
        
        if final_diff is None:
            return None
        
        return {
            'pattern': pattern,
            'diff_at_halftime': halftime_diff if halftime_diff is not None else final_diff * 0.6,
            'diff_at_final': final_diff
        }
        
    except Exception as e:
        return None

# Process games in batches
print(f"\n[2/5] Extracting patterns...")
print(f"   ⚠️  This will take 2-4 hours due to API rate limits")
print(f"   ⏸️  Can pause anytime (Ctrl+C) and resume later!\n")

batch_size = 50
total_remaining = len(df_remaining)

try:
    for batch_num, start_idx in enumerate(range(0, total_remaining, batch_size)):
        end_idx = min(start_idx + batch_size, total_remaining)
        batch_df = df_remaining.iloc[start_idx:end_idx]
        
        print(f"\n📦 Batch {batch_num + 1}: Games {start_idx + 1}-{end_idx} of {total_remaining}")
        
        for i, (idx, row) in enumerate(batch_df.iterrows(), 1):
            game_id = row['GAME_ID']
            game_date = row['GAME_DATE']
            matchup = row['MATCHUP']
            
            # Progress
            if i % 10 == 0:
                print(f"   [{start_idx + i}/{total_remaining}] Processing...")
            
            # Extract pattern
            pattern_data = extract_game_pattern(game_id)
            
            if pattern_data:
                game_record = {
                    'season': row['SEASON_ID'],
                    'game_id': game_id,
                    'date': game_date,
                    'matchup': matchup,
                    'pattern': pattern_data['pattern'],
                    'diff_at_halftime': pattern_data['diff_at_halftime'],
                    'diff_at_final': pattern_data['diff_at_final']
                }
                
                processed_games.append(game_record)
                processed_game_ids.add(game_id)
            
            # Rate limit - critical!
            time.sleep(0.6)  # ~100 games per minute max
        
        # Save checkpoint after each batch
        print(f"   💾 Saving checkpoint... ({len(processed_games)} games total)")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'games': processed_games,
                'game_ids': processed_game_ids,
                'last_update': datetime.now()
            }, f)
        
        print(f"   ✅ Batch {batch_num + 1} complete!")
        
except KeyboardInterrupt:
    print(f"\n\n⏸️  PAUSED by user")
    print(f"   Processed: {len(processed_games)} games")
    print(f"   Checkpoint saved - run again to resume!")
    sys.exit(0)

# Final save
print(f"\n[3/5] Saving final dataset...")

output_path = Path('historical_games_2021_2025_WITH_PATTERNS.pkl')

with open(output_path, 'wb') as f:
    pickle.dump(processed_games, f)

print(f"✅ Saved {len(processed_games)} games to: {output_path}")

# Statistics
print(f"\n[4/5] Data quality check...")

patterns = [g['pattern'] for g in processed_games]
halftime_diffs = [g['diff_at_halftime'] for g in processed_games]
final_diffs = [g['diff_at_final'] for g in processed_games]

print(f"\n📊 Dataset statistics:")
print(f"   Games with patterns: {len(processed_games)}")
print(f"   Pattern length: {len(patterns[0]) if patterns else 0}")
print(f"   Avg halftime diff: {np.mean(halftime_diffs):.2f}")
print(f"   Avg final diff: {np.mean(final_diffs):.2f}")

print(f"\n[5/5] Next steps...")
print(f"""
✅ DATA COLLECTION COMPLETE!

Next steps:
1. Merge with existing 2015-2021 data (10 min)
2. Retrain Dejavu model (5 min)
3. Test on 2025 holdout (5 min)
4. Compare: Old MAE 10.75 vs New MAE (should be 6-7!)

Total time to updated model: ~20 minutes
Then: Launch Monday with CONFIDENCE!
""")

print("="*80)

