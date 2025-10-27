#!/usr/bin/env python3
"""
🔥 PATTERN EXTRACTION WITH LIVE PROGRESS

Extracts patterns from 6,914 games with:
- Live progress updates every 10 games
- Checkpoint saves every 50 games
- Resume capability if interrupted
- ETA calculations
- Success rate tracking

SAFE TO:
- Use another Cursor agent while running
- Interrupt anytime (Ctrl+C)
- Resume later from checkpoint

TIME: 2-4 hours
OUTPUT: Training-ready dataset with patterns
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta

from nba_api.stats.endpoints import playbyplayv2

# Progress tracking
class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = datetime.now()
        
    def update(self, success=True):
        self.processed += 1
        if success:
            self.successful += 1
        else:
            self.failed += 1
    
    def print_progress(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.processed / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.processed) / rate if rate > 0 else 0
        eta = datetime.now() + timedelta(seconds=remaining)
        
        pct = (self.processed / self.total * 100) if self.total > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"📊 PROGRESS UPDATE")
        print(f"{'='*80}")
        print(f"   Processed: {self.processed}/{self.total} ({pct:.1f}%)")
        print(f"   Successful: {self.successful}")
        print(f"   Failed: {self.failed}")
        print(f"   Success rate: {self.successful/self.processed*100:.1f}%")
        print(f"   Speed: {rate*60:.1f} games/min")
        print(f"   Time elapsed: {elapsed/60:.1f} min")
        print(f"   ETA: {eta.strftime('%I:%M %p')} ({remaining/60:.0f} min remaining)")
        print(f"{'='*80}\n")

def extract_pattern(game_id):
    """Extract 18-min pattern from play-by-play"""
    try:
        pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
        plays_df = pbp.get_data_frames()[0]
        
        if len(plays_df) == 0:
            return None
        
        differentials = [0]
        halftime_diff = 0
        final_diff = 0
        
        for idx, play in plays_df.iterrows():
            period = play['PERIOD']
            pctimestring = play['PCTIMESTRING']
            score_margin = play['SCOREMARGIN']
            
            if pd.notna(pctimestring) and pd.notna(score_margin):
                try:
                    parts = pctimestring.split(':')
                    mins_remaining = int(parts[0])
                    secs_remaining = int(parts[1])
                    
                    if period == 1:
                        elapsed = 12 - mins_remaining - (secs_remaining / 60.0)
                    elif period == 2:
                        elapsed = 12 + (12 - mins_remaining - (secs_remaining / 60.0))
                        if mins_remaining == 0 and secs_remaining <= 5:
                            halftime_diff = 0 if score_margin == 'TIE' else int(score_margin)
                    elif period == 4:
                        if mins_remaining == 0 and secs_remaining <= 5:
                            final_diff = 0 if score_margin == 'TIE' else int(score_margin)
                        continue
                    else:
                        continue
                    
                    if elapsed > 18:
                        continue
                    
                    diff = 0 if score_margin == 'TIE' else int(score_margin)
                    minute = int(elapsed)
                    
                    if 0 <= minute <= 18:
                        while len(differentials) <= minute:
                            differentials.append(differentials[-1])
                        differentials[minute] = diff
                
                except:
                    continue
        
        while len(differentials) < 18:
            differentials.append(differentials[-1])
        
        return {
            'pattern': differentials[:18],
            'diff_at_halftime': halftime_diff if halftime_diff != 0 else None,
            'diff_at_final': final_diff if final_diff != 0 else None
        }
        
    except Exception as e:
        return None

# Main execution
print("="*80)
print("🔥 PATTERN EXTRACTION - Live Progress Mode")
print("="*80)

# Load games list
print("\nLoading game list...")
games_df = pd.read_csv('historical_games_2021_2025_basic.csv')
games_df = games_df.drop_duplicates(subset=['GAME_ID'], keep='first')

print(f"✅ {len(games_df)} unique games to process")

# Check for checkpoint
checkpoint_file = Path('pattern_checkpoint.pkl')
processed_games = []
processed_ids = set()

if checkpoint_file.exists():
    print("\n📂 Loading checkpoint...")
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
        processed_games = checkpoint['games']
        processed_ids = checkpoint['ids']
    
    print(f"   ✅ Resuming from {len(processed_games)} games")
    
    # Filter already processed
    games_df = games_df[~games_df['GAME_ID'].isin(processed_ids)]
    print(f"   Remaining: {len(games_df)} games")

# Initialize tracker
tracker = ProgressTracker(len(games_df))

print(f"\n⏱️  ESTIMATED TIME: {len(games_df) * 0.6 / 60:.0f} minutes")
print(f"   (Can pause/resume anytime with Ctrl+C)")
print(f"\n🚀 Starting extraction...\n")

try:
    for idx, (_, row) in enumerate(games_df.iterrows(), 1):
        game_id = row['GAME_ID']
        
        # Extract pattern
        pattern_data = extract_pattern(game_id)
        
        if pattern_data and pattern_data['diff_at_final'] is not None:
            game_record = {
                'season': row['SEASON_ID'],
                'game_id': game_id,
                'date': row['GAME_DATE'],
                'matchup': row['MATCHUP'],
                'pattern': pattern_data['pattern'],
                'diff_at_halftime': pattern_data['diff_at_halftime'],
                'diff_at_final': pattern_data['diff_at_final']
            }
            
            processed_games.append(game_record)
            processed_ids.add(game_id)
            tracker.update(success=True)
        else:
            tracker.update(success=False)
        
        # Live updates every 10 games
        if idx % 10 == 0:
            tracker.print_progress()
        
        # Checkpoint every 50 games
        if idx % 50 == 0:
            print(f"💾 Saving checkpoint... ({len(processed_games)} games)")
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'games': processed_games, 'ids': processed_ids}, f)
        
        # Rate limit - CRITICAL!
        time.sleep(0.6)  # ~100 games/hour max (respect NBA API)

except KeyboardInterrupt:
    print(f"\n\n⏸️  PAUSED!")
    print(f"   Processed: {len(processed_games)} games")
    
    # Save checkpoint
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({'games': processed_games, 'ids': processed_ids}, f)
    
    print(f"   ✅ Checkpoint saved")
    print(f"   Run again to resume from here!")
    sys.exit(0)

# Final save
print(f"\n✅ EXTRACTION COMPLETE!")
print(f"   Total: {len(processed_games)} games with patterns")

output_path = Path('games_2021_2025_WITH_PATTERNS.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(processed_games, f)

print(f"   💾 Saved to: {output_path}")

# Summary
tracker.print_progress()

print(f"\n{'='*80}")
print(f"NEXT: Merge with existing data and retrain model!")
print(f"{'='*80}")

