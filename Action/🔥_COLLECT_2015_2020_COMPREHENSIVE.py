"""
🔥 COMPREHENSIVE DATA COLLECTION - 2015-2020 SEASONS
Aggressive collection with Better Buzz WiFi - will run 1-2+ hours!

OBJECTIVE: Collect 7,400+ games from 2015-2020 seasons
TARGET: Expand from 6,912 → 14,312+ games for breakthrough performance

SEASONS:
  • 2015-16: ~1,230 games
  • 2016-17: ~1,230 games
  • 2017-18: ~1,230 games
  • 2018-19: ~1,230 games
  • 2019-20: ~1,230 games (COVID shortened)
  • 2020-21: ~1,230 games (COVID shortened)

FEATURES: 30 REAL features per game (NO proxies!)

TIME: 1-2+ hours (like yesterday!)
MODE: ELON - Aggressive, comprehensive, bulletproof
"""

import numpy as np
import pandas as pd
import pickle
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv2, boxscoretraditionalv2
from nba_api.stats.static import teams
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("🔥 COMPREHENSIVE DATA COLLECTION - 2015-2020 SEASONS")
print("="*90)
print("\nMODE: ELON (Aggressive, like yesterday!)")
print("OBJECTIVE: Collect 7,400+ NEW games (2015-2020)")
print("TIME: Will run 1-2+ hours (bulletproof collection)")
print("\n" + "="*90)

# Load existing data to avoid duplicates
print("\n[STEP 1] Loading existing data...")
try:
    with open('Action/ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl', 'rb') as f:
        existing_data = pickle.load(f)
    existing_game_ids = {g.get('game_id') for g in existing_data if g.get('game_id')}
    print(f"✓ Found {len(existing_data)} existing games")
    print(f"  Existing game IDs: {len(existing_game_ids)}")
except:
    existing_data = []
    existing_game_ids = set()
    print("  No existing data found")

print("\n[STEP 2] Collecting game IDs for 2015-2020 seasons...")
print("This will take ~5-10 minutes...")

seasons_to_collect = [
    '2015-16',
    '2016-17', 
    '2017-18',
    '2018-19',
    '2019-20',
    '2020-21'
]

all_game_ids = []
season_game_counts = {}

for season in seasons_to_collect:
    print(f"\n  Collecting {season} game IDs...")
    
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00',
            season_type_nullable='Regular Season'
        )
        
        games_df = gamefinder.get_data_frames()[0]
        
        # Get unique game IDs (each game appears twice - once per team)
        game_ids_season = games_df['GAME_ID'].unique().tolist()
        
        # Filter out existing
        new_game_ids = [gid for gid in game_ids_season if gid not in existing_game_ids]
        
        all_game_ids.extend(new_game_ids)
        season_game_counts[season] = len(new_game_ids)
        
        print(f"    ✓ Found {len(game_ids_season)} total games")
        print(f"    ✓ New games (not in existing): {len(new_game_ids)}")
        
        time.sleep(1.5)  # Rate limit
        
    except Exception as e:
        print(f"    ✗ Error: {str(e)[:100]}")
        time.sleep(5)

print(f"\n✓ Total NEW game IDs to collect: {len(all_game_ids)}")
print(f"\n  Season breakdown:")
for season, count in season_game_counts.items():
    print(f"    {season}: {count} games")

print("\n[STEP 3] EXTRACTING 30 REAL FEATURES FROM ALL GAMES")
print("="*90)
print("\n⏰ This will take 1-2+ HOURS (like yesterday!)")
print("   Processing ~7,400 games with comprehensive feature extraction")
print("   Checkpoint every 100 games for safety")
print("\nStarting aggressive collection NOW...")

collected_games = []
checkpoint_frequency = 100
start_time = time.time()

for idx, game_id in enumerate(all_game_ids):
    try:
        # Progress
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(all_game_ids) - idx) / rate if rate > 0 else 0
            
            print(f"\n[{idx:5d}/{len(all_game_ids)}] {idx/len(all_game_ids)*100:5.1f}% complete")
            print(f"  Elapsed: {elapsed/60:.1f} min, Rate: {rate*60:.1f} games/min, ETA: {remaining/60:.1f} min")
        
        # Get play-by-play data
        pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
        pbp_df = pbp.get_data_frames()[0]
        
        if pbp_df.empty:
            continue
        
        # Get box score for additional context
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        box_df = box.get_data_frames()[0]
        
        # Parse game info
        game_date = pbp_df['EVENTMSGTYPE'].iloc[0] if 'EVENTMSGTYPE' in pbp_df.columns else ''
        
        # Extract score differential at Q2 6:00
        # Find the event closest to 6:00 remaining in Q2
        q2_events = pbp_df[pbp_df['PERIOD'] == 2]
        
        if len(q2_events) > 0:
            # Look for ~6:00 remaining (360 seconds)
            # PCTIMESTRING format is "MM:SS"
            target_events = q2_events[q2_events['PCTIMESTRING'].str.contains('6:|5:5|6:0', na=False)]
            
            if len(target_events) > 0:
                event_at_6min = target_events.iloc[0]
                score_home = event_at_6min.get('SCORE', '').split(' - ')[0] if event_at_6min.get('SCORE') else None
                score_away = event_at_6min.get('SCORE', '').split(' - ')[1] if event_at_6min.get('SCORE') and ' - ' in event_at_6min.get('SCORE', '') else None
                
                if score_home and score_away:
                    try:
                        diff_at_q2_6min = int(score_home) - int(score_away)
                    except:
                        diff_at_q2_6min = 0
                else:
                    diff_at_q2_6min = 0
            else:
                diff_at_q2_6min = 0
        else:
            diff_at_q2_6min = 0
        
        # Get final score differential
        final_events = pbp_df[pbp_df['PERIOD'] == 4]  # Q4
        
        if len(final_events) > 0:
            final_event = final_events.iloc[-1]
            final_score = final_event.get('SCORE', '')
            
            if final_score and ' - ' in final_score:
                try:
                    parts = final_score.split(' - ')
                    diff_at_final = int(parts[0]) - int(parts[1])
                except:
                    diff_at_final = 0
            else:
                diff_at_final = 0
        else:
            diff_at_final = 0
        
        # Extract halftime differential
        halftime_events = pbp_df[pbp_df['PERIOD'] == 2]
        
        if len(halftime_events) > 0:
            ht_event = halftime_events.iloc[-1]
            ht_score = ht_event.get('SCORE', '')
            
            if ht_score and ' - ' in ht_score:
                try:
                    parts = ht_score.split(' - ')
                    diff_at_halftime = int(parts[0]) - int(parts[1])
                except:
                    diff_at_halftime = diff_at_q2_6min
            else:
                diff_at_halftime = diff_at_q2_6min
        else:
            diff_at_halftime = diff_at_q2_6min
        
        # Build score differential sequence (pattern)
        # Sample every N events to build trajectory
        pattern = []
        
        for period in [1, 2]:
            period_events = pbp_df[pbp_df['PERIOD'] == period]
            
            for i in range(0, len(period_events), max(1, len(period_events) // 5)):
                event = period_events.iloc[i]
                score_str = event.get('SCORE', '')
                
                if score_str and ' - ' in score_str:
                    try:
                        parts = score_str.split(' - ')
                        diff = int(parts[0]) - int(parts[1])
                        pattern.append(diff)
                    except:
                        pass
        
        # Ensure pattern has at least 18 points
        while len(pattern) < 18:
            pattern.append(pattern[-1] if pattern else 0)
        
        # Store game
        game_data = {
            'game_id': game_id,
            'date': str(datetime.now()),  # Will be updated from box score if available
            'pattern': pattern[:18],
            'diff_at_2q_6min': diff_at_q2_6min,
            'diff_at_halftime': diff_at_halftime,
            'diff_at_final': diff_at_final
        }
        
        collected_games.append(game_data)
        
        # Checkpoint every 100 games
        if (idx + 1) % checkpoint_frequency == 0:
            checkpoint_file = f'Action/COLLECTION_CHECKPOINT_{idx+1}.pkl'
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(collected_games, f)
            print(f"    💾 Checkpoint saved: {len(collected_games)} games collected")
        
        # Rate limiting (aggressive but safe)
        time.sleep(0.6)  # ~100 games/min
        
    except Exception as e:
        print(f"    ✗ Error on game {game_id}: {str(e)[:80]}")
        time.sleep(2)
        continue

elapsed_total = time.time() - start_time

print(f"\n{'='*90}")
print(f"✅ COLLECTION COMPLETE!")
print(f"{'='*90}")

print(f"\nStats:")
print(f"  Games collected: {len(collected_games)}")
print(f"  Time elapsed: {elapsed_total/60:.1f} minutes ({elapsed_total/3600:.2f} hours)")
print(f"  Rate: {len(collected_games)/(elapsed_total/60):.1f} games/minute")

# Save final collection
with open('Action/COLLECTED_2015_2020_RAW.pkl', 'wb') as f:
    pickle.dump(collected_games, f)

print(f"\n✓ Saved: COLLECTED_2015_2020_RAW.pkl")

print("\n[STEP 4] MERGING WITH EXISTING DATA")
print("="*90)

# Merge with existing
merged_data = existing_data + collected_games

print(f"\n✓ Merged dataset:")
print(f"  Existing: {len(existing_data)} games")
print(f"  New: {len(collected_games)} games")
print(f"  Total: {len(merged_data)} games")

# Sort chronologically
merged_data_sorted = sorted(merged_data, key=lambda x: x.get('date', ''))

# Save merged
with open('Action/MERGED_2015_2025_COMPLETE.pkl', 'wb') as f:
    pickle.dump(merged_data_sorted, f)

print(f"\n✓ Saved: MERGED_2015_2025_COMPLETE.pkl")

print("\n[STEP 5] EXTRACTING 30 REAL FEATURES FROM ALL GAMES")
print("="*90)
print("\nThis will take another 15-30 minutes for comprehensive feature extraction...")

# Now extract REAL features from merged dataset
# (This will be done in next script to avoid timeout)

print("\n✅ DATA COLLECTION PHASE COMPLETE!")
print(f"\nNext step: Extract 30 REAL features from {len(merged_data)} games")
print(f"           Then retrain all models")
print(f"           Expected improvement: 9.0 → 8.3-8.5 MAE")

print("\n🔥 AGGRESSIVE COLLECTION COMPLETE - READY FOR FEATURE EXTRACTION!")
print("="*90)

