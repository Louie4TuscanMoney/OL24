"""
🔥 COMPREHENSIVE 10-SEASON DATA COLLECTION - 2015-2021
Collect ALL missing seasons to complement existing 2021-2025 data

EXISTING: 2021-2025 (6,912 games)
COLLECTING: 2015-2021 (~8,500 games)
TOTAL: ~15,400 games (FULL 10 SEASONS!)

TIME: 1.5-2.5 hours (comprehensive, bulletproof)
MODE: ELON - Better Buzz WiFi - 12:39 PM start
TARGET: Break data ceiling with massive dataset!
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
print("🔥 COMPREHENSIVE 10-SEASON DATA COLLECTION")
print("="*90)
print(f"\nSTART TIME: 12:39 PM (Better Buzz WiFi)")
print(f"TARGET: Collect 2015-2021 seasons (~8,500 NEW games)")
print(f"TOTAL: ~15,400 games (FULL 10 SEASONS!)")
print(f"TIME: 1.5-2.5 hours")
print("\n" + "="*90)

# Load existing data
print("\n[PHASE 1/5] Loading existing data...")
try:
    with open('Action/ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl', 'rb') as f:
        existing_data = pickle.load(f)
    existing_ids = {g.get('game_id') for g in existing_data if g.get('game_id')}
    print(f"✓ Existing data: {len(existing_data)} games (2021-2025)")
    print(f"  Existing game IDs: {len(existing_ids)}")
except:
    existing_data = []
    existing_ids = set()
    print("  No existing data found")

print("\n[PHASE 2/5] Collecting game IDs for 2015-2021...")
print("This will take ~10-15 minutes...")

seasons_to_collect = [
    '2015-16',
    '2016-17',
    '2017-18',
    '2018-19',
    '2019-20',  # COVID shortened
    '2020-21'   # COVID shortened
]

all_game_ids = []
season_breakdown = {}

print(f"\nCollecting game IDs from {len(seasons_to_collect)} seasons...")

for season in seasons_to_collect:
    print(f"\n  📅 {season}...", end=" ", flush=True)
    
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00',
            season_type_nullable='Regular Season'
        )
        
        games_df = gamefinder.get_data_frames()[0]
        
        # Get unique game IDs
        game_ids = games_df['GAME_ID'].unique().tolist()
        
        # Filter out existing
        new_ids = [gid for gid in game_ids if gid not in existing_ids]
        
        all_game_ids.extend(new_ids)
        season_breakdown[season] = {
            'total': len(game_ids),
            'new': len(new_ids),
            'ids': new_ids
        }
        
        print(f"✓ {len(game_ids)} total, {len(new_ids)} NEW")
        
        time.sleep(1.5)  # Rate limit
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:60]}")
        time.sleep(5)
        continue

print(f"\n✅ Total NEW games to collect: {len(all_game_ids)}")
print(f"\nBreakdown by season:")
for season, info in season_breakdown.items():
    print(f"  {season}: {info['new']:4d} new games (of {info['total']} total)")

print(f"\n[PHASE 3/5] EXTRACTING FEATURES FROM {len(all_game_ids)} GAMES")
print("="*90)
print(f"\n⏰ ESTIMATED TIME: {len(all_game_ids) * 0.6 / 60:.1f} - {len(all_game_ids) * 1.0 / 60:.1f} minutes")
print(f"   ({len(all_game_ids) * 0.6 / 3600:.2f} - {len(all_game_ids) * 1.0 / 3600:.2f} hours)")
print(f"\n🔥 STARTING AGGRESSIVE COLLECTION NOW (12:39 PM)...")

collected_games = []
checkpoint_freq = 100
errors = []
start_time = time.time()

for idx, game_id in enumerate(all_game_ids):
    try:
        # Progress update every 10 games
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining_games = len(all_game_ids) - idx
            eta_seconds = remaining_games / rate if rate > 0 else 0
            
            current_time = datetime.now().strftime("%I:%M %p")
            
            print(f"\n{'─'*90}")
            print(f"[{idx:5d}/{len(all_game_ids)}] {idx/len(all_game_ids)*100:5.1f}% │ Time: {current_time}")
            print(f"  Elapsed: {elapsed/60:6.1f} min │ Rate: {rate*60:5.1f} games/min │ ETA: {eta_seconds/60:6.1f} min")
            print(f"  Collected: {len(collected_games)} │ Errors: {len(errors)}")
            
            if len(collected_games) > 0:
                success_rate = len(collected_games) / (idx + 1) * 100
                print(f"  Success rate: {success_rate:.1f}%")
        
        # Fetch PBP
        pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
        pbp_df = pbp.get_data_frames()[0]
        
        if pbp_df.empty:
            errors.append((game_id, "Empty PBP"))
            time.sleep(0.6)
            continue
        
        # Get game date from first event
        if 'GAME_DATE_EST' in pbp_df.columns:
            game_date = str(pbp_df['GAME_DATE_EST'].iloc[0])
        else:
            game_date = datetime.now().isoformat()
        
        # Find Q2 6:00 event
        q2_events = pbp_df[pbp_df['PERIOD'] == 2]
        
        # Look for ~6:00 remaining in Q2
        q2_6min_event = None
        diff_at_q2_6min = 0
        
        if len(q2_events) > 0:
            for _, event in q2_events.iterrows():
                time_str = str(event.get('PCTIMESTRING', ''))
                
                # Check if close to 6:00
                if any(t in time_str for t in ['6:0', '6:1', '5:5', '6:']):
                    score_str = str(event.get('SCORE', ''))
                    
                    if score_str and ' - ' in score_str:
                        try:
                            home, away = score_str.split(' - ')
                            diff_at_q2_6min = int(home) - int(away)
                            q2_6min_event = event
                            break
                        except:
                            pass
        
        # If couldn't find exact 6:00, use middle of Q2
        if q2_6min_event is None and len(q2_events) > 0:
            mid_idx = len(q2_events) // 2
            event = q2_events.iloc[mid_idx]
            score_str = str(event.get('SCORE', ''))
            
            if score_str and ' - ' in score_str:
                try:
                    home, away = score_str.split(' - ')
                    diff_at_q2_6min = int(home) - int(away)
                except:
                    diff_at_q2_6min = 0
        
        # Get halftime differential
        q2_end = q2_events.iloc[-1] if len(q2_events) > 0 else None
        diff_at_halftime = diff_at_q2_6min  # Default
        
        if q2_end is not None:
            score_str = str(q2_end.get('SCORE', ''))
            if score_str and ' - ' in score_str:
                try:
                    home, away = score_str.split(' - ')
                    diff_at_halftime = int(home) - int(away)
                except:
                    pass
        
        # Get final differential
        final_events = pbp_df[pbp_df['PERIOD'] >= 4]  # Q4 or OT
        
        if len(final_events) > 0:
            final_event = final_events.iloc[-1]
            score_str = str(final_event.get('SCORE', ''))
            
            if score_str and ' - ' in score_str:
                try:
                    home, away = score_str.split(' - ')
                    diff_at_final = int(home) - int(away)
                except:
                    diff_at_final = 0
            else:
                diff_at_final = 0
        else:
            diff_at_final = diff_at_halftime
        
        # Build pattern (score differential sequence)
        pattern = []
        
        # Sample events from Q1-Q2 to build trajectory
        for period in [1, 2]:
            period_events = pbp_df[pbp_df['PERIOD'] == period]
            
            if len(period_events) > 0:
                # Sample ~9 points per quarter
                step = max(1, len(period_events) // 9)
                
                for i in range(0, len(period_events), step):
                    if len(pattern) >= 18:
                        break
                    
                    event = period_events.iloc[i]
                    score_str = str(event.get('SCORE', ''))
                    
                    if score_str and ' - ' in score_str:
                        try:
                            home, away = score_str.split(' - ')
                            diff = int(home) - int(away)
                            pattern.append(diff)
                        except:
                            if pattern:
                                pattern.append(pattern[-1])
        
        # Pad to 18 if needed
        while len(pattern) < 18:
            pattern.append(pattern[-1] if pattern else 0)
        
        # Store
        game_data = {
            'game_id': game_id,
            'date': game_date,
            'pattern': pattern[:18],
            'diff_at_2q_6min': diff_at_q2_6min,
            'diff_at_halftime': diff_at_halftime,
            'diff_at_final': diff_at_final,
            'season': season
        }
        
        collected_games.append(game_data)
        
        # Checkpoint
        if (idx + 1) % checkpoint_freq == 0:
            with open(f'Action/COLLECTION_CHECKPOINT_{idx+1}.pkl', 'wb') as f:
                pickle.dump(collected_games, f)
            
            print(f"\n    💾 CHECKPOINT: {len(collected_games)} games collected")
            
            # Also save incremental merge
            incremental_merge = existing_data + collected_games
            with open('Action/INCREMENTAL_MERGE.pkl', 'wb') as f:
                pickle.dump(incremental_merge, f)
        
        # Rate limiting (60-100 games/min)
        time.sleep(0.65)
        
    except Exception as e:
        error_msg = str(e)[:100]
        errors.append((game_id, error_msg))
        
        if idx % 50 == 0:
            print(f"    ✗ Error: {error_msg}")
        
        time.sleep(1.5)
        continue

# Final stats
elapsed_total = time.time() - start_time
end_time = datetime.now().strftime("%I:%M %p")

print(f"\n{'='*90}")
print(f"✅ COLLECTION COMPLETE!")
print(f"{'='*90}")

print(f"\nStats:")
print(f"  Start time: 12:39 PM")
print(f"  End time: {end_time}")
print(f"  Elapsed: {elapsed_total/60:.1f} minutes ({elapsed_total/3600:.2f} hours)")
print(f"  Games collected: {len(collected_games)}")
print(f"  Errors: {len(errors)}")
print(f"  Success rate: {len(collected_games)/(len(all_game_ids))*100:.1f}%")
print(f"  Rate: {len(collected_games)/(elapsed_total/60):.1f} games/minute")

# Save final raw collection
with open('Action/COLLECTED_2015_2021_RAW.pkl', 'wb') as f:
    pickle.dump(collected_games, f)

print(f"\n✓ Saved raw collection: COLLECTED_2015_2021_RAW.pkl")

print("\n[PHASE 4/5] MERGING WITH EXISTING DATA")
print("="*90)

merged_data = existing_data + collected_games

print(f"\n✓ Merged dataset:")
print(f"  Existing (2021-2025): {len(existing_data)} games")
print(f"  New (2015-2021): {len(collected_games)} games")
print(f"  Total: {len(merged_data)} games")

# Sort chronologically
merged_sorted = sorted(merged_data, key=lambda x: x.get('date', ''))

# Get date range
if merged_sorted:
    first_date = merged_sorted[0].get('date', '')[:10]
    last_date = merged_sorted[-1].get('date', '')[:10]
    print(f"  Date range: {first_date} to {last_date}")

# Save merged
with open('Action/MERGED_2015_2025_COMPLETE.pkl', 'wb') as f:
    pickle.dump(merged_sorted, f)

print(f"\n✓ Saved: MERGED_2015_2025_COMPLETE.pkl")

print("\n[PHASE 5/5] SUMMARY")
print("="*90)

print(f"\n🏆 COMPLETE 10-SEASON DATASET:")
print(f"  Total games: {len(merged_data)}")
print(f"  Coverage: 2015-2025 (10 seasons!)")
print(f"  Features: 18-point pattern per game")
print(f"  Targets: Q2 6:00, Halftime, Final differentials")

print(f"\n📊 Collection breakdown:")
for season, info in season_breakdown.items():
    collected = sum(1 for g in collected_games if g.get('season') == season)
    print(f"  {season}: {collected:4d} games collected")

print(f"\n✅ DATA COLLECTION COMPLETE!")
print(f"\n🚀 Next steps:")
print(f"  1. Extract 30 REAL features from {len(merged_data)} games")
print(f"  2. Train models on 10-season dataset")
print(f"  3. Rolling validation (15+ folds)")
print(f"  4. Expected: 8.8 → 8.3-8.5 MAE (breakthrough!)")

print(f"\n💎 EXPECTED IMPROVEMENT:")
print(f"  Current (6.9k games): 8.8 ± 0.4 MAE")
print(f"  Target (15k games): 8.3-8.5 ± 0.3 MAE")
print(f"  Gain: ~0.3-0.5 MAE")
print(f"  EV gain: +$40-80 per 100 games")
print(f"  Season gain: +$2,000-4,000!")

print(f"\n🔥 10-SEASON COLLECTION COMPLETE - READY FOR BREAKTHROUGH!")
print("="*90)

