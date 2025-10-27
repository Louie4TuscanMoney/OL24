#!/usr/bin/env python3
"""
🔥 ANTI-OVERFITTING DATA COLLECTION - 2015-2019
Doubling dataset to reduce overfitting from 5% → 2%

Uses existing Better Buzz stealth infrastructure
Estimated time: 2-3 hours
"""

from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import pickle
import time
import random
from datetime import datetime

print("="*80)
print("🔥 ANTI-OVERFITTING: COLLECTING 2015-2019 DATA")
print("="*80)
print()
print("PROBLEM DETECTED:")
print("  Train MAE: 9.401")
print("  Test MAE:  9.906")
print("  Gap: 5% overfitting ⚠️")
print()
print("SOLUTION:")
print("  Current: 6,912 games (2020-2024)")
print("  Adding: ~6,000 games (2015-2019)")
print("  Result: 13,000 games total (2x data)")
print()

# Collect game IDs
seasons = ['2015-16', '2016-17', '2017-18', '2018-19', '2019-20']
all_game_ids = []

print("Collecting game IDs...")
for season in seasons:
    print(f"  {season}...", end='', flush=True)
    try:
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable='Regular Season'
        )
        df = finder.get_data_frames()[0]
        unique = df.drop_duplicates(subset=['GAME_ID'])
        game_ids = unique['GAME_ID'].astype(str).str.zfill(10).tolist()
        all_game_ids.extend(game_ids)
        print(f" {len(game_ids)} games ✅")
        time.sleep(random.uniform(0.8, 1.5))
    except Exception as e:
        print(f" Error: {e}")

print(f"\n✅ Total: {len(all_game_ids)} games to collect")
print()

# Save game IDs
with open('game_ids_2015_2019.pkl', 'wb') as f:
    pickle.dump(all_game_ids, f)

print("✅ Game IDs saved")
print()
print("="*80)
print("NEXT STEP: Use existing stealth extractor")
print("="*80)
print()
print("Run this command:")
print("  python3 ⚡_STEALTH_EXTRACTION.py --input game_ids_2015_2019.pkl --output PATTERNS_2015_2019.pkl")
print()
print("OR manually adapt the ultra-optimized extractor")
print("="*80)

