"""
Test: Why did we only get 5 games on Oct 27 instead of 11?
"""

import time
from nba_api.stats.endpoints import leaguegamelog

print("="*80)
print("🔍 TESTING OCT 27, 2025 GAMES")
print("="*80)

# Test for 2025-26 season
time.sleep(0.6)
games_df = leaguegamelog.LeagueGameLog(
    season='2025-26',
    season_type_all_star='Regular Season',
    date_from_nullable='10/27/2025',
    date_to_nullable='10/27/2025'
).get_data_frames()[0]

print(f"\n📊 Total rows in game log: {len(games_df)}")
print(f"📊 Unique game IDs: {len(games_df['GAME_ID'].unique())}")
print(f"📊 Unique teams: {len(games_df['TEAM_ABBREVIATION'].unique())}")

print("\n🏀 Games on Oct 27, 2025:")
for game_id in games_df['GAME_ID'].unique():
    game_rows = games_df[games_df['GAME_ID'] == game_id]
    teams = sorted(game_rows['TEAM_ABBREVIATION'].unique())
    if len(teams) >= 2:
        away = teams[0]
        home = teams[1]
        print(f"   {game_id}: {away} @ {home}")
    else:
        print(f"   {game_id}: Only 1 team? {teams}")

print("\n" + "="*80)

