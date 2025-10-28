"""
TEST NBA API - Check if we can fetch Oct 27 games
"""

import time
from datetime import datetime, timedelta

try:
    from nba_api.stats.endpoints import leaguegamelog
    print("✅ nba_api imported successfully\n")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

print("="*80)
print("🏀 TESTING NBA API - FETCHING OCT 27 GAMES")
print("="*80)

# Test different date formats
test_dates = [
    ('10/27/2024', '10/27/2024'),
    ('10/22/2024', '10/28/2024'),  # Week range
    ('10/01/2024', '10/31/2024'),  # Whole month
]

for date_from, date_to in test_dates:
    print(f"\n📅 Testing: {date_from} → {date_to}")
    
    try:
        time.sleep(0.6)
        games_df = leaguegamelog.LeagueGameLog(
            season='2024-25',
            season_type_all_star='Regular Season',
            date_from_nullable=date_from,
            date_to_nullable=date_to
        ).get_data_frames()[0]
        
        game_count = len(games_df['GAME_ID'].unique())
        print(f"   ✅ Found {game_count} games")
        
        if game_count > 0:
            print(f"   📊 Sample games:")
            for game_id in games_df['GAME_ID'].unique()[:3]:
                game_rows = games_df[games_df['GAME_ID'] == game_id]
                teams = game_rows['TEAM_ABBREVIATION'].unique()
                date = game_rows['GAME_DATE'].iloc[0]
                print(f"      {game_id}: {teams[0]} vs {teams[1] if len(teams) > 1 else '?'} on {date}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*80)
print("🔍 DIAGNOSIS COMPLETE")
print("="*80)

