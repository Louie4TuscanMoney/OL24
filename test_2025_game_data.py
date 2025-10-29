"""
Test if 2025-26 games have actual box score data
"""

from nba_api.stats.endpoints import boxscoretraditionalv2, leaguegamefinder
from datetime import datetime

print("="*80)
print("🏀 TESTING 2025-26 GAME DATA AVAILABILITY")
print("="*80)
print()

# Get recent games
print("1️⃣  Fetching recent 2025-26 games...")
games = leaguegamefinder.LeagueGameFinder(
    season_nullable='2025-26',
    season_type_nullable='Regular Season'
)
games_df = games.get_data_frames()[0]

print(f"   Found {len(games_df)} team-game entries")
print(f"   Unique games: {len(games_df['GAME_ID'].unique())}")

# Show first few games with their dates
print()
print("📅 First 5 games:")
for _, row in games_df.head(5).iterrows():
    print(f"   {row['GAME_ID']} - {row['GAME_DATE']} - {row['MATCHUP']} - {row['WL']}")

print()
print("="*80)
print("2️⃣  Testing box score data for first game...")
print("="*80)

first_game = games_df.iloc[0]['GAME_ID']
print(f"   Game ID: {first_game}")
print()

try:
    box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=first_game)
    player_stats = box.get_data_frames()[0]
    
    print(f"   Players in box score: {len(player_stats)}")
    
    if len(player_stats) > 0:
        print()
        print("   ✅ DATA EXISTS! Sample:")
        print(player_stats[['TEAM_ID', 'PLAYER_NAME', 'MIN', 'PTS', 'REB', 'AST']].head(5))
        print()
        print(f"   Total stats available:")
        print(f"     Columns: {list(player_stats.columns)}")
    else:
        print()
        print("   ❌ EMPTY! Box score has no player data.")
        print()
        print("   This means:")
        print("     - Game was scheduled but not played yet")
        print("     - Or NBA API hasn't updated 2025-26 data")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

print()
print("="*80)
print("🔍 DIAGNOSIS")
print("="*80)
print()

# Check if WL (Win/Loss) is populated
games_with_results = games_df[games_df['WL'].notna()]
print(f"Games with W/L results: {len(games_with_results)}")

if len(games_with_results) == 0:
    print()
    print("❌ NO GAMES HAVE BEEN PLAYED YET!")
    print()
    print("Solution: Use 2024-25 data to populate and test the system")
    print()
    print("Run this instead:")
    print("  python3 backend/services/populate_possessions_hybrid.py 2024-25 30")
else:
    print()
    print(f"✅ {len(games_with_results)} games have been completed")
    print(f"   Date range: {games_with_results['GAME_DATE'].min()} to {games_with_results['GAME_DATE'].max()}")

