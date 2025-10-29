"""
Test if 2025-26 NBA season data is available
"""

from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime

print("="*80)
print("🏀 2025-26 NBA SEASON DATA VERIFICATION")
print("="*80)
print(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
print()

# Test 1: Player Stats
try:
    print("📊 Testing Player Stats...")
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season='2025-26',
        season_type_all_star='Regular Season',
        per_mode_detailed='PerGame'
    )
    df = stats.get_data_frames()[0]
    print(f"✅ Players with stats: {len(df)}")
    if len(df) > 0:
        print(f"   Sample: {df['PLAYER_NAME'].head(5).tolist()}")
        print(f"   Games played range: {df['GP'].min()}-{df['GP'].max()}")
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Test 2: Team Stats
try:
    print("🏀 Testing Team Stats...")
    team_stats = leaguedashteamstats.LeagueDashTeamStats(
        season='2025-26',
        season_type_all_star='Regular Season'
    )
    team_df = team_stats.get_data_frames()[0]
    print(f"✅ Teams with stats: {len(team_df)}")
    if len(team_df) > 0:
        print(f"   Games played range: {team_df['GP'].min()}-{team_df['GP'].max()}")
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Test 3: Today's Games
try:
    print("📅 Testing Today's Scoreboard...")
    scoreboard = scoreboardv2.ScoreboardV2()
    games_df = scoreboard.get_data_frames()[0]
    print(f"✅ Games today: {len(games_df)}")
    if len(games_df) > 0:
        print(f"   Games: {games_df[['GAMECODE', 'GAME_STATUS_TEXT']].head(3).to_dict('records')}")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("="*80)
print("CONCLUSION:")
print("  - If player stats = 0: Season hasn't started or API not updated")
print("  - If player stats > 0: Data is available, populate will work")
print("="*80)

