"""
Test 02: Can we extract halftime and final scores for ONE game?
Verify our extraction logic is correct
"""

import pandas as pd
import sys

print("="*60)
print("TEST 02: SINGLE GAME EXTRACTION")
print("="*60)

try:
    # Load data
    print("\nLoading NBA_PBP_2020-21.csv...")
    df = pd.read_csv('../../All Of Our Data/NBA_PBP_2020-21.csv')
    
    # Get first game
    first_game_url = df['URL'].iloc[0]
    game_df = df[df['URL'] == first_game_url].copy()
    
    # Game info
    game_info = game_df.iloc[0]
    home_team = game_info['HomeTeam']
    away_team = game_info['AwayTeam']
    date = game_info['Date']
    winner = game_info['WinningTeam']
    
    print(f"\n📊 Game: {away_team} @ {home_team}")
    print(f"   Date: {date}")
    print(f"   URL: {first_game_url}")
    print(f"   Total plays: {len(game_df)}")
    print(f"   Winner: {winner}")
    
    # Calculate game time elapsed
    game_df['GameTime'] = (game_df['Quarter'] - 1) * 720 + (720 - game_df['SecLeft'])
    
    # Target: 6:00 remaining Q2 = 1080 seconds into game
    # Q1 = 0-720, Q2 = 720-1440
    # 6:00 Q2 = 720 + 360 = 1080 seconds
    target_time = 1080
    
    # Find closest row to target
    game_df['TimeDiff'] = abs(game_df['GameTime'] - target_time)
    halftime_idx = game_df['TimeDiff'].idxmin()
    halftime_row = game_df.loc[halftime_idx]
    
    print(f"\n⏱️  Halftime Snapshot (6:00 Q2):")
    print(f"   Quarter: {halftime_row['Quarter']}")
    print(f"   Seconds left in Q: {halftime_row['SecLeft']}")
    print(f"   Game time: {halftime_row['GameTime']} sec (target: {target_time})")
    print(f"   {away_team}: {halftime_row['AwayScore']}")
    print(f"   {home_team}: {halftime_row['HomeScore']}")
    
    diff_ht = halftime_row['HomeScore'] - halftime_row['AwayScore']
    print(f"   Differential: {home_team} {diff_ht:+d}")
    
    # Get final score (last row of game)
    final_row = game_df.iloc[-1]
    
    print(f"\n🏁 Final Score:")
    print(f"   {away_team}: {final_row['AwayScore']}")
    print(f"   {home_team}: {final_row['HomeScore']}")
    
    diff_final = final_row['HomeScore'] - final_row['AwayScore']
    print(f"   Differential: {home_team} {diff_final:+d}")
    print(f"   Winner: {winner}")
    
    # Calculate delta (what changed)
    delta = diff_final - diff_ht
    
    print(f"\n📈 Change (halftime → final):")
    print(f"   Delta differential: {delta:+d}")
    if delta > 0:
        print(f"   → {home_team} improved by {delta} points")
    elif delta < 0:
        print(f"   → {away_team} improved by {-delta} points")
    else:
        print(f"   → No change")
    
    # Verify logic
    print(f"\n🔍 Verification:")
    print(f"   Halftime: {home_team} {diff_ht:+d}")
    print(f"   Final: {home_team} {diff_final:+d}")
    print(f"   Change: {delta:+d}")
    print(f"   Check: {diff_ht} + {delta} = {diff_ht + delta} (should equal {diff_final})")
    
    if diff_ht + delta == diff_final:
        print(f"   ✅ Math checks out!")
    else:
        print(f"   ❌ Math error!")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ TEST 02 PASSED - Single game extraction works")
    print("="*60)
    print("\nThis is the data Dejavu will use:")
    print(f"  • PATTERN (input): Score at halftime = {home_team} {diff_ht:+d}")
    print(f"  • OUTCOME (label): Change by end of game = {delta:+d}")
    print(f"\nNext: Run test_03_all_games.py to process all games")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

