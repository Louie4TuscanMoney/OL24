"""
Test 03: Extract ALL games from ONE season
Verify extraction works for entire season before scaling to 6 seasons
"""

import pandas as pd
import numpy as np
import sys

print("="*60)
print("TEST 03: ALL GAMES EXTRACTION (2020-21 SEASON)")
print("="*60)

try:
    # Load 2020-21 season
    print("\nLoading NBA_PBP_2020-21.csv...")
    df = pd.read_csv('../../All Of Our Data/NBA_PBP_2020-21.csv')
    print(f"✅ Loaded {len(df):,} plays")
    
    # Calculate game time for all rows
    df['GameTime'] = (df['Quarter'] - 1) * 720 + (720 - df['SecLeft'])
    
    # Target: 6:00 Q2
    target_time = 1080
    
    # Process each game
    print(f"\nProcessing {df['URL'].nunique()} games...")
    
    games = []
    errors = 0
    
    for game_url in df['URL'].unique():
        try:
            game_df = df[df['URL'] == game_url].copy()
            
            # Get game info
            info = game_df.iloc[0]
            home_team = info['HomeTeam']
            away_team = info['AwayTeam']
            date = info['Date']
            winner = info['WinningTeam']
            
            # Find halftime
            game_df['TimeDiff'] = abs(game_df['GameTime'] - target_time)
            halftime_idx = game_df['TimeDiff'].idxmin()
            ht_row = game_df.loc[halftime_idx]
            
            # Get final
            final_row = game_df.iloc[-1]
            
            # Extract scores
            home_ht = ht_row['HomeScore']
            away_ht = ht_row['AwayScore']
            home_final = final_row['HomeScore']
            away_final = final_row['AwayScore']
            
            # Calculate differentials
            diff_ht = home_ht - away_ht
            diff_final = home_final - away_final
            delta = diff_final - diff_ht
            
            games.append({
                'game_id': game_url,
                'date': date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score_ht': home_ht,
                'away_score_ht': away_ht,
                'home_score_final': home_final,
                'away_score_final': away_final,
                'differential_ht': diff_ht,
                'differential_final': diff_final,
                'delta_differential': delta,
                'winner': winner
            })
            
        except Exception as e:
            errors += 1
            print(f"   ⚠️ Error in game {game_url}: {e}")
    
    # Create dataframe
    games_df = pd.DataFrame(games)
    
    print(f"✅ Extracted {len(games_df)} games")
    if errors > 0:
        print(f"   ⚠️ {errors} games had errors")
    
    # Display summary statistics
    print(f"\n📊 SUMMARY STATISTICS")
    print(f"   Games: {len(games_df)}")
    print(f"   Date range: {games_df['date'].min()} to {games_df['date'].max()}")
    print(f"   Teams: {games_df['home_team'].nunique()}")
    
    print(f"\n   Halftime differential:")
    print(f"      Mean: {games_df['differential_ht'].mean():+.1f}")
    print(f"      Std: {games_df['differential_ht'].std():.1f}")
    print(f"      Range: [{games_df['differential_ht'].min():+.0f}, {games_df['differential_ht'].max():+.0f}]")
    
    print(f"\n   Final differential:")
    print(f"      Mean: {games_df['differential_final'].mean():+.1f}")
    print(f"      Std: {games_df['differential_final'].std():.1f}")
    print(f"      Range: [{games_df['differential_final'].min():+.0f}, {games_df['differential_final'].max():+.0f}]")
    
    print(f"\n   Delta (change halftime → final):")
    print(f"      Mean: {games_df['delta_differential'].mean():+.1f}")
    print(f"      Std: {games_df['delta_differential'].std():.1f}")
    print(f"      Range: [{games_df['delta_differential'].min():+.0f}, {games_df['delta_differential'].max():+.0f}]")
    
    # Show first 5 games
    print(f"\n📋 First 5 games:")
    for i, row in games_df.head(5).iterrows():
        print(f"\n   {row['away_team']} @ {row['home_team']} ({row['date']})")
        print(f"      Halftime: {row['home_team']} {row['differential_ht']:+d}")
        print(f"      Final: {row['home_team']} {row['differential_final']:+d}")
        print(f"      Delta: {row['delta_differential']:+d}")
        print(f"      Winner: {row['winner']}")
    
    # Save for next test
    print(f"\n💾 Saving to test_games.csv...")
    games_df.to_csv('test_games.csv', index=False)
    print(f"   ✅ Saved {len(games_df)} games")
    
    print("\n" + "="*60)
    print("✅ TEST 03 PASSED - All games extracted successfully")
    print("="*60)
    print(f"\nReady to:")
    print(f"  1. Scale to all 6 seasons")
    print(f"  2. Build Dejavu model")
    print(f"\nVerify test_games.csv looks correct, then continue.")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

