"""
Test 01: Can we load the data?
Simple test to verify CSV files are readable
"""

import pandas as pd
import sys

print("="*60)
print("TEST 01: DATA LOADING")
print("="*60)

try:
    # Try loading just 2020-21 (smallest file)
    print("\nAttempting to load NBA_PBP_2020-21.csv...")
    df = pd.read_csv('../../All Of Our Data/NBA_PBP_2020-21.csv')
    
    print(f"✅ Successfully loaded!")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    # Check key columns exist
    required_cols = ['URL', 'Date', 'Quarter', 'SecLeft', 'HomeTeam', 'AwayTeam', 'HomeScore', 'AwayScore', 'WinningTeam']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"\n❌ Missing required columns: {missing}")
        sys.exit(1)
    
    print(f"✅ All required columns present")
    
    # Check unique games
    n_games = df['URL'].nunique()
    print(f"   Unique games: {n_games:,}")
    
    # Check date range
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Check teams
    teams = sorted(df['HomeTeam'].dropna().unique())
    print(f"   Teams: {len(teams)} ({', '.join(teams[:5])}, ...)")
    
    # Check for missing scores
    print(f"\nData quality check:")
    print(f"   Missing HomeScore: {df['HomeScore'].isnull().sum():,}")
    print(f"   Missing AwayScore: {df['AwayScore'].isnull().sum():,}")
    print(f"   Missing Quarter: {df['Quarter'].isnull().sum():,}")
    
    print("\n" + "="*60)
    print("✅ TEST 01 PASSED - Data loads correctly")
    print("="*60)
    print("\nNext: Run test_02_single_game.py")
    
except FileNotFoundError:
    print("\n❌ ERROR: CSV file not found")
    print("   Check path: ../../All Of Our Data/NBA_PBP_2020-21.csv")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

