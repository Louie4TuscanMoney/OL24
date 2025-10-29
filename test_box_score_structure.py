"""
Test boxscoretraditionalv2 structure to find team stats
"""

from nba_api.stats.endpoints import boxscoretraditionalv2
import pandas as pd

# Test with a recent game
game_id = '0022500001'  # First game of 2025-26

print(f"Testing game: {game_id}")
print("="*80)

try:
    box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    dfs = box.get_data_frames()
    
    print(f"Number of dataframes returned: {len(dfs)}")
    print()
    
    for i, df in enumerate(dfs):
        print(f"DataFrame {i}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"  First row sample: {df.iloc[0][['TEAM_ID', 'PLAYER_ID', 'PLAYER_NAME']].to_dict() if 'PLAYER_NAME' in df.columns else df.iloc[0].head(5).to_dict()}")
        else:
            print(f"  ⚠️  EMPTY DATAFRAME")
        print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("This might mean the game doesn't have data yet.")
    print("Try with a game from 2024-25 season:")
    print()
    
    # Try last season
    game_id = '0022301230'
    try:
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        dfs = box.get_data_frames()
        
        print(f"Testing 2024-25 game: {game_id}")
        print(f"Number of dataframes: {len(dfs)}")
        print()
        
        for i, df in enumerate(dfs):
            print(f"DataFrame {i}: {df.shape} - {list(df.columns)[:5]}")
            
    except Exception as e2:
        print(f"❌ Still error: {e2}")

