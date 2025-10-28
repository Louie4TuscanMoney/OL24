"""
Check team box score columns
"""

import time
from nba_api.stats.endpoints import boxscoretraditionalv2

game_id = '0022400100'

print(f"Testing game: {game_id}\n")

time.sleep(0.6)
dfs = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).get_data_frames()

print(f"Got {len(dfs)} dataframes:\n")

for i, df in enumerate(dfs):
    print(f"DataFrame {i}: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst row:")
    print(df.iloc[0])
    print("\n" + "="*80 + "\n")

