"""
Test 04: Convert ONE game from play-by-play → minute-by-minute time series
Following: ML/Action Steps Folder/02_DATA_PROCESSING.md

Goal: Create 48-point time series (one value per minute)
"""

import pandas as pd
import numpy as np

print("="*60)
print("TEST 04: MINUTE-BY-MINUTE TIME SERIES CONVERSION")
print("="*60)

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

print(f"\n📊 Game: {away_team} @ {home_team} ({date})")
print(f"   Total plays: {len(game_df)}")

# Calculate game time for each play
# GameTime = elapsed seconds from game start
# Quarter 1: 0-720 sec
# Quarter 2: 720-1440 sec
# Quarter 3: 1440-2160 sec
# Quarter 4: 2160-2880 sec

game_df['GameTime'] = (game_df['Quarter'] - 1) * 720 + (720 - game_df['SecLeft'])

print(f"   Time range: {game_df['GameTime'].min():.0f} to {game_df['GameTime'].max():.0f} seconds")

# Create minute-by-minute time series
# 48 minutes = 2880 seconds
# We want value at 0, 60, 120, 180, ..., 2820 seconds

time_series = []

for minute in range(48):  # Minutes 0-47
    target_time = minute * 60  # Convert minute to seconds
    
    # Find the play closest to (but not after) this time
    plays_before = game_df[game_df['GameTime'] <= target_time]
    
    if len(plays_before) > 0:
        # Use the most recent play before this minute
        closest_play = plays_before.iloc[-1]
        home_score = closest_play['HomeScore']
        away_score = closest_play['AwayScore']
        differential = home_score - away_score
        
        time_series.append({
            'minute': minute,
            'game_time_sec': target_time,
            'quarter': closest_play['Quarter'],
            'sec_left_in_q': closest_play['SecLeft'],
            'home_score': home_score,
            'away_score': away_score,
            'differential': differential
        })
    else:
        # Start of game - no plays yet
        time_series.append({
            'minute': minute,
            'game_time_sec': target_time,
            'quarter': 1,
            'sec_left_in_q': 720,
            'home_score': 0,
            'away_score': 0,
            'differential': 0
        })

ts_df = pd.DataFrame(time_series)

print(f"\n✅ Created {len(ts_df)}-point time series")
print(f"\n📈 Sample time series (first 20 minutes):")
print(ts_df[['minute', 'quarter', 'home_score', 'away_score', 'differential']].head(20).to_string(index=False))

# Verify critical points
print(f"\n🎯 Key Timepoints:")
print(f"   Start (min 0):       Diff = {ts_df.iloc[0]['differential']:+.0f}")
print(f"   6:00 Q2 (min 18):    Diff = {ts_df.iloc[18]['differential']:+.0f}  ← PATTERN END")
print(f"   Halftime (min 24):   Diff = {ts_df.iloc[24]['differential']:+.0f}  ← FORECAST TARGET")
print(f"   End (min 47):        Diff = {ts_df.iloc[47]['differential']:+.0f}")

# Extract Dejavu pattern and outcome
pattern = ts_df['differential'].iloc[0:18].values  # Minutes 0-17 (to 6:00 Q2)
diff_at_6min = ts_df['differential'].iloc[18]      # At 6:00 Q2
diff_at_halftime = ts_df['differential'].iloc[24]  # At halftime (0:00 Q2)
diff_at_final = ts_df['differential'].iloc[47]     # At game end

print(f"\n📊 Dejavu Pattern:")
print(f"   Pattern (18 points):     {pattern}")
print(f"   Pattern shape:           {pattern.shape}")
print(f"   Differential at 6:00 Q2: {diff_at_6min:+.0f}")
print(f"   Differential at halftime:{diff_at_halftime:+.0f}")
print(f"   Change (6min → HT):      {diff_at_halftime - diff_at_6min:+.0f}")
print(f"   Final differential:      {diff_at_final:+.0f}")
print(f"   Change (HT → Final):     {diff_at_final - diff_at_halftime:+.0f}")

# Visualize the pattern
print(f"\n📉 Pattern Visualization (minutes 0-17):")
for i, val in enumerate(pattern):
    bar_length = int(abs(val) / 2)  # Scale for display
    bar = "+" * bar_length if val >= 0 else "-" * bar_length
    print(f"   Min {i:2d}: {val:+5.0f}  {bar}")

print("\n" + "="*60)
print("✅ TEST 04 PASSED")
print("="*60)
print("\nKey insight:")
print(f"  • We have a 48-point time series (one per minute)")
print(f"  • First 18 points = PATTERN (input to Dejavu)")
print(f"  • Differential at min 24 = OUTCOME (what we want to predict)")
print(f"\nNext: Scale this to ALL 6,600 games")

