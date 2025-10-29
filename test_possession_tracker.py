"""
TEST POSSESSION TRACKER
Validate against real NBA API play-by-play data
"""

from nba_api.stats.endpoints import playbyplayv2, boxscoreadvancedv2
import pandas as pd

print("="*80)
print("🏀 TESTING POSSESSION TRACKER WITH REAL NBA DATA")
print("="*80)
print()

# Test with a recent completed game
# Let's use a game from 2024-25 season (we know this data exists)
test_game_id = '0022400001'  # First game of 2024-25 season

print(f"📊 Fetching play-by-play for game: {test_game_id}")
print()

# 1. Get play-by-play data
try:
    pbp = playbyplayv2.PlayByPlayV2(game_id=test_game_id)
    plays_df = pbp.get_data_frames()[0]
    print(f"✅ Fetched {len(plays_df)} plays")
    print()
    
    # Show sample plays
    print("Sample plays:")
    print(plays_df[['PERIOD', 'PCTIMESTRING', 'EVENTMSGTYPE', 'HOMEDESCRIPTION', 'VISITORDESCRIPTION']].head(20))
    print()
    
except Exception as e:
    print(f"❌ Error fetching play-by-play: {e}")
    print("Trying with a different game...")
    
    # Try a more recent game
    test_game_id = '0022301230'  # Late season game from 2023-24
    try:
        pbp = playbyplayv2.PlayByPlayV2(game_id=test_game_id)
        plays_df = pbp.get_data_frames()[0]
        print(f"✅ Fetched {len(plays_df)} plays")
    except Exception as e2:
        print(f"❌ Still error: {e2}")
        print("Note: Play-by-play may not be available for current season yet")
        import sys
        sys.exit(1)

print()

# 2. Get advanced box score (has actual possession count!)
try:
    advanced = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=test_game_id)
    team_advanced = advanced.get_data_frames()[0]  # Team stats
    
    print("✅ Advanced box score (GROUND TRUTH):")
    print(team_advanced[['TEAM_NAME', 'POSS', 'PACE']].to_string(index=False))
    print()
    
    # Extract ground truth possessions
    ground_truth_possessions = {}
    for _, row in team_advanced.iterrows():
        ground_truth_possessions[row['TEAM_ID']] = row['POSS']
    
except Exception as e:
    print(f"⚠️  Could not fetch advanced stats: {e}")
    ground_truth_possessions = None

print()
print("="*80)
print("EVENT TYPE BREAKDOWN")
print("="*80)

# Analyze event types in the data
event_counts = plays_df['EVENTMSGTYPE'].value_counts().sort_index()
print("\nEvent types present:")
for event_type, count in event_counts.items():
    event_name = {
        1: 'Made Shot',
        2: 'Missed Shot',
        3: 'Free Throw',
        4: 'Rebound',
        5: 'Turnover',
        6: 'Foul',
        7: 'Violation',
        8: 'Substitution',
        9: 'Timeout',
        10: 'Jump Ball',
        12: 'Start Period',
        13: 'End Period'
    }.get(event_type, f'Unknown ({event_type})')
    print(f"  {event_type}: {event_name:20s} - {count:4d} occurrences")

print()
print("="*80)
print("POSSESSION TRACKING LOGIC TEST")
print("="*80)
print()

# Manual possession counting using simplified logic
def count_possessions_simple(plays_df):
    """
    Count possessions using simplified rules:
    1. Made shot ends possession (unless followed by OREB)
    2. Defensive rebound ends possession (and switches team)
    3. Turnover ends possession
    4. End of period ends possession
    """
    
    team_possessions = {}
    current_possession_team = None
    
    for i, play in plays_df.iterrows():
        event_type = play['EVENTMSGTYPE']
        home_desc = str(play['HOMEDESCRIPTION']) if pd.notna(play['HOMEDESCRIPTION']) else ''
        away_desc = str(play['VISITORDESCRIPTION']) if pd.notna(play['VISITORDESCRIPTION']) else ''
        
        # Determine which team is involved
        team_id = play['PLAYER1_TEAM_ID']
        
        # Skip non-possession events
        if event_type in [6, 8, 9, 12]:  # Foul, Sub, Timeout, Start Period
            continue
        
        # Initialize possession if needed
        if current_possession_team is None:
            current_possession_team = team_id
        
        # Check for possession-ending events
        possession_ended = False
        
        # 1. Made shot (type 1)
        if event_type == 1:
            # Check if next play is offensive rebound
            if i + 1 < len(plays_df):
                next_play = plays_df.iloc[i + 1]
                next_desc = str(next_play['HOMEDESCRIPTION']) if pd.notna(next_play['HOMEDESCRIPTION']) else ''
                next_desc += str(next_play['VISITORDESCRIPTION']) if pd.notna(next_play['VISITORDESCRIPTION']) else ''
                
                if 'Offensive' not in next_desc or 'Rebound' not in next_desc:
                    possession_ended = True
            else:
                possession_ended = True
        
        # 2. Defensive rebound (type 4)
        elif event_type == 4:
            if 'Defensive' in home_desc or 'Defensive' in away_desc:
                possession_ended = True
                # Switch possession to rebounding team
                current_possession_team = team_id
        
        # 3. Turnover (type 5)
        elif event_type == 5:
            possession_ended = True
        
        # 4. End of period (type 13)
        elif event_type == 13:
            possession_ended = True
        
        # Record possession end
        if possession_ended and current_possession_team:
            if current_possession_team not in team_possessions:
                team_possessions[current_possession_team] = 0
            team_possessions[current_possession_team] += 1
            
            # Reset possession (will be set on next play)
            current_possession_team = None
    
    return team_possessions

# Run simplified counting
calculated_possessions = count_possessions_simple(plays_df)

print("RESULTS:")
print()

# Get team names
team_names = {}
for _, play in plays_df.iterrows():
    if pd.notna(play['PLAYER1_TEAM_ID']):
        # Try to get team name from play descriptions
        if play['HOMEDESCRIPTION'] and pd.notna(play['PLAYER1_TEAM_ID']):
            team_id = play['PLAYER1_TEAM_ID']
            if team_id not in team_names:
                team_names[team_id] = f"Team {team_id}"

for team_id, poss_count in calculated_possessions.items():
    team_name = team_names.get(team_id, f"Team {team_id}")
    print(f"  {team_name}: {poss_count} possessions")

print()

if ground_truth_possessions:
    print("COMPARISON WITH GROUND TRUTH:")
    print()
    
    for team_id, true_poss in ground_truth_possessions.items():
        calc_poss = calculated_possessions.get(team_id, 0)
        diff = calc_poss - true_poss
        pct_diff = (diff / true_poss * 100) if true_poss > 0 else 0
        
        status = "✅" if abs(diff) <= 2 else "⚠️"  # Within 2 possessions is acceptable
        
        print(f"{status} Team {team_id}:")
        print(f"     Ground truth: {true_poss}")
        print(f"     Calculated:   {calc_poss}")
        print(f"     Difference:   {diff:+d} ({pct_diff:+.1f}%)")
        print()

print("="*80)
print("CONCLUSION:")
print("="*80)
print()
print("If difference is within ±2 possessions: ✅ ACCURATE")
print("If difference is 3-5 possessions: ⚠️  NEEDS REFINEMENT")
print("If difference is >5 possessions: ❌ LOGIC ERROR")
print()
print("Common issues:")
print("  - Free throw sequences not handled correctly")
print("  - Offensive rebounds not detected properly")
print("  - End-of-quarter possessions double-counted")
print("  - Technical fouls creating phantom possessions")

