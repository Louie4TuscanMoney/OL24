#!/usr/bin/env python3
"""
⚡ MERGE TEAM FEATURES WITH PBP PATTERNS
Enhance extracted patterns with team context
Time: 30 min
"""

import pickle
import pandas as pd
import re

print("="*60)
print("⚡ MERGING TEAM FEATURES")
print("="*60)

# Load extracted patterns
print("\n[1/3] Loading extracted patterns...")
with open('ULTRA_OPTIMIZED_PATTERNS_2021_2025.pkl', 'rb') as f:
    patterns = pickle.load(f)

print(f"✅ Loaded {len(patterns)} games")

# Load team stats
print("\n[2/3] Loading team stats...")
with open('team_stats_2024_25.pkl', 'rb') as f:
    team_stats = pickle.load(f)

print(f"✅ Loaded {len(team_stats)} teams")

def parse_matchup(matchup_str):
    """
    Parse matchup string to extract teams
    Examples: "LAL vs. GSW", "BOS @ MIA", "PHX vs CHI"
    """
    # Common patterns
    if ' vs. ' in matchup_str:
        parts = matchup_str.split(' vs. ')
        return parts[0].strip(), parts[1].strip()
    elif ' @ ' in matchup_str:
        parts = matchup_str.split(' @ ')
        return parts[1].strip(), parts[0].strip()  # Away @ Home → Home, Away
    elif ' vs ' in matchup_str:
        parts = matchup_str.split(' vs ')
        return parts[0].strip(), parts[1].strip()
    else:
        # Can't parse
        return None, None

# Merge features
print("\n[3/3] Merging team features...")

enhanced_patterns = []
success = 0
failed = 0

for pattern in patterns:
    matchup = pattern.get('matchup', '')
    
    home_team, away_team = parse_matchup(matchup)
    
    if home_team and away_team:
        # Get team stats (with defaults)
        home_stats = team_stats.get(home_team, {
            'offensive_rating': 110, 'defensive_rating': 110,
            'net_rating': 0, 'pace': 100, 'true_shooting_pct': 0.56
        })
        
        away_stats = team_stats.get(away_team, {
            'offensive_rating': 110, 'defensive_rating': 110,
            'net_rating': 0, 'pace': 100, 'true_shooting_pct': 0.56
        })
        
        # Add team features
        pattern['team_features'] = {
            'home_off_rating': home_stats['offensive_rating'],
            'home_def_rating': home_stats['defensive_rating'],
            'home_net_rating': home_stats['net_rating'],
            'home_pace': home_stats['pace'],
            
            'away_off_rating': away_stats['offensive_rating'],
            'away_def_rating': away_stats['defensive_rating'],
            'away_net_rating': away_stats['net_rating'],
            'away_pace': away_stats['pace'],
            
            # Derived features
            'net_rating_diff': home_stats['net_rating'] - away_stats['net_rating'],
            'pace_avg': (home_stats['pace'] + away_stats['pace']) / 2,
            'off_vs_def': home_stats['offensive_rating'] - away_stats['defensive_rating']
        }
        
        success += 1
    else:
        # Failed to parse - use defaults
        pattern['team_features'] = {
            'home_off_rating': 110, 'home_def_rating': 110,
            'home_net_rating': 0, 'home_pace': 100,
            'away_off_rating': 110, 'away_def_rating': 110,
            'away_net_rating': 0, 'away_pace': 100,
            'net_rating_diff': 0, 'pace_avg': 100, 'off_vs_def': 0
        }
        failed += 1
    
    enhanced_patterns.append(pattern)
    
    if (success + failed) % 500 == 0:
        print(f"  Processed {success + failed} games...")

print(f"\n✅ Merged {success} games successfully")
print(f"⚠️  Used defaults for {failed} games")

# Save enhanced patterns
output_file = 'ENHANCED_PATTERNS_WITH_TEAM.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(enhanced_patterns, f)

print(f"\n✅ Saved to: {output_file}")

print(f"\n📊 Feature Count:")
print(f"   PBP features: 57")
print(f"   Team features: 11")
print(f"   Total: 68 features")

print(f"\n🚀 Next: python3 ⚡_3_add_players_simple.py")

print("="*60)

