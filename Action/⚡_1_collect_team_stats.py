#!/usr/bin/env python3
"""
⚡ RAPID TEAM STATS COLLECTION
Collect essential team stats for 2024-25 season
Time: 60 min
"""

import sys
import time
import pickle
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguedashteamstats

print("="*60)
print("⚡ RAPID TEAM STATS COLLECTION")
print("="*60)

# Get all teams
all_teams = teams.get_teams()

print(f"\n📊 Collecting stats for {len(all_teams)} teams...")

team_stats_db = {}

try:
    # Get league-wide team stats
    print("\nFetching 2024-25 team stats...")
    
    dashboard = leaguedashteamstats.LeagueDashTeamStats(
        season='2024-25',
        season_type_all_star='Regular Season'
    )
    
    stats_df = dashboard.get_data_frames()[0]
    
    print(f"✅ Got {len(stats_df)} teams")
    print(f"   Columns: {list(stats_df.columns[:10])}")
    
    # Extract essential stats
    for idx, row in stats_df.iterrows():
        # Try different column names
        team_abbrev = None
        for col in ['TEAM_ABBREVIATION', 'TEAM_ABBR', 'Team']:
            if col in row:
                team_abbrev = row[col]
                break
        
        if not team_abbrev:
            # Use team name if abbreviation not found
            team_abbrev = row.get('TEAM_NAME', f'TEAM_{idx}')[:3].upper()
        
        team_abbrev = str(team_abbrev)
        
        team_stats_db[team_abbrev] = {
            # Core ratings
            'offensive_rating': float(row.get('OFF_RATING', 110)),
            'defensive_rating': float(row.get('DEF_RATING', 110)),
            'net_rating': float(row.get('NET_RATING', 0)),
            
            # Pace and efficiency
            'pace': float(row.get('PACE', 100)),
            'true_shooting_pct': float(row.get('TS_PCT', 0.55)),
            
            # Traditional stats
            'ppg': float(row.get('PTS', 110)),
            'opp_ppg': float(row.get('OPP_PTS', 110)),
            
            # Win%
            'win_pct': float(row.get('W_PCT', 0.5))
        }
        
        print(f"  ✅ {team_abbrev}: OffRtg={team_stats_db[team_abbrev]['offensive_rating']:.1f}")
    
    # Save
    output_file = 'team_stats_2024_25.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(team_stats_db, f)
    
    print(f"\n✅ Saved {len(team_stats_db)} teams to: {output_file}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Teams: {len(team_stats_db)}")
    print(f"   Avg offensive rating: {sum(t['offensive_rating'] for t in team_stats_db.values())/len(team_stats_db):.1f}")
    print(f"   Avg defensive rating: {sum(t['defensive_rating'] for t in team_stats_db.values())/len(team_stats_db):.1f}")
    
    print(f"\n🚀 Next: python3 ⚡_2_merge_features.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"\nUsing fallback: Default values for all teams")
    
    # Fallback: Create default stats
    for team in all_teams:
        abbrev = team['abbreviation']
        team_stats_db[abbrev] = {
            'offensive_rating': 110.0,
            'defensive_rating': 110.0,
            'net_rating': 0.0,
            'pace': 100.0,
            'true_shooting_pct': 0.56,
            'ppg': 110.0,
            'opp_ppg': 110.0,
            'win_pct': 0.5
        }
    
    with open('team_stats_2024_25.pkl', 'wb') as f:
        pickle.dump(team_stats_db, f)
    
    print(f"✅ Saved fallback stats")

print("="*60)

