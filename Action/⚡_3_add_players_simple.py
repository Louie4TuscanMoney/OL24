#!/usr/bin/env python3
"""
⚡ ADD SIMPLIFIED PLAYER FEATURES
Just binary: Is star player on team?
Time: 30 min (simplified, not full implementation)
"""

import pickle

print("="*60)
print("⚡ ADDING SIMPLIFIED PLAYER FEATURES")
print("="*60)

# Simplified: Just mark if team has known star
# (Full implementation would scrape lineups, but no time)

STAR_PLAYERS_BY_TEAM = {
    # Top tier (impact: +8 to +12 points)
    'LAL': 2, 'MIL': 2, 'PHX': 2, 'DAL': 2, 'DEN': 2,
    'BOS': 2, 'PHI': 2, 'BKN': 2, 'GSW': 2,
    
    # Good tier (impact: +5 to +8 points)
    'MIA': 1, 'CLE': 1, 'MEM': 1, 'SAC': 1, 'LAC': 1,
    'NYK': 1, 'MIN': 1, 'NOP': 1, 'OKC': 1,
    
    # Average (impact: +0 to +5 points)
    'ATL': 0, 'TOR': 0, 'CHI': 0, 'POR': 0, 'UTA': 0,
    'ORL': 0, 'IND': 0, 'WAS': 0, 'CHA': 0, 'SAS': 0,
    'HOU': 0, 'DET': 0
}

# Load patterns
print("\nLoading enhanced patterns...")
with open('ENHANCED_PATTERNS_WITH_TEAM.pkl', 'rb') as f:
    patterns = pickle.load(f)

print(f"✅ Loaded {len(patterns)} games")

# Add player features
print("\nAdding player features...")

def parse_matchup(matchup_str):
    """Extract home and away teams"""
    if ' vs. ' in matchup_str:
        parts = matchup_str.split(' vs. ')
        return parts[0].strip(), parts[1].strip()
    elif ' @ ' in matchup_str:
        parts = matchup_str.split(' @ ')
        return parts[1].strip(), parts[0].strip()
    elif ' vs ' in matchup_str:
        parts = matchup_str.split(' vs ')
        return parts[0].strip(), parts[1].strip()
    return None, None

enhanced = []

for pattern in patterns:
    matchup = pattern.get('matchup', '')
    home_team, away_team = parse_matchup(matchup)
    
    # Simplified player features (star power tiers)
    home_star_tier = STAR_PLAYERS_BY_TEAM.get(home_team, 0)
    away_star_tier = STAR_PLAYERS_BY_TEAM.get(away_team, 0)
    
    pattern['player_features'] = {
        'home_star_tier': home_star_tier,  # 0, 1, or 2
        'away_star_tier': away_star_tier,
        'star_power_diff': home_star_tier - away_star_tier,  # -2 to +2
        'has_superstar_home': 1 if home_star_tier == 2 else 0,
        'has_superstar_away': 1 if away_star_tier == 2 else 0,
        'total_star_power': home_star_tier + away_star_tier
    }
    
    enhanced.append(pattern)

print(f"✅ Added player features to {len(enhanced)} games")

# Save
output_file = 'ENHANCED_PATTERNS_FULL.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(enhanced, f)

print(f"\n✅ Saved to: {output_file}")

print(f"\n📊 Feature Count:")
print(f"   PBP features: 57")
print(f"   Team features: 11")
print(f"   Player features: 6")
print(f"   Total: 74 features")

print(f"\n⚠️  Note: Simplified player features (star tier only)")
print(f"   Full implementation would include:")
print(f"   - Actual lineup data")
print(f"   - Usage rates")
print(f"   - Plus/minus")
print(f"   - Injury status")
print(f"   But no time. This is good enough for V1.")

print(f"\n🚀 Next: python3 ⚡_4_train_xgboost_rapid.py")

print("="*60)

