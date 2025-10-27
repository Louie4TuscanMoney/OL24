#!/usr/bin/env python3
"""
🔍 REAL 2025 MAE - Parse Play-by-Play Data

Extract actual minute-by-minute differentials from today's 8 games
Calculate REAL 2025 MAE to inform Monday launch strategy
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "1. ML/1. Dejavu"))

from nba_api.stats.endpoints import playbyplayv2

# Today's games
TODAYS_GAMES = [
    {"id": "0012500065", "away": "BKN", "home": "TOR", "away_final": 114, "home_final": 119},
    {"id": "0012500071", "away": "MIN", "home": "PHI", "away_final": 110, "home_final": 126},
    {"id": "0012500066", "away": "CHA", "home": "NYK", "away_final": 108, "home_final": 113},
    {"id": "0012500006", "away": "MEM", "home": "MIA", "away_final": 141, "home_final": 125},
    {"id": "0012500067", "away": "DEN", "home": "OKC", "away_final": 91, "home_final": 94},
    {"id": "0012500068", "away": "IND", "home": "SAS", "away_final": 104, "home_final": 133},
    {"id": "0012500069", "away": "LAC", "home": "GSW", "away_final": 106, "home_final": 103},
    {"id": "0012500007", "away": "SAC", "home": "LAL", "away_final": 117, "home_final": 116},
]

print("="*80)
print("🔍 PARSING REAL PLAY-BY-PLAY DATA - 2025 MAE CALCULATION")
print("="*80)

# Load model
print("\n[1/5] Loading model...")
from dejavu_model import DejavuForecaster
import pickle

model_path = Path(__file__).parent / "1. ML/1. Dejavu Deployment/dejavu_k500.pkl"
sys.modules['__main__'].DejavuForecaster = DejavuForecaster

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"✅ Model: {len(model.database)} patterns, k={model.k}")

# Parse play-by-play
print("\n[2/5] Fetching and parsing play-by-play data...")
print("(~2 minutes for 8 games)\n")

def extract_18min_pattern(game_id, home_team, away_team):
    """
    Extract score differential at each minute 0-18
    Returns: 18-element array of differentials
    """
    try:
        # Fetch play-by-play
        pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
        plays_df = pbp.get_data_frames()[0]
        
        # Parse score margin
        # SCOREMARGIN format: "TIE", "+5", "-3", etc.
        differentials = [0]  # Start at 0-0
        
        # Go through plays chronologically
        for idx, play in plays_df.iterrows():
            period = play['PERIOD']
            pctimestring = play['PCTIMESTRING']  # e.g., "11:45", "0:23"
            score_margin = play['SCOREMARGIN']
            
            # Only process Q1 and Q2 (first 24 minutes)
            if period > 2:
                break
            
            # Parse time remaining in period
            if pd.notna(pctimestring):
                try:
                    parts = pctimestring.split(':')
                    mins_remaining = int(parts[0])
                    secs_remaining = int(parts[1])
                    
                    # Calculate elapsed game time
                    if period == 1:
                        elapsed = 12 - mins_remaining - (secs_remaining / 60.0)
                    else:  # period == 2
                        elapsed = 12 + (12 - mins_remaining - (secs_remaining / 60.0))
                    
                    # Only care about first 18 minutes
                    if elapsed > 18:
                        break
                    
                    # Parse score margin
                    if pd.notna(score_margin):
                        if score_margin == 'TIE':
                            diff = 0
                        else:
                            # Format: "+5" or "-3" (from HOME team perspective)
                            diff = int(score_margin)
                        
                        # Store (minute, differential) pair
                        minute = int(elapsed)
                        if 0 <= minute <= 18 and len(differentials) <= minute + 1:
                            while len(differentials) <= minute:
                                differentials.append(differentials[-1])  # Fill gaps
                            differentials[minute] = diff
                
                except (ValueError, IndexError, AttributeError):
                    continue
        
        # Ensure we have exactly 18 values
        while len(differentials) < 18:
            differentials.append(differentials[-1])
        
        pattern = np.array(differentials[:18])
        
        return pattern
        
    except Exception as e:
        print(f"      ❌ Parsing failed: {e}")
        return None

# Test each game
predictions = []
actuals = []
errors = []
successful_games = []

for i, game in enumerate(TODAYS_GAMES, 1):
    print(f"Game {i}/8: {game['away']} @ {game['home']}")
    
    try:
        # Extract 18-minute pattern
        pattern = extract_18min_pattern(game['id'], game['home'], game['away'])
        
        if pattern is None:
            print(f"   ❌ Could not extract pattern")
            continue
        
        print(f"   ✅ Pattern extracted: {pattern[:5]}... (first 5 mins)")
        
        # Make prediction
        prediction = model.predict(pattern)
        
        # Actual final differential
        actual = game['home_final'] - game['away_final']
        error = abs(prediction - actual)
        
        predictions.append(prediction)
        actuals.append(actual)
        errors.append(error)
        successful_games.append(game)
        
        print(f"   Predicted: {prediction:+.1f}, Actual: {actual:+d}, Error: {error:.1f}")
        
        # Rate limit - be nice to NBA API
        time.sleep(1)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        time.sleep(1)
        continue

# Calculate MAE
print("\n" + "="*80)
print("[3/5] CALCULATING REAL 2025 MAE")
print("="*80)

if len(errors) > 0:
    mae = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    print(f"\n📊 REAL 2025 PERFORMANCE:")
    print(f"   Games successfully tested: {len(errors)}/{len(TODAYS_GAMES)}")
    print(f"   Mean Absolute Error (MAE): {mae:.2f} points")
    print(f"   Median Error: {median_error:.2f} points")
    print(f"   Min Error: {min_error:.2f} points")
    print(f"   Max Error: {max_error:.2f} points")
    
    print(f"\n🎯 COMPARISON:")
    print(f"   Training MAE (2015-2021): 5.39 points (documented)")
    print(f"   Test Set MAE (2015-2021): 6.00 points (just tested)")
    print(f"   2025 Preseason MAE: {mae:.2f} points ← REAL DATA!")
    print(f"   Drift: {mae - 6.00:+.2f} points")
    
    # Assessment
    print("\n" + "="*80)
    print("[4/5] ACCURACY ASSESSMENT")
    print("="*80)
    
    if mae < 7:
        print(f"\n✅ EXCELLENT - Model has NOT drifted!")
        print(f"   MAE {mae:.2f} is very close to test set 6.00")
        print(f"   Game has not changed significantly")
        print(f"   LAUNCH WITH CONFIDENCE!")
        readiness = 95
    elif mae < 9:
        print(f"\n✅ GOOD - Acceptable drift")
        print(f"   MAE {mae:.2f} vs test set 6.00")
        print(f"   Drift of {mae-6.00:.2f} points is reasonable")
        print(f"   Launch conservatively, monitor closely")
        readiness = 90
    elif mae < 11:
        print(f"\n⚠️  ACCEPTABLE - Moderate drift")
        print(f"   MAE {mae:.2f} vs test set 6.00")  
        print(f"   Model has drifted {mae-6.00:.2f} points")
        print(f"   Launch in conservative mode")
        print(f"   Consider recalibration after Week 1")
        readiness = 85
    else:
        print(f"\n❌ CONCERNING - Significant drift")
        print(f"   MAE {mae:.2f} is much higher than 6.00")
        print(f"   Drift of {mae-6.00:.2f} points is large")
        print(f"   Recommend paper trading Week 1")
        print(f"   Collect data and recalibrate")
        readiness = 75
    
    # Show individual games
    print(f"\n📋 Game-by-Game Results:")
    for i, (game, pred, actual, err) in enumerate(zip(successful_games, predictions, actuals, errors), 1):
        status = "✅" if err < 8 else "⚠️" if err < 12 else "❌"
        print(f"   {status} {game['away']} @ {game['home']}: "
              f"Pred {pred:+.1f}, Actual {actual:+d}, Error {err:.1f}")
    
    # Launch recommendation
    print("\n" + "="*80)
    print("[5/5] MONDAY LAUNCH DECISION")
    print("="*80)
    
    print(f"\n📊 System Readiness: {readiness}%")
    
    if readiness >= 90:
        print(f"\n✅ GO FOR LAUNCH - Standard Mode")
        print(f"   Use normal Kelly sizing")
        print(f"   Max bet: $750")
        print(f"   Min edge: 3 points")
    elif readiness >= 85:
        print(f"\n✅ GO FOR LAUNCH - Conservative Mode")
        print(f"   Use 50-75% Kelly sizing")
        print(f"   Max bet: $375-500")
        print(f"   Min edge: 4 points")
        print(f"   Monitor closely Week 1")
    else:
        print(f"\n⚠️  GO FOR LAUNCH - Paper Trade Mode")
        print(f"   Make predictions but don't bet")
        print(f"   Collect Week 1 data")
        print(f"   Real betting Week 2 after validation")
    
    print(f"\n💡 Learnings for Tomorrow's Live Test:")
    print(f"   - Validate these results with fresh games")
    print(f"   - Check if MAE is consistent")
    print(f"   - Adjust strategy based on results")
    
else:
    print("\n❌ No games successfully tested")
    print("   Need to debug play-by-play parsing")

print("\n" + "="*80)
print("REAL 2025 MAE CALCULATION COMPLETE!")
print("="*80)

