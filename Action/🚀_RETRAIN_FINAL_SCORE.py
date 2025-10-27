#!/usr/bin/env python3
"""
🚀 RETRAIN MODEL FOR FINAL SCORE PREDICTION

CURRENT: Predict halftime differential (not useful for betting!)
NEW: Predict FINAL differential (what we actually bet on!)

This will:
1. Load complete game data with final scores
2. Extract 18-minute patterns (same)
3. Change target from halftime → FINAL score
4. Retrain Dejavu with k=500
5. Test on held-out data
6. Save as dejavu_FINAL_k500.pkl
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time

sys.path.insert(0, str(Path(__file__).parent / "1. ML/1. Dejavu"))

print("="*80)
print("🚀 RETRAINING DEJAVU FOR FINAL SCORE PREDICTION")
print("="*80)

# Step 1: Load complete game data
print("\n[1/6] Loading complete game data with FINAL scores...")

data_path = Path(__file__).parent / "1. ML/1. Dejavu Deployment/complete_games.csv"
df = pd.read_csv(data_path)

print(f"✅ Loaded {len(df)} games")
print(f"   Columns: {list(df.columns)}")

# Step 2: Check data quality
print("\n[2/6] Checking data quality...")

print(f"   Games with halftime data: {df['differential_ht'].notna().sum()}")
print(f"   Games with final data: {df['differential_final'].notna().sum()}")

# Sample
sample = df.iloc[0]
print(f"\n   Sample game:")
print(f"   {sample['away_team']} @ {sample['home_team']} ({sample['date']})")
print(f"   Halftime: {sample['differential_ht']:+.0f}")
print(f"   Final: {sample['differential_final']:+.0f}")

# Step 3: Load the pattern data
print("\n[3/6] Loading minute-by-minute patterns...")

# Load the timeseries data that has patterns
timeseries_path = Path(__file__).parent / "1. ML/1. Dejavu Deployment/complete_timeseries.pkl"

with open(timeseries_path, 'rb') as f:
    timeseries_data = pickle.load(f)

print(f"✅ Loaded timeseries data")
print(f"   Type: {type(timeseries_data)}")

# Check structure
if isinstance(timeseries_data, pd.DataFrame):
    print(f"   Shape: {timeseries_data.shape}")
    print(f"   Columns: {list(timeseries_data.columns)[:10]}...")
    
    # Extract patterns and final scores
    print("\n[4/6] Creating training data with FINAL score targets...")
    
    training_data = []
    
    for idx, row in timeseries_data.iterrows():
        if 'pattern' in row and 'differential_final' in row:
            training_data.append({
                'pattern': row['pattern'],
                'diff_at_final': row['differential_final'],  # Changed from halftime!
                'game_id': row.get('game_id', idx),
                'date': row.get('date', ''),
                'home_team': row.get('home_team', ''),
                'away_team': row.get('away_team', '')
            })
        elif idx < 10:  # Debug first few
            print(f"   Row {idx} keys: {list(row.keys())[:5]}...")
    
    print(f"✅ Created {len(training_data)} training examples")
    
    if len(training_data) == 0:
        print("\n⚠️  No patterns extracted - checking data structure...")
        print(f"   First row type: {type(timeseries_data.iloc[0])}")
        print(f"   First row: {timeseries_data.iloc[0]}")
    
elif isinstance(timeseries_data, dict):
    print(f"   Keys: {list(timeseries_data.keys())}")
    
    # Try to extract
    print("\n[4/6] Extracting patterns from dictionary structure...")
    
    # This depends on how data is structured
    # Will need to adapt based on actual structure

else:
    print(f"   Unknown structure: {type(timeseries_data)}")

print("\n" + "="*80)
print("🔍 CURRENT STATUS")
print("="*80)

print(f"""
WHAT WE DISCOVERED:
   ✅ complete_games.csv has 'differential_final' column
   ✅ We can load the data
   ⚠️  Need to check timeseries structure for patterns
   
NEXT STEPS:
   1. Examine timeseries_data structure (2 min)
   2. Extract (pattern_18min, diff_final) pairs
   3. Create train/test split
   4. Train new Dejavu model
   5. Test performance
   6. Save dejavu_FINAL_k500.pkl
   
EXPECTED OUTCOME:
   Model that predicts: 18-min pattern → Final score
   Can compare to live spread at 6:00 2Q
   Find betting edges!
   
TIME TO COMPLETE: 30-60 minutes
""")

