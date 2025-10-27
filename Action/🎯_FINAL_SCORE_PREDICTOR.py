#!/usr/bin/env python3
"""
🎯 FINAL SCORE PREDICTOR - Predict from 6:00 2Q to End of Game

NEW PREDICTION TARGET:
- Input: Score at 6:00 remaining in 2Q (18 minutes into game)
- Output: FINAL score differential (not halftime!)

EXAMPLE:
At 6:00 2Q:
- Current: LAL 55, CHI 48 (LAL -7)
- Live spread: LAL -7.5, CHI +7.5
- Model predicts: LAL will win by 10 (final)
- Edge: 10 vs 7.5 = 2.5 points
- Bet: LAL -7.5 (we think they'll cover)

This is MORE VALUABLE than halftime prediction!
- Can bet on FULL GAME outcome
- More time for model to be right
- Bigger edges available
"""

import sys
import numpy as np
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).parent / "1. ML/1. Dejavu"))

print("="*80)
print("🎯 FINAL SCORE PREDICTOR - 18 Min → Final Differential")
print("="*80)

# Load model
print("\n[1/3] Loading Dejavu model...")
from dejavu_model import DejavuForecaster

model_path = Path(__file__).parent / "1. ML/1. Dejavu Deployment/dejavu_k500.pkl"
sys.modules['__main__'].DejavuForecaster = DejavuForecaster

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"✅ Model loaded: {len(model.database)} patterns")

# Check what model actually predicts
print("\n[2/3] Checking model's target variable...")

# Load training data to see what it was trained on
with open(Path(__file__).parent / "1. ML/1. Dejavu Deployment/splits/train.pkl", 'rb') as f:
    train_df = pickle.load(f)

print(f"✅ Training data: {len(train_df)} games")

# Check first few examples
print(f"\n📊 What does the model predict?")
sample = train_df.iloc[0]

print(f"\n   Sample game: {sample['away_team']} @ {sample['home_team']}")
print(f"   Pattern (18 min differentials): {sample['pattern']}")
print(f"   Target (diff_at_halftime): {sample['diff_at_halftime']}")

# Check if we have final score data
if 'diff_final' in train_df.columns or 'differential_final' in train_df.columns:
    print(f"\n   ✅ Training data HAS final score!")
    print(f"   We can retarget to predict final instead of halftime!")
elif 'home_score_final' in train_df.columns:
    print(f"\n   ✅ Can calculate final differential!")
else:
    print(f"\n   ⚠️  Only have halftime differential in training data")
    print(f"   Need to load full game data with final scores")

# Check cleaned_games.parquet for final scores
print("\n[3/3] Checking for full game data...")

try:
    import pandas as pd
    
    # Try CSV first (since parquet won't install)
    games_path = Path(__file__).parent / "1. ML/1. Dejavu Deployment/complete_games.csv"
    
    if games_path.exists():
        print(f"✅ Found complete_games.csv")
        df = pd.read_csv(games_path, nrows=5)
        print(f"   Columns: {list(df.columns)}")
        
        if 'differential_final' in df.columns or 'home_score_final' in df.columns:
            print(f"\n   ✅ PERFECT! We have final scores!")
            print(f"   Can retrain model to predict FINAL instead of HALFTIME")
        else:
            print(f"\n   Checking what we have...")
            print(df.head())
    
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*80)
print("🔧 SOLUTION: Retarget Model to Predict Final Score")
print("="*80)

print(f"""
CURRENT MODEL:
   Input:  18-minute differential pattern
   Output: Halftime differential (24 minutes)
   
NEW MODEL (What you want):
   Input:  18-minute differential pattern  
   Output: FINAL differential (48 minutes)
   
IMPLEMENTATION:

1. Load complete_games.csv (has final scores)

2. Create new training data:
   pattern = differentials[0:18]  # First 18 minutes
   target = final_differential     # Full game result
   
3. Train new Dejavu model:
   dejavu_final = DejavuForecaster(k=500)
   dejavu_final.fit(data_with_final_targets)
   
4. Save as dejavu_final_score.pkl

TIME REQUIRED: 30-60 minutes

EXPECTED MAE: 8-12 points (harder than halftime!)
   - More time = more uncertainty
   - But also more profitable (full game bets!)
   
BENEFITS:
   ✅ Bet on final outcome (more liquid markets)
   ✅ Compare to live spread (find edges)
   ✅ More time for model to be correct
   ✅ Bigger potential edges
""")

print("\n🎯 DECISION:")
print("   A. Use current model (halftime prediction) → Launch Monday")
print("   B. Retrain for final score (30-60 min) → Better system")
print("   C. Do both (ensemble halftime + final) → Best but complex")

print("\n💡 MY RECOMMENDATION:")
print("   Option B - Retrain for final score TONIGHT")
print("   This is what you actually want to bet on!")
print("   30-60 minutes well spent for launch Monday")

