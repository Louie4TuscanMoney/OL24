#!/usr/bin/env python3
"""
🎯 DUAL BRANCH PREDICTION SYSTEM

At 6:00 2Q (18 minutes into game):
┌─────────────────────────────────────────────┐
│ Input: 18-minute differential pattern       │
│ Current: LAL -7 vs CHI                     │
└──────────────┬──────────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
  ┌─────▼─────┐  ┌──▼──────┐
  │ BRANCH A  │  │BRANCH B │
  │ Halftime  │  │ Final   │
  │ (6 min)   │  │ (30 min)│
  └─────┬─────┘  └──┬──────┘
        │           │
  ┌─────▼─────┐  ┌──▼──────┐
  │Pred: -8.5 │  │Pred: -10│
  │1H Spread  │  │Full Game│
  └─────┬─────┘  └──┬──────┘
        │           │
  ┌─────▼─────────┬─▼───────┐
  │ BET OPTIONS:            │
  │ 1. LAL 1H -7.5 (edge!)  │
  │ 2. LAL FG -8.0 (edge!)  │
  │ 3. Both (diversified!)  │
  └─────────────────────────┘

BENEFITS:
✅ 2x betting opportunities per game
✅ Diversified predictions (uncorrelated)
✅ Different market liquidity
✅ Halftime bets settle faster
✅ Final bets have more time to be right

STRATEGY:
- Halftime model: Higher confidence, smaller edges
- Final model: Lower confidence, bigger edges
- Bet on both when both show edges!
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "1. ML/1. Dejavu"))

from dejavu_model import DejavuForecaster

print("="*80)
print("🎯 BUILDING DUAL BRANCH PREDICTION SYSTEM")
print("="*80)

print(f"""
ARCHITECTURE:
   Input: 18-minute pattern (@ 6:00 2Q)
   
   Branch A (Halftime):
   - Target: Differential at halftime (24 min)
   - Prediction window: 6 minutes
   - Use case: 1H spread betting
   - Expected MAE: 5-6 points (shorter window)
   
   Branch B (Final):
   - Target: Differential at final (48 min)
   - Prediction window: 30 minutes
   - Use case: Full game spread betting
   - Expected MAE: 8-12 points (longer window)
   
IMPLEMENTATION PLAN:
""")

# Step 1: Load data with both targets
print("[1/7] Loading complete game data...")

timeseries_path = Path(__file__).parent / "1. ML/1. Dejavu Deployment/complete_timeseries.pkl"

with open(timeseries_path, 'rb') as f:
    df = pickle.load(f)

print(f"✅ Loaded {len(df)} games")

# Check what we have
sample = df.iloc[0]
print(f"\n📊 Data structure check:")
print(f"   Type: {type(sample)}")

# Access the data properly
if 'pattern' in df.columns:
    print(f"   ✅ Has 'pattern' column")
if 'diff_at_halftime' in df.columns:
    print(f"   ✅ Has 'diff_at_halftime' (Branch A target)")
if 'diff_at_final' in df.columns:
    print(f"   ✅ Has 'diff_at_final' (Branch B target)")

# Step 2: Create Branch A training data (Halftime)
print("\n[2/7] Creating BRANCH A (Halftime) training data...")

branch_a_data = []

for idx, row in df.iterrows():
    if 'pattern' in row and 'diff_at_halftime' in row:
        pattern = row['pattern']
        if isinstance(pattern, (list, np.ndarray)) and len(pattern) >= 18:
            branch_a_data.append({
                'pattern': np.array(pattern[:18]) if isinstance(pattern, list) else pattern[:18],
                'diff_at_halftime': row['diff_at_halftime'],
                'game_id': row.get('game_id', idx),
                'date': row.get('date', ''),
                'home_team': row.get('home_team', ''),
                'away_team': row.get('away_team', ''),
                'season': row.get('season', '')
            })

branch_a_df = pd.DataFrame(branch_a_data)
print(f"✅ Branch A: {len(branch_a_df)} games")

# Step 3: Create Branch B training data (Final)
print("\n[3/7] Creating BRANCH B (Final) training data...")

branch_b_data = []

for idx, row in df.iterrows():
    if 'pattern' in row and 'diff_at_final' in row:
        pattern = row['pattern']
        if isinstance(pattern, (list, np.ndarray)) and len(pattern) >= 18:
            branch_b_data.append({
                'pattern': np.array(pattern[:18]) if isinstance(pattern, list) else pattern[:18],
                'diff_at_halftime': row['diff_at_final'],  # Note: using diff_at_halftime key but final value!
                'game_id': row.get('game_id', idx),
                'date': row.get('date', ''),
                'home_team': row.get('home_team', ''),
                'away_team': row.get('away_team', ''),
                'season': row.get('season', '')
            })

branch_b_df = pd.DataFrame(branch_b_data)
print(f"✅ Branch B: {len(branch_b_df)} games")

# Step 4: Split data (80% train, 20% test)
print("\n[4/7] Splitting data...")

from sklearn.model_selection import train_test_split

# Branch A split
train_a, test_a = train_test_split(branch_a_df, test_size=0.2, random_state=42)
print(f"✅ Branch A: {len(train_a)} train, {len(test_a)} test")

# Branch B split
train_b, test_b = train_test_split(branch_b_df, test_size=0.2, random_state=42)
print(f"✅ Branch B: {len(train_b)} train, {len(test_b)} test")

# Step 5: Train Branch A (Halftime predictor)
print("\n[5/7] Training BRANCH A: Halftime Predictor...")

model_a = DejavuForecaster(k=500)
model_a.fit(train_a)

print(f"✅ Branch A trained: {len(model_a.database)} patterns")

# Test Branch A
print(f"\n   Testing Branch A on {len(test_a)} games...")

preds_a = []
actuals_a = []

for idx, row in test_a.iterrows():
    pred = model_a.predict(row['pattern'])
    actual = row['diff_at_halftime']
    preds_a.append(pred)
    actuals_a.append(actual)

mae_a = np.mean(np.abs(np.array(preds_a) - np.array(actuals_a)))
print(f"   ✅ Branch A MAE: {mae_a:.2f} points (Halftime prediction)")

# Step 6: Train Branch B (Final predictor)
print("\n[6/7] Training BRANCH B: Final Score Predictor...")

model_b = DejavuForecaster(k=500)
model_b.fit(train_b)

print(f"✅ Branch B trained: {len(model_b.database)} patterns")

# Test Branch B
print(f"\n   Testing Branch B on {len(test_b)} games...")

preds_b = []
actuals_b = []

for idx, row in test_b.iterrows():
    pred = model_b.predict(row['pattern'])
    actual = row['diff_at_halftime']  # This is actually final due to how we built branch_b_data
    preds_b.append(pred)
    actuals_b.append(actual)

mae_b = np.mean(np.abs(np.array(preds_b) - np.array(actuals_b)))
print(f"   ✅ Branch B MAE: {mae_b:.2f} points (Final score prediction)")

# Step 7: Save both models
print("\n[7/7] Saving dual branch models...")

model_a.save('1. ML/1. Dejavu Deployment/dejavu_HALFTIME_k500.pkl')
model_b.save('1. ML/1. Dejavu Deployment/dejavu_FINAL_k500.pkl')

print("\n" + "="*80)
print("🎉 DUAL BRANCH SYSTEM COMPLETE!")
print("="*80)

print(f"""
RESULTS:

Branch A (Halftime):
   MAE: {mae_a:.2f} points
   Use for: 1H spread betting
   Settlement: 24 minutes (halftime)
   
Branch B (Final):
   MAE: {mae_b:.2f} points  
   Use for: Full game spread betting
   Settlement: 48 minutes (final)

BETTING STRATEGY:

At 6:00 2Q, you get TWO predictions:
1. Halftime spread: Use Branch A
2. Final spread: Use Branch B

Example:
   Current: LAL -7 vs CHI
   Branch A predicts: LAL -8.5 at halftime
   Branch B predicts: LAL -10 final
   
   Check odds:
   - 1H spread: LAL -7.5 → Edge of 1.0 pt → BET!
   - FG spread: LAL -8.0 → Edge of 2.0 pts → BET!
   
   Place BOTH bets (diversified exposure!)

EXPECTED PERFORMANCE:
   Total bets per night: 8-12 (4-6 per branch)
   Combined MAE: ~{(mae_a + mae_b)/2:.1f} points
   Diversification: Reduces variance
   
LAUNCH STRATEGY:
   ✅ Use both branches Monday
   ✅ Test performance separately
   ✅ Adjust weights if one performs better
   ✅ Feedback loop adapts over time!
""")

print("\n📝 FILES CREATED:")
print(f"   - dejavu_HALFTIME_k500.pkl (MAE: {mae_a:.2f})")
print(f"   - dejavu_FINAL_k500.pkl (MAE: {mae_b:.2f})")

print("\n🚀 READY FOR MONDAY LAUNCH!")

