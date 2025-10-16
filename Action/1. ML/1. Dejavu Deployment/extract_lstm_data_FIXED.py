"""
Extract 7-step outcomes for LSTM training (FIXED)
LSTM must predict minutes 18-24 (7 steps to reach halftime at minute 24)

CRITICAL FIX: forecast_horizon = 7 (not 6!)
"""

import pickle
import numpy as np
from pathlib import Path

print("="*80)
print("EXTRACTING 7-STEP OUTCOMES FOR LSTM (CORRECTED)")
print("="*80)

# Load all splits
splits_dir = Path('splits/')

print("\nLoading splits...")
with open(splits_dir / 'train.pkl', 'rb') as f:
    train_df = pickle.load(f)
with open(splits_dir / 'validation.pkl', 'rb') as f:
    val_df = pickle.load(f)
with open(splits_dir / 'calibration.pkl', 'rb') as f:
    cal_df = pickle.load(f)
with open(splits_dir / 'test.pkl', 'rb') as f:
    test_df = pickle.load(f)

print(f"✅ Loaded all splits")

# Extract 7-step outcomes for all splits
print(f"\nExtracting 7-step outcomes (minutes 18-24)...")
print(f"  THIS INCLUDES HALFTIME AT MINUTE 24!")

for split_name, df in [('train', train_df), ('validation', val_df), 
                        ('calibration', cal_df), ('test', test_df)]:
    outcomes_7step = []
    
    for idx, row in df.iterrows():
        # Full timeseries has 48 points (minutes 0-47)
        # Pattern is minutes 0-17 (indices 0-17)
        # Outcome should be minutes 18-24 (indices 18-24) - 7 STEPS
        timeseries = row['timeseries']
        outcome = timeseries[18:25]  # Extract minutes 18-24 (7 points)
        
        # Verify
        if len(outcome) != 7:
            print(f"ERROR: outcome length = {len(outcome)}, should be 7!")
            exit(1)
        
        outcomes_7step.append(outcome)
    
    # Add to dataframe
    df['outcome_7step'] = outcomes_7step
    
    print(f"   {split_name}: extracted {len(outcomes_7step)} outcomes, shape {outcomes_7step[0].shape}")

# Verify extraction
print(f"\n🔍 Verifying extraction (first training game):")
first_game = train_df.iloc[0]
print(f"   Game: {first_game['away_team']} @ {first_game['home_team']} ({first_game['date']})")
print(f"   Pattern (min 0-17):   {first_game['pattern']}  (18 points)")
print(f"   Outcome (min 18-24):  {first_game['outcome_7step']}  (7 points)")
print(f"   diff_at_halftime:     {first_game['diff_at_halftime']}  (minute 24)")
print(f"\n   ✅ Check: outcome_7step[-1] = {first_game['outcome_7step'][-1]} == diff_at_halftime = {first_game['diff_at_halftime']}? {first_game['outcome_7step'][-1] == first_game['diff_at_halftime']}")

if first_game['outcome_7step'][-1] != first_game['diff_at_halftime']:
    print(f"\n   ❌ ERROR: Mismatch! outcome_7step[-1] should equal diff_at_halftime!")
    exit(1)

# Save updated splits
print(f"\n💾 Saving updated splits with 7-step outcomes...")
with open(splits_dir / 'train.pkl', 'wb') as f:
    pickle.dump(train_df, f)
with open(splits_dir / 'validation.pkl', 'wb') as f:
    pickle.dump(val_df, f)
with open(splits_dir / 'calibration.pkl', 'wb') as f:
    pickle.dump(cal_df, f)
with open(splits_dir / 'test.pkl', 'wb') as f:
    pickle.dump(test_df, f)

print(f"   ✅ All splits updated and saved")

print(f"\n{'='*80}")
print("✅ DATA CORRECTED FOR LSTM")
print("="*80)
print(f"\nEach game now has:")
print(f"  • pattern (18,): Minutes 0-17 (input to LSTM)")
print(f"  • outcome_7step (7,): Minutes 18-24 (target for LSTM)")
print(f"  • outcome_7step[-1] = diff_at_halftime = minute 24 = TRUE HALFTIME")
print(f"\nNow LSTM will predict the SAME target as Dejavu!")
print(f"\nNext: Retrain LSTM with forecast_horizon=7")

