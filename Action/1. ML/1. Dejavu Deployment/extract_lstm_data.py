"""
Extract 6-step outcomes for LSTM training
LSTM predicts minutes 18-23 (6 steps from pattern end to halftime)

Following: ML/Action Steps Folder/06_INFORMER_TRAINING.md
"""

import pickle
import numpy as np
from pathlib import Path

print("="*80)
print("EXTRACTING 6-STEP OUTCOMES FOR LSTM")
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
print(f"   Training: {len(train_df)}")
print(f"   Validation: {len(val_df)}")
print(f"   Calibration: {len(cal_df)}")
print(f"   Test: {len(test_df)}")

# Extract 6-step outcomes for all splits
print(f"\nExtracting 6-step outcomes (minutes 18-23)...")

for split_name, df in [('train', train_df), ('validation', val_df), 
                        ('calibration', cal_df), ('test', test_df)]:
    outcomes_6step = []
    
    for idx, row in df.iterrows():
        # Full timeseries has 48 points (minutes 0-47)
        # Pattern is minutes 0-17 (indices 0-17)
        # Outcome should be minutes 18-23 (indices 18-23)
        timeseries = row['timeseries']
        outcome = timeseries[18:24]  # Extract minutes 18-23
        
        outcomes_6step.append(outcome)
    
    # Add to dataframe
    df['outcome_6step'] = outcomes_6step
    
    print(f"   {split_name}: extracted {len(outcomes_6step)} outcomes, shape {outcomes_6step[0].shape}")

# Verify extraction
print(f"\n🔍 Verifying extraction (first training game):")
first_game = train_df.iloc[0]
print(f"   Game: {first_game['away_team']} @ {first_game['home_team']} ({first_game['date']})")
print(f"   Pattern (min 0-17):   {first_game['pattern']}")
print(f"   Outcome (min 18-23):  {first_game['outcome_6step']}")
print(f"   Halftime (min 24):    {first_game['diff_at_halftime']}")
print(f"\n   Check: outcome[-1] at min 23 = {first_game['outcome_6step'][-1]}")
print(f"          Should be close to halftime at min 24 = {first_game['diff_at_halftime']}")

# Save updated splits
print(f"\n💾 Saving updated splits with 6-step outcomes...")
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
print("✅ DATA READY FOR LSTM")
print("="*80)
print(f"\nEach game now has:")
print(f"  • pattern (18,): Input to LSTM")
print(f"  • outcome_6step (6,): Target for LSTM")
print(f"\nNext: Build LSTM model with exact specifications:")
print(f"  • hidden_size = 64")
print(f"  • num_layers = 2")
print(f"  • dropout = 0.1")
print(f"  • forecast_horizon = 6")

