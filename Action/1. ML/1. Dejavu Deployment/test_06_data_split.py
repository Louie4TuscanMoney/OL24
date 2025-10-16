"""
Test 06: Split data into train/validation/calibration/test sets
Following: ML/Action Steps Folder/03_DATA_SPLITTING.md

CRITICAL REQUIREMENTS (from research):
1. Chronological split only (NO SHUFFLING!)
2. 60% train / 10% val / 15% cal / 15% test
3. Validate no temporal overlap
4. Minimum samples: train≥500, val≥100, cal≥100, test≥100
"""

import pandas as pd
import pickle
from pathlib import Path

print("="*80)
print("DATA SPLITTING (RESEARCH-BASED)")
print("="*80)

# Load complete dataset
print("\nLoading complete_timeseries.pkl...")
with open('complete_timeseries.pkl', 'rb') as f:
    games_df = pickle.load(f)

print(f"✅ Loaded {len(games_df)} games")
print(f"   Date range: {games_df['date'].min()} to {games_df['date'].max()}")

# CRITICAL: Convert date strings to datetime and sort (line 58-59 from 03_DATA_SPLITTING.md)
print("\nConverting dates to datetime...")
games_df['game_date'] = pd.to_datetime(games_df['date'])
print(f"✅ Converted dates to datetime")

print("\nSorting chronologically by date...")
games_df = games_df.sort_values('game_date').reset_index(drop=True)
print(f"✅ Sorted {len(games_df)} games chronologically")

# Split ratios (lines 33-36 from research)
train_ratio = 0.60
val_ratio = 0.10
calibration_ratio = 0.15
test_ratio = 0.15

# Verify ratios sum to 1.0
assert abs(train_ratio + val_ratio + calibration_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0!"

# Calculate split indices (lines 64-66)
n = len(games_df)
train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))
cal_end = int(n * (train_ratio + val_ratio + calibration_ratio))

# Split (lines 69-72)
print("\nCreating splits...")
train_df = games_df.iloc[:train_end].copy()
val_df = games_df.iloc[train_end:val_end].copy()
calibration_df = games_df.iloc[val_end:cal_end].copy()
test_df = games_df.iloc[cal_end:].copy()

print(f"✅ Created 4 splits")

# Validate no temporal overlap (lines 75-77)
print("\nValidating temporal ordering...")
try:
    assert train_df['game_date'].max() < val_df['game_date'].min(), "Train-Val overlap!"
    assert val_df['game_date'].max() < calibration_df['game_date'].min(), "Val-Cal overlap!"
    assert calibration_df['game_date'].max() < test_df['game_date'].min(), "Cal-Test overlap!"
    print("✅ No temporal overlap - all splits properly ordered")
except AssertionError as e:
    print(f"❌ ERROR: {e}")
    print(f"   Train last date: {train_df['game_date'].max()}")
    print(f"   Val first date: {val_df['game_date'].min()}")
    print(f"   Val last date: {val_df['game_date'].max()}")
    print(f"   Cal first date: {calibration_df['game_date'].min()}")
    print(f"   Cal last date: {calibration_df['game_date'].max()}")
    print(f"   Test first date: {test_df['game_date'].min()}")
    exit(1)

# Print split information (lines 80-88)
print(f"\n📊 DATA SPLIT (Chronological):")
print(f"  Training:    {len(train_df):4d} games ({train_ratio:.1%}) "
      f"[{train_df['game_date'].min().strftime('%Y-%m-%d')} to {train_df['game_date'].max().strftime('%Y-%m-%d')}]")
print(f"  Validation:  {len(val_df):4d} games ({val_ratio:.1%}) "
      f"[{val_df['game_date'].min().strftime('%Y-%m-%d')} to {val_df['game_date'].max().strftime('%Y-%m-%d')}]")
print(f"  Calibration: {len(calibration_df):4d} games ({calibration_ratio:.1%}) "
      f"[{calibration_df['game_date'].min().strftime('%Y-%m-%d')} to {calibration_df['game_date'].max().strftime('%Y-%m-%d')}]")
print(f"  Test:        {len(test_df):4d} games ({test_ratio:.1%}) "
      f"[{test_df['game_date'].min().strftime('%Y-%m-%d')} to {test_df['game_date'].max().strftime('%Y-%m-%d')}]")

# Verify minimum sample sizes (lines 219-224)
print(f"\n🔍 Verifying minimum sample requirements...")
min_requirements = {
    'Training': (train_df, 500),
    'Validation': (val_df, 100),
    'Calibration': (calibration_df, 100),
    'Test': (test_df, 100)
}

all_passed = True
for split_name, (split_df, min_required) in min_requirements.items():
    actual_count = len(split_df)
    passed = actual_count >= min_required
    status = "✅" if passed else "❌"
    print(f"  {status} {split_name}: {actual_count} games (min: {min_required})")
    if not passed:
        all_passed = False

if not all_passed:
    print("\n❌ ERROR: Some splits don't meet minimum requirements!")
    exit(1)

print("\n✅ All splits meet minimum sample requirements")

# Check for game ID overlaps (lines 188-201)
print(f"\n🔍 Checking for game ID overlaps...")
train_ids = set(train_df['game_id'])
val_ids = set(val_df['game_id'])
cal_ids = set(calibration_df['game_id'])
test_ids = set(test_df['game_id'])

overlaps = []
if train_ids & val_ids:
    overlaps.append(f"Train-Val: {len(train_ids & val_ids)} games")
if train_ids & cal_ids:
    overlaps.append(f"Train-Cal: {len(train_ids & cal_ids)} games")
if train_ids & test_ids:
    overlaps.append(f"Train-Test: {len(train_ids & test_ids)} games")
if val_ids & cal_ids:
    overlaps.append(f"Val-Cal: {len(val_ids & cal_ids)} games")
if val_ids & test_ids:
    overlaps.append(f"Val-Test: {len(val_ids & test_ids)} games")
if cal_ids & test_ids:
    overlaps.append(f"Cal-Test: {len(cal_ids & test_ids)} games")

if overlaps:
    print("❌ OVERLAP DETECTED:")
    for overlap in overlaps:
        print(f"  - {overlap}")
    exit(1)
else:
    print("✅ No game ID overlaps - each game in exactly one split")

# Save splits (lines 122-126)
print(f"\n💾 Saving splits...")
output_dir = Path('splits/')
output_dir.mkdir(exist_ok=True)

train_df.to_pickle(output_dir / 'train.pkl')
val_df.to_pickle(output_dir / 'validation.pkl')
calibration_df.to_pickle(output_dir / 'calibration.pkl')
test_df.to_pickle(output_dir / 'test.pkl')

print(f"   ✅ splits/train.pkl ({len(train_df)} games)")
print(f"   ✅ splits/validation.pkl ({len(val_df)} games)")
print(f"   ✅ splits/calibration.pkl ({len(calibration_df)} games)")
print(f"   ✅ splits/test.pkl ({len(test_df)} games)")

# Save metadata (lines 131-148)
metadata = {
    'split_date': pd.Timestamp.now().isoformat(),
    'total_games': len(games_df),
    'train_games': len(train_df),
    'val_games': len(val_df),
    'calibration_games': len(calibration_df),
    'test_games': len(test_df),
    'train_ratio': train_ratio,
    'val_ratio': val_ratio,
    'calibration_ratio': calibration_ratio,
    'test_ratio': test_ratio,
    'train_date_range': [train_df['game_date'].min().isoformat(), train_df['game_date'].max().isoformat()],
    'val_date_range': [val_df['game_date'].min().isoformat(), val_df['game_date'].max().isoformat()],
    'calibration_date_range': [calibration_df['game_date'].min().isoformat(), calibration_df['game_date'].max().isoformat()],
    'test_date_range': [test_df['game_date'].min().isoformat(), test_df['game_date'].max().isoformat()]
}

import json
with open(output_dir / 'split_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   ✅ splits/split_metadata.json")

# Summary statistics
print(f"\n{'='*80}")
print("✅ DATA SPLIT COMPLETE")
print("="*80)

print(f"\nSplit Summary:")
print(f"  Total games:      {len(games_df):,}")
print(f"  Training:         {len(train_df):,} ({len(train_df)/len(games_df)*100:.1f}%)")
print(f"  Validation:       {len(val_df):,} ({len(val_df)/len(games_df)*100:.1f}%)")
print(f"  Calibration:      {len(calibration_df):,} ({len(calibration_df)/len(games_df)*100:.1f}%)")
print(f"  Test:             {len(test_df):,} ({len(test_df)/len(games_df)*100:.1f}%)")

print(f"\nUsage by Model:")
print(f"  Dejavu:")
print(f"    - Build database: Training set ({len(train_df):,} patterns)")
print(f"    - Evaluate: Test set ({len(test_df):,} games)")
print(f"  LSTM (future):")
print(f"    - Train: Training set ({len(train_df):,} games)")
print(f"    - Validate: Validation set ({len(val_df):,} games)")
print(f"    - Test: Test set ({len(test_df):,} games)")
print(f"  Conformal (future):")
print(f"    - Calibrate: Calibration set ({len(calibration_df):,} games)")
print(f"    - Test: Test set ({len(test_df):,} games)")

print(f"\n{'='*80}")
print("✅ READY FOR DEJAVU MODEL BUILDING")
print("="*80)
print(f"\nNext: Build Dejavu K-NN forecaster using training set")
print(f"      ({len(train_df):,} patterns for the database)")

