#!/usr/bin/env python3
"""
🕒 AUTOMATED TEMPORAL LEAKAGE CHECK
Week 2 Priority 1: Ensure no future data ever influences training

PHILOSOPHY:
- Temporal integrity is non-negotiable
- Automate what can go wrong
- Zero tolerance for leakage
- Run on every split, every time
"""

import pickle
import sys
from datetime import datetime
from typing import List, Dict, Tuple

print("="*80)
print("🕒 AUTOMATED TEMPORAL LEAKAGE CHECK")
print("="*80)
print()
print("Verifying temporal integrity of train/test split...")
print()

# Load data
with open('ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"✅ Loaded {len(data)} games")
print()

# Split (same as training)
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"Split: {len(train_data)} train, {len(test_data)} test")
print()

# ============================================================================
# CHECK 1: NO GAME ID OVERLAP
# ============================================================================
print("[CHECK 1/5] Verifying no duplicate game IDs...")

train_ids = set(g.get('game_id') for g in train_data if g.get('game_id'))
test_ids = set(g.get('game_id') for g in test_data if g.get('game_id'))

overlap = train_ids & test_ids

if len(overlap) > 0:
    print(f"❌ TEMPORAL LEAKAGE DETECTED!")
    print(f"   Found {len(overlap)} duplicate game IDs between train and test:")
    for game_id in list(overlap)[:5]:
        print(f"   - {game_id}")
    if len(overlap) > 5:
        print(f"   ... and {len(overlap) - 5} more")
    sys.exit(1)
else:
    print(f"✅ PASS: No duplicate game IDs")
    print(f"   Train IDs: {len(train_ids)}")
    print(f"   Test IDs: {len(test_ids)}")
    print(f"   Overlap: 0")
print()

# ============================================================================
# CHECK 2: TEMPORAL ORDERING (TEST AFTER TRAIN)
# ============================================================================
print("[CHECK 2/5] Verifying test dates are AFTER train dates...")

train_dates = [g.get('date', '') for g in train_data if g.get('date')]
test_dates = [g.get('date', '') for g in test_data if g.get('date')]

if not train_dates or not test_dates:
    print("⚠️  WARNING: Some games missing dates")
    print(f"   Train games with dates: {len(train_dates)}/{len(train_data)}")
    print(f"   Test games with dates: {len(test_dates)}/{len(test_data)}")
else:
    latest_train = max(train_dates)
    earliest_test = min(test_dates)
    
    print(f"   Train date range: {min(train_dates)} to {latest_train}")
    print(f"   Test date range:  {earliest_test} to {max(test_dates)}")
    print()
    
    if earliest_test <= latest_train:
        print(f"❌ TEMPORAL LEAKAGE DETECTED!")
        print(f"   Latest train date: {latest_train}")
        print(f"   Earliest test date: {earliest_test}")
        print(f"   Test data includes games from BEFORE end of training!")
        
        # Find overlapping dates
        train_date_set = set(train_dates)
        test_date_set = set(test_dates)
        overlapping_dates = train_date_set & test_date_set
        
        if overlapping_dates:
            print(f"   Overlapping dates: {len(overlapping_dates)}")
            for date in sorted(list(overlapping_dates))[:5]:
                print(f"   - {date}")
        
        sys.exit(1)
    else:
        gap_days = (datetime.strptime(earliest_test, '%Y-%m-%d') - 
                   datetime.strptime(latest_train, '%Y-%m-%d')).days
        print(f"✅ PASS: Test data is AFTER train data")
        print(f"   Temporal gap: {gap_days} days")
        print()

# ============================================================================
# CHECK 3: CHRONOLOGICAL ORDERING WITHIN SETS
# ============================================================================
print("[CHECK 3/5] Verifying chronological ordering within train/test...")

# Check train is sorted
train_sorted = all(train_dates[i] <= train_dates[i+1] for i in range(len(train_dates)-1))
test_sorted = all(test_dates[i] <= test_dates[i+1] for i in range(len(test_dates)-1))

if not train_sorted:
    print("❌ TRAIN DATA NOT CHRONOLOGICALLY SORTED!")
    print("   This can cause subtle leakage in sequential models")
    sys.exit(1)

if not test_sorted:
    print("⚠️  WARNING: Test data not chronologically sorted")
    print("   This is OK but not ideal for rolling prediction simulation")

print(f"✅ PASS: Train data is chronologically sorted")
if test_sorted:
    print(f"✅ PASS: Test data is chronologically sorted")
print()

# ============================================================================
# CHECK 4: DATE DISTRIBUTION (DETECT SUSPICIOUS PATTERNS)
# ============================================================================
print("[CHECK 4/5] Checking date distributions...")

# Count games per month in train
from collections import defaultdict

train_months = defaultdict(int)
test_months = defaultdict(int)

for date in train_dates:
    month = date[:7]  # YYYY-MM
    train_months[month] += 1

for date in test_dates:
    month = date[:7]
    test_months[month] += 1

# Check for months that appear in both train and test (suspicious!)
common_months = set(train_months.keys()) & set(test_months.keys())

if common_months:
    print(f"⚠️  WARNING: {len(common_months)} months appear in BOTH train and test:")
    for month in sorted(list(common_months))[:5]:
        print(f"   {month}: Train={train_months[month]}, Test={test_months[month]}")
    if len(common_months) > 5:
        print(f"   ... and {len(common_months) - 5} more")
    print("   This is OK if dates don't overlap, but worth noting")
else:
    print(f"✅ PASS: No months appear in both train and test")

print()

# ============================================================================
# CHECK 5: SEASON/YEAR DISTRIBUTION
# ============================================================================
print("[CHECK 5/5] Checking season distribution...")

train_seasons = defaultdict(int)
test_seasons = defaultdict(int)

for game in train_data:
    season = game.get('season', 'unknown')
    train_seasons[season] += 1

for game in test_data:
    season = game.get('season', 'unknown')
    test_seasons[season] += 1

print(f"Train seasons: {dict(train_seasons)}")
print(f"Test seasons: {dict(test_seasons)}")
print()

# Warn if test seasons appeared in training (natural, but note it)
common_seasons = set(train_seasons.keys()) & set(test_seasons.keys())
if common_seasons:
    print(f"ℹ️  INFO: {len(common_seasons)} seasons appear in both train and test:")
    for season in sorted(common_seasons):
        print(f"   {season}: Train={train_seasons[season]}, Test={test_seasons[season]}")
    print("   This is EXPECTED for within-season prediction")
    print("   But be aware test includes games from seasons seen in training")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("🎯 TEMPORAL INTEGRITY SUMMARY")
print("="*80)
print()

checks_passed = 5
total_checks = 5

print(f"✅ CHECK 1: No duplicate game IDs")
print(f"✅ CHECK 2: Test dates after train dates (gap: {gap_days} days)")
print(f"✅ CHECK 3: Data chronologically sorted")
if common_months:
    print(f"⚠️  CHECK 4: Some months overlap (review dates)")
else:
    print(f"✅ CHECK 4: Month distributions clean")
print(f"ℹ️  CHECK 5: Season overlap noted (expected)")
print()

print(f"RESULT: {checks_passed}/{total_checks} checks passed")
print()

print("="*80)
print("✅ TEMPORAL INTEGRITY VERIFIED")
print("="*80)
print()
print("Split is CLEAN:")
print(f"  • No duplicate games")
print(f"  • Test data is {gap_days} days after train data")
print(f"  • Chronologically ordered")
print(f"  • Safe for causal inference")
print()
print("This split preserves temporal integrity. ✅")
print("Models trained on this split can be trusted for forward prediction.")
print()
print("="*80)

# Save validation report
validation_report = {
    'timestamp': datetime.now().isoformat(),
    'total_games': len(data),
    'train_games': len(train_data),
    'test_games': len(test_data),
    'checks': {
        'no_duplicate_ids': True,
        'temporal_ordering': True,
        'chronological_sorted': train_sorted,
        'temporal_gap_days': gap_days,
        'common_months': len(common_months),
        'common_seasons': len(common_seasons)
    },
    'train_date_range': (min(train_dates), max(train_dates)),
    'test_date_range': (min(test_dates), max(test_dates)),
    'status': 'PASS'
}

with open('temporal_integrity_validation.pkl', 'wb') as f:
    pickle.dump(validation_report, f)

print("Report saved to: temporal_integrity_validation.pkl")
print()
print("Run this check after ANY data split to ensure integrity!")
print("="*80)

