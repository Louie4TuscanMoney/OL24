# Step 3: Data Splitting for All Three Models

**Objective:** Split data into train/validation/calibration/test sets with proper temporal ordering

**Duration:** 1 hour  
**Prerequisites:** Completed Step 2 (processed time series)  
**Output:** Four datasets ready for model development

---

## Action Items

### 3.1 Chronological Split Strategy

**File:** `scripts/split_data.py`

```python
"""
Split data for Informer, Conformal, and Dejavu
CRITICAL: Chronological split only (no shuffling!)
"""

import pandas as pd
import numpy as np
from pathlib import Path

class ChronologicalDataSplitter:
    """
    Split NBA data chronologically for time series forecasting
    """
    def __init__(
        self,
        train_ratio=0.60,
        val_ratio=0.10,
        calibration_ratio=0.15,
        test_ratio=0.15
    ):
        """
        Four-way split for conformal prediction
        """
        assert abs(train_ratio + val_ratio + calibration_ratio + test_ratio - 1.0) < 1e-6
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.calibration_ratio = calibration_ratio
        self.test_ratio = test_ratio
    
    def split(self, df):
        """
        Chronological split by game date
        
        Args:
            df: Halftime prediction dataset
        
        Returns:
            train_df, val_df, calibration_df, test_df
        """
        # Sort by date (CRITICAL!)
        df = df.sort_values('game_date').reset_index(drop=True)
        
        n = len(df)
        
        # Calculate split indices
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        cal_end = int(n * (self.train_ratio + self.val_ratio + self.calibration_ratio))
        
        # Split
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        calibration_df = df.iloc[val_end:cal_end].copy()
        test_df = df.iloc[cal_end:].copy()
        
        # Validate no temporal overlap
        assert train_df['game_date'].max() < val_df['game_date'].min()
        assert val_df['game_date'].max() < calibration_df['game_date'].min()
        assert calibration_df['game_date'].max() < test_df['game_date'].min()
        
        # Print split information
        print(f"Data Split (Chronological):")
        print(f"  Training:    {len(train_df):4d} games ({self.train_ratio:.1%}) "
              f"[{train_df['game_date'].min()} to {train_df['game_date'].max()}]")
        print(f"  Validation:  {len(val_df):4d} games ({self.val_ratio:.1%}) "
              f"[{val_df['game_date'].min()} to {val_df['game_date'].max()}]")
        print(f"  Calibration: {len(calibration_df):4d} games ({self.calibration_ratio:.1%}) "
              f"[{calibration_df['game_date'].min()} to {calibration_df['game_date'].max()}]")
        print(f"  Test:        {len(test_df):4d} games ({self.test_ratio:.1%}) "
              f"[{test_df['game_date'].min()} to {test_df['game_date'].max()}]")
        
        return train_df, val_df, calibration_df, test_df


def split_and_save_data():
    """
    Main function to split and save datasets
    """
    print("=" * 80)
    print("DATA SPLITTING FOR ALL THREE MODELS")
    print("=" * 80)
    
    # Load halftime dataset
    print("\nLoading data...")
    df = pd.read_parquet('data/processed/halftime_prediction_dataset.parquet')
    print(f"Loaded {len(df)} games")
    
    # Split
    print("\nSplitting chronologically...")
    splitter = ChronologicalDataSplitter(
        train_ratio=0.60,
        val_ratio=0.10,
        calibration_ratio=0.15,
        test_ratio=0.15
    )
    
    train_df, val_df, cal_df, test_df = splitter.split(df)
    
    # Create output directory
    output_dir = Path('data/splits/')
    output_dir.mkdir(exist_ok=True)
    
    # Save each split
    print("\nSaving splits...")
    train_df.to_parquet(output_dir / 'train.parquet')
    val_df.to_parquet(output_dir / 'validation.parquet')
    cal_df.to_parquet(output_dir / 'calibration.parquet')
    test_df.to_parquet(output_dir / 'test.parquet')
    
    print(f"✓ Saved to: {output_dir}")
    
    # Save split metadata
    metadata = {
        'split_date': pd.Timestamp.now().isoformat(),
        'total_games': len(df),
        'train_games': len(train_df),
        'val_games': len(val_df),
        'calibration_games': len(cal_df),
        'test_games': len(test_df),
        'train_date_range': [str(train_df['game_date'].min()), str(train_df['game_date'].max())],
        'val_date_range': [str(val_df['game_date'].min()), str(val_df['game_date'].max())],
        'calibration_date_range': [str(cal_df['game_date'].min()), str(cal_df['game_date'].max())],
        'test_date_range': [str(test_df['game_date'].min()), str(test_df['game_date'].max())]
    }
    
    import json
    with open(output_dir / 'split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Metadata saved")
    
    return train_df, val_df, cal_df, test_df

if __name__ == "__main__":
    train_df, val_df, cal_df, test_df = split_and_save_data()
    
    print("\n" + "=" * 80)
    print("SPLIT COMPLETE")
    print("=" * 80)
```

---

### 3.2 Data Split Validation (15 minutes)

**File:** `scripts/validate_splits.py`

```python
"""
Validate data splits have no leakage
"""

import pandas as pd

def validate_splits():
    """
    Validate train/val/calibration/test splits
    """
    print("Validating data splits...")
    
    # Load splits
    train_df = pd.read_parquet('data/splits/train.parquet')
    val_df = pd.read_parquet('data/splits/validation.parquet')
    cal_df = pd.read_parquet('data/splits/calibration.parquet')
    test_df = pd.read_parquet('data/splits/test.parquet')
    
    issues = []
    
    # Check 1: No game ID overlap
    train_ids = set(train_df['game_id'])
    val_ids = set(val_df['game_id'])
    cal_ids = set(cal_df['game_id'])
    test_ids = set(test_df['game_id'])
    
    if train_ids & val_ids:
        issues.append(f"Train-Val overlap: {len(train_ids & val_ids)} games")
    if train_ids & cal_ids:
        issues.append(f"Train-Calibration overlap: {len(train_ids & cal_ids)} games")
    if train_ids & test_ids:
        issues.append(f"Train-Test overlap: {len(train_ids & test_ids)} games")
    if val_ids & test_ids:
        issues.append(f"Val-Test overlap: {len(val_ids & test_ids)} games")
    
    print(f"[1] Game ID Overlap: {'✓ None' if not issues else '✗ Found'}")
    
    # Check 2: Temporal ordering
    temporal_checks = [
        ("Train before Val", train_df['game_date'].max() < val_df['game_date'].min()),
        ("Val before Cal", val_df['game_date'].max() < cal_df['game_date'].min()),
        ("Cal before Test", cal_df['game_date'].max() < test_df['game_date'].min())
    ]
    
    print(f"[2] Temporal Ordering:")
    for check_name, passed in temporal_checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            issues.append(check_name)
    
    # Check 3: Sufficient samples
    min_requirements = {
        'train': 500,
        'val': 100,
        'calibration': 100,
        'test': 100
    }
    
    actual = {
        'train': len(train_df),
        'val': len(val_df),
        'calibration': len(cal_df),
        'test': len(test_df)
    }
    
    print(f"[3] Sample Size Requirements:")
    for split_name, min_required in min_requirements.items():
        actual_count = actual[split_name]
        passed = actual_count >= min_required
        status = "✓" if passed else "✗"
        print(f"  {status} {split_name}: {actual_count} (min: {min_required})")
        
        if not passed:
            issues.append(f"{split_name} insufficient: {actual_count} < {min_required}")
    
    # Summary
    print("\n" + "=" * 80)
    if issues:
        print(f"✗ VALIDATION FAILED - {len(issues)} issues")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ VALIDATION PASSED - All splits valid")
        return True

if __name__ == "__main__":
    is_valid = validate_splits()
    
    if is_valid:
        print("\n✓ Proceed to Step 4: Model-Specific Preparation")
    else:
        print("\n✗ Fix split issues before proceeding")
```

---

## Expected Split Results

**From 5,000 games:**

| Split | Games | % | Date Range (Typical) | Purpose |
|-------|-------|---|---------------------|----------|
| **Training** | 3,000 | 60% | Oct 2020 - Apr 2023 | Train Informer, build Dejavu DB |
| **Validation** | 500 | 10% | Apr 2023 - Oct 2023 | Tune hyperparameters |
| **Calibration** | 750 | 15% | Oct 2023 - Apr 2024 | Fit Conformal predictor |
| **Test** | 750 | 15% | Apr 2024 - Oct 2025 | Final evaluation |

**Critical Properties:**
- ✅ No temporal overlap
- ✅ Chronological ordering maintained
- ✅ Sufficient samples for each model
- ✅ Calibration set independent from training

---

## Next Step

Proceed to **Step 4: Model-Specific Data Preparation** to format data for Informer, Conformal, and Dejavu.

---

*Action Step 3 of 10 - Data Splitting*

