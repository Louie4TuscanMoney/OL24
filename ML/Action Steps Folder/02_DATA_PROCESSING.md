# Step 2: Data Processing & Time Series Creation

**Objective:** Convert event-based play-by-play data into minute-by-minute time series

**Duration:** 2-3 hours  
**Prerequisites:** Completed Step 1 (raw play-by-play data)  
**Output:** Clean minute-by-minute differential time series for 5,000+ games

---

## Action Items

### 2.1 Create Time Series Converter (1 hour)

**File:** `processors/timeseries_converter.py`

```python
"""
Convert play-by-play events to minute-by-minute time series
"""

import pandas as pd
import numpy as np
from typing import List

class TimeSeriesConverter:
    """
    Convert NBA play-by-play to minute-by-minute differential series
    """
    def __init__(self):
        self.minutes_per_quarter = 12
        self.total_quarters = 4
    
    def convert_game(self, pbp_df: pd.DataFrame, game_id: str) -> pd.DataFrame:
        """
        Convert single game to minute-by-minute format
        
        Args:
            pbp_df: Play-by-play DataFrame
            game_id: Game identifier
        
        Returns:
            Minute-by-minute DataFrame (48 rows)
        """
        timeseries = []
        
        # For each minute of the game
        for quarter in range(1, 5):  # Quarters 1-4
            for minute_remaining in range(12, 0, -1):  # 12:00 down to 0:01
                
                # Find plays in this minute
                mask = (pbp_df['quarter'] == quarter) & \
                       (pbp_df['time'].str.startswith(f"{minute_remaining}:"))
                
                plays_in_minute = pbp_df[mask]
                
                if len(plays_in_minute) > 0:
                    # Use last play in this minute (closest to minute boundary)
                    last_play = plays_in_minute.iloc[-1]
                    differential = last_play['differential']
                    score_home = last_play['score_home']
                    score_away = last_play['score_away']
                else:
                    # No plays in this minute - interpolate
                    differential, score_home, score_away = self._interpolate_score(
                        pbp_df, quarter, minute_remaining
                    )
                
                # Calculate game minute (0-47)
                game_minute = (quarter - 1) * 12 + (12 - minute_remaining)
                
                timeseries.append({
                    'game_id': game_id,
                    'minute': game_minute,
                    'quarter': quarter,
                    'time_remaining_quarter': minute_remaining,
                    'score_home': score_home,
                    'score_away': score_away,
                    'differential': differential
                })
        
        return pd.DataFrame(timeseries)
    
    def _interpolate_score(self, pbp_df, quarter, minute):
        """
        Interpolate score if no plays in this minute
        """
        # Find nearest plays before and after
        before_mask = (pbp_df['quarter'] < quarter) | \
                     ((pbp_df['quarter'] == quarter) & \
                      (pbp_df['time'] > f"{minute}:00"))
        
        after_mask = (pbp_df['quarter'] > quarter) | \
                    ((pbp_df['quarter'] == quarter) & \
                     (pbp_df['time'] <= f"{minute}:00"))
        
        before_plays = pbp_df[before_mask]
        after_plays = pbp_df[after_mask]
        
        if len(before_plays) > 0 and len(after_plays) > 0:
            # Linear interpolation
            score_before = before_plays.iloc[-1]['differential']
            score_after = after_plays.iloc[0]['differential']
            differential = (score_before + score_after) / 2
            
            home_before = before_plays.iloc[-1]['score_home']
            home_after = after_plays.iloc[0]['score_home']
            score_home = int((home_before + home_after) / 2)
            
            away_before = before_plays.iloc[-1]['score_away']
            away_after = after_plays.iloc[0]['score_away']
            score_away = int((away_before + away_after) / 2)
        
        elif len(before_plays) > 0:
            # Use last known score
            differential = before_plays.iloc[-1]['differential']
            score_home = before_plays.iloc[-1]['score_home']
            score_away = before_plays.iloc[-1]['score_away']
        
        else:
            # Start of game - zeros
            differential = 0
            score_home = 0
            score_away = 0
        
        return differential, score_home, score_away


def process_all_games(input_dir='data/raw_pbp/', output_dir='data/processed/'):
    """
    Process all games to minute-by-minute format
    """
    from pathlib import Path
    from tqdm import tqdm
    
    converter = TimeSeriesConverter()
    
    # Get all raw play-by-play files
    pbp_files = list(Path(input_dir).glob('*.parquet'))
    print(f"Processing {len(pbp_files)} games...")
    
    processed_games = []
    failed = []
    
    for pbp_file in tqdm(pbp_files, desc="Converting to time series"):
        game_id = pbp_file.stem
        
        try:
            # Load raw play-by-play
            pbp_df = pd.read_parquet(pbp_file)
            
            # Convert to minute-by-minute
            ts_df = converter.convert_game(pbp_df, game_id)
            
            # Validate (should have 48 minutes)
            if len(ts_df) == 48:
                processed_games.append(ts_df)
            else:
                failed.append((game_id, f"Wrong length: {len(ts_df)}"))
        
        except Exception as e:
            failed.append((game_id, str(e)))
    
    # Save processed games
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Individual files
    for ts_df in processed_games:
        game_id = ts_df['game_id'].iloc[0]
        ts_df.to_parquet(f"{output_dir}/{game_id}.parquet")
    
    # Combined dataset
    combined = pd.concat(processed_games, ignore_index=True)
    combined.to_parquet(f"{output_dir}/all_games_timeseries.parquet")
    
    print(f"\n✓ Processed: {len(processed_games)} games")
    print(f"✗ Failed: {len(failed)} games")
    print(f"✓ Saved to: {output_dir}")
    
    if failed:
        failed_df = pd.DataFrame(failed, columns=['game_id', 'error'])
        failed_df.to_csv(f"{output_dir}/processing_failures.csv", index=False)
    
    return combined

if __name__ == "__main__":
    combined_df = process_all_games()
    
    # Quick stats
    print("\nDataset Statistics:")
    print(f"Total data points: {len(combined_df):,}")
    print(f"Unique games: {combined_df['game_id'].nunique()}")
    print(f"Differential range: [{combined_df['differential'].min()}, {combined_df['differential'].max()}]")
    print(f"Differential std: {combined_df['differential'].std():.2f}")
```

---

### 2.2 Add Temporal Features (30 minutes)

**File:** `processors/feature_engineering.py`

```python
"""
Add temporal and derived features
"""

import pandas as pd
import numpy as np

def add_temporal_features(df, game_dates_dict):
    """
    Add temporal features required by Informer
    
    Args:
        df: Minute-by-minute DataFrame
        game_dates_dict: {game_id: datetime} mapping
    
    Returns:
        DataFrame with temporal features
    """
    # Add game date
    df['game_date'] = df['game_id'].map(game_dates_dict)
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Extract temporal features
    df['month'] = df['game_date'].dt.month
    df['day'] = df['game_date'].dt.day
    df['weekday'] = df['game_date'].dt.dayofweek
    df['hour'] = df['game_date'].dt.hour
    
    # Game time features (within game)
    df['is_first_quarter'] = (df['quarter'] == 1).astype(int)
    df['is_second_quarter'] = (df['quarter'] == 2).astype(int)
    df['is_halftime_approaching'] = (df['minute'] >= 18).astype(int)
    
    return df

def add_momentum_features(df):
    """
    Add momentum-related features
    """
    # Group by game
    df = df.sort_values(['game_id', 'minute'])
    
    # Rate of change (points per minute)
    df['diff_velocity'] = df.groupby('game_id')['differential'].diff()
    
    # Acceleration
    df['diff_acceleration'] = df.groupby('game_id')['diff_velocity'].diff()
    
    # Rolling statistics (3-minute window)
    df['rolling_mean_3'] = df.groupby('game_id')['differential'].rolling(3).mean().reset_index(0, drop=True)
    df['rolling_std_3'] = df.groupby('game_id')['differential'].rolling(3).std().reset_index(0, drop=True)
    
    # Lead size categories
    df['lead_size'] = pd.cut(
        df['differential'],
        bins=[-100, -15, -5, 5, 15, 100],
        labels=['large_deficit', 'small_deficit', 'close', 'small_lead', 'large_lead']
    )
    
    # Fill NaN from diff/rolling
    df = df.fillna(method='bfill')
    
    return df

if __name__ == "__main__":
    # Load processed time series
    df = pd.read_parquet('data/processed/all_games_timeseries.parquet')
    
    # Load game dates from game_ids CSV
    game_ids_df = pd.read_csv('data/game_ids_2020_2025.csv')
    game_dates = dict(zip(game_ids_df['game_id'], game_ids_df['date']))
    
    # Add features
    df = add_temporal_features(df, game_dates)
    df = add_momentum_features(df)
    
    # Save enhanced dataset
    df.to_parquet('data/processed/all_games_with_features.parquet')
    print(f"✓ Added features: {list(df.columns)}")
```

---

### 2.3 Data Quality Validation (30 minutes)

**File:** `scripts/validate_data.py`

```python
"""
Validate processed data quality
"""

import pandas as pd
import numpy as np

def validate_dataset(df):
    """
    Comprehensive data validation
    """
    print("=" * 80)
    print("DATA VALIDATION REPORT")
    print("=" * 80)
    
    issues = []
    
    # Check 1: Correct structure
    n_games = df['game_id'].nunique()
    expected_rows = n_games * 48
    actual_rows = len(df)
    
    print(f"\n[1] Structure Check:")
    print(f"  Unique games: {n_games}")
    print(f"  Total rows: {actual_rows} (expected: {expected_rows})")
    
    if actual_rows != expected_rows:
        issues.append(f"Row count mismatch: {actual_rows} vs {expected_rows}")
    
    # Check 2: No missing differentials
    missing_diff = df['differential'].isna().sum()
    print(f"\n[2] Missing Values:")
    print(f"  Missing differentials: {missing_diff} ({missing_diff/len(df)*100:.2f}%)")
    
    if missing_diff > 0:
        issues.append(f"{missing_diff} missing differential values")
    
    # Check 3: Reasonable ranges
    diff_stats = df['differential'].describe()
    print(f"\n[3] Differential Statistics:")
    print(diff_stats)
    
    if abs(diff_stats['min']) > 60 or abs(diff_stats['max']) > 60:
        issues.append(f"Extreme differentials detected: [{diff_stats['min']}, {diff_stats['max']}]")
    
    # Check 4: Temporal features present
    temporal_cols = ['month', 'day', 'weekday', 'hour']
    missing_temporal = [col for col in temporal_cols if col not in df.columns]
    
    print(f"\n[4] Temporal Features:")
    if missing_temporal:
        print(f"  ✗ Missing: {missing_temporal}")
        issues.append(f"Missing temporal features: {missing_temporal}")
    else:
        print(f"  ✓ All temporal features present")
    
    # Check 5: Halftime data exists
    halftime_data = df[df['minute'] == 23]  # 0:00 2Q = minute 23
    print(f"\n[5] Halftime Data:")
    print(f"  Halftime rows: {len(halftime_data)} (should equal {n_games})")
    
    if len(halftime_data) != n_games:
        issues.append(f"Halftime data incomplete: {len(halftime_data)}/{n_games}")
    
    # Summary
    print("\n" + "=" * 80)
    if issues:
        print(f"✗ VALIDATION FAILED - {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ VALIDATION PASSED - Dataset ready for modeling")
        return True

if __name__ == "__main__":
    df = pd.read_parquet('data/processed/all_games_with_features.parquet')
    is_valid = validate_dataset(df)
    
    if is_valid:
        print("\n✓ Proceed to Step 3: Data Splitting")
    else:
        print("\n✗ Fix data issues before proceeding")
```

---

### 2.4 Extract Halftime-Specific Dataset (30 minutes)

**File:** `scripts/extract_halftime_dataset.py`

```python
"""
Extract dataset for halftime prediction task
Input: Minutes 0-17 (start to 6:00 2Q)
Output: Minutes 18-23 (6:00 2Q to 0:00 2Q halftime)
"""

import pandas as pd
import numpy as np

def extract_halftime_patterns(df):
    """
    Extract (pattern, outcome) pairs for halftime prediction
    
    Returns:
        DataFrame with pattern and outcome columns
    """
    pattern_length = 18  # Minutes 0-17 (to 6:00 2Q)
    forecast_horizon = 6  # Minutes 18-23 (to halftime)
    
    halftime_data = []
    
    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id].sort_values('minute')
        
        # Validate complete game
        if len(game_df) < pattern_length + forecast_horizon:
            continue
        
        # Extract pattern (first 18 minutes)
        pattern = game_df['differential'].iloc[:pattern_length].values
        
        # Extract outcome (next 6 minutes to halftime)
        outcome = game_df['differential'].iloc[pattern_length:pattern_length+forecast_horizon].values
        
        # Extract temporal features (use time at 6:00 2Q)
        temporal_row = game_df.iloc[pattern_length - 1]
        
        halftime_data.append({
            'game_id': game_id,
            'pattern': pattern,
            'outcome': outcome,
            'halftime_differential': outcome[-1],  # Final value at 0:00 2Q
            'differential_at_6min_2q': pattern[-1],  # Value at 6:00 2Q
            'month': temporal_row['month'],
            'day': temporal_row['day'],
            'weekday': temporal_row['weekday'],
            'hour': temporal_row['hour']
        })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(halftime_data)
    
    print(f"Extracted {len(result_df)} halftime patterns")
    print(f"Pattern shape: {result_df['pattern'].iloc[0].shape}")
    print(f"Outcome shape: {result_df['outcome'].iloc[0].shape}")
    
    return result_df

if __name__ == "__main__":
    # Load processed data
    df = pd.read_parquet('data/processed/all_games_with_features.parquet')
    
    # Extract halftime-specific dataset
    halftime_df = extract_halftime_patterns(df)
    
    # Save
    halftime_df.to_parquet('data/processed/halftime_prediction_dataset.parquet')
    
    print("\n✓ Halftime dataset ready")
    print(f"✓ Saved to: data/processed/halftime_prediction_dataset.parquet")
    
    # Quick statistics
    print("\nDataset Statistics:")
    print(f"  Games: {len(halftime_df)}")
    print(f"  Halftime differential mean: {halftime_df['halftime_differential'].mean():.2f}")
    print(f"  Halftime differential std: {halftime_df['halftime_differential'].std():.2f}")
```

---

### 2.5 Validation Checklist

**Before proceeding to Step 3:**

- [ ] ✅ All raw play-by-play converted to minute-by-minute
- [ ] ✅ Each game has 48 minutes (or documented exceptions)
- [ ] ✅ No missing differentials in critical range (minutes 0-23)
- [ ] ✅ Temporal features added (month/day/weekday/hour)
- [ ] ✅ Halftime dataset extracted (pattern length=18, horizon=6)
- [ ] ✅ Data validation passed
- [ ] ✅ ~5,000 games processed successfully

**Quality Metrics:**
```python
# Run quick check
df = pd.read_parquet('data/processed/all_games_with_features.parquet')

print(f"Games: {df['game_id'].nunique()}")
print(f"Complete: {(df.groupby('game_id').size() == 48).sum()} games")
print(f"Differential range: [{df['differential'].min():.1f}, {df['differential'].max():.1f}]")
print(f"Missing values: {df['differential'].isna().sum()}")
```

---

## Expected Output

```
data/processed/
├── all_games_timeseries.parquet          ← 48 mins × 5,000 games
├── all_games_with_features.parquet       ← With temporal features
├── halftime_prediction_dataset.parquet   ← Ready for models
└── processing_failures.csv                ← Failed games (if any)
```

**Dataset Size:**
- Rows: ~240,000 (48 minutes × 5,000 games)
- Columns: ~15-20 features
- Storage: ~30-50 MB (Parquet compressed)

---

## Troubleshooting

**Problem:** Games with fewer than 48 minutes

**Solution:**
- Overtime games have more minutes - truncate to 48 or handle separately
- Some games may be missing data - log and skip
- Typical success rate: >95%

**Problem:** Score interpolation creating artifacts

**Solution:**
- Verify interpolation logic
- Check edge cases (start/end of quarters)
- Use forward-fill instead of interpolation if needed

**Problem:** Temporal features not matching games

**Solution:**
- Ensure game_dates_dict is complete
- Check date parsing format
- Validate game_id to date mapping

---

## Next Step

Proceed to **Step 3: Data Splitting** to create train/val/calibration/test sets for all three models.

---

*Action Step 2 of 10 - Data Processing*

