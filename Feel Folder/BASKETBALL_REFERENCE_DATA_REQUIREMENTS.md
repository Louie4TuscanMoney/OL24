# Basketball-Reference Data Requirements for Forecasting Models

**Complete Data Specification for Informer, Conformal, and Dejavu**

**Date:** October 14, 2025  
**Objective:** Extract optimal data from basketball-reference.com for NBA forecasting (2020-2021 onwards)  
**Task:** Predict halftime (0:00 2Q) score differential from game state at 6:00 2Q

---

## Executive Summary

This document specifies **exactly what data to collect** from Basketball-Reference to implement:
- **Informer** (model-centric forecasting)
- **Conformal Prediction** (uncertainty quantification)
- **Dejavu** (pattern matching)

**Data Coverage:** 2020-2021 season through present (~4+ seasons, 5,000+ games)

**Primary Sources on Basketball-Reference:**
1. Play-by-play data
2. Box score data
3. Team statistics
4. Player statistics

---

## Table of Contents

1. [Core Data Requirements](#core-data-requirements)
2. [Basketball-Reference Data Sources](#basketball-reference-data-sources)
3. [Data Collection Strategy](#data-collection-strategy)
4. [Data Schema for Each Model](#data-schema-for-each-model)
5. [Feature Engineering from Raw Data](#feature-engineering-from-raw-data)
6. [Implementation Code](#implementation-code)

---

## Core Data Requirements

### Essential: Play-by-Play Data

**URL Pattern:**
```
https://www.basketball-reference.com/boxscores/pbp/[GAME_ID].html

Example:
https://www.basketball-reference.com/boxscores/pbp/202010220LAL.html
```

**Required Fields from Play-by-Play:**

| Field | Description | Format | Example | Why Needed |
|-------|-------------|--------|---------|------------|
| `game_id` | Unique game identifier | String | "202010220LAL" | Primary key |
| `date` | Game date | YYYY-MM-DD | "2020-10-22" | Temporal features |
| `time` | Game clock time | MM:SS Q | "6:00 2Q" | Timestamp |
| `quarter` | Quarter number | 1-4 | 2 | Period identifier |
| `score_away` | Away team score | Integer | 45 | Score differential calculation |
| `score_home` | Home team score | Integer | 50 | Score differential calculation |
| `play_description` | What happened | Text | "LeBron James made 3-pt" | Event parsing |
| `team` | Team that performed action | String | "LAL" | Team tracking |

**Derived Field (Critical):**
```python
score_differential = score_home - score_away  # Or Team1 - Team2
```

### Supporting: Box Score Data

**URL Pattern:**
```
https://www.basketball-reference.com/boxscores/[GAME_ID].html
```

**Required Fields:**

| Field | Description | Why Needed |
|-------|-------------|------------|
| `team_home` | Home team code | Identify teams |
| `team_away` | Away team code | Identify teams |
| `final_score_home` | Final home score | Validation |
| `final_score_away` | Final away score | Validation |
| `date` | Game date | Temporal features |
| `season` | NBA season | Dataset versioning |

### Optional: Enhanced Features

**Team Statistics (Season Averages):**
```
https://www.basketball-reference.com/teams/[TEAM]/[YEAR].html
```

| Field | Description | Use Case |
|-------|-------------|----------|
| `pts_per_game` | Points per game | Team strength baseline |
| `opp_pts_per_game` | Opponent points | Defensive rating |
| `pace` | Possessions per game | Game speed factor |
| `off_rtg` | Offensive rating | Offensive efficiency |
| `def_rtg` | Defensive rating | Defensive efficiency |

**Player Statistics (If using advanced features):**
```
https://www.basketball-reference.com/players/[PLAYER_INITIAL]/[PLAYER_ID].html
```

| Field | Description | Use Case |
|-------|-------------|----------|
| `player_name` | Player name | On-court tracking |
| `minutes` | Minutes played | Fatigue modeling |
| `plus_minus` | +/- rating | Impact metric |

---

## Basketball-Reference Data Sources

### 1. Play-by-Play Data (PRIMARY - ESSENTIAL)

**Access Method:**
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_play_by_play(game_id):
    """
    Scrape play-by-play data from basketball-reference
    
    Args:
        game_id: Game ID (e.g., "202010220LAL")
    
    Returns:
        DataFrame with play-by-play data
    """
    url = f"https://www.basketball-reference.com/boxscores/pbp/{game_id}.html"
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find play-by-play table
    pbp_table = soup.find('table', {'id': 'pbp'})
    
    # Parse table to DataFrame
    # (basketball-reference uses specific HTML structure)
    
    return pbp_df
```

**Data Frequency:** Event-based (every play)

**What We Need:** Aggregate to 1-minute intervals
```python
# From event-based to time-series
time_series_df = create_minute_by_minute(pbp_df)
```

### 2. Schedule Data (For Collecting All Games)

**URL:**
```
https://www.basketball-reference.com/leagues/NBA_[YEAR]_games.html

Examples:
- 2020-21: https://www.basketball-reference.com/leagues/NBA_2021_games.html
- 2021-22: https://www.basketball-reference.com/leagues/NBA_2022_games.html
- 2022-23: https://www.basketball-reference.com/leagues/NBA_2023_games.html
- 2023-24: https://www.basketball-reference.com/leagues/NBA_2024_games.html
- 2024-25: https://www.basketball-reference.com/leagues/NBA_2025_games.html
```

**Required Fields:**

| Field | Example | Purpose |
|-------|---------|---------|
| `date` | "2020-10-22" | Game identification |
| `home_team` | "LAL" | Team identification |
| `away_team` | "LAC" | Team identification |
| `game_id` | "202010220LAL" | Construct play-by-play URL |

**Use:** Build list of all games to scrape (2020-2025 = ~6,000 games)

---

## Data Collection Strategy

### Phase 1: Collect Game IDs (2020-2021 onwards)

```python
def collect_all_game_ids(start_year=2021, end_year=2025):
    """
    Collect all NBA game IDs from basketball-reference
    
    Args:
        start_year: Starting season (2021 = 2020-21 season)
        end_year: Ending season
    
    Returns:
        List of game IDs
    """
    all_game_ids = []
    
    for year in range(start_year, end_year + 1):
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games.html"
        
        # Scrape schedule page
        schedule_df = scrape_schedule(url)
        
        # Extract game IDs
        game_ids = schedule_df['game_id'].tolist()
        all_game_ids.extend(game_ids)
        
        print(f"Season {year-1}-{year}: {len(game_ids)} games")
    
    print(f"Total games collected: {len(all_game_ids)}")
    return all_game_ids

# Expected output:
# Season 2020-2021: ~1,080 games (COVID shortened)
# Season 2021-2022: ~1,230 games
# Season 2022-2023: ~1,230 games
# Season 2023-2024: ~1,230 games
# Season 2024-2025: ~600 games (partial, as of Oct 2025)
# Total: ~5,400 games
```

### Phase 2: Scrape Play-by-Play Data

```python
def scrape_all_games(game_ids, rate_limit=2.0):
    """
    Scrape play-by-play for all games
    
    Args:
        game_ids: List of game IDs to scrape
        rate_limit: Seconds between requests (respect server)
    
    Returns:
        List of DataFrames (one per game)
    """
    import time
    
    all_games = []
    
    for i, game_id in enumerate(game_ids):
        try:
            pbp_df = scrape_play_by_play(game_id)
            all_games.append(pbp_df)
            
            # Progress tracking
            if (i + 1) % 100 == 0:
                print(f"Scraped {i+1}/{len(game_ids)} games")
            
            # Rate limiting (be respectful!)
            time.sleep(rate_limit)
        
        except Exception as e:
            print(f"Error scraping {game_id}: {e}")
            continue
    
    print(f"Successfully scraped {len(all_games)}/{len(game_ids)} games")
    return all_games

# Estimated time: 5,400 games × 2 seconds = 3 hours
```

### Phase 3: Convert to Minute-by-Minute Time Series

```python
def create_minute_by_minute_series(pbp_df):
    """
    Convert event-based play-by-play to 1-minute time series
    
    Args:
        pbp_df: Play-by-play DataFrame
    
    Returns:
        Time series DataFrame with 1-minute intervals
    """
    # Initialize 48-minute series (4 quarters × 12 minutes)
    minutes = []
    
    for quarter in [1, 2, 3, 4]:
        for minute in range(12, 0, -1):  # 12:00 down to 0:00
            time_key = f"{minute}:00 {quarter}Q"
            
            # Find score at this time
            mask = (pbp_df['quarter'] == quarter) & \
                   (pbp_df['time'] >= f"{minute}:00") & \
                   (pbp_df['time'] < f"{minute+1}:00")
            
            if mask.any():
                # Get score at or near this minute
                score_row = pbp_df[mask].iloc[0]
                differential = score_row['score_home'] - score_row['score_away']
            else:
                # Interpolate if no exact match
                differential = interpolate_score(pbp_df, quarter, minute)
            
            minutes.append({
                'minute': (quarter - 1) * 12 + (12 - minute),  # 0-47
                'quarter': quarter,
                'time_remaining_quarter': minute,
                'differential': differential
            })
    
    return pd.DataFrame(minutes)
```

---

## Data Schema for Each Model

### Informer Requirements

**Input Sequence Format:**

```python
# For predicting 0:00 2Q from 6:00 2Q
informer_data = {
    # Encoder input (sequence up to 6:00 2Q)
    'x_enc': np.array([
        differential_at_minute_0,   # Game start
        differential_at_minute_1,
        differential_at_minute_2,
        ...
        differential_at_minute_17   # 6:00 2Q (18th minute)
    ]),  # Shape: (18,) for univariate or (18, n_features) for multivariate
    
    # Temporal features (x_mark_enc)
    'temporal_features': np.array([
        [month, day, weekday, hour],  # For minute 0
        [month, day, weekday, hour],  # For minute 1
        ...
        [month, day, weekday, hour]   # For minute 17
    ]),  # Shape: (18, 4)
    
    # Target (decoder output)
    'y': np.array([
        differential_at_minute_18,  # 5:00 2Q
        differential_at_minute_19,  # 4:00 2Q
        differential_at_minute_20,  # 3:00 2Q
        differential_at_minute_21,  # 2:00 2Q
        differential_at_minute_22,  # 1:00 2Q
        differential_at_minute_23   # 0:00 2Q (halftime)
    ])  # Shape: (6,) - predict 6 minutes to halftime
}
```

**Data Collection for Informer:**

```python
def prepare_informer_dataset(all_games, seq_len=18, pred_len=6):
    """
    Prepare dataset for Informer from basketball-reference data
    
    Args:
        all_games: List of game DataFrames (minute-by-minute)
        seq_len: Input sequence length (18 minutes to 6:00 2Q)
        pred_len: Prediction horizon (6 minutes to halftime)
    
    Returns:
        Dataset ready for Informer training
    """
    samples = []
    
    for game_df in all_games:
        # Extract sequence (minute 0-17)
        x_enc = game_df['differential'].iloc[:seq_len].values
        
        # Extract temporal features
        game_datetime = pd.to_datetime(game_df['game_date'].iloc[0])
        temporal = np.array([
            [
                game_datetime.month,
                game_datetime.day,
                game_datetime.weekday(),
                game_datetime.hour
            ]
        ] * seq_len)
        
        # Extract target (minute 18-23, i.e., 6:00 2Q to halftime)
        y = game_df['differential'].iloc[seq_len:seq_len+pred_len].values
        
        if len(y) == pred_len:  # Validate complete sequence
            samples.append({
                'x_enc': x_enc,
                'x_mark_enc': temporal,
                'y': y
            })
    
    print(f"Prepared {len(samples)} samples for Informer")
    return samples
```

**Minimum Data:**
- **Training:** 1,000+ games (2 seasons)
- **Validation:** 200+ games (0.5 season)
- **Calibration:** 150+ games (playoffs + regular season mix)
- **Test:** 200+ games (most recent season)

**Total:** ~1,550+ games from basketball-reference

---

### Conformal Prediction Requirements

**Calibration Set Structure:**

```python
conformal_calibration = {
    # Historical patterns (input at 6:00 2Q)
    'X_cal': [
        [diff_0, diff_1, ..., diff_17],  # Game 1 pattern
        [diff_0, diff_1, ..., diff_17],  # Game 2 pattern
        ...
    ],  # Shape: (n_cal_games, 18)
    
    # Multi-step outcomes (6:00 2Q to halftime)
    'Y_cal': [
        [diff_18, diff_19, diff_20, diff_21, diff_22, diff_23],  # Game 1
        [diff_18, diff_19, diff_20, diff_21, diff_22, diff_23],  # Game 2
        ...
    ],  # Shape: (n_cal_games, 6)
    
    # Metadata for adaptive weighting
    'timestamps': [date1, date2, ...]  # For recency weighting
}
```

**Data Collection for Conformal:**

```python
def prepare_conformal_calibration(games, seq_len=18, horizon=6):
    """
    Prepare calibration set for Conformal Prediction
    
    Args:
        games: List of game DataFrames (held-out from training)
        seq_len: Pattern length
        horizon: Forecast horizon
    
    Returns:
        X_cal, Y_cal for conformal calibration
    """
    X_cal = []
    Y_cal = []
    timestamps = []
    
    for game_df in games:
        # Pattern: First 18 minutes
        pattern = game_df['differential'].iloc[:seq_len].values
        
        # Outcome: Next 6 minutes (to halftime)
        outcome = game_df['differential'].iloc[seq_len:seq_len+horizon].values
        
        if len(pattern) == seq_len and len(outcome) == horizon:
            X_cal.append(pattern)
            Y_cal.append(outcome)
            timestamps.append(game_df['game_date'].iloc[0])
    
    return np.array(X_cal), np.array(Y_cal), timestamps
```

**Minimum Data:**
- **Calibration:** 100-200 games minimum
- **Recommended:** 300-500 games for stable quantiles
- **From:** Most recent 1-2 seasons (for relevance)

---

### Dejavu Requirements

**Pattern Database Structure:**

```python
dejavu_database = [
    {
        'pattern': [diff_0, diff_1, ..., diff_17],  # 18-minute pattern
        'outcome': [diff_18, diff_19, ..., diff_23],  # 6-minute outcome
        'halftime_differential': diff_23,  # Final target
        'game_id': "202010220LAL",
        'date': "2020-10-22",
        'teams': "LAL vs LAC",
        'metadata': {
            'home_team': "LAL",
            'away_team': "LAC",
            'season': "2020-21",
            'game_type': "regular"  # regular, playoffs
        }
    },
    # ... 500-5000 games
]
```

**Data Collection for Dejavu:**

```python
def prepare_dejavu_database(all_games, pattern_length=18, forecast_horizon=6):
    """
    Build Dejavu pattern database from basketball-reference data
    
    Args:
        all_games: All scraped games
        pattern_length: 18 minutes (to 6:00 2Q)
        forecast_horizon: 6 minutes (to halftime)
    
    Returns:
        Pattern database for Dejavu
    """
    database = []
    
    for game_df in all_games:
        pattern = game_df['differential'].iloc[:pattern_length].values
        outcome = game_df['differential'].iloc[pattern_length:pattern_length+forecast_horizon].values
        
        if len(pattern) == pattern_length and len(outcome) == forecast_horizon:
            database.append({
                'pattern': pattern.astype(np.float32),
                'outcome': outcome.astype(np.float32),
                'halftime_differential': outcome[-1],  # Value at 0:00 2Q
                'game_id': game_df['game_id'].iloc[0],
                'date': game_df['game_date'].iloc[0],
                'teams': f"{game_df['home_team'].iloc[0]} vs {game_df['away_team'].iloc[0]}",
                'metadata': {
                    'season': game_df['season'].iloc[0],
                    'game_type': game_df['game_type'].iloc[0]
                }
            })
    
    print(f"Dejavu database: {len(database)} games")
    return database
```

**Minimum Data:**
- **Baseline:** 500 games (1 season)
- **Good:** 2,000 games (2-3 seasons)
- **Excellent:** 5,000 games (4-5 seasons)

**Why More is Better:** 
More historical patterns → better matches → better forecasts

---

## Feature Engineering from Raw Data

### 1. Basic Features (ESSENTIAL)

**Score Differential Time Series:**

```python
def extract_differential_timeseries(pbp_df):
    """
    Extract minute-by-minute differential from play-by-play
    
    Returns:
        48-minute time series (or 24 for half-game analysis)
    """
    timeseries = []
    
    # For each minute of game
    for minute in range(48):  # 0-47 (full game)
        quarter = (minute // 12) + 1
        time_in_quarter = 12 - (minute % 12)
        
        # Find score at this minute
        score_at_minute = get_score_at_time(pbp_df, quarter, time_in_quarter)
        
        timeseries.append({
            'minute': minute,
            'quarter': quarter,
            'time_remaining_quarter': time_in_quarter,
            'score_home': score_at_minute['home'],
            'score_away': score_at_minute['away'],
            'differential': score_at_minute['home'] - score_at_minute['away']
        })
    
    return pd.DataFrame(timeseries)
```

### 2. Enhanced Features (OPTIONAL for Informer)

**Momentum Indicators:**

```python
def add_momentum_features(game_df):
    """
    Add momentum-related features
    """
    # Rate of change (points per minute)
    game_df['differential_velocity'] = game_df['differential'].diff()
    
    # Acceleration (momentum change)
    game_df['differential_acceleration'] = game_df['differential_velocity'].diff()
    
    # Rolling statistics
    game_df['rolling_avg_3min'] = game_df['differential'].rolling(3).mean()
    game_df['rolling_std_3min'] = game_df['differential'].rolling(3).std()
    
    # Lead changes
    game_df['lead_changed'] = (np.sign(game_df['differential']) != 
                               np.sign(game_df['differential'].shift(1))).astype(int)
    
    return game_df
```

**Game Context Features:**

```python
def add_context_features(game_df, team_stats):
    """
    Add team and context features
    """
    # Team strength differential
    game_df['team_strength_diff'] = (
        team_stats[game_df['home_team']]['off_rtg'] -
        team_stats[game_df['away_team']]['def_rtg']
    )
    
    # Pace differential
    game_df['pace_diff'] = (
        team_stats[game_df['home_team']]['pace'] -
        team_stats[game_df['away_team']]['pace']
    )
    
    # Home court advantage (encode)
    game_df['home_court'] = 1  # Always 1 if using home differential
    
    # Day of week (back-to-back effects)
    game_df['day_of_week'] = pd.to_datetime(game_df['date']).dt.dayofweek
    game_df['is_back_to_back'] = 0  # Would need schedule analysis
    
    return game_df
```

### 3. Temporal Features (REQUIRED for Informer)

```python
def extract_temporal_features(game_date):
    """
    Extract temporal features for Informer
    
    Args:
        game_date: Game date (datetime)
    
    Returns:
        [month, day, weekday, hour] for each minute
    """
    dt = pd.to_datetime(game_date)
    
    temporal = {
        'month': dt.month,        # 1-12 (season progression)
        'day': dt.day,            # 1-31
        'weekday': dt.weekday(),  # 0-6 (Monday=0)
        'hour': dt.hour           # Game start time (affects venue, fatigue)
    }
    
    return temporal
```

---

## Complete Data Collection Pipeline

### End-to-End Implementation

```python
class BasketballReferenceDataPipeline:
    """
    Complete pipeline for collecting NBA data from basketball-reference
    """
    def __init__(self, start_year=2021, end_year=2025):
        self.start_year = start_year
        self.end_year = end_year
        self.all_games = []
    
    def run(self, output_dir='./nba_data/'):
        """
        Run complete data collection pipeline
        """
        print("=" * 80)
        print("BASKETBALL-REFERENCE DATA COLLECTION PIPELINE")
        print("=" * 80)
        
        # Step 1: Collect game IDs
        print("\n[1/4] Collecting game IDs...")
        game_ids = self.collect_game_ids()
        print(f"  Found {len(game_ids)} games")
        
        # Step 2: Scrape play-by-play
        print("\n[2/4] Scraping play-by-play data...")
        print("  (This will take ~3 hours at 2 sec/game)")
        raw_pbp_games = self.scrape_all_pbp(game_ids)
        
        # Step 3: Convert to minute-by-minute
        print("\n[3/4] Converting to minute-by-minute series...")
        minute_games = []
        for pbp in raw_pbp_games:
            minute_df = create_minute_by_minute_series(pbp)
            minute_games.append(minute_df)
        
        # Step 4: Save processed data
        print("\n[4/4] Saving processed data...")
        self.save_data(minute_games, output_dir)
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Collected: {len(minute_games)} games")
        print(f"Saved to: {output_dir}")
        
        return minute_games
    
    def collect_game_ids(self):
        """Collect all game IDs from schedules"""
        return collect_all_game_ids(self.start_year, self.end_year)
    
    def scrape_all_pbp(self, game_ids):
        """Scrape play-by-play for all games"""
        return scrape_all_games(game_ids, rate_limit=2.0)
    
    def save_data(self, games, output_dir):
        """Save data in multiple formats"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual games (Parquet for efficiency)
        for i, game in enumerate(games):
            game.to_parquet(f"{output_dir}/game_{i:04d}.parquet")
        
        # Save combined dataset
        combined_df = pd.concat(games, ignore_index=True)
        combined_df.to_parquet(f"{output_dir}/all_games_combined.parquet")
        
        # Save metadata
        metadata = {
            'n_games': len(games),
            'start_year': self.start_year,
            'end_year': self.end_year,
            'collected_at': datetime.now().isoformat()
        }
        
        import json
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved {len(games)} games")
        print(f"  Individual files: {output_dir}/game_*.parquet")
        print(f"  Combined: {output_dir}/all_games_combined.parquet")
```

---

## Specific Basketball-Reference Tables

### Table 1: Play-by-Play Table Structure

**HTML Element:** `<table id="pbp">`

**Columns to Extract:**

| Column | HTML Class | Description | Data Type |
|--------|-----------|-------------|-----------|
| Time | `time` | Game clock | String (MM:SS) |
| Team | `team` | Acting team | String (3-letter code) |
| Score | `score` | Current score | String (XX-YY) |
| Play | `play` | Play description | String |

**Parsing Example:**

```python
def parse_play_by_play_table(soup):
    """
    Parse basketball-reference play-by-play table
    """
    pbp_table = soup.find('table', {'id': 'pbp'})
    rows = pbp_table.find_all('tr')
    
    data = []
    for row in rows[1:]:  # Skip header
        cols = row.find_all('td')
        if len(cols) >= 6:
            # Extract quarter from section
            quarter_section = row.find_previous('tr', class_='thead')
            quarter = extract_quarter_from_header(quarter_section)
            
            # Parse score (format: "45-50")
            score_text = cols[2].text.strip()
            if '-' in score_text:
                score_away, score_home = map(int, score_text.split('-'))
            
            data.append({
                'time': cols[0].text.strip(),
                'quarter': quarter,
                'score_away': score_away,
                'score_home': score_home,
                'differential': score_home - score_away,
                'play': cols[5].text.strip()
            })
    
    return pd.DataFrame(data)
```

### Table 2: Box Score Summary

**HTML Element:** `<table id="box-[TEAM]-game-basic">`

**Columns to Extract:**

| Column | Purpose |
|--------|---------|
| `PTS` | Total points (validation) |
| `FG%` | Field goal percentage |
| `3P%` | Three-point percentage |
| `FT%` | Free throw percentage |
| `TRB` | Total rebounds |
| `AST` | Assists |
| `TOV` | Turnovers |

**Use:** Optional features for enhanced Informer model

---

## Data Quality Requirements

### Validation Checks

```python
class NBADataValidator:
    """
    Validate collected NBA data quality
    """
    def __init__(self):
        self.issues = []
    
    def validate_game(self, game_df):
        """
        Validate single game data
        """
        issues = []
        
        # Check 1: Correct number of minutes (24 for half, 48 for full)
        if len(game_df) != 24 and len(game_df) != 48:
            issues.append(f"Wrong length: {len(game_df)} minutes")
        
        # Check 2: No missing differentials
        if game_df['differential'].isna().any():
            issues.append("Missing differential values")
        
        # Check 3: Reasonable differential range (-50 to +50 typical)
        if game_df['differential'].abs().max() > 50:
            issues.append(f"Extreme differential: {game_df['differential'].abs().max()}")
        
        # Check 4: Monotonic time progression
        if not game_df['minute'].is_monotonic_increasing:
            issues.append("Time not monotonic")
        
        # Check 5: Halftime differential exists
        if 'halftime_differential' in game_df.columns:
            if pd.isna(game_df['halftime_differential'].iloc[0]):
                issues.append("Missing halftime differential")
        
        return len(issues) == 0, issues
    
    def validate_dataset(self, all_games):
        """
        Validate entire dataset
        """
        valid_games = []
        invalid_games = []
        
        for game in all_games:
            is_valid, issues = self.validate_game(game)
            
            if is_valid:
                valid_games.append(game)
            else:
                invalid_games.append((game, issues))
        
        print(f"Validation Results:")
        print(f"  Valid: {len(valid_games)} games")
        print(f"  Invalid: {len(invalid_games)} games")
        
        if invalid_games:
            print("\nInvalid game examples:")
            for game, issues in invalid_games[:5]:
                print(f"  Game {game['game_id'].iloc[0]}: {issues}")
        
        return valid_games
```

---

## Data Requirements Summary

### Minimum Viable Dataset (MVP)

**For proof-of-concept:**

| Component | Minimum Games | Seasons | Source |
|-----------|--------------|---------|--------|
| Informer Training | 500 | 1.5 | 2023-24 + partial 2024-25 |
| Conformal Calibration | 100 | 0.5 | 2024 playoffs |
| Dejavu Database | 500 | 1.5 | Same as Informer training |
| Test Set | 100 | 0.5 | Recent 2024-25 games |
| **Total Unique Games** | **600-700** | **2** | **2023-present** |

**Collection Time:** ~2-3 hours (with rate limiting)

### Recommended Production Dataset

**For production deployment:**

| Component | Recommended Games | Seasons | Source |
|-----------|------------------|---------|--------|
| Informer Training | 2,500 | 3 | 2021-22, 2022-23, 2023-24 |
| Conformal Calibration | 300 | 0.5 | 2023-24 playoffs + recent regular |
| Dejavu Database | 3,000 | 4 | 2020-21 through 2023-24 |
| Test Set | 300 | 0.5 | 2024-25 season |
| **Total Unique Games** | **3,500+** | **4-5** | **2020-present** |

**Collection Time:** ~7-10 hours (with rate limiting)

### Optimal Research Dataset

**For maximum performance:**

| Component | Games | Coverage |
|-----------|-------|----------|
| Full Historical | 10,000+ | 2015-present (10 seasons) |
| Training | 7,000 | 2015-2023 |
| Calibration | 500 | 2023-24 |
| Test | 500 | 2024-25 |

**Collection Time:** ~20-25 hours

---

## Basketball-Reference API Considerations

### Rate Limiting

**Be respectful of basketball-reference servers:**

```python
import time
import random

def respectful_scrape(url, min_delay=2.0, max_delay=4.0):
    """
    Scrape with respectful rate limiting
    """
    response = requests.get(url, headers={
        'User-Agent': 'Research Project (your.email@university.edu)'
    })
    
    # Random delay between requests
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)
    
    return response
```

**Recommended:**
- 2-3 seconds between requests
- Identify your scraper in User-Agent
- Run during off-peak hours
- Consider basketball-reference's data usage policies

### Caching Strategy

```python
import hashlib
import pickle
from pathlib import Path

class BasketballReferenceCache:
    """
    Cache scraped data to avoid re-scraping
    """
    def __init__(self, cache_dir='./cache/'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, game_id):
        """Generate cache file path for game"""
        return self.cache_dir / f"{game_id}.pkl"
    
    def get(self, game_id):
        """Get cached game data"""
        cache_path = self.get_cache_path(game_id)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, game_id, data):
        """Cache game data"""
        cache_path = self.get_cache_path(game_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def scrape_with_cache(self, game_id, scrape_func):
        """Scrape with caching"""
        # Check cache first
        cached = self.get(game_id)
        if cached is not None:
            return cached
        
        # Scrape if not cached
        data = scrape_func(game_id)
        
        # Cache for future
        self.set(game_id, data)
        
        return data
```

---

## Data Schema Documentation

### Final Dataset Structure

**File:** `nba_games_2020_2025.parquet`

```python
schema = {
    # Identifiers
    'game_id': 'string',           # e.g., "202010220LAL"
    'game_date': 'datetime',       # Game date
    'season': 'string',            # e.g., "2020-21"
    'game_type': 'string',         # "regular" or "playoffs"
    
    # Teams
    'home_team': 'string',         # 3-letter code
    'away_team': 'string',         # 3-letter code
    
    # Time series (minute-by-minute)
    'minute': 'int',               # 0-47 (game minute)
    'quarter': 'int',              # 1-4
    'time_remaining_quarter': 'int',  # 12-0 within quarter
    
    # Scores
    'score_home': 'int',           # Home team score at this minute
    'score_away': 'int',           # Away team score
    'differential': 'float',       # score_home - score_away (PRIMARY)
    
    # Optional: Enhanced features
    'differential_velocity': 'float',      # Rate of change
    'differential_acceleration': 'float',  # Momentum
    'rolling_avg_3min': 'float',           # 3-minute average
    
    # Temporal features (for Informer)
    'month': 'int',                # 1-12
    'day': 'int',                  # 1-31
    'weekday': 'int',              # 0-6
    'hour': 'int'                  # Game start hour
}
```

### Expected Dataset Size

**Per Game:**
- 48 rows (minutes) × ~15 columns = ~720 values
- Storage: ~5-10 KB per game (Parquet)

**Total Dataset:**
- 5,000 games × 10 KB = ~50 MB (very manageable!)
- Fits in memory easily
- Fast to load and process

---

## Implementation Checklist

### Data Collection
- [ ] Set up basketball-reference scraper with rate limiting
- [ ] Collect game IDs for 2020-2025 seasons (~5,400 games)
- [ ] Scrape play-by-play data (allow 3-10 hours)
- [ ] Cache scraped data to avoid re-scraping
- [ ] Convert to minute-by-minute format

### Data Processing
- [ ] Extract score differentials
- [ ] Create temporal features (month/day/weekday/hour)
- [ ] Add optional momentum features
- [ ] Validate all games (completeness, quality)
- [ ] Save in Parquet format

### Dataset Preparation
- [ ] Split for Informer: train/val/cal/test (60/10/15/15)
- [ ] Prepare Dejavu database (all historical games)
- [ ] Create Conformal calibration set (held-out games)
- [ ] Validate schema matches model requirements

### Model-Specific Prep
- [ ] **Informer:** Create sequences (18-min input, 6-min output)
- [ ] **Conformal:** Extract (pattern, outcome) for calibration
- [ ] **Dejavu:** Build searchable pattern database
- [ ] Verify data shapes match specifications

---

## Quick Start Script

```python
#!/usr/bin/env python3
"""
Quick start: Collect NBA data from basketball-reference
"""

if __name__ == "__main__":
    # Configuration
    START_YEAR = 2021  # 2020-21 season
    END_YEAR = 2025    # Through 2024-25 season
    
    # Run pipeline
    pipeline = BasketballReferenceDataPipeline(
        start_year=START_YEAR,
        end_year=END_YEAR
    )
    
    games = pipeline.run(output_dir='./nba_data/')
    
    print("\n" + "=" * 80)
    print("DATA READY FOR MODELS")
    print("=" * 80)
    
    # Prepare for each model
    print("\nPreparing datasets...")
    
    # Informer
    informer_data = prepare_informer_dataset(games, seq_len=18, pred_len=6)
    print(f"  Informer: {len(informer_data)} training samples")
    
    # Conformal calibration
    X_cal, Y_cal, timestamps = prepare_conformal_calibration(games[-300:])
    print(f"  Conformal: {len(X_cal)} calibration samples")
    
    # Dejavu database
    dejavu_db = prepare_dejavu_database(games, pattern_length=18, forecast_horizon=6)
    print(f"  Dejavu: {len(dejavu_db)} patterns")
    
    print("\n✓ All datasets ready for model training/deployment!")
```

---

## Expected Data Statistics

### From 2020-2025 Collection

**Games Available:**
- 2020-21: ~1,080 games (COVID shortened)
- 2021-22: ~1,230 games
- 2022-23: ~1,230 games
- 2023-24: ~1,230 games
- 2024-25: ~600 games (partial season as of Oct 2025)
- **Total: ~5,400 games**

**After Filtering:**
- Remove overtime games? (affects halftime patterns): -5%
- Remove incomplete data: -2%
- **Usable: ~5,000 games**

**Train/Val/Cal/Test Split:**
- Training: 3,000 games (60%)
- Validation: 500 games (10%)
- Calibration: 750 games (15%)
- Test: 750 games (15%)

### Differential Statistics (Expected)

**At 6:00 2Q (18 minutes):**
- Mean differential: ~0 points (balanced)
- Std differential: ~8-10 points
- Range: -30 to +30 points (95% of games)

**At Halftime (0:00 2Q, 24 minutes):**
- Mean differential: ~0 points
- Std differential: ~10-12 points
- Range: -35 to +35 points (95% of games)

**Change from 6:00 2Q to Halftime:**
- Mean change: ~0 points
- Std change: ~4-6 points (inherent uncertainty)
- This ±5 point uncertainty is what models try to reduce

---

## Alternative Data Sources

### If Basketball-Reference Access Limited

**Option 1: NBA Stats API**
```
stats.nba.com API endpoints (unofficial, may have rate limits)
```

**Option 2: Kaggle Datasets**
```
Search: "NBA play-by-play data"
Available: Pre-scraped datasets for 2015-2023
```

**Option 3: Commercial APIs**
```
SportsRadar, ESPN API (paid, but structured)
```

**Option 4: Manual Download**
```
Basketball-reference allows CSV download for some tables
```

---

## Data Engineering Best Practices

### 1. Incremental Collection

```python
def incremental_update(existing_data_dir, current_date):
    """
    Update dataset with new games only
    """
    # Load existing game IDs
    existing_ids = load_existing_game_ids(existing_data_dir)
    
    # Get new games since last collection
    new_game_ids = get_games_since_date(current_date, existing_ids)
    
    # Scrape only new games
    new_games = scrape_all_pbp(new_game_ids)
    
    # Append to dataset
    save_games(new_games, existing_data_dir, mode='append')
    
    print(f"Added {len(new_games)} new games")
```

### 2. Data Versioning

```python
# Version datasets for reproducibility
dataset_versions = {
    'v1.0': 'nba_data_2020_2023.parquet',  # Initial collection
    'v1.1': 'nba_data_2020_2024.parquet',  # Added 2023-24 season
    'v2.0': 'nba_data_2020_2025.parquet'   # Current (2024-25 partial)
}

# Use versioned data for experiments
train_data = pd.read_parquet(dataset_versions['v2.0'])
```

### 3. Data Quality Monitoring

```python
# Track data quality over time
quality_log = {
    'collection_date': '2025-10-14',
    'n_games_collected': 5400,
    'n_games_valid': 5124,
    'validation_pass_rate': 0.949,
    'avg_missing_values_per_game': 0.02,
    'issues': ['12 games missing Q4 data', '3 games with extreme differentials']
}
```

---

## Summary

### What You Need from Basketball-Reference

**Essential (Minimum):**
1. ✅ Play-by-play data for 500-1,000 games (2023-2025)
2. ✅ Score at each minute (or interpolate from events)
3. ✅ Game dates for temporal features
4. ✅ Team identifiers

**Recommended (Better Performance):**
1. ✅ 3,000-5,000 games (2020-2025)
2. ✅ Team statistics (offensive/defensive ratings)
3. ✅ Game context (playoffs vs regular season)

**Optional (Enhanced Models):**
1. ✅ Player statistics
2. ✅ Possession-level data
3. ✅ Play-type classifications

### Data Collection Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| Setup scraper | 2-4 hours | Working scraper with cache |
| Collect game IDs | 10 minutes | ~5,400 game IDs |
| Scrape play-by-play | 3-10 hours | Raw play-by-play HTML |
| Parse & convert | 1-2 hours | Minute-by-minute CSV/Parquet |
| Validate & clean | 30 minutes | Clean dataset |
| **Total** | **7-17 hours** | **Production-ready dataset** |

### Storage Requirements

- **Raw HTML:** ~500 MB (5,000 games × 100 KB)
- **Parsed CSV:** ~100 MB
- **Parquet (compressed):** ~50 MB
- **Total (with cache):** ~650 MB

**Fits easily on any modern machine!**

---

## Ready to Collect

Run this pipeline and you'll have everything needed for:
- ✅ Informer training (long-sequence forecasting)
- ✅ Conformal calibration (uncertainty intervals)
- ✅ Dejavu database (pattern matching)

**The data is out there on basketball-reference. Go collect it!** 🏀

---

*Version 1.0.0 - October 14, 2025*

