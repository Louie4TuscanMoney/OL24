# Dejavu - Step by Step Build Guide

**Philosophy:** Build carefully, test thoroughly, verify each step  
**Date:** October 15, 2025

---

## 🎯 Our Goal

Build Dejavu model to predict NBA game outcomes at halftime (6:00 Q2)

**Input:** Score at 6:00 Q2 + team context  
**Output:** Predicted final score differential  
**Method:** K-NN pattern matching (k=500, no training needed)

---

## 📊 Understanding Our Data

### What We Have

**6 CSV files from Basketball Reference:**
- NBA_PBP_2015-16.csv (131 MB)
- NBA_PBP_2016-17.csv (131 MB)
- NBA_PBP_2017-18.csv (130 MB)
- NBA_PBP_2018-19.csv (134 MB)
- NBA_PBP_2019-20.csv (118 MB)
- NBA_PBP_2020-21.csv (21 MB - COVID shortened)

**Total:** 665 MB play-by-play data

### CSV Structure

**Each row = one play**

Key columns:
- `URL`: Game identifier (/boxscores/202012220BRK.html)
- `Date`: December 22 2020
- `Quarter`: 1, 2, 3, 4
- `SecLeft`: Seconds remaining in quarter (720 to 0)
- `HomeTeam`: BRK
- `AwayTeam`: GSW
- `HomeScore`: Current home score
- `AwayScore`: Current away score
- `WinningTeam`: Final winner

**Total rows:** ~600,000+ plays across all files

---

## 🛠️ Build Steps (Do These IN ORDER)

### Step 1: Test Data Loading (5 minutes)

**Goal:** Make sure we can load and parse one season

**Test file:** `test_01_loading.py`

```python
import pandas as pd

# Load just 2020-21 (smallest file for testing)
df = pd.read_csv('../../All Of Our Data/NBA_PBP_2020-21.csv')

print(f"Loaded {len(df):,} rows")
print(f"Columns: {len(df.columns)}")
print(f"Unique games: {df['URL'].nunique()}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Teams: {sorted(df['HomeTeam'].unique())}")

# Check for missing data
print(f"\nMissing data:")
print(df[['Quarter', 'SecLeft', 'HomeScore', 'AwayScore']].isnull().sum())
```

**Run this first, verify it works, THEN move to Step 2**

---

### Step 2: Extract One Game (10 minutes)

**Goal:** Process one game from start to finish

**Test file:** `test_02_single_game.py`

```python
import pandas as pd

# Load data
df = pd.read_csv('../../All Of Our Data/NBA_PBP_2020-21.csv')

# Get first game
first_game_url = df['URL'].iloc[0]
game_df = df[df['URL'] == first_game_url].copy()

print(f"Game: {first_game_url}")
print(f"Teams: {game_df['AwayTeam'].iloc[0]} @ {game_df['HomeTeam'].iloc[0]}")
print(f"Plays: {len(game_df)}")

# Calculate game time
game_df['GameTime'] = (game_df['Quarter'] - 1) * 720 + (720 - game_df['SecLeft'])

# Find halftime (6:00 Q2 = 1080 seconds)
target_time = 1080
game_df['TimeDiff'] = abs(game_df['GameTime'] - target_time)
halftime_idx = game_df['TimeDiff'].idxmin()
halftime_row = game_df.loc[halftime_idx]

print(f"\nHalftime (6:00 Q2):")
print(f"  Quarter: {halftime_row['Quarter']}")
print(f"  Time left: {halftime_row['SecLeft']} seconds")
print(f"  Score: {halftime_row['AwayTeam']} {halftime_row['AwayScore']}, {halftime_row['HomeTeam']} {halftime_row['HomeScore']}")
print(f"  Differential: {halftime_row['HomeScore'] - halftime_row['AwayScore']:+d}")

# Get final score
final_row = game_df.iloc[-1]
print(f"\nFinal:")
print(f"  Score: {final_row['AwayTeam']} {final_row['AwayScore']}, {final_row['HomeTeam']} {final_row['HomeScore']}")
print(f"  Differential: {final_row['HomeScore'] - final_row['AwayScore']:+d}")
print(f"  Winner: {final_row['WinningTeam']}")

# Calculate delta
delta = (final_row['HomeScore'] - final_row['AwayScore']) - (halftime_row['HomeScore'] - halftime_row['AwayScore'])
print(f"\nDelta (halftime → final): {delta:+d}")
```

**Verify this outputs correct scores, THEN move to Step 3**

---

### Step 3: Process ALL Games (15 minutes)

**Goal:** Extract halftime and final scores for ALL games in ONE season

**Only after Steps 1-2 work perfectly**

---

### Step 4: Build Reference Set (20 minutes)

**Goal:** Create patterns.npy and outcomes.npy from processed games

**Only after Step 3 produces good data**

---

### Step 5: Test K-NN (30 minutes)

**Goal:** Load reference set, run one prediction, verify it makes sense

**Only after Step 4 creates valid reference set**

---

### Step 6: Batch Testing (30 minutes)

**Goal:** Run 100 predictions, check accuracy, timing

**Only after Step 5 works**

---

## ⏰ Realistic Timeline

**Today:** Steps 1-3 (data loading and extraction)  
**Tomorrow:** Steps 4-6 (model building and testing)  
**Day 3:** Polish and optimize  
**Day 4:** Move to LSTM

**Don't rush. Build it right.** ✅

---

## 🎯 What To Do RIGHT NOW

1. Create `test_01_loading.py` (code above)
2. Run it: `python test_01_loading.py`
3. Verify output makes sense
4. **STOP and show me the output**
5. Only proceed to Step 2 when Step 1 is verified

---

**Let's build this carefully, one step at a time.** 🐢➡️🏆

