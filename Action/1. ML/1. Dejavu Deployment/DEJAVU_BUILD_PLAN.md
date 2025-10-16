# Dejavu Build Plan (Research-Based)

**Following:** Kang et al. "Déjà vu: A data-centric forecasting approach" (arXiv 2020)

---

## Research Requirements

### Dejavu 7-Step Methodology:
1. **Seasonal Adjustment** - Remove seasonality if detected (NBA: likely minimal)
2. **Smoothing** - Loess trend extraction to remove noise
3. **Scaling** - Divide by forecast origin for comparability  
4. **Similarity** - Distance measurement (L1/L2/DTW, paper shows L2 fastest)
5. **Aggregation** - k=500 neighbors (paper sweet spot)
6. **Inverse Scaling** - Return to original scale
7. **Reseasonalize** - Add seasonality back (if removed)

### NBA Context:
- **Pattern length (h):** 18 data points (game start → 6:00 Q2)
- **Forecast horizon (H):** 1 value (final score differential change)
- **Database:** All historical games (~6,600 games across 6 seasons)
- **Distance metric:** Start with L2 (Euclidean), paper shows it's fastest

---

## Data Requirements

### From 02_DATA_PROCESSING.md:
We need **minute-by-minute time series**, not just halftime snapshots.

### Required Structure:
```python
# Per game, we need:
{
    'game_id': str,
    'timeseries': [
        # Minute 0, 1, 2, ..., 47
        differential_0,
        differential_1,
        ...
        differential_47
    ],
    'pattern': timeseries[0:18],  # First 18 minutes (to 6:00 Q2)
    'outcome': differential_final - differential_halftime  # Change
}
```

---

## Build Steps (SLOW & CAREFUL)

### Step 1: Data Loading ✅ DONE
- Test: `test_01_loading.py` - Verify CSV loads correctly
- Status: PASSED

### Step 2: Single Game Time Series ✅ DONE  
- Test: `test_02_single_game.py` - Extract one game's halftime/final
- Status: PASSED

### Step 3: Minute-by-Minute Converter ⏳ NEXT
- **Goal:** Convert play-by-play → minute-by-minute differential
- **Reference:** `ML/Action Steps Folder/02_DATA_PROCESSING.md`
- **Output:** 48-minute time series per game
- **Why:** Dejavu needs regular-interval time series, not events

### Step 4: Pattern Extraction
- **Goal:** Extract 18-minute patterns from each game
- **Reference:** `DATA_ENGINEERING_DEJAVU.md` - PatternExtractor
- **Validation:** Check pattern quality (no NaN, sufficient variance)

### Step 5: Normalization/Scaling
- **Goal:** Z-score normalization for pattern matching
- **Reference:** Dejavu paper Step 3 (scaling)
- **Method:** `(x - mean) / std` per pattern

### Step 6: Build Reference Database
- **Goal:** Store all 6,600 games as searchable patterns
- **Structure:** {'pattern': array(18), 'outcome': float, 'metadata': dict}
- **Size:** ~6,600 patterns

### Step 7: K-NN Implementation
- **Goal:** Find k=500 most similar patterns
- **Distance:** Euclidean (L2 norm) 
- **Aggregation:** Median of k outcomes

### Step 8: Evaluation
- **Goal:** Validate on hold-out games
- **Metrics:** MAE, RMSE
- **Target:** MAE < 8 points (baseline)

---

## Current Status

✅ **COMPLETED:**
- test_01: Data loading works
- test_02: Single game extraction works  
- test_03: All games from one season (209 games 2020-21 partial)

❌ **ISSUE IDENTIFIED:**
- 2020-21 file only has 209 games (partial season, COVID)
- Need to process ALL 6 seasons (~6,600 games)

⏳ **NEXT:**
- Build minute-by-minute time series converter (following 02_DATA_PROCESSING.md)
- Convert play-by-play events → regular time series
- Apply proper Dejavu preprocessing

---

## Key Research Insights

### From MATH_BREAKDOWN.txt:
```
Pattern: x^h = [x_t, x_{t+1}, ..., x_{t+h-1}]  (length h=18)
Outcome: y^H = [y_{t+h}, ..., y_{t+h+H-1}]     (length H=1 for us)

Distance (L2): d(x_i, x_j) = √(Σ(x_i - x_j)²)
Similarity: sim = exp(-d / σ)
Forecast: ŷ = median{y_1, y_2, ..., y_k}  (k=500)
```

### From DEJAVU_IMPLEMENTATION_SPEC.md:
- **Pattern length:** NBA has short games (48 min) → Perfect for Dejavu
- **Database size:** Paper tested 1K-100K patterns → We have ~6.6K (good fit)
- **k=500:** Paper shows diminishing returns after k>100, k=500 is sweet spot
- **Normalization:** Critical for pattern matching across different game scales

### From RESEARCH_BREAKDOWN.txt:
- **L2 vs DTW:** L2 is 27× faster, only 2% less accurate
- **Preprocessing:** 28% MASE improvement with seasonal adjustment + smoothing
- **Limited data:** Dejavu wins when series < 6 years (we have exactly 5-6 years!)

---

## What NOT to Do

❌ Skip time series conversion (we need regular intervals)
❌ Use raw scores without normalization
❌ Rush pattern extraction without validation
❌ Use k<100 or k>1000 (paper shows k=500 optimal)
❌ Skip the 7-step preprocessing pipeline

---

## Next Immediate Action

**Build:** `test_04_minute_by_minute.py`
- Convert one game from play-by-play → 48-minute time series
- Follow `TimeSeriesConverter` from 02_DATA_PROCESSING.md
- Validate: 48 data points, one per minute
- Show the differential at each minute

**After that works:**
- Scale to all games
- Extract 18-minute patterns
- Build reference database
- Implement k=500 K-NN
- Test predictions

---

*Following research methodology - one step at a time*

