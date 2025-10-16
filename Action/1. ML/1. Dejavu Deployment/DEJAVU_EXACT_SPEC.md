# Dejavu EXACT Specification (Research-Verified)

**NO ASSUMPTIONS. NO GUESSING. ONLY RESEARCH FACTS.**

---

## DEFINITIVE SPECIFICATIONS (From Research)

### 1. K VALUE: **k = 500**

**Source:** MATH_BREAKDOWN.txt line 83, DEJAVU_MODEL.md line 235, RESEARCH_BREAKDOWN.txt line 141

**Exact Quote from Paper:**
> "we identify a sweet point at k = 500"
> "improvements seem to tapper off when k > 100"

**✅ DECISION: Use k=500**

---

### 2. AGGREGATION METHOD: **MEDIAN**

**Source:** MATH_BREAKDOWN.txt line 81

**Exact Quote:**
> "Forecast: median of their future paths (per horizon)"

**✅ DECISION: Use median (NOT weighted average)**

---

### 3. DISTANCE METRIC: **Euclidean (L2)**

**Source:** MATH_BREAKDOWN.txt lines 74-77, DEJAVU_MODEL.md line 236

**Paper Finding:**
> "L1 and L2 perform almost indistinguishable across all frequencies"
> "DTW outperforms... but is ×6 to ×27 slower"

**For NBA (speed-critical):**
- L2 (Euclidean) is fastest
- Performance difference with DTW is small (10^-2)
- DTW is 27× slower for monthly-like data

**✅ DECISION: Use Euclidean (L2) for speed**

---

### 4. NORMALIZATION: **Z-Score**

**Source:** MATH_BREAKDOWN.txt lines 124-136, 04_DEJAVU_DEPLOYMENT.md lines 100-101

**Paper Preprocessing:**
- Seasonal adjustment (if seasonal) - NBA: NO
- Smoothing (Loess) - NBA: NO (already minute-by-minute)
- Scaling by forecast origin - OR z-score normalization

**Action Step Code (line 100-101):**
```python
p1_norm = (pattern1 - np.mean(pattern1)) / (np.std(pattern1) + 1e-10)
p2_norm = (pattern2 - np.mean(pattern2)) / (np.std(pattern2) + 1e-10)
```

**✅ DECISION: Use z-score normalization (simpler, works well)**

---

## EXACT IMPLEMENTATION REQUIREMENTS

### Pattern Database Structure

**From our data (test_05_all_seasons_timeseries.py):**
```python
{
    'pattern': np.array (18,),          # Minutes 0-17 differential
    'outcome': float,                    # Halftime differential (minute 24)
    'game_id': str,
    'date': datetime,
    'season': str,
    'home_team': str,
    'away_team': str
}
```

**Database size:** 4,003 patterns (training set)

---

### Algorithm (EXACT)

**Step 1:** Load training set (4,003 games)

**Step 2:** For each query pattern:
  1. Normalize query: `query_norm = (query - mean(query)) / std(query)`
  2. For each database pattern:
     - Normalize: `db_norm = (db_pattern - mean(db_pattern)) / std(db_pattern)`
     - Compute Euclidean distance: `d = √(Σ(query_norm - db_norm)²)`
  3. Find k=500 smallest distances
  4. Get their outcomes (halftime differentials)
  5. **Return MEDIAN** of the 500 outcomes

**Step 3:** Evaluate on test set (893 games)

---

## CODE STRUCTURE

**File:** `dejavu_model.py`

```python
class DejavuForecaster:
    def __init__(self, k=500):
        self.k = 500  # Paper-verified optimal
        self.database = []
    
    def fit(self, train_df):
        """Build database from 4,003 training games"""
        for idx, row in train_df.iterrows():
            self.database.append({
                'pattern': row['pattern'],  # 18-point array
                'outcome': row['diff_at_halftime'],  # What we predict
                'metadata': {...}
            })
    
    def _normalize(self, pattern):
        """Z-score normalization"""
        return (pattern - np.mean(pattern)) / (np.std(pattern) + 1e-10)
    
    def _euclidean_distance(self, p1, p2):
        """Euclidean distance (L2)"""
        p1_norm = self._normalize(p1)
        p2_norm = self._normalize(p2)
        return np.sqrt(np.sum((p1_norm - p2_norm) ** 2))
    
    def predict(self, query_pattern):
        """Forecast using k=500 nearest neighbors + median"""
        # Compute distances to all 4,003 patterns
        distances = []
        for entry in self.database:
            d = self._euclidean_distance(query_pattern, entry['pattern'])
            distances.append((d, entry['outcome']))
        
        # Sort and select k=500 nearest
        distances.sort(key=lambda x: x[0])
        top_k = distances[:self.k]
        
        # Get outcomes
        outcomes = [outcome for (dist, outcome) in top_k]
        
        # Return MEDIAN (not mean!)
        forecast = np.median(outcomes)
        
        return forecast
```

---

## VALIDATION CHECKLIST

Before running, verify:
- ✅ k = 500 (not 10, not 100)
- ✅ Euclidean distance (not DTW, not Manhattan)
- ✅ Z-score normalization applied to BOTH patterns
- ✅ Median aggregation (not mean, not weighted average)
- ✅ Database has 4,003 patterns from training set
- ✅ Test on 893 games from test set

---

## EXPECTED PERFORMANCE

**From MODELSYNERGY.md line 852:**
- Dejavu alone: MAE ~6.0 points

**This is our target.** If we get MAE significantly different from 6.0, something is wrong.

---

*Ready to implement with ZERO assumptions*

