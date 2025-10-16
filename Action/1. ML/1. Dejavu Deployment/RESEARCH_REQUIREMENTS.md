# Dejavu Research Requirements (EXACT)

**Sources Reviewed:**
1. `MATH_BREAKDOWN.txt` - Mathematical foundations
2. `DEJAVU_IMPLEMENTATION_SPEC.md` - Implementation guide
3. `04_DEJAVU_DEPLOYMENT.md` - Action steps

---

## Critical Research Facts

### From MATH_BREAKDOWN.txt (Lines 55-91): 7-Step Methodology

**The paper's full methodology:**

1. **Seasonal Adjustment** (if seasonal)
   - Test: `|ACF_s| > 1.645√(1 + 2Σ ACF_i²) / √n̂`
   - Method: STL decomposition
   - Apply Box-Cox if multiplicative

2. **Smoothing**
   - Method: Loess (Local Regression)
   - Removes noise and outliers
   - span parameter: h (forecast horizon)

3. **Scaling**
   - Divide by forecast origin (last historical value)
   - Makes series comparable

4. **Measure Similarity**
   - L1: `d_L1 = Σ|ỹ_t - Q̃^(i)_t|`
   - L2: `d_L2 = √(Σ(ỹ_t - Q̃^(i)_t)²)`
   - DTW: Dynamic programming alignment

5. **Aggregate k Most Similar**
   - **Select k series with smallest distances**
   - **Forecast: MEDIAN of their future paths**
   - **Paper tested: k ∈ {1, 5, 10, 50, 100, 500, 1000}**
   - **✅ OPTIMAL: k = 500** (line 83)

6. **Inverse Scaling**
   - Multiply forecasts by forecast origin

7. **Reseasonalize** (if seasonal)
   - Apply latest seasonal cycle
   - Inverse Box-Cox

### From 04_DEJAVU_DEPLOYMENT.md (Lines 54-72): Implementation Code

**The action steps show:**

```python
def __init__(
    self,
    K=10,  # ❓ DISCREPANCY: Code shows K=10, paper says k=500
    similarity_method='euclidean',
    weighting='gaussian',
    sigma=1.0
):
```

**Distance computation (lines 97-111):**
- Uses **z-score normalization** (not the paper's scaling by forecast origin)
- Euclidean: `√(Σ(p1_norm - p2_norm)²)`
- Manhattan: `Σ|p1_norm - p2_norm|`
- Correlation: `1 - |corr(pattern1, pattern2)|`

**Aggregation (lines 136-147):**
- Uses **weighted average** (not median as paper suggests)
- Weights: uniform, inverse distance, or Gaussian

---

## KEY DISCREPANCIES

| Aspect | Paper (MATH_BREAKDOWN.txt) | Code (04_DEJAVU_DEPLOYMENT.md) |
|--------|----------------------------|----------------------------------|
| **k value** | k = 500 (optimal) | K = 10 |
| **Normalization** | Scaling by forecast origin | Z-score normalization |
| **Aggregation** | Median | Weighted average |
| **7 steps** | Full methodology | Simplified (only similarity) |

---

## NBA-Specific Considerations

**Our Task:**
- Pattern: 18-minute differential sequence [min 0-17]
- Outcome: halftime differential (at minute 24)
- Database: 4,003 training games

**Questions to resolve:**

1. **K value:** Use k=500 (paper optimal) or K=10 (action step code)?
   - Paper explicitly states k=500 is optimal (line 83)
   - Action step example uses K=10 for demonstration

2. **Normalization:** Z-score or scaling by forecast origin?
   - Paper uses scaling (Step 3, line 70-72)
   - Code uses z-score (line 100-101)

3. **Aggregation:** Median or weighted average?
   - Paper uses median (line 81)
   - Code uses weighted average (line 147)

4. **Full 7 steps or simplified?**
   - Steps 1-2 (seasonal, smoothing): NBA likely doesn't need
   - Steps 3-5 (scale, similarity, aggregate): REQUIRED
   - Steps 6-7 (inverse, reseasonalize): Only if we do 1-3

---

## Recommended Approach

**Option A: Full Paper Implementation**
- k = 500
- Scaling by forecast origin
- Median aggregation
- Implement all 7 steps

**Option B: Simplified Production Version**
- K = 500 (use paper's optimal value)
- Z-score normalization (simpler, works well)
- Median aggregation (paper's method)
- Skip seasonal/smoothing (NBA doesn't need)

**Option C: Action Steps as Written**
- K = 10 (quick demo)
- Z-score normalization
- Weighted average
- Simplified

---

## What We MUST Decide

Before writing ANY code, we need to decide:

1. ✅ **K value:** 500 (paper optimal) or 10 (quick test)?
2. ✅ **Normalization:** Z-score or scaling?
3. ✅ **Aggregation:** Median or weighted average?
4. ✅ **Preprocessing:** Full 7 steps or simplified?

**NO ASSUMPTIONS. NO GUESSING.**

Let me know which approach to implement, and I'll build it exactly as specified.

---

*Waiting for clarification before proceeding*

