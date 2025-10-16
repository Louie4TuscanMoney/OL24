# What is Dejavu?

**Type:** Data-Centric Forecasting Model

**Authors:** Google Research (2023) - Arik et al.

---

## Definition

**Dejavu** is a forecasting method that predicts future values by finding **similar historical patterns** and reusing their outcomes.

Instead of learning complex model parameters, Dejavu uses a **memory-based approach**: store past time series patterns and retrieve the most similar ones when making predictions.

---

## Core Concept

```
"Show me games that started like this one → What happened in those games?"
```

Dejavu operates on the principle that **similar past patterns lead to similar future outcomes**.

---

## How It Works

1. **Database Construction**: Store all historical time series patterns with their outcomes
2. **Similarity Search**: When given a new pattern, find K most similar historical patterns
3. **Weighted Aggregation**: Combine outcomes of similar patterns (weighted by similarity)
4. **Forecast**: Return aggregated outcome as prediction

---

## Key Features

- ✅ **Interpretable**: Shows which historical games/patterns influenced the prediction
- ✅ **Non-parametric**: No training required, just store and retrieve
- ✅ **Fast inference**: Retrieval-based prediction (<5ms)
- ✅ **Adaptive**: Automatically incorporates new data into database
- ✅ **Cross-similarity**: Finds matches across different time series
- ✅ **Data-centric**: Quality depends on database size and diversity

---

## The "Dejavu" Metaphor

The name comes from the feeling of "I've seen this before":
- Current game pattern → "This reminds me of..."
- Historical database → Similar games from the past
- Prediction → What happened in those similar situations

---

## Technical Approach

**Distance Metric:**
- Dynamic Time Warping (DTW) or Euclidean distance
- Finds patterns with similar shape, not just exact matches

**Aggregation:**
```python
prediction = Σ(similarity_i × outcome_i) / Σ(similarity_i)
```

Closer matches get higher weight in the final prediction.

---

## Strengths

- **Highly interpretable**: "Your prediction is based on games X, Y, Z"
- **No overfitting**: No parameters to overfit
- **Handles anomalies**: Can find rare but relevant patterns
- **Real-time updates**: Add new games immediately

---

## Limitations

- **Requires large database**: Performance improves with more historical data
- **Curse of dimensionality**: Struggles with very high-dimensional patterns
- **Computational cost**: Similarity search scales with database size
- **Cold start**: Poor performance with limited historical data

---

## In Our System

Dejavu serves as:
- **Pattern matcher** for NBA games (finds games with similar 18-minute starts)
- **Baseline model** (~6 pts MAE)
- **Interpretability layer** (shows similar historical games)
- **Ensemble component** (40% weight in final prediction)

---

## Example

**Query:** Lakers lead by 8 points at 6:00 in Q2
**Dejavu Response:**
- "Similar to Game A (2023-02-15): Led by 7 → Won by 12"
- "Similar to Game B (2023-01-10): Led by 9 → Won by 15"  
- "Similar to Game C (2022-12-20): Led by 8 → Won by 10"
- **Prediction:** +12.3 points at halftime

---

**Bottom Line:** Dejavu is a memory-based forecaster that predicts by finding and reusing similar historical patterns.

