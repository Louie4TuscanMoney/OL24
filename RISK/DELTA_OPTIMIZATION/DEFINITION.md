# Delta Optimization - Definition

**Purpose:** Correlation-based hedging between ML probabilities and market odds  
**Foundation:** Delta hedging (options theory) + Correlation coefficients  
**Application:** Treating ML model and market as correlated assets  
**Date:** October 15, 2025

---

## What is Delta Optimization?

**Delta Optimization** treats ML predictions and market odds as two correlated assets, using their correlation coefficient to optimize position sizing and hedging strategies.

**Core Analogy:** Like rubber bands between two moving objects

---

## The Rubber Band Concept

```
ML Prediction (Asset 1)          Market Odds (Asset 2)
     +15.1                            -7.5
       ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━●
       
When gap is WIDE (rubber band stretched):
→ Strong tension
→ One will move toward the other
→ Opportunity to bet on convergence

When gap is NARROW (rubber band relaxed):
→ Low tension
→ Agreement between ML and market
→ Reduced opportunity

Correlation coefficient ρ measures "stiffness" of rubber band
High ρ (0.8-0.95): Stiff band, strong mean reversion
Low ρ (0.3-0.5): Loose band, can diverge
```

---

## Core Problem

**Given:**
- ML prediction: +15.1 (expects LAL to lead by 15.1 at halftime)
- Market spread: -7.5 (expects LAL to win by 7.5 full game)
- Historical correlation: ρ = 0.85 between ML and market
- Gap: 19.2 points (large divergence)

**Question:** How do we use correlation to optimize position?

---

## The Three Delta Concepts

### 1. **Delta as Sensitivity** (Options Theory)

**In options:**
\[
\Delta = \frac{\partial C}{\partial S}
\]
(Change in option price per change in underlying)

**In sports betting:**
\[
\Delta_{ML} = \frac{\partial P_{win}}{\partial \text{ML forecast}}
\]

**Example:**
```python
# ML forecast changes from +15.1 to +16.1 (+1 point)
# Win probability changes from 0.75 to 0.78 (+0.03)

Delta_ML = 0.03 / 1.0 = 0.03

# Interpretation: Each point in ML forecast adds 3% win probability
```

---

### 2. **Correlation as Risk Measure**

**Correlation coefficient:**
\[
\rho = \frac{\text{Cov}(ML, Market)}{\sigma_{ML} \times \sigma_{Market}}
\]

**Example:**
```python
# Historical data (last 50 games):
ML predictions:   [+10.5, +12.2, +8.5, +11.0, +13.5, ...]
Market spreads:   [-6.0, -7.5, -5.0, -6.5, -8.0, ...]

Correlation:      ρ = 0.85

# Interpretation:
# - High correlation (0.85)
# - ML and market usually agree
# - Current 19.2pt gap is UNUSUAL
# - Strong mean reversion expected
```

**Correlation levels:**
- ρ > 0.8: Very strong (tight rubber band)
- ρ = 0.5-0.8: Moderate (normal rubber band)
- ρ < 0.5: Weak (loose rubber band)

---

### 3. **Mean Reversion Strength**

**Gap analysis:**
\[
\text{Z-score} = \frac{\text{Current Gap} - \mu_{gap}}{\sigma_{gap}}
\]

**Example:**
```python
# Historical gaps (ML - Market implied):
Historical gaps: [+2, -1, +3, +1, -2, +2, +1, ...]
Mean gap: +1.2 points
Std dev: 3.5 points

# Current gap
Current gap: 19.2 points

# Z-score
z = (19.2 - 1.2) / 3.5 = 5.14

# Interpretation:
# - Gap is 5.14 standard deviations from mean
# - This is EXTREMELY unusual (>99.999% percentile)
# - Strong mean reversion expected
# - High probability one will move toward the other
```

---

## Delta Hedging Strategies

### Strategy 1: **Bet on Mean Reversion**

**When:** Large gap (z-score > 2.0), high correlation (ρ > 0.7)

**Action:**
```python
# Gap: ML=+15.1, Market implied=-4
# ML expects stronger LAL performance

# Bet on LAL (side ML favors)
# Size: Proportional to gap size and correlation
bet_size = base_kelly × (1 + 0.1 × z_score) × ρ

Example:
  base_kelly = $272 (from Kelly calculation)
  z_score = 5.14
  ρ = 0.85
  
  bet_size = $272 × (1 + 0.1 × 5.14) × 0.85
           = $272 × 1.514 × 0.85
           = $350
  
  Bet $350 on LAL (increased from $272 due to gap)
```

---

### Strategy 2: **Butterfly Spread** (Both Sides)

**When:** Moderate gap, moderate correlation, want to profit from convergence

**Setup:**
```python
# Bet on both mean reversion scenarios:

# Scenario A: Market moves toward ML (LAL gets stronger)
Bet 1: $200 on LAL -7.5 @ -110
  (Profits if LAL covers, which ML predicts)

# Scenario B: ML was overconfident (LAL weaker than predicted)
Bet 2: $50 on BOS +7.5 @ -110
  (Hedge in case ML wrong)

# Net: $150 bullish on LAL, with $50 hedge
# Like butterfly spread in options
```

**Payoff profile:**
```
LAL wins by 12:     Bet 1 wins +$182, Bet 2 loses -$50 = +$132 net
LAL wins by 6:      Both lose = -$250 net (worst case)
BOS wins by 2:      Bet 1 loses -$200, Bet 2 wins +$45 = -$155 net

Expected value: Positive if ML edge is real
Risk: Capped at $250 (total bet)
```

---

### Strategy 3: **Correlation-Adjusted Sizing**

**When:** Multiple correlated bets (same night, multiple games)

**Adjustment:**
```python
# Without correlation adjustment:
Game 1: Kelly says bet $300
Game 2: Kelly says bet $280
Game 3: Kelly says bet $260
Total: $840 (16.8% of $5,000)

# With correlation (ρ=0.20 between games on same night):
adjusted_total = $840 / sqrt(1 + (3-1)×0.20)
               = $840 / sqrt(1.40)
               = $840 / 1.183
               = $710

# Reduce total from $840 to $710 due to correlation
# Allocate proportionally:
Game 1: $710 × (300/840) = $254
Game 2: $710 × (280/840) = $237
Game 3: $710 × (260/840) = $219
```

---

## Integration with ML System

### ML Predictions (From Conformal)

```python
# ML gives:
prediction = {
    'point_forecast': 15.1,      # Best estimate
    'interval_lower': 11.3,      # 95% CI lower
    'interval_upper': 18.9,      # 95% CI upper
    'volatility': 1.939          # Derived: (18.9-11.3)/3.92
}

# Convert to delta:
delta_ml = change_in_forecast / change_in_probability
         = 1.0 / 0.03
         = 33.3

# Interpretation: Need 33-point forecast change to change prob by 100%
```

---

### Market Odds (From BetOnline)

```python
# Market gives:
market_odds = {
    'spread': -7.5,
    'odds': -110,
    'implied_probability': 0.524
}

# Convert to delta:
delta_market = change_in_spread / change_in_probability
             = 1.0 / 0.02
             = 50.0

# Interpretation: Need 50-point spread change to change prob by 100%
```

---

## Delta-Neutral Hedging

**Concept:** Balance position to be neutral to small movements

**Example:**
```python
# Want $1000 exposure to LAL, but hedge against small swings

# Calculate delta ratio
delta_ratio = delta_ml / delta_market = 33.3 / 50.0 = 0.666

# Position sizing for delta-neutral:
Position_ML_side = $1000
Position_Market_hedge = $1000 × (1 - delta_ratio) = $334

# Bet:
# $666 on LAL -7.5 (ML side)
# $334 on BOS +7.5 (Market hedge)

# Net exposure: $332 bullish LAL
# Protected against: Small divergences
# Profit from: Large moves in LAL direction
```

---

## Correlation Tracking

### Real-Time Correlation Calculation

```python
class CorrelationTracker:
    """
    Track correlation between ML and market in real-time
    """
    
    def __init__(self, window_size: int = 50):
        self.ml_history = []
        self.market_history = []
        self.window_size = window_size
    
    def update(self, ml_forecast: float, market_spread: float):
        """Add new observation"""
        self.ml_history.append(ml_forecast)
        self.market_history.append(market_spread)
        
        # Keep only recent window
        if len(self.ml_history) > self.window_size:
            self.ml_history.pop(0)
            self.market_history.pop(0)
    
    def get_correlation(self) -> float:
        """
        Calculate current correlation coefficient
        
        Returns:
            ρ ∈ [-1, 1]
        
        Time: <5ms
        """
        if len(self.ml_history) < 10:
            return 0.70  # Default assumption
        
        return np.corrcoef(self.ml_history, self.market_history)[0, 1]
    
    def get_gap_statistics(self) -> Dict:
        """
        Calculate gap statistics (for z-score)
        
        Returns:
            {
                'mean_gap': 1.2,
                'std_gap': 3.5,
                'current_gap': 19.2,
                'z_score': 5.14
            }
        """
        if len(self.ml_history) < 10:
            return {}
        
        gaps = [ml - mkt * 0.55 for ml, mkt in zip(self.ml_history, self.market_history)]
        
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        current_gap = gaps[-1]
        z_score = (current_gap - mean_gap) / std_gap if std_gap > 0 else 0
        
        return {
            'mean_gap': mean_gap,
            'std_gap': std_gap,
            'current_gap': current_gap,
            'z_score': z_score
        }
```

---

## Performance Requirements

### Real-Time Calculations

| Operation | Target | Complexity |
|-----------|--------|------------|
| Correlation update | <5ms | O(N), N≤100 |
| Delta calculation | <2ms | O(1) |
| Gap z-score | <3ms | O(N), N≤100 |
| Hedge ratio | <1ms | O(1) |
| Position sizing | <2ms | O(1) |
| **Total** | **<15ms** | **Real-time** |

**Critical:** Must not slow down system (adds to 20ms risk optimization)

---

## File Structure

```
DELTA_OPTIMIZATION/
├── DEFINITION.md                    ← This file
├── MATH_BREAKDOWN.txt               ← Delta formulas, correlation
├── RESEARCH_BREAKDOWN.txt           ← Academic foundations
├── DELTA_IMPLEMENTATION_SPEC.md     ← Implementation guide
├── DATA_ENGINEERING_DELTA.md        ← Data pipelines
└── Applied Model/
    ├── correlation_tracker.py
    ├── delta_calculator.py
    ├── hedge_optimizer.py
    └── butterfly_spreader.py
```

---

## Integration Points

### Input 1: ML Ensemble
```python
ml_prediction = {
    'point_forecast': 15.1,
    'interval': [11.3, 18.9],
    'volatility': 1.939
}
```

### Input 2: BetOnline
```python
market_odds = {
    'spread': -7.5,
    'odds': -110,
    'implied_prob': 0.524
}
```

### Output: Hedged Position
```python
optimal_position = {
    'primary_bet': 272.50,      # On LAL (ML side)
    'hedge_bet': 82.50,         # On BOS (market side)
    'net_exposure': 190.00,     # Net bullish LAL
    'delta_neutral': False,     # Directional bet
    'correlation': 0.85,        # High correlation
    'expected_convergence': True  # Gap likely to close
}
```

---

**Delta Optimization ensures:** Optimal hedging based on correlation, reduced risk from divergence

---

*Last Updated: October 15, 2025*  
*Part of ML Research - Risk Management Layer*

