# Risk Optimization - Definition

**Purpose:** Quantitative risk management for NBA betting portfolio  
**Foundation:** Kelly Criterion + Black-Scholes + Bankroll Management  
**Date:** October 15, 2025

---

## What is Risk Optimization?

**Risk Optimization** is the mathematical framework for determining optimal bet sizing given edge opportunities, confidence levels, and bankroll constraints.

---

## Core Problem

**Given:**
- Bankroll: $5,000
- ML prediction: LAL +15.1 [+11.3, +18.9] at halftime
- BetOnline odds: LAL -7.5 (full game)
- Detected edge: STRONG_POSITIVE (19.2 point gap)

**Question:** How much should we bet?

**Naive approach:** "Bet $500 (10% of bankroll)"  
**Risk-optimized approach:** "Bet $X based on Kelly Criterion with confidence adjustment"

---

## The Three Pillars

### 1. **Kelly Criterion** (Optimal Bet Sizing)

**Formula:**
\[
f^* = \frac{p(b+1) - 1}{b}
\]

Where:
- \( f^* \) = Optimal fraction of bankroll to bet
- \( p \) = Probability of winning (from ML model)
- \( b \) = Odds received (converted from American odds)

**Purpose:** Maximize long-term growth rate while avoiding ruin

**For NBA betting:**
```python
# ML model gives probability
p = 0.65  # 65% chance of LAL covering

# BetOnline gives American odds
american_odds = -110  # Standard vig

# Convert to decimal odds
b = american_to_decimal(american_odds) - 1  # = 0.909

# Kelly fraction
f_kelly = (p * (b + 1) - 1) / b
# = (0.65 * 1.909 - 1) / 0.909
# = 0.187 (18.7% of bankroll)

# Bet size
bet_size = $5000 * 0.187 = $935
```

**Result:** Optimal bet is $935 (18.7% of $5,000 bankroll)

---

### 2. **Confidence Adjustment** (Conformal Intervals)

**Problem:** ML confidence varies (narrow vs wide intervals)

**Solution:** Adjust Kelly fraction by confidence:
```python
# ML gives interval
interval_width = 18.9 - 11.3 = 7.6 points

# Calculate confidence factor
confidence_factor = max(0, 1 - (interval_width / 20))
# = max(0, 1 - 7.6/20)
# = 0.62

# Adjusted Kelly
f_adjusted = f_kelly * confidence_factor
# = 0.187 * 0.62
# = 0.116 (11.6% of bankroll)

# Adjusted bet size
bet_size_adjusted = $5000 * 0.116 = $580
```

**Result:** Bet $580 instead of $935 (adjusted for uncertainty)

---

### 3. **Black-Scholes Adaptation** (Volatility)

**Traditional Black-Scholes (Options):**
\[
C = S_0 N(d_1) - K e^{-rT} N(d_2)
\]

**Adapted for Sports Betting (Implied Volatility):**
```python
# Measure volatility of ML predictions over time
historical_predictions = [14.5, 15.2, 14.8, 15.1, 15.0, ...]
volatility = np.std(historical_predictions)  # σ = 0.5

# Measure volatility of market odds
historical_spreads = [-7.5, -7.0, -7.5, -8.0, -7.5, ...]
market_volatility = np.std(historical_spreads)  # σ = 0.3

# High volatility → Reduce position size
volatility_factor = 1 / (1 + volatility)
# = 1 / 1.5 = 0.67

# Final bet size
bet_size_final = bet_size_adjusted * volatility_factor
# = $580 * 0.67 = $389
```

**Result:** Bet $389 (accounts for Kelly + confidence + volatility)

---

## Key Concepts

### 1. **Bankroll Management**

**Never risk entire bankroll:**
- Maximum single bet: 20% of bankroll
- Typical bet: 5-15% (Kelly-adjusted)
- Minimum reserve: 30% (for drawdowns)

---

### 2. **Expected Value (EV)**

**Formula:**
\[
EV = (P_{win} \times \text{Win Amount}) - (P_{loss} \times \text{Loss Amount})
\]

**Example:**
```python
# Bet $400 on LAL -110
p_win = 0.65  # From ML model
p_loss = 0.35

win_amount = $400 * (100/110) = $363.64
loss_amount = $400

EV = (0.65 * $363.64) - (0.35 * $400)
   = $236.36 - $140
   = $96.36

# Positive EV! Expected to profit $96.36 per bet
```

---

### 3. **Implied Probability from American Odds**

**Formula:**

For negative odds (favorites):
\[
P_{implied} = \frac{-\text{Odds}}{-\text{Odds} + 100}
\]

For positive odds (underdogs):
\[
P_{implied} = \frac{100}{\text{Odds} + 100}
\]

**Example:**
```python
# LAL -110 (favorite)
p_implied = 110 / (110 + 100) = 0.524 (52.4%)

# BOS +120 (underdog)
p_implied = 100 / (120 + 100) = 0.455 (45.5%)

# Note: 52.4% + 45.5% = 97.9% (vig is ~2.1%)
```

---

## Integration with ML System

### ML Model Output

**From:** Dejavu + LSTM + Conformal

```python
prediction = {
    'point_forecast': 15.1,
    'interval_lower': 11.3,
    'interval_upper': 18.9,
    'coverage_probability': 0.95
}
```

**Convert to probability:**
```python
# ML says LAL will lead by 15.1 at halftime
# Market spread is LAL -7.5

# If LAL leads by 15.1 at halftime, they likely cover -7.5 full game
# Historical correlation: ~0.85 (strong)

# Probability LAL covers
p_ml = 0.75  # Estimated from ML interval and historical correlation
```

---

### Market Odds (BetOnline)

```python
market_odds = {
    'spread': -7.5,
    'odds': -110  # Standard vig
}

# Implied probability
p_market = 110 / (110 + 100) = 0.524
```

---

### Edge Calculation

```python
# Edge exists when ML probability > Market probability
p_ml = 0.75
p_market = 0.524

edge = p_ml - p_market = 0.226 (22.6% edge!)

# This is a HUGE edge in sports betting
```

---

## Risk Optimization Output

**Given inputs above:**

```python
{
  'bankroll': 5000,
  'ml_probability': 0.75,
  'market_probability': 0.524,
  'edge': 0.226,
  'ml_interval_width': 7.6,
  'volatility': 0.5,
  
  'optimal_bet_size': 389,        # Kelly + confidence + volatility
  'optimal_fraction': 0.078,      # 7.8% of bankroll
  'expected_value': 96.36,        # Expected profit per bet
  'risk_of_ruin': 0.001,          # 0.1% chance of losing bankroll
  'max_drawdown_expected': 0.15,  # Expect 15% drawdowns
  
  'recommendation': 'BET',
  'confidence': 'HIGH',
  'reason': 'Large edge (22.6%), ML confidence acceptable'
}
```

---

## Performance Requirements

### Real-Time Calculations

| Operation | Target | Purpose |
|-----------|--------|---------|
| Convert odds to probability | <1ms | Instant |
| Calculate Kelly fraction | <5ms | Optimization |
| Adjust for confidence | <5ms | Risk management |
| Apply volatility factor | <5ms | Black-Scholes |
| **Total** | **<20ms** | **Real-time** |

**Critical:** Must not slow down 5-second BetOnline scraping cycle

---

## Integration Points

### Input 1: From ML Ensemble

```python
# ML prediction at 6:00 Q2
ml_prediction = {
    'point_forecast': 15.1,
    'interval_lower': 11.3,
    'interval_upper': 18.9,
    'coverage_probability': 0.95
}

# Convert to probability (using historical correlation)
p_win = convert_ml_to_probability(ml_prediction, market_spread=-7.5)
# Returns: 0.75 (75% probability LAL covers)
```

---

### Input 2: From BetOnline

```python
# Market odds from scraper
market_odds = {
    'spread': -7.5,
    'odds': -110,
    'total': 215.5
}

# Convert to implied probability
p_market = american_to_implied_probability(-110)
# Returns: 0.524 (52.4%)
```

---

### Output: Optimal Bet Size

```python
# Risk optimization calculates
optimal_bet = {
    'bet_size': 389,
    'fraction': 0.078,
    'expected_value': 96.36,
    'kelly_fraction': 0.187,
    'confidence_adjustment': 0.62,
    'volatility_adjustment': 0.67
}

# Send to trade execution system
```

---

## Why This Matters

### Without Risk Optimization

**Naive betting:**
- Bet same amount every time ($500)
- Ignore confidence levels
- Ignore edge size
- **Result:** Suboptimal growth, unnecessary risk

---

### With Risk Optimization

**Optimal betting:**
- Bet more when edge is large and confidence is high
- Bet less when uncertain or small edge
- Never over-bet (risk of ruin)
- **Result:** Maximum long-term growth, controlled risk

**Mathematical advantage:** Kelly Criterion proven to maximize logarithmic growth

---

## File Structure (Matches ML Model Folders)

```
RISK_OPTIMIZATION/
├── DEFINITION.md                    ← This file
├── MATH_BREAKDOWN.txt               ← Kelly, Black-Scholes formulas
├── RESEARCH_BREAKDOWN.txt           ← Academic foundations
├── RISK_IMPLEMENTATION_SPEC.md      ← Code specifications
├── DATA_ENGINEERING_RISK.md         ← Data pipelines
└── Applied Model/
    ├── kelly_calculator.py
    ├── confidence_adjuster.py
    ├── volatility_estimator.py
    └── risk_optimizer.py
```

---

## Next Steps

1. Read **MATH_BREAKDOWN.txt** (complete formulas)
2. Read **RESEARCH_BREAKDOWN.txt** (academic foundations)
3. Read **RISK_IMPLEMENTATION_SPEC.md** (implementation guide)

---

**Risk Optimization ensures:** Bet optimally, maximize growth, avoid ruin

---

*Last Updated: October 15, 2025*  
*Part of ML Research - Risk Management Layer*

