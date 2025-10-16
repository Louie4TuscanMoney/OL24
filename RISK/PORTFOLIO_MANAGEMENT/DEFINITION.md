# Portfolio Management - Definition

**Purpose:** Hedge fund-level portfolio optimization for multi-game betting  
**Foundation:** Modern Portfolio Theory (Markowitz) + Sharpe Ratio Optimization  
**Application:** Manage $5,000 bankroll across multiple NBA games as proprietary trading desk  
**Date:** October 15, 2025

---

## What is Portfolio Management?

**Portfolio Management** applies institutional-grade optimization techniques to allocate capital across multiple simultaneous betting opportunities, maximizing risk-adjusted returns.

**Core Philosophy:** Treat sports betting like a hedge fund manages a portfolio of assets

---

## The Portfolio Problem

**Scenario:** 10 NBA games tonight, ML identifies edges in 6 games

**Naive approach:**
- Bet same amount on each edge ($833 × 6 = $5,000)
- **Problem:** Ignores correlation, over-concentrates risk, suboptimal

**Portfolio approach:**
- Optimize allocation across all 6 opportunities
- Account for correlation between games
- Maximize Sharpe ratio (risk-adjusted return)
- **Result:** Optimal diversification, better risk/reward

---

## Core Concepts

### 1. **Efficient Frontier**

**From:** Markowitz, H. (1952). "Portfolio Selection." Journal of Finance, 7(1), 77-91.

```
Expected Return
     ↑
     |     ●  Optimal portfolio
     |    /
     |   /  Efficient frontier
     |  /●
     | / ●
     |/● ●●●
     ●───●───●───● → Risk (Std Dev)
     
Point: Choose portfolio on efficient frontier
Goal: Maximum return for given risk level
```

**For NBA betting:**
- X-axis: Portfolio volatility (risk)
- Y-axis: Expected return
- Points: Different allocation strategies
- Optimal: Highest Sharpe ratio on frontier

---

### 2. **Sharpe Ratio** (Risk-Adjusted Returns)

**Formula:**
\[
\text{Sharpe} = \frac{E[R] - R_f}{\sigma_R}
\]

**Example:**
```python
# Portfolio of 6 bets
Expected return: 15% per game night
Volatility: 25%
Risk-free rate: 0%

Sharpe = (0.15 - 0) / 0.25 = 0.60

# Interpretation:
# 0.60 Sharpe = Moderate risk-adjusted returns
# Target: >1.0 for excellent strategy
```

---

### 3. **Kelly-Markowitz Synthesis**

**Combine:**
- Kelly Criterion (optimal bet sizing)
- Markowitz Portfolio Theory (optimal allocation)

**Result:** Optimal allocation across correlated opportunities

**Example:**
```python
# 3 games with edges:
Game 1: Kelly=10%, Expected return=12%, σ=20%
Game 2: Kelly=8%,  Expected return=10%, σ=18%
Game 3: Kelly=12%, Expected return=15%, σ=25%

# Correlation matrix:
        Game1  Game2  Game3
Game1   1.00   0.20   0.15
Game2   0.20   1.00   0.25
Game3   0.15   0.25   1.00

# Optimize allocation to maximize Sharpe ratio:
Optimal weights: [35%, 28%, 37%]
Total allocation: 30% of bankroll (not 30% = 10%+8%+12%)

# Reason: Correlation reduces diversification benefit
```

---

## Portfolio Optimization Methods

### Method 1: **Mean-Variance Optimization**

**Objective:** Maximize expected return for given risk

\[
\max_w \quad w^T \mu - \frac{\lambda}{2} w^T \Sigma w
\]

Where:
- \( w \) = Portfolio weights (allocation to each game)
- \( \mu \) = Expected returns vector
- \( \Sigma \) = Covariance matrix
- \( \lambda \) = Risk aversion parameter

---

### Method 2: **Maximum Sharpe Ratio**

**Objective:** Maximize risk-adjusted returns

\[
\max_w \quad \frac{w^T \mu}{\sqrt{w^T \Sigma w}}
\]

Subject to: \( \sum w_i = 1 \) (weights sum to 1)

---

### Method 3: **Risk Parity**

**Objective:** Equal risk contribution from each position

\[
w_i \propto \frac{1}{\sigma_i}
\]

**Result:** Allocate more to low-volatility bets, less to high-volatility

---

## Portfolio Constraints

### Hard Constraints

```python
constraints = {
    'max_single_bet': 0.20,        # Max 20% on any game
    'max_total_exposure': 0.80,    # Max 80% of bankroll at risk
    'min_bet_size': 10,            # Minimum $10 per bet
    'max_simultaneous_bets': 10,   # Max 10 games at once
}
```

---

### Soft Constraints (Preferences)

```python
preferences = {
    'target_sharpe': 1.0,          # Aim for Sharpe > 1.0
    'max_drawdown_tolerance': 0.20, # Accept 20% drawdowns
    'min_win_rate': 0.55,          # Prefer >55% win rate
    'concentration_limit': 0.40,   # No sector >40% (e.g., home favorites)
}
```

---

## Real-Time Portfolio Optimization

**Performance:** <50ms for 10-game optimization

```python
class PortfolioOptimizer:
    """
    Optimize allocation across multiple betting opportunities
    
    Performance: <50ms for up to 10 games
    """
    
    def optimize_portfolio(
        self,
        opportunities: List[Dict],
        bankroll: float,
        risk_aversion: float = 2.0
    ) -> Dict:
        """
        Optimize allocation across opportunities
        
        Args:
            opportunities: List of {ml_pred, market_odds, kelly_size}
            bankroll: Current bankroll
            risk_aversion: Higher = more conservative
        
        Returns:
            {
                'allocations': [...],
                'total_exposure': 0.35,
                'expected_return': 0.12,
                'portfolio_sharpe': 0.85,
                'max_single_bet': 450.00
            }
        
        Time: <50ms for 10 games
        """
        # Use quadratic programming to solve
        # Maximize: Sharpe ratio
        # Subject to: Constraints
        
        # Implementation in PORTFOLIO_MANAGEMENT folder
        pass
```

---

## File Structure

```
PORTFOLIO_MANAGEMENT/
├── DEFINITION.md                         ← This file
├── MATH_BREAKDOWN.txt                    ← Markowitz, Sharpe formulas
├── RESEARCH_BREAKDOWN.txt                ← Academic foundations
├── PORTFOLIO_IMPLEMENTATION_SPEC.md      ← Implementation guide
├── DATA_ENGINEERING_PORTFOLIO.md         ← Data pipelines
└── Applied Model/
    ├── portfolio_optimizer.py
    ├── sharpe_calculator.py
    ├── efficient_frontier.py
    ├── risk_parity.py
    └── covariance_estimator.py
```

---

## Integration with Complete System

```
NBA_API (Scores)
        ↓
ML Ensemble (Predictions for 10 games)
        ↓
BetOnline (Odds for 10 games)
        ↓
RISK OPTIMIZATION (Kelly sizing for each game)
        ↓ Individual Kelly sizes
DELTA OPTIMIZATION (Correlation adjustments)
        ↓ Correlation-adjusted sizes
PORTFOLIO MANAGEMENT (Optimal allocation) ← THIS LAYER
        ↓ Final portfolio weights
TRADE EXECUTION (Place bets)
```

---

**Portfolio Management ensures:** Optimal diversification, maximum Sharpe ratio, controlled risk across all positions

---

*Last Updated: October 15, 2025*  
*Part of ML Research - Risk Management Layer*

