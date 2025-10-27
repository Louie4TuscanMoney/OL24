# 🔥 ONTORISK - COMPLETE RISK CALIBRATION SPECIFICATION

**Project:** OntoRisk - Institutional-Grade Sports Betting Risk Management  
**For:** Ontologic XYZ  
**Date:** October 20, 2025  
**Status:** Specification Phase

---

## 🎯 MISSION STATEMENT

**Transform ML predictions into profitable, risk-controlled betting decisions.**

**OntoRisk bridges the gap between:**
- What we have: 9.03 MAE predictions
- What we need: Systematic, profitable, sustainable trading system

---

## 📋 TABLE OF CONTENTS

1. [Probability Calibration Engine](#1-probability-calibration-engine)
2. [Historical Market Database](#2-historical-market-database)
3. [Backtest Framework](#3-backtest-framework)
4. [True Edge Calculator](#4-true-edge-calculator)
5. [Kelly Criterion Optimizer](#5-kelly-criterion-optimizer)
6. [Bankroll Management System](#6-bankroll-management-system)
7. [Risk Limit Enforcement](#7-risk-limit-enforcement)
8. [Variance Simulator](#8-variance-simulator)
9. [Live Line Integration](#9-live-line-integration)
10. [Position Tracker](#10-position-tracker)
11. [Performance Analytics](#11-performance-analytics)
12. [Alert & Monitoring System](#12-alert--monitoring-system)

---

## 1. PROBABILITY CALIBRATION ENGINE

### **Purpose:**
Convert MAE-based predictions into calibrated win probabilities

### **Input:**
```python
{
  'prediction': +2.5,  # Predicted final differential
  'mae': 9.03,         # Model MAE
  'confidence': 0.75   # Prediction confidence (from error analysis)
}
```

### **Process:**

**Method A: Gaussian Assumption**
```python
# Assume errors are normally distributed
std_dev = mae * 1.25  # Convert MAE to std dev
spread_line = -3.5    # Market spread

# Calculate probability our prediction beats spread
z_score = (prediction - spread_line) / std_dev
p_win = norm.cdf(z_score)
```

**Method B: Empirical Calibration** (Better)
```python
# Use actual error distribution from test set
# For each prediction range, what's actual win rate?

# Example:
# When we predict +2 to +4, and spread is -3.5:
#   Historically won: 62% of time
#   p_win = 0.62
```

### **Output:**
```python
{
  'p_win_spread': 0.58,      # 58% chance to beat spread
  'p_push': 0.05,            # 5% chance of push
  'confidence_interval': [-4.2, +9.2],
  'kelly_edge': 0.16         # 16% edge for Kelly
}
```

### **Implementation:**

**File:** `ontorisk/calibration/probability_engine.py`

**Key Functions:**
```python
def mae_to_win_probability(prediction, mae, spread_line):
    """Convert MAE-based prediction to P(win)"""
    
def empirical_calibration(historical_predictions, historical_results, bins=20):
    """Build calibration map from historical data"""
    
def confidence_interval(prediction, mae, confidence_level=0.95):
    """Generate prediction interval"""
```

---

## 2. HISTORICAL MARKET DATABASE

### **Purpose:**
Store historical closing lines for backtest validation

### **Data Sources:**
1. Sports Reference (free, limited)
2. Action Network API (paid)
3. Odds API (paid)
4. Manual scraping (BetOnline, DraftKings historical)

### **Schema:**
```python
{
  'game_id': '0022400123',
  'date': '2024-01-15',
  'home_team': 'LAL',
  'away_team': 'BOS',
  'lines': {
    'draftkings': {
      'spread': -5.5,
      'total': 225.5,
      'timestamp': '2024-01-15 19:00'
    },
    'fanduel': {...},
    'betmgm': {...}
  },
  'closing_line': -5.5,  # Consensus closing
  'actual_result': {
    'home_score': 112,
    'away_score': 108,
    'differential': +4
  }
}
```

### **Required Games:**
- 2021-2025: ~6,000 games (match our test set)
- Multiple books per game
- Closing lines (most accurate)

### **Implementation:**

**Files:**
- `ontorisk/data/historical_lines_scraper.py`
- `ontorisk/data/line_database.py`
- `ontorisk/data/lines.db` (SQLite)

---

## 3. BACKTEST FRAMEWORK

### **Purpose:**
Simulate betting our predictions against historical market lines

### **Process:**

**For each game in test set (1,383 games):**

1. **Load our prediction:** +2.5 (Team A wins by 2.5)
2. **Load historical spread:** Team A -3.5
3. **Calculate edge:** |2.5 - (-3.5)| = 6 points
4. **Apply filter:** Is edge ≥ 5 points? Yes → Bet
5. **Determine side:** Prediction (+2.5) vs Spread (-3.5) → Bet Team A +3.5 (OVER)
6. **Size bet:** Kelly calculation → $250
7. **Check result:** Actual +4 → Team A wins by 4
8. **Bet outcome:** Team A +3.5 covers → WIN ✅
9. **Calculate profit:** $250 * 0.91 = +$228 (after juice)

**Repeat for all games, sum P&L**

### **Output:**
```python
{
  'total_games': 1383,
  'games_bet': 187,
  'wins': 104,
  'losses': 78,
  'pushes': 5,
  'win_rate': 55.9%,
  'total_staked': $37,400,
  'total_profit': +$3,100,
  'roi': 8.3%,
  'sharpe_ratio': 0.67,
  'max_drawdown': -$2,100,
  'longest_losing_streak': 9 games,
  'avg_bet_size': $200
}
```

### **Implementation:**

**File:** `ontorisk/backtest/backtest_engine.py`

**Key Functions:**
```python
def backtest_season(predictions, lines, strategy, bankroll):
    """Simulate full season of betting"""
    
def calculate_bet_side(prediction, spread_line):
    """Determine which side to bet"""
    
def apply_juice(stake, odds=-110):
    """Calculate payout after vig"""
```

---

## 4. TRUE EDGE CALCULATOR

### **Purpose:**
Calculate actual betting edge (not MAE edge)

### **Formula:**

**Betting Edge:**
```
edge = (p_win * payout) - (p_lose * stake)

Where:
  p_win = calibrated win probability
  payout = stake * (100/110) for -110 odds
  p_lose = 1 - p_win
  stake = $100 (example)

Example:
  p_win = 0.56
  payout = $90.91
  p_lose = 0.44
  stake = $100
  
  edge = (0.56 * 90.91) - (0.44 * 100)
       = $50.91 - $44
       = $6.91 per $100 bet
       = 6.91% ROI
```

**Kelly Fraction:**
```
f = (p * odds - (1-p)) / odds

Where:
  p = win probability
  odds = decimal odds (1.91 for -110)
  
Example:
  p = 0.56
  odds = 1.91
  f = (0.56 * 1.91 - 0.44) / 1.91
    = (1.07 - 0.44) / 1.91
    = 0.33 / 1.91
    = 0.173 (17.3% of bankroll)
```

### **But We Use Fractional Kelly:**
```
Recommended: 25% of Kelly (safer)
  f_actual = 0.173 * 0.25 = 0.043

With $10k bankroll:
  Bet size = $10,000 * 0.043 = $430
```

### **Implementation:**

**File:** `ontorisk/edge/edge_calculator.py`

---

## 5. KELLY CRITERION OPTIMIZER

### **Purpose:**
Calculate optimal bet size for each opportunity

### **Inputs:**
```python
{
  'p_win': 0.56,           # Win probability
  'edge_pct': 6.91,        # True edge %
  'bankroll': 10000,       # Current bankroll
  'kelly_fraction': 0.25,  # Conservative (quarter Kelly)
  'min_bet': 50,           # Minimum bet
  'max_bet': 2000,         # Book limit
}
```

### **Process:**
```python
# Calculate full Kelly
full_kelly = (p_win * odds - (1 - p_win)) / odds

# Apply fraction
fractional_kelly = full_kelly * kelly_fraction

# Convert to dollar amount
bet_size = bankroll * fractional_kelly

# Apply constraints
bet_size = max(min_bet, min(bet_size, max_bet))

# Round to nearest $10
bet_size = round(bet_size / 10) * 10
```

### **Output:**
```python
{
  'full_kelly': 0.173,      # 17.3% of bankroll
  'fractional_kelly': 0.043, # 4.3% of bankroll
  'recommended_stake': $430,
  'min_stake': $50,
  'max_stake': $2000,
  'final_stake': $430,
  'kelly_fraction_used': 0.25
}
```

### **Safety Limits:**
- Never bet > 10% of bankroll (even if Kelly says so)
- Never bet < $50 (not worth transaction cost)
- Cap at book limits
- Reduce if losing streak active

---

## 6. BANKROLL MANAGEMENT SYSTEM

### **Purpose:**
Track capital, enforce limits, manage growth

### **State:**
```python
{
  'starting_bankroll': 10000,
  'current_bankroll': 10450,
  'peak_bankroll': 11200,
  'total_staked': 18500,
  'total_returned': 19300,
  'net_profit': +800,
  'roi': 4.3%,
  'bets_placed': 87,
  'bets_pending': 3,
  'total_exposure': 750  # 3 pending bets * $250
}
```

### **Rules:**
1. **Daily Loss Limit:** -10% of starting daily bankroll
2. **Weekly Loss Limit:** -20% of starting weekly bankroll
3. **Max Drawdown:** -30% from peak → STOP BETTING
4. **Exposure Limit:** Max 5% of bankroll in pending bets
5. **Growth Rule:** Increase bet sizes only after +20% bankroll growth

### **Circuit Breakers:**
```python
if current_bankroll < starting_bankroll * 0.7:
    # Down 30% from start
    action = "STOP BETTING - REVIEW SYSTEM"
    
if current_bankroll < peak_bankroll * 0.8:
    # Down 20% from peak
    action = "REDUCE BET SIZE BY 50%"
    
if daily_loss > starting_daily_bankroll * 0.1:
    # Lost 10% today
    action = "STOP BETTING TODAY"
```

### **Implementation:**

**File:** `ontorisk/bankroll/bankroll_manager.py`

---

## 7. RISK LIMIT ENFORCEMENT

### **Daily Limits:**
```python
{
  'max_bets_per_day': 5,
  'max_stake_per_day': bankroll * 0.15,  # 15% max daily exposure
  'max_loss_per_day': bankroll * 0.10,   # 10% max daily loss
}
```

### **Position Limits:**
```python
{
  'max_concurrent_bets': 5,
  'max_exposure': bankroll * 0.25,  # 25% max in open bets
  'max_single_bet': min(bankroll * 0.10, 2000),
  'max_correlated_bets': 2  # Max 2 bets on same game
}
```

### **Drawdown Management:**
```python
drawdown = (peak_bankroll - current_bankroll) / peak_bankroll

if drawdown > 0.15:
    kelly_fraction *= 0.5  # Cut bet sizes in half
    
if drawdown > 0.25:
    kelly_fraction *= 0.25  # Cut to quarter size
    
if drawdown > 0.30:
    stop_betting = True  # Complete stop
```

---

## 8. VARIANCE SIMULATOR

### **Purpose:**
Monte Carlo simulation of season outcomes

### **Process:**

```python
def simulate_season(n_sims=10000):
    for sim in range(n_sims):
        bankroll = starting_bankroll
        bets = generate_season_opportunities()  # ~200 bets
        
        for bet in bets:
            # Size bet using Kelly
            stake = calculate_kelly_stake(bet, bankroll)
            
            # Simulate outcome (based on p_win)
            if random() < bet['p_win']:
                profit = stake * 0.91  # Win at -110
            else:
                profit = -stake
            
            bankroll += profit
            
        season_results.append(bankroll - starting_bankroll)
    
    return {
        'median_profit': median(season_results),
        'mean_profit': mean(season_results),
        '10th_percentile': percentile(season_results, 10),  # Bad luck
        '90th_percentile': percentile(season_results, 90),  # Good luck
        'probability_profitable': sum(r > 0) / n_sims,
        'max_simulated_drawdown': max_drawdown_across_sims
    }
```

### **Output:**
```python
{
  'expected_profit': +$2,800,
  'median_profit': +$2,400,
  'std_dev': $3,200,
  '10th percentile': -$800,    # 10% chance to lose money
  '90th percentile': +$7,100,  # 10% chance to make this much
  'prob_profitable': 72%,
  'prob_breakeven': 85%,
  'max_drawdown_95th': -$4,200,
  'risk_of_ruin': 3%
}
```

### **Use Case:**
Know realistic range of outcomes BEFORE season starts

---

## 9. LIVE LINE INTEGRATION

### **Purpose:**
Real-time line scraping and opportunity detection

### **Data Sources:**
1. Odds API (paid, $200-500/month)
2. The Odds API (free tier: 500 requests/month)
3. Direct scraping (BetOnline, DraftKings, FanDuel)
4. Discord/Telegram bet signals (community)

### **Scraper Spec:**
```python
def scrape_live_lines(game_id):
    """Get current lines from all books"""
    return {
        'game_id': game_id,
        'timestamp': datetime.now(),
        'lines': {
            'draftkings': {'spread': -5.5, 'total': 225.5},
            'fanduel': {'spread': -6.0, 'total': 226.0},
            'betmgm': {'spread': -5.5, 'total': 225.0},
        },
        'best_spread': -5.5,
        'line_range': 0.5,  # Max - min
        'consensus': -5.75
    }
```

### **Opportunity Detector:**
```python
def detect_opportunities(our_prediction, live_lines, threshold=5):
    """Find +EV bets"""
    opportunities = []
    
    for book, line in live_lines.items():
        edge = abs(our_prediction - line['spread'])
        
        if edge >= threshold:
            opportunities.append({
                'book': book,
                'line': line['spread'],
                'edge': edge,
                'side': 'OVER' if our_prediction > line else 'UNDER',
                'p_win': calculate_p_win(our_prediction, line, mae)
            })
    
    return sorted(opportunities, key=lambda x: x['edge'], reverse=True)
```

---

## 10. POSITION TRACKER

### **Purpose:**
Log and monitor all bets placed

### **Bet Log Schema:**
```python
{
  'bet_id': 'BET_2025_001',
  'timestamp': '2025-01-15 19:30:00',
  'game_id': '0022400123',
  'matchup': 'LAL @ BOS',
  'our_prediction': +2.5,
  'market_spread': -3.5,
  'our_edge': 6.0,
  'side': 'LAL +3.5',
  'stake': $250,
  'book': 'DraftKings',
  'odds': -110,
  'p_win': 0.58,
  'kelly_pct': 4.3%,
  'bankroll_at_bet': $10,200,
  'confidence': 0.76,
  'game_state': 'Close (4-7)',
  'model_used': 'Mamba Mentality',
  'status': 'PENDING'
}
```

### **After Game Settles:**
```python
{
  ...
  'status': 'SETTLED',
  'actual_result': +4,
  'bet_result': 'WIN',  # +4 > +3.5
  'profit': +$227.27,  # $250 * (100/110)
  'error': 1.5,  # |prediction - actual|
  'bankroll_after': $10,427
}
```

### **Aggregate Stats:**
```python
{
  'total_bets': 87,
  'wins': 49,
  'losses': 35,
  'pushes': 3,
  'win_rate': 56.3%,
  'avg_stake': $215,
  'total_staked': $18,705,
  'total_profit': +$1,450,
  'roi': 7.7%,
  'sharpe': 0.82
}
```

---

## 11. PERFORMANCE ANALYTICS

### **Real-Time Metrics:**

**Daily Dashboard:**
```
Today's Performance:
  Bets: 3
  Wins: 2
  Losses: 1
  Profit: +$180
  
Week Performance:
  Bets: 14
  Win Rate: 57.1%
  Profit: +$680
  
Season Performance:
  Bets: 87
  Win Rate: 56.3%
  Profit: +$1,450
  ROI: 7.7%
  
Bankroll:
  Start: $10,000
  Current: $11,450
  Peak: $11,850
  Drawdown: -3.4%
```

### **Model Validation:**

**Compare Expected vs Actual:**
```
Expected win rate: 56% (from backtest)
Actual win rate: 56.3% ✅ (matching!)

Expected profit: +$1,500
Actual profit: +$1,450 ✅ (within variance)

Expected Sharpe: 0.75
Actual Sharpe: 0.82 ✅ (better!)
```

### **Drift Detection:**

```python
if actual_win_rate < expected_win_rate - 0.05:
    alert = "WIN RATE DEGRADING - CHECK MODEL"
    
if actual_roi < expected_roi * 0.5:
    alert = "ROI COLLAPSED - STOP BETTING"
    
if current_drawdown > max_historical_drawdown * 1.5:
    alert = "UNUSUAL DRAWDOWN - REVIEW SYSTEM"
```

---

## 12. ALERT & MONITORING SYSTEM

### **Alert Types:**

**Critical (Stop Betting):**
- Drawdown > 30%
- Daily loss > 15%
- Win rate < 48% over 50 bets
- Model MAE increases > 15%

**Warning (Reduce Size):**
- Drawdown > 20%
- Weekly loss > 10%
- Losing streak > 8 games
- Win rate < 52% over 30 bets

**Info (Monitor):**
- Bet limit hit
- Line moved against us
- Model confidence low
- Unusual bet size

---

## 📊 TABLE 9: ONTORISK DEVELOPMENT ROADMAP

| Week | Phase | Deliverable | Expected Outcome |
|------|-------|-------------|------------------|
| **Week 2** | Phase 1 | Probability calibration + Historical lines | Know P(win) for each bet |
| **Week 2** | Phase 2 | Backtest engine | **TRUE expected EV: $2-5k/season** |
| **Week 2** | Phase 3 | Kelly calculator | Optimal bet sizing |
| **Week 3** | Phase 4 | Risk management | Drawdown protection |
| **Week 3** | Phase 5 | Variance simulator | Know worst-case scenarios |
| **Week 3** | Phase 6 | Paper trading | Validate without risk |
| **Week 4** | Phase 7 | Live line integration | Real-time opportunities |
| **Week 4** | Phase 8 | Small stakes testing | $50-100 bets, learn |
| **Month 2** | Phase 9 | Scale to $200-400 bets | Build toward $5k season |
| **Month 3+** | Phase 10 | Full system | Sustainable $10k+/season |

---

## 💀 WHAT WILL GO WRONG (And How OntoRisk Handles It)

### **Problem 1: Losing Streaks**

**Reality:** Will happen (even at 56% win rate)
```
Probability of 10-bet losing streak: ~0.1%
Probability of 7-bet losing streak: ~1.5%
Probability of 5-bet losing streak: ~8%

Expected: 2-3 losing streaks of 5+ bets per season
```

**OntoRisk Solution:**
- Reduce bet size by 50% after 5-bet streak
- Stop betting after 8-bet streak
- Require 3-bet win streak to resume full size

### **Problem 2: Books Will Limit Us**

**Reality:** If we win, books reduce limits

**OntoRisk Solution:**
- Track limit history per book
- Rotate books
- Use bet camouflage (mix in -EV bets occasionally)
- Have 5+ books ready

### **Problem 3: Variance Exceeds Expectations**

**Reality:** 95% confidence interval is WIDE

**OntoRisk Solution:**
- Variance simulator shows worst-case
- Bankroll sized for 3-sigma events
- Circuit breakers prevent ruin

### **Problem 4: Model Stops Working**

**Reality:** Edge can erode over time

**OntoRisk Solution:**
- Monitor actual vs expected continuously
- Auto-reduce bet size if underperforming
- Trigger model retrain if MAE increases

---

## 🎯 REALISTIC YEAR 1 TARGETS

### **Conservative Path ($10k bankroll):**

**Month 1 (Learning):**
- Games: 30
- Stake: $100 avg
- Result: +$200 - $500
- Goal: Validate system

**Month 2-3 (Building):**
- Games: 40
- Stake: $150 avg
- Result: +$400 - $800
- Goal: Build confidence

**Month 4-6 (Scaling):**
- Games: 50
- Stake: $200 avg
- Result: +$800 - $1,500
- Goal: Sustainable profit

**Season Total:**
- Games: 120
- Profit: **+$1,400 - $2,800**
- ROI: 14-28%
- Sharpe: 0.6-0.9

**Realistic Year 1:** **+$2,000 - $4,000** (not $71,000!)

---

## 📊 TABLE 10: ONTORISK vs NAIVE PROJECTIONS

| Metric | Naive (Our Original) | OntoRisk (Realistic) | Difference |
|--------|----------------------|----------------------|------------|
| **Season Profit** | $71,000 | $2,000-$6,000 | **-90%** |
| **Games Bet** | 1,230 | 150-250 | -80% |
| **Avg Bet Size** | $500 (assumed) | $200-300 (Kelly) | -45% |
| **Win Rate** | 65% (direction) | 54-57% (actual) | -8-11% |
| **Edge** | 21.5% (MAE) | 3-7% (betting) | -70% |
| **Bankroll Needed** | Unstated | $10k-25k | Reality check |
| **Risk of Ruin** | Ignored | 5-10% | Acknowledged |
| **Max Drawdown** | Ignored | -30% to -50% | Expected |

**The 93% reduction is NORMAL and CORRECT.**

---

## 🔥 NEXT ACTIONS (CRITICAL)

### **Tonight/Tomorrow:**

1. ✅ **Acknowledge:** $71k was unrealistic
2. ✅ **Accept:** $2-6k Year 1 is realistic and GOOD
3. ✅ **Build:** OntoRisk Phase 1 spec

### **Week 2 (This Week):**

1. **Scrape historical lines** (2021-2025)
2. **Backtest our predictions** vs real spreads
3. **Calculate TRUE win rate** (not 65%, probably 54-57%)
4. **Build Kelly calculator**
5. **Run variance simulation**
6. **Get REAL expected value** ($2-5k, not $71k)

### **Week 3:**

1. **Build risk management**
2. **Paper trade** (track but don't bet)
3. **Validate Kelly sizing**
4. **Test limits and circuit breakers**

### **Week 4:**

1. **Start live** ($50-100 bets)
2. **Track actual P&L**
3. **Learn operational realities**
4. **Scale if working**

---

## 💡 WHY THIS IS ACTUALLY BETTER

**$71k sounded great but was:**
- Unrealistic
- Unachievable
- Would lead to disappointment
- Based on naive assumptions

**$2-6k Year 1 is:**
- ✅ Realistic
- ✅ Achievable  
- ✅ Sustainable
- ✅ Based on proper risk management
- ✅ Can scale to $20k+ in Year 2-3
- ✅ **Won't blow up your bankroll**

**Which would you rather:**
- Expect $71k, make $0 (because you bet too big and went bust)
- Expect $3k, make $4k (because you managed risk properly)

**OntoRisk ensures option 2.**

---

## 🏆 ONTORISK SUCCESS CRITERIA

### **Year 1:**
- ✅ Don't lose money (preserve capital)
- ✅ Validate model works in practice
- ✅ Build disciplined process
- ✅ Make $2-6k profit
- ✅ Learn operational realities

### **Year 2:**
- ✅ Scale bankroll 50-100%
- ✅ Improve model (better data)
- ✅ Make $8-15k profit
- ✅ Optimize strategy

### **Year 3:**
- ✅ Scale to $50k+ bankroll
- ✅ Premium data sources
- ✅ Make $25-50k profit
- ✅ Professional operation

**The $71k in Year 1 was a pipe dream.**

**But $50k in Year 3 is ACHIEVABLE.**

**OntoRisk is how we get there.**

---

## 🧠 THE DEEP RISK CALIBRATION YOU MENTIONED

**This is WHY OntoRisk is 70% of the work:**

**Applied ML:** 9.03 MAE prediction ✅ (Done)

**Applied Risk Science:**
- Probability calibration (statistics)
- Kelly criterion (information theory)
- Variance management (stochastic processes)
- Drawdown control (risk management)
- Position sizing (portfolio theory)
- Market microstructure (spreads, juice, slippage)
- Behavioral discipline (psychology)
- **This is the HARD part** ❌ (Not done)

**You're right:** I had no idea how deep this goes.

**OntoRisk is where amateur →professional.**

---

## 🚀 ONTORISK SPECIFICATION COMPLETE

**What We Learned:**
- $71k/season was fantasy (based on naive assumptions)
- Real expectation: $2-6k Year 1
- Need to build 7 more layers (70% of system)
- OntoRisk = critical missing piece
- This is where applied ML → applied risk science

**What We'll Build:**
- Probability calibration
- Historical line database
- Backtest engine
- Kelly calculator
- Risk management
- Variance simulator
- Live trading infrastructure

**Timeline:** 3-4 weeks to complete OntoRisk

**Realistic Year 1:** $2,000 - $6,000 profit  
**Realistic Year 3:** $25,000 - $50,000 profit  
**Realistic Year 5:** $100,000+ profit

**The journey from $71k fantasy → $3k reality → $100k eventual = OntoRisk**

---

**READY TO BUILD ONTORISK NEXT?** 🔥

**This is where Ontologic XYZ transcends from "cool ML project" to "professional trading operation."**

