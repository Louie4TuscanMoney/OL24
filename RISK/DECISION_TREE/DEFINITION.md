# Decision Tree Risk Management - Definition

**Purpose:** Finite mathematics decision tree for loss recovery and progression betting  
**Foundation:** Probability theory, Conditional probability, Kelly-adjusted Martingale  
**Application:** Progressive betting with mathematical safeguards  
**Date:** October 15, 2025

---

## What is Decision Tree Risk Management?

**Decision Tree Risk Management** uses finite mathematics probability theory to calculate optimal progression betting after losses, ensuring each level can recover previous losses while maintaining Kelly Criterion safety limits.

**Core Philosophy:** If you lose once, the probability of losing consecutively decreases geometrically - use this to recover losses with controlled risk.

---

## The Core Problem

**Scenario:** You bet $272.50 and lose

**Questions:**
1. Should you increase your next bet to recover?
2. By how much?
3. What's the probability of losing again?
4. How many times can you safely progress before hitting ruin?

**Traditional approaches:**
- **Fixed betting:** Bet same amount (ignores recovery opportunity)
- **Martingale:** Double after loss (leads to ruin)
- **Kelly only:** Bet same Kelly fraction (slow recovery)

**Decision Tree approach:**
- Calculate conditional probability of consecutive losses
- Use Kelly Criterion to limit progression
- Set strict depth limits (max 3-5 levels)
- Each level covers previous + aims for original target

---

## Finite Mathematics Foundation

### Decision Tree Probability

**Concept from finite mathematics:**

```
Game 1: P(Loss) = 0.40 (40% chance to lose)
│
├─ WIN (60%) → Return to base betting
│
└─ LOSE (40%) → Progress to Level 2
    │
    Game 2: P(Loss | Lost 1) = 0.40
    │       P(Lose both) = 0.40 × 0.40 = 0.16 (16%)
    │
    ├─ WIN (60%) → Recover + profit, return to base
    │
    └─ LOSE (40%) → Progress to Level 3
        │
        Game 3: P(Loss | Lost 2) = 0.40
        │       P(Lose 3 in row) = 0.40³ = 0.064 (6.4%)
        │
        ├─ WIN (60%) → Recover all + profit
        │
        └─ LOSE (40%) → STOP (Max depth reached)
```

**Key insight:** P(Lose N consecutive) = P(Loss)^N

**Example:**
- P(Lose 1) = 0.40 (40%)
- P(Lose 2 consecutive) = 0.16 (16%)
- P(Lose 3 consecutive) = 0.064 (6.4%)
- P(Lose 4 consecutive) = 0.026 (2.6%)
- P(Lose 5 consecutive) = 0.010 (1.0%)

**At each level, probability of continued losses decreases geometrically.**

---

## Progressive Betting Formula

### Level 1 (Base Betting)

**From RISK_OPTIMIZATION:**
```python
bet_1 = kelly_optimal = $272.50
target_profit = $247 (win amount at -110)
```

**If lose:** Loss = -$272.50

---

### Level 2 (First Recovery)

**Goal:** Recover Level 1 loss + achieve original target profit

**Required win amount:**
```
total_needed = previous_loss + target_profit
             = $272.50 + $247
             = $519.50

At -110 odds (win 0.909 × bet):
bet_2 = $519.50 / 0.909 = $571

But apply Kelly limit:
bet_2_kelly_adjusted = min($571, kelly_max)
```

**Probability of losing both:**
- P(Lose Level 1) = 0.40
- P(Lose Level 2) = 0.40
- P(Lose both) = 0.16 (16%)

**If lose:** Cumulative loss = -$272.50 - $571 = -$843.50

---

### Level 3 (Second Recovery)

**Goal:** Recover all previous losses + achieve original target

**Required win amount:**
```
total_needed = cumulative_loss + target_profit
             = $843.50 + $247
             = $1,090.50

bet_3 = $1,090.50 / 0.909 = $1,200

Kelly-adjusted:
bet_3_kelly_adjusted = min($1,200, kelly_max, 0.20 × bankroll)
```

**Probability of losing all three:**
- P(Lose 3 consecutive) = 0.40³ = 0.064 (6.4%)

**Risk:** Cumulative loss would be -$2,043.50

---

### Maximum Depth Limit

**Stop at Level 3** (or when hitting limits)

**Reasons:**
1. P(Lose 4 consecutive) = 2.6% (rare but possible)
2. Cumulative bet sizes become large relative to bankroll
3. Kelly limits prevent over-betting
4. Risk of ruin increases exponentially

**Safety rule:** Never progress beyond Level 3

---

## Kelly-Adjusted Progression

**Pure Martingale (DANGEROUS):**
```
Level 1: $272
Level 2: $544 (double)
Level 3: $1,088 (double)
Level 4: $2,176 (double)
Problem: Ignores edge, can lead to ruin
```

**Kelly-Adjusted Progression (SAFE):**
```python
# Each level respects Kelly limits
def progressive_bet(level, cumulative_loss, target_profit, bankroll):
    # Calculate required bet to recover
    required_win = cumulative_loss + target_profit
    bet_needed = required_win / 0.909  # At -110 odds
    
    # Apply Kelly limits
    kelly_fraction = calculate_kelly(win_prob, odds)
    kelly_max = bankroll × kelly_fraction
    
    # Hard limits
    max_single_bet = bankroll × 0.20  # Never more than 20%
    max_progression = bankroll × 0.30  # Never more than 30% in progression
    
    # Final bet (smallest of all limits)
    bet_final = min(
        bet_needed,           # What's needed to recover
        kelly_max,            # Kelly optimal
        max_single_bet,       # 20% hard limit
        max_progression       # 30% progression limit
    )
    
    return bet_final
```

**Result:** Progression is capped by Kelly + hard limits

---

## State Management

### Betting States

```python
class BettingState:
    """
    Track current progression state
    """
    
    def __init__(self):
        self.level = 1              # Current progression level (1-3)
        self.cumulative_loss = 0.0  # Total losses in current sequence
        self.target_profit = 0.0    # Original target profit
        self.games_in_sequence = 0  # Number of games in current sequence
        self.max_level = 3          # Maximum progression depth
    
    def record_loss(self, bet_size):
        """Record a loss and progress to next level"""
        self.cumulative_loss += bet_size
        self.level += 1
        self.games_in_sequence += 1
        
        if self.level > self.max_level:
            self.reset()  # Hit max depth, reset to base
    
    def record_win(self, profit):
        """Record a win and reset to base level"""
        self.reset()
    
    def reset(self):
        """Reset to base level (Level 1)"""
        self.level = 1
        self.cumulative_loss = 0.0
        self.target_profit = 0.0
        self.games_in_sequence = 0
```

---

## Finite Mathematics Decision Tree

### Complete 3-Level Tree

```
                    START (Level 1)
                    Bet: $272.50
                    P(Win) = 0.60
                         |
        ┌────────────────┴────────────────┐
        |                                  |
    WIN (60%)                          LOSE (40%)
    +$247                              -$272.50
    Return to Level 1              Progress to Level 2
                                         |
                                   Bet: $571
                                   P(Win) = 0.60
                                   P(Here) = 0.40
                                         |
                        ┌────────────────┴────────────────┐
                        |                                  |
                    WIN (60%)                          LOSE (40%)
                    +$519 (recover all)                -$571
                    Return to Level 1              Progress to Level 3
                                                         |
                                                   Bet: $1,200
                                                   P(Win) = 0.60
                                                   P(Here) = 0.16
                                                         |
                                        ┌────────────────┴────────────────┐
                                        |                                  |
                                    WIN (60%)                          LOSE (40%)
                                    +$1,091 (recover all)              -$1,200
                                    Return to Level 1                  STOP
                                                                    Reset to Level 1
```

**Expected values at each node:**

**Level 1:**
- EV = 0.60 × (+$247) - 0.40 × ($272.50) = +$39

**Level 2 (given reached):**
- EV = 0.60 × (+$519) - 0.40 × ($571) = +$83

**Level 3 (given reached):**
- EV = 0.60 × (+$1,091) - 0.40 × ($1,200) = +$175

**Overall EV of system:**
- Level 1: 60% win = +$247
- Level 2: 24% reach = 60% of 40% win here = +$125 (amortized)
- Level 3: 9.6% reach = 60% of 16% win here = +$63 (amortized)
- Total EV: +$435 over 3-level sequence

---

## Risk of Ruin Analysis

### Probability of Maximum Loss

**Worst case:** Lose all 3 levels

```python
# Probability of losing all 3
P_ruin_3_level = P(Loss)³ = 0.40³ = 0.064 (6.4%)

# Maximum loss
max_loss = $272.50 + $571 + $1,200 = $2,043.50

# As percentage of $5,000 bankroll
max_loss_pct = $2,043.50 / $5,000 = 40.9%
```

**Risk assessment:**
- 6.4% chance of losing 41% of bankroll in single sequence
- This is HIGH RISK but mathematically constrained
- Only acceptable if edge is real and large

---

### Comparison to Fixed Betting

**Fixed betting (no progression):**
- Bet $272.50 per game
- 3 consecutive losses: -$817.50 (16.4% of bankroll)
- Risk: Slower recovery but lower peak loss

**Progressive betting (3 levels):**
- Bet $272.50, $571, $1,200
- 3 consecutive losses: -$2,043.50 (40.9% of bankroll)
- Risk: Higher peak loss but faster recovery when win

**Trade-off:** Higher variance for faster recovery

---

## Integration with Other Risk Models

### With RISK_OPTIMIZATION (Kelly)

**Role:** Kelly calculates base bet size

```python
# Level 1 uses Kelly
base_bet = kelly_optimizer.calculate_optimal_bet(
    bankroll=5000,
    ml_prediction=ml_pred,
    market_odds=market
)  # Returns $272.50

# Level 2-3 use Kelly as LIMIT
progressive_bet_2 = min(
    required_bet,
    kelly_max_for_current_bankroll
)
```

**Integration:** Kelly is the foundation, progression is overlay

---

### With DELTA_OPTIMIZATION (Correlation)

**Role:** Delta adjusts for correlation during progression

```python
# In progression, still use correlation hedge
if level == 2:
    correlation = delta_tracker.get_correlation()
    
    if correlation > 0.8 and z_score > 3.0:
        # High correlation, large gap
        # Apply hedge even in progression
        hedge_bet = progressive_bet_2 × 0.30
```

**Integration:** Hedging strategies apply at all levels

---

### With PORTFOLIO_MANAGEMENT (Multi-Game)

**Role:** Portfolio limits across all progressions

```python
# Check total progression exposure
total_progression_exposure = sum(all_active_progressions)

if total_progression_exposure > 0.50 × bankroll:
    # Too much in progression across portfolio
    # Reduce or skip new progressions
    skip_progression = True
```

**Integration:** Portfolio caps total progression risk

---

## Safety Mechanisms

### 1. Maximum Depth Limit

```python
MAX_PROGRESSION_LEVEL = 3

if current_level > MAX_PROGRESSION_LEVEL:
    # Stop progression
    reset_to_base()
```

**Reason:** P(Lose 4+) is low but non-zero, cap exposure

---

### 2. Kelly Limits at Each Level

```python
# Never bet more than Kelly allows
bet = min(calculated_bet, kelly_max)
```

**Reason:** Respect optimal growth criteria

---

### 3. Bankroll Percentage Caps

```python
# Never risk more than 20% on single bet
bet = min(bet, 0.20 × bankroll)

# Never have more than 50% in active progressions
total_progression = sum(active_progressions)
if total_progression > 0.50 × bankroll:
    skip_new_progression()
```

**Reason:** Prevent overexposure

---

### 4. Cooldown After Max Depth

```python
if hit_max_depth:
    # Reset to base
    # Wait for 2-3 wins before allowing progression again
    cooldown_counter = 3
```

**Reason:** Avoid chasing losses indefinitely

---

### 5. Session Loss Limits

```python
if session_loss > 0.30 × starting_bankroll:
    # Hit 30% session loss
    # Stop all progression betting
    # Return to base Kelly only
    disable_progression()
```

**Reason:** Circuit breaker for bad sessions

---

## Performance Requirements

### Real-Time Calculations

| Operation | Target | Complexity |
|-----------|--------|------------|
| State check | <1ms | O(1) |
| Progression calculation | <5ms | O(1) |
| Kelly limit check | <2ms | O(1) |
| Risk of ruin calc | <3ms | O(N) levels |
| Decision tree traversal | <5ms | O(depth) |
| **Total** | **<20ms** | **Real-time** |

**Critical:** Must not slow down system (adds to <100ms total)

---

## Expected Performance Impact

### With $5,000 Bankroll

**Base Kelly (no progression):**
- Average bet: $250
- Recovery time after loss: 1-2 wins
- Variance: Moderate

**Kelly + Decision Tree (with progression):**
- Level 1 bet: $250
- Level 2-3: Progressively larger
- Recovery time after loss: Immediate on any win in sequence
- Variance: Higher (but controlled)

**Expected improvement:**
- Faster recovery: 40% faster return to peak bankroll
- Higher variance: 30% increase in volatility
- Better bankroll utilization: Use geometric probability to our advantage

---

## File Structure

```
DECISION_TREE/
├── DEFINITION.md                    ← This file
├── MATH_BREAKDOWN.txt               ← Decision tree formulas
├── RESEARCH_BREAKDOWN.txt           ← Academic foundations
├── DECISION_TREE_IMPLEMENTATION_SPEC.md ← Code specifications
├── README.md                        ← Navigation
└── Applied Model/
    ├── decision_tree.py
    ├── state_manager.py
    ├── progression_calculator.py
    └── risk_analyzer.py
```

---

## When to Use Decision Tree Progression

### ✅ Use When:
- Edge is real and significant (>10%)
- Bankroll is healthy (>80% of starting)
- Win probability is high (>55%)
- Not already in multiple progressions
- Session loss <20%

### ❌ Don't Use When:
- Edge is small (<5%)
- Bankroll is depleted (<60% of starting)
- Win probability is marginal (<52%)
- Already in 2+ active progressions
- Hit session loss limit (>30%)
- Recently hit max depth (cooldown active)

---

## Example Scenario

**Game 1 (Level 1):**
- ML: +15.1 [+11.3, +18.9]
- Market: LAL -7.5 @ -110
- Kelly bet: $272.50
- **LOSE** (-$272.50)

**Game 2 (Level 2):**
- ML: +12.8 [+10.0, +15.6]
- Market: GSW -8.0 @ -110
- Cumulative loss: $272.50
- Target recovery: $272.50 + $247 = $519.50
- Progressive bet: $571 (Kelly-adjusted)
- P(Lose both) = 16%
- **WIN** (+$519, net: +$247 after recovering)

**Result:** Recovered loss + achieved original target in 2 games instead of 3-4

---

**Decision Tree Risk Management ensures:** Fast recovery from losses using finite mathematics probability theory, with strict safeguards to prevent ruin.

---

*Last Updated: October 15, 2025*  
*Part of ML Research - Risk Management Layer 4 of 4*  
*Sequential: After Risk, Delta, Portfolio - Before Trade Execution*

