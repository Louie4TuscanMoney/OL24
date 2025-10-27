# 🧭 WEEK 2: OVERFITTING REDUCTION & INTEGRITY HARDENING

**Date:** Sunday October 20, 2025  
**Phase:** Post-Launch Improvement  
**Focus:** Reduce overfitting from 2.9-7.3% to <2%, improve final score branch  
**Philosophy:** Integrity over speed, one change per iteration, measure everything

---

## 🎯 **CURRENT STATUS (End of Week 1)**

### Systems Built
```
MIT:      2.9% / 7.3% overfit   🏆 BEST
Stanford: 3.5% / 8.3% overfit   ⭐ Good
Mamba:    81% / 100% overfit    ❌ Memorization
Strive:   89% / 92% overfit     ❌ Memorization

Current Launch System:
  Ensemble: MIT 53%, Stanford 44%, Others 3%
  Expected: 5.5-6.5 / 10.5-11.5 MAE Monday
  Edge: 28-39% halftime, 0-9% final
```

### Problems to Solve
1. ✅ Temporal leakage: FIXED
2. ⚠️  Overfitting: MIT at 2.9-7.3% (good but can improve)
3. ❌ Final score: 0-9% edge (marginal, needs work)
4. ⚠️  Feature redundancy: Using 40-73 features (can prune)
5. ⚠️  Model stability: Need to test on more time windows

---

## 📋 **WEEK 2 ACTION PLAN (8 PRIORITIES)**

### 🕒 **Priority 1: Reinforce Temporal Validation Protocol**

**Goal:** Ensure no future data ever influences training (ironclad protection)

**Current State:**
- ✅ Chronological split implemented
- ✅ Train: 2021-2024, Test: Dec 2024-Apr 2025
- ⚠️  No automated leakage checks
- ⚠️  No locked golden validation set

**Week 2 Tasks:**
```python
# 1.1 Automated Leakage Check
def check_temporal_leakage(train_data, test_data):
    """
    Verify no temporal leakage in split
    - Check no game ID overlaps
    - Verify test dates are ALL after train dates
    - Flag any suspicious patterns
    """
    latest_train = max(g['date'] for g in train_data)
    earliest_test = min(g['date'] for g in test_data)
    assert earliest_test > latest_train, "TEMPORAL LEAKAGE DETECTED!"
    
    # Check for duplicate IDs
    train_ids = set(g['game_id'] for g in train_data)
    test_ids = set(g['game_id'] for g in test_data)
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"DUPLICATE GAME IDS: {overlap}"
    
    return True

# 1.2 Golden Validation Set
# Lock 2025 preseason (75 games) as permanent benchmark
# Never use for training, only for final validation
```

**Success Metrics:**
- [ ] Automated check runs on every split
- [ ] Zero overlap detected
- [ ] Golden set locked and documented
- [ ] Temporal gap > 0 days (currently 1 day)

**Timeline:** Monday-Tuesday (2 hours)

---

### ✂️ **Priority 2: Prune & Prioritize Features**

**Goal:** Reduce from 73 → 40-50 high-signal features

**Current State:**
- MIT uses 40/73 (Lasso selected)
- Stanford/Strive use all 73
- Likely redundancy/collinearity
- 6 features have zero variance

**Week 2 Tasks:**
```python
# 2.1 Feature Importance Analysis
# - Run permutation importance on MIT/Stanford
# - Identify features with low/negative contribution
# - Check correlation matrix for redundancy

# 2.2 Feature Stability Check
# - Test feature distributions across time windows
# - Remove features with high drift
# - Keep only features stable over time

# 2.3 Feature Pruning
# - Target: 40-50 core features
# - Remove: zero variance, high correlation, low importance
# - Retrain all systems on pruned set
```

**Success Metrics:**
- [ ] Feature count: 73 → 40-50
- [ ] Feature importance documented
- [ ] Correlation matrix < 0.9 for all pairs
- [ ] Stability score > 0.8 for all features
- [ ] Overfitting gap: Expect 2.9% → <2%

**Timeline:** Wednesday-Thursday (4 hours)

---

### 📊 **Priority 3: Regularization & Model Stability**

**Goal:** Make models more robust to unseen data

**Current State:**
- MIT: Already heavily regularized (LASSO, ElasticNet)
- Stanford: Moderate regularization
- Mamba/Strive: Minimal regularization (explains 81-100% overfit)

**Week 2 Tasks:**
```python
# 3.1 Increase Regularization Strength
# MIT:
#   - alpha: 1.0 → 2.0 (stronger L1/L2 penalty)
#   - SVR C: 0.1 → 0.05 (stricter)
#   - Trees: max_depth 5 → 3 (shallower)

# 3.2 Early Stopping
# - Add validation-based early stopping to all models
# - Stop when validation MAE stops improving
# - Prevent training too long (memorization)

# 3.3 Ensemble Diversity
# - Add more diverse models (different architectures)
# - Weight by validation performance, not train
```

**Success Metrics:**
- [ ] Overfitting gap: 2.9-7.3% → <2% / <5%
- [ ] Test MAE: Accept slight increase if gap reduces
- [ ] Training stops earlier (fewer epochs/iterations)
- [ ] Model coefficients smaller (stronger regularization)

**Timeline:** Friday-Saturday (4 hours)

---

### 🧪 **Priority 4: Cross-Validation Aligned with Deployment**

**Goal:** CV mirrors production reality (time-based, not random)

**Current State:**
- ✅ Used 5-fold time series CV for Stanford (validated!)
- ⚠️  Not systematic across all systems
- ⚠️  No documentation of fold variance

**Week 2 Tasks:**
```python
# 4.1 Rolling Window CV
# - 6 monthly windows (Oct 2023-Apr 2024, Nov 2023-May 2024, etc.)
# - Train on past N months, test on next month
# - Track MAE variance across folds

# 4.2 Expanding Window CV
# - Start with 1 year, expand by 1 month each fold
# - Tests how model scales with more data
# - Detects if recent data is different

# 4.3 Document Fold Variance
# - If CV MAE varies a lot → model is unstable
# - If CV MAE is consistent → model is robust
# - Target: Std dev < 10% of mean MAE
```

**Success Metrics:**
- [ ] 6 time windows tested per model
- [ ] CV MAE variance documented
- [ ] Std dev < 10% of mean (stability threshold)
- [ ] CV MAE ≈ Test MAE (confirms no overfitting)

**Timeline:** Saturday-Sunday Week 2 (4 hours)

---

### 🧠 **Priority 5: Stress Test Against Live-Like Conditions**

**Goal:** Prove model works on truly unseen data

**Current State:**
- ✅ Test set: Dec 2024-Apr 2025 (4 months)
- ⚠️  No testing on late 2025 data
- ⚠️  No scenario analysis

**Week 2 Tasks:**
```python
# 5.1 Simulate Live Prediction
# - Collect May-Oct 2025 games (not in training)
# - Run predictions as if live
# - Compare to test set MAE

# 5.2 Scenario Analysis
# - High-scoring games vs low-scoring
# - Top teams vs bottom teams
# - Close games vs blowouts
# - Identify where model fails

# 5.3 Edge Case Testing
# - Back-to-back games
# - Playoff games
# - Games with key injuries
# - See where model is unreliable
```

**Success Metrics:**
- [ ] Live-like MAE within 10% of test MAE
- [ ] Scenario MAE documented per category
- [ ] Edge cases identified and flagged
- [ ] Prediction confidence scores calibrated

**Timeline:** Week 2 Monday-Tuesday (3 hours)

---

### 🪜 **Priority 6: Rebuild Final Score Branch**

**Goal:** Improve 0-9% edge to 15-20%+ edge

**Current State:**
- Final score: 10.5-11.5 expected vs 11.5 baseline (marginal)
- Uses same features as halftime (may not be optimal)
- 7.3% overfitting (better than 92%, but still high)

**Week 2 Tasks:**
```python
# 6.1 Separate Feature Engineering
# - Final score may need different features than halftime
# - Focus on: momentum, late-game patterns, fatigue
# - Don't just copy halftime features

# 6.2 Different Model Architecture
# - Try sequential model (predict halftime → use for final)
# - Cascade: Halftime prediction as feature for final
# - May capture progression better

# 6.3 More Aggressive Pruning
# - Final score might need even sparser model
# - Try 20-30 features instead of 40-50
# - Extreme regularization (alpha=5.0)

# 6.4 Longer Horizon Prediction
# - Current: Q2 6:00 → Final (30 min ahead)
# - Try: Q3 end → Final (15 min ahead)
# - Shorter horizon = easier to predict = better edge
```

**Success Metrics:**
- [ ] Final MAE: 10.5 → 9-10 (improvement)
- [ ] Edge: 0-9% → 15-20%+ (viable)
- [ ] Overfitting: 7.3% → <5%
- [ ] Independent validation on live games

**Timeline:** Week 2 Wednesday-Friday (6 hours)

---

### 🧭 **Priority 7: Establish Performance Gates**

**Goal:** Prevent weak models from reaching production

**Current State:**
- ⚠️  No formal gates (manual judgment)
- ⚠️  No automated rejection
- ⚠️  No performance logging

**Week 2 Tasks:**
```python
# 7.1 Define Hard Thresholds
PERFORMANCE_GATES = {
    'max_test_mae_halftime': 6.5,
    'max_test_mae_final': 11.0,
    'min_edge_halftime': 0.20,  # 20% better than baseline
    'min_edge_final': 0.15,     # 15% better than baseline
    'max_overfitting_gap_halftime': 0.05,  # 5%
    'max_overfitting_gap_final': 0.08,     # 8%
    'min_cv_stability': 0.90,  # CV std < 10% of mean
    'max_temporal_leakage': 0,  # Zero tolerance
}

# 7.2 Automated Gate Checking
def check_performance_gates(model, validation_results):
    """
    Return True if model passes all gates, False otherwise
    Log which gates failed for debugging
    """
    pass

# 7.3 Performance Logging
# - Log every model's metrics to CSV/DB
# - Track over time
# - Detect degradation trends
```

**Success Metrics:**
- [ ] Performance gates documented
- [ ] Automated checking implemented
- [ ] Rejection logs saved
- [ ] Historical performance tracked

**Timeline:** Week 2 Monday (2 hours)

---

### 🪙 **Priority 8: Iterate Conservatively**

**Goal:** One major change per iteration, measure everything

**Current State:**
- ✅ Built 4 systems in one session (necessary for initial build)
- ⚠️  Now need to slow down for refinement

**Week 2 Process:**
```
MONDAY:
  • Implement automated leakage check
  • Run on all systems
  • Document results
  • STOP (don't change anything else)

TUESDAY:
  • Verify leakage check working
  • IF clean → proceed
  • IF issues → fix first

WEDNESDAY:
  • Feature importance analysis
  • Document findings
  • STOP (don't prune yet)

THURSDAY:
  • Prune features based on Wednesday analysis
  • Retrain MIT only
  • Compare new vs old
  • STOP

FRIDAY:
  • IF Thursday improved → apply to Stanford
  • IF Thursday degraded → revert and investigate
  • STOP

SATURDAY:
  • Final score branch rebuild (separate effort)
  • Independent of other changes
  • STOP

SUNDAY:
  • Week 2 review
  • Document all changes
  • Measure cumulative impact
  • Plan Week 3
```

**Success Metrics:**
- [ ] One major change per day
- [ ] Full validation after each change
- [ ] Impact documented (before/after)
- [ ] Can revert any change independently

**Timeline:** All week (structured process)

---

## 📊 **EXPECTED OUTCOMES (End of Week 2)**

### Overfitting Reduction
```
CURRENT:
  MIT:      2.9% / 7.3%
  Stanford: 3.5% / 8.3%

TARGET (Week 2):
  MIT:      <2% / <5%
  Stanford: <3% / <6%

Method:
  • Feature pruning (73 → 40-50)
  • Stronger regularization
  • Early stopping
```

### Final Score Improvement
```
CURRENT:
  MAE: 10.5-11.5 vs 11.5 baseline (0-9% edge)
  Overfitting: 7.3%

TARGET (Week 2):
  MAE: 9-10 vs 11.5 baseline (13-22% edge)
  Overfitting: <5%

Method:
  • Separate feature engineering
  • Cascade architecture
  • Aggressive pruning
```

### System Integrity
```
CURRENT:
  ✅ Temporal leakage fixed
  ✅ Chronological split
  ⚠️  Manual validation
  ⚠️  No automated gates

TARGET (Week 2):
  ✅ Automated leakage check
  ✅ Performance gates
  ✅ Golden validation set
  ✅ Performance logging
```

---

## 🎯 **SUCCESS CRITERIA (Week 2)**

### Must Have
- [ ] Automated temporal leakage check (Priority 1)
- [ ] Performance gates defined and implemented (Priority 7)
- [ ] Feature count reduced to 40-50 (Priority 2)
- [ ] Overfitting gap: MIT <2%, Stanford <3% (Priority 3)

### Should Have
- [ ] Rolling window CV completed (Priority 4)
- [ ] Final score MAE < 10 (Priority 6)
- [ ] Stress tests on May-Oct 2025 data (Priority 5)

### Nice to Have
- [ ] Final score edge > 15% (Priority 6)
- [ ] Golden validation set locked (Priority 1)
- [ ] All models logged to DB (Priority 7)

---

## 🔄 **ITERATION FRAMEWORK**

### Before Each Change
```
1. Document current baseline metrics
2. State hypothesis (what will improve)
3. Define success criteria
4. Commit code (can revert)
```

### After Each Change
```
1. Run full validation suite
2. Compare to baseline
3. Document impact (better/worse/same)
4. Decide: keep, revert, or iterate
```

### Weekly Review
```
1. List all changes made
2. Measure cumulative impact
3. Identify what worked / didn't work
4. Adjust priorities for next week
```

---

## 📈 **PERFORMANCE TRACKING**

### Metrics to Log (Every Change)
```
Model Performance:
  - Train MAE (halftime / final)
  - Test MAE (halftime / final)
  - Overfitting gap (%)
  - CV MAE (mean ± std)

System Integrity:
  - Temporal leakage check (pass/fail)
  - Performance gates (passed/failed which ones)
  - Feature count
  - Model complexity (depth, parameters)

Business Metrics:
  - Expected edge vs baseline
  - Confidence level
  - Recommended bet sizing
```

### Dashboard (Create Simple)
```python
# Week 2 Monday: Create simple performance dashboard
# - CSV log of all metrics
# - Plot overfitting gap over time
# - Plot MAE over time
# - Flag when gates fail
```

---

## 🧭 **GUIDING PRINCIPLES**

### Integrity Over Speed
```
Don't rush improvements
Better to launch conservative Week 1
Then improve steadily Week 2+
Than launch broken system Week 1
```

### One Change Per Iteration
```
Changing multiple things = can't tell what worked
Change one thing, measure, document
This is slower but more reliable
Science, not guessing
```

### Measure Everything
```
If you can't measure it, you can't improve it
Log all metrics before/after every change
Build intuition about what works
Data-driven iteration
```

### Trust the Process
```
Week 1: Build fast, test hard, launch conservative
Week 2: Improve systematically, measure rigorously
Week 3+: Scale if validated, iterate if not
Long-term thinking
```

---

## 🎓 **LESSONS APPLIED**

### From Saturday Night Audit
```
✅ Temporal leakage breaks integrity → Automated checks
✅ Overfitting is deceptive → Reduce to <2%
✅ Test performance ≠ real performance → Stress tests
✅ Must validate honestly → Performance gates
```

### From MIT Principles
```
✅ Sparsity generalizes better → Feature pruning
✅ Extreme regularization works → Strengthen it
✅ Robust methods handle drift → Keep them
✅ Uncertainty quantification → Track confidence
```

### From Your Teaching
```
✅ "Temporal leakage & overfitting break integrity"
    → Automated checks + aggressive reduction

✅ "Model that performs slightly worse but is honestly validated
    is far more valuable"
    → Accept small MAE increase if overfitting drops

✅ "Can't bet on overfitted models"
    → Performance gates prevent weak models
```

---

## 📋 **WEEK 2 CHECKLIST**

### Monday
- [ ] Implement automated temporal leakage check
- [ ] Define performance gates
- [ ] Create performance logging system

### Tuesday
- [ ] Verify leakage check on all systems
- [ ] Document baseline metrics
- [ ] Start live-like stress tests

### Wednesday
- [ ] Feature importance analysis
- [ ] Identify redundant/low-signal features
- [ ] Document pruning candidates

### Thursday
- [ ] Prune features (73 → 40-50)
- [ ] Retrain MIT
- [ ] Validate improvement

### Friday
- [ ] Apply pruning to Stanford if MIT improved
- [ ] Increase regularization strength
- [ ] Validate overfitting reduction

### Saturday
- [ ] Rebuild final score branch
- [ ] Separate feature engineering
- [ ] Test cascade architecture

### Sunday
- [ ] Week 2 review
- [ ] Document all changes and impacts
- [ ] Plan Week 3 priorities

---

## ✅ **READY FOR WEEK 2**

**Philosophy:** Integrity over speed, one change per iteration, measure everything

**Goal:** MIT overfitting 2.9% → <2%, Final edge 0-9% → 15-20%+

**Timeline:** 7 days, systematic improvement

**Confidence:** HIGH (methodical approach, learned from Week 1)

**Status:** PLANNED ✅

---

**"Week 1: Build fast, launch conservative. Week 2: Improve systematically, validate rigorously. Week 3+: Scale if profitable, iterate if not."**

**ONTOLOGIC XYZ - FAIL FORWARD - CONTINUOUS IMPROVEMENT** 🧭

