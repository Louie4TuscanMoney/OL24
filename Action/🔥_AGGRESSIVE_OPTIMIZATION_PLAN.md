# 🔥 AGGRESSIVE OPTIMIZATION PLAN
## Target: 4-5 MAE (Not 8.22 - That's Amateur Hour)

**Current:** 8.22 MAE (XGBoost basic)  
**Target:** 4-5 MAE (Deep Ensemble)  
**Benchmark:** Research shows 5.37 MAE is achievable  
**Timeline:** Tonight + Tomorrow (don't launch until it's RIGHT)  

---

## 🎯 RESEARCH BENCHMARKS

### **What Top Models Achieve:**

From your ML_ENHANCEMENTS research:

1. **XGBoost (optimized):** R² = 0.999, RMSE = 0.198 (~2-3 MAE equivalent)
2. **Extra Trees ensemble:** 34.14% WAPE (~5-6 MAE)
3. **Random Forest + Gradient Boosting:** 29.81% MAPE (~4-5 MAE)
4. **Stacked ensemble (RF + XGB + LGBM):** Best performance

**YOUR system has 5.37 MAE somewhere - we need to find it or beat it.**

---

## 💀 WHY 8.22 IS UNACCEPTABLE

**8.22 MAE means:**
- Predicting Lakers +5 could be Lakers +13 or -3
- That's ±8 points of error
- **Betting edge is basically zero**
- You'd need HUGE mispricing in odds to be profitable
- Week 1 would likely be -EV (negative expected value)

**4-5 MAE means:**
- Predicting Lakers +5 → actual is Lakers +9 or +1
- That's ±4-5 points of error
- **You can find +EV bets easily**
- If odds are Lakers +8 and you predict +5, that's 3-point edge
- Week 1 would be +EV with proper sizing

**We need 4-5 MAE to actually make money.** Not settle for 8.22.

---

## 🔥 OPTIMIZATION ROADMAP (24-48 HOURS)

### **PHASE 1: Advanced Feature Engineering (4-6 hours)**

**What we're missing:**

1. **Lag features** (critical in research)
   - Last 1, 2, 3 games performance
   - Last 5, 10 games rolling averages
   - Season trends (improving vs declining)

2. **Temporal derivatives**
   - Velocity (rate of change in score differential)
   - Acceleration (change in momentum)
   - Moving averages with decay

3. **Spectral features** (we skipped these!)
   - FFT (Fast Fourier Transform) for pattern cycles
   - Dominant frequencies
   - Spectral entropy

4. **Player interaction features**
   - Star player on/off court impact
   - Matchup-specific performance (LeBron vs Warriors historically)
   - Lineup combinations

5. **Context features**
   - Rest days (back-to-back kills performance)
   - Home/away streaks
   - Recent form (last 10 games)
   - Playoff implications

**Implementation:**
```python
# Extract ALL features from research
from tsfresh import extract_features
from scipy.fft import fft
from scipy.stats import entropy

# 100+ features total (not 35)
```

**Expected gain:** 8.22 → 6.5 MAE (20% improvement)

---

### **PHASE 2: Hyperparameter Optimization (6-8 hours)**

**What we did WRONG:**
- Used default XGBoost params (lazy)
- No cross-validation (overfitting risk)
- No systematic search

**What research recommends:**

1. **Bayesian Optimization** (HEBO or Optuna)
   ```python
   import optuna
   
   def objective(trial):
       params = {
           'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
           'max_depth': trial.suggest_int('max_depth', 4, 12),
           'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
           'subsample': trial.suggest_float('subsample', 0.6, 1.0),
           'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
           'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
           'gamma': trial.suggest_float('gamma', 0, 5),
           'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
           'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
       }
       model = xgb.XGBRegressor(**params)
       scores = cross_val_score(model, X_train, y_train, cv=TimeSeriesSplit(5), scoring='neg_mean_absolute_error')
       return -scores.mean()
   
   study = optuna.create_study(direction='minimize')
   study.optimize(objective, n_trials=100)  # 100 iterations
   ```

2. **GridSearch for ensemble weights**
3. **Cross-validation** (TimeSeriesSplit, not random)
4. **Early stopping** (prevent overfitting)

**Expected gain:** 6.5 → 5.5 MAE (15% improvement)

---

### **PHASE 3: Deep Ensemble (4-6 hours)**

**What we're NOT doing (but should):**

1. **Stack multiple models:**
   - Layer 1: XGBoost, LightGBM, CatBoost, ExtraTrees, Random Forest
   - Layer 2: Neural network meta-learner
   - Weighted average based on validation performance

2. **Diversity in models:**
   - Tree-based: XGBoost, RF, Extra Trees
   - KNN-based: Dejavu (already have)
   - Deep learning: LSTM or Transformer
   - Linear: Ridge regression with polynomial features

3. **Ensemble strategy:**
   ```python
   # Train 5 diverse models
   models = {
       'xgboost': XGBRegressor(**best_params),
       'lightgbm': LGBMRegressor(**best_params),
       'extra_trees': ExtraTreesRegressor(n_estimators=500),
       'random_forest': RandomForestRegressor(n_estimators=500),
       'dejavu': DejavuForecaster(k=500)
   }
   
   # Train all
   for name, model in models.items():
       model.fit(X_train, y_train)
   
   # Meta-learner (stacking)
   from sklearn.ensemble import StackingRegressor
   stacked = StackingRegressor(
       estimators=[(name, model) for name, model in models.items()],
       final_estimator=Ridge()
   )
   ```

4. **Weighted by inverse MAE:**
   ```python
   # Weight models by performance
   weights = {
       'xgboost': 1/6.5,  # If MAE is 6.5
       'lightgbm': 1/6.2,
       'extra_trees': 1/6.8,
       ...
   }
   
   # Weighted prediction
   pred = sum(weight * model.predict(X) for model, weight in zip(models, weights)) / sum(weights)
   ```

**Expected gain:** 5.5 → 4.5 MAE (18% improvement)

---

### **PHASE 4: Advanced DL Integration (8-12 hours)**

**What research recommends (Phase 8 from StartHere.md):**

1. **LSTM for temporal patterns:**
   ```python
   import torch
   import torch.nn as nn
   
   class NBAPredictor(nn.Module):
       def __init__(self):
           super().__init__()
           self.lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=3, dropout=0.2)
           self.fc = nn.Linear(128, 1)
       
       def forward(self, x):
           lstm_out, _ = self.lstm(x)
           return self.fc(lstm_out[:, -1, :])
   ```

2. **Transformer for attention:**
   - Model which parts of the 18-minute pattern matter most
   - Attention to critical moments (runs, momentum shifts)

3. **Bayesian Neural Network:**
   - Uncertainty quantification
   - Combine with Conformal prediction

**Expected gain:** 4.5 → 4.0 MAE (11% improvement)

---

### **PHASE 5: Vine Copulas for Dependencies (6-8 hours)**

**From VINECOPULA.md paper:**

Model joint distributions:
- Home score AND away score (not just differential)
- Rebounds AND turnovers (defensive performance)
- Shooting % AND pace (offensive efficiency)

**Implementation:**
```python
from pyvinecopulib import Vinecop

# Model multivariate dependencies
data = np.column_stack([
    home_scores,
    away_scores,
    rebounds,
    turnovers,
    pace
])

# Fit vine copula
cop = Vinecop(data)

# Use for probabilistic prediction
prob_distribution = cop.pdf(new_game_data)
```

**Expected gain:** 4.0 → 3.8 MAE (5% improvement from better uncertainty)

---

## 🎯 COMPLETE OPTIMIZATION PIPELINE

### **TONIGHT (4-8 hours):**

**Step 1: Feature Engineering (4 hours)**
```bash
python3 🔥_1_ADVANCED_FEATURES.py
# Extract 100+ features:
# - Lag features (1,2,3,5,10 games)
# - Rolling averages
# - Spectral (FFT, entropy)
# - Player interactions
# - Context (rest, streaks)
```

**Step 2: Hyperparameter Optimization (4 hours)**
```bash
python3 🔥_2_BAYESIAN_OPTIMIZATION.py
# Optuna with 200 trials
# TimeSeriesSplit CV
# Early stopping
# Find BEST params for XGBoost, LightGBM, ExtraTrees
```

**Step 3: Train Optimized Models (2 hours)**
```bash
python3 🔥_3_TRAIN_ENSEMBLE.py
# Train 5 models with best params:
# - XGBoost
# - LightGBM  
# - ExtraTrees
# - RandomForest
# - Dejavu (already have)
```

**Step 4: Stacked Ensemble (2 hours)**
```bash
python3 🔥_4_STACK_ENSEMBLE.py
# Meta-learner (Ridge or Neural Net)
# Weighted by inverse MAE
# Test on holdout
```

**Expected MAE after tonight: ~5-6 MAE**

---

### **TOMORROW (6-10 hours):**

**Step 5: LSTM Integration (4 hours)**
```bash
python3 🔥_5_LSTM_TEMPORAL.py
# PyTorch LSTM
# 3 layers, 128 hidden units
# Trained on sequential patterns
# Add to ensemble
```

**Step 6: Vine Copula (3 hours)**
```bash
python3 🔥_6_VINE_COPULA.py
# Model joint distributions
# Probabilistic predictions
# Uncertainty quantification
```

**Step 7: Final Ensemble (3 hours)**
```bash
python3 🔥_7_FINAL_ENSEMBLE.py
# Combine ALL models:
# - 5 tree-based
# - LSTM
# - Vine copula
# Meta-learner with Bayesian weights
```

**Expected MAE after tomorrow: ~4-4.5 MAE**

---

### **FINAL TEST (2 hours):**

```bash
python3 🔥_8_VALIDATE_ON_2025.py
# Test on 2025 preseason
# Calculate MAE
# Compare to 5.37 benchmark
# If <5 MAE: LAUNCH
# If 5-7 MAE: CAUTIOUS
# If >7 MAE: KEEP OPTIMIZING
```

---

## 🎯 IMPLEMENTATION STRATEGY

### **Feature Engineering - What We're Adding:**

**From XGBoost paper (R² = 0.999):**
- Lag features: PTS_lag_1, PTS_lag_2, PTS_lag_3
- Rolling averages: PTS_rolling_2, PTS_rolling_3
- Trend features: PTS_trend (current - previous)
- Points_per_minute
- Efficiency metrics

**From comparative study (34% WAPE):**
- AST/TO ratio
- EFG% (Effective Field Goal %)
- True Shooting %
- 4 Factors (shooting, turnovers, rebounds, free throws)
- NETRTG (Net Rating)
- PIE (Player Impact Estimate)
- Plus/Minus
- USG% (Usage Rate)

**From StartHere.md (Phase 0-8):**
- Vine copula features (joint distributions)
- Dynamic Bayesian Network features
- MCTS-optimized feature selection
- NBA-specific tracking data

**Total features: 100-150 (not 35)**

---

### **Model Ensemble - What We're Building:**

**Layer 1: Base Models (5-7 models)**
1. XGBoost (optimized with Bayesian)
2. LightGBM (faster, different regularization)
3. CatBoost (handles categoricals better)
4. ExtraTrees (best performer in research)
5. RandomForest (robust, low variance)
6. Dejavu (KNN-based, already have)
7. LSTM (temporal patterns)

**Layer 2: Meta-Learner**
- Ridge regression OR
- Neural network OR
- Another gradient booster

**Layer 3: Bayesian Weighting**
- Weight each model by inverse MAE
- Update weights dynamically
- Uncertainty quantification

---

### **Hyperparameter Optimization - What We're Doing:**

**NOT this (what we did):**
```python
# Amateur hour
model = XGBRegressor()  # Defaults
model.fit(X, y)  # No CV
# Result: 8.22 MAE (trash)
```

**THIS (what research does):**
```python
import optuna

def objective(trial):
    # Search space (from research)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }
    
    model = xgb.XGBRegressor(**params, random_state=42)
    
    # TimeSeriesSplit CV (not random - this is TIME SERIES!)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
    
    return -scores.mean()

# 200-500 trials (8-12 hours compute)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

# Result: Optimized params, expected 15-25% MAE reduction
```

---

## 🚀 TONIGHT'S EXECUTION PLAN

### **6:00 PM - 10:00 PM (4 hours)**

**Task 1: Extract REAL features (2 hours)**
```bash
python3 🔥_EXTRACT_ALL_FEATURES.py

# This will:
# - Reload all 6,912 games
# - Add lag features (last 1,2,3,5,10 games per team)
# - Add rolling averages
# - Add spectral features (FFT)
# - Add interaction features
# - Add context (rest, streaks, home/away)
# Output: 100-150 features per game
```

**Task 2: Start Bayesian Optimization (2 hours setup, runs overnight)**
```bash
python3 🔥_BAYESIAN_HYPEROPT.py

# This will:
# - Set up Optuna study
# - Define search space (9 hyperparameters)
# - Run 200 trials (can run overnight)
# - Save best params
```

### **TOMORROW (Sunday) 8:00 AM - 6:00 PM (10 hours)**

**Task 3: Train Optimized Ensemble (3 hours)**
```bash
python3 🔥_TRAIN_OPTIMIZED_ENSEMBLE.py

# Train with BEST params:
# - XGBoost (optimized)
# - LightGBM (optimized)
# - ExtraTrees (optimized)
# - RandomForest (optimized)
# - CatBoost (optimized)
```

**Task 4: Add LSTM (4 hours)**
```bash
python3 🔥_TRAIN_LSTM.py

# PyTorch LSTM:
# - 3 layers
# - 128 hidden units
# - Dropout 0.3
# - Trained on sequential patterns
```

**Task 5: Stack Everything (2 hours)**
```bash
python3 🔥_FINAL_STACKED_ENSEMBLE.py

# Meta-learner combines:
# - 5 tree-based models
# - LSTM
# - Dejavu
# Weighted by inverse MAE
```

**Task 6: Test on 2025 (1 hour)**
```bash
python3 🔥_FINAL_VALIDATION.py

# Test on 2025 preseason
# Calculate MAE
# Target: <5.0 MAE
```

---

## 📊 EXPECTED RESULTS

### **Progression:**

```
Current:  8.22 MAE (basic XGBoost, 35 features)
          ↓ Add 100+ features
Step 1:   6.50 MAE (20% gain)
          ↓ Bayesian hyperparameter optimization
Step 2:   5.50 MAE (15% gain)
          ↓ 5-model ensemble
Step 3:   5.00 MAE (10% gain)
          ↓ Add LSTM
Step 4:   4.50 MAE (10% gain)
          ↓ Stacked meta-learner
Target:   4.00-4.50 MAE ✅
```

**Total improvement: 45-50% (8.22 → 4.0-4.5)**

---

## 💪 WHY THIS IS DOABLE

**Evidence from research:**

1. **XGBoost paper:** R² = 0.999 (near-perfect) with:
   - Lag features
   - Rolling averages
   - Proper CV
   - GridSearch optimization

2. **Comparative study:** 34% WAPE (~5-6 MAE) with:
   - ExtraTrees ensemble
   - 18 advanced stats
   - 90 player case studies

3. **Your benchmark:** 5.37 MAE (you said we have this somewhere)

**We have:**
- ✅ 6,912 games (2020-2024) + 10,000 (2015-2021) = 17,000 games
- ✅ Real play-by-play data
- ✅ Team stats
- ✅ Player stats
- ✅ All the compute we need

**We DON'T have (yet):**
- ❌ 100+ features (we have 35)
- ❌ Optimized hyperparameters (we used defaults)
- ❌ Proper ensemble (we have 1 model)
- ❌ LSTM/DL (we skipped it)
- ❌ Proper CV (we did random split, not time series split)

**FIX THESE → GET TO 4-5 MAE** ✅

---

## 🔥 LET'S FUCKING BUILD THIS

**No more settling for 8.22.**

**No more "launch Monday and improve later."**

**We have 48 hours. Let's get to 4-5 MAE BEFORE we launch.**

**Research shows it's possible. Your benchmark shows 5.37. Let's beat it.**

---

## 📋 IMMEDIATE ACTION (RIGHT NOW):

I'll create:
1. `🔥_EXTRACT_ALL_FEATURES.py` - 100+ features
2. `🔥_BAYESIAN_HYPEROPT.py` - Optuna optimization
3. `🔥_TRAIN_OPTIMIZED_ENSEMBLE.py` - 5-model ensemble
4. `🔥_TRAIN_LSTM.py` - Deep learning
5. `🔥_FINAL_STACKED_ENSEMBLE.py` - Meta-learner
6. `🔥_FINAL_VALIDATION.py` - Test on 2025

**Run these sequentially. Target: 4-5 MAE by Sunday night.**

**THEN launch Monday with a REAL edge.** 🚀

---

**Ready to build this properly?** No more amateur hour.

