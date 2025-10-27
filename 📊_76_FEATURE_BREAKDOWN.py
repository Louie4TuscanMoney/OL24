#!/usr/bin/env python3
"""
COMPLETE MODEL BREAKDOWN - 76 FEATURES
Comprehensive analysis of model performance, distributions, and insights
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime

print("=" * 100)
print("📊 COMPLETE MODEL BREAKDOWN - 76 FEATURES")
print("=" * 100)
print()

# Load data
print("[1/10] Loading data...")
with open("Action/ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl", 'rb') as f:
    data = pickle.load(f)
    
df = pd.DataFrame(data) if isinstance(data, list) else data
print(f"✅ Loaded {len(df)} games with {df.shape[1]} columns")
print()

# Prepare features
print("[2/10] Preparing features...")
exclude_cols = ['game_id', 'date', 'season', 'home_team', 'away_team', 'pattern', 
                'diff_at_final', 'diff_at_halftime', 'diff_at_2q_6min']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Filter numeric only
numeric_features = []
for col in feature_cols:
    if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
        numeric_features.append(col)
feature_cols = numeric_features

print(f"✅ Selected {len(feature_cols)} numeric features")
print()

# Split chronologically
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

X_train = df_train[feature_cols].values
y_train = df_train['diff_at_final'].values
X_test = df_test[feature_cols].values
y_test = df_test['diff_at_final'].values

# Handle NaN values
X_train = np.nan_to_num(X_train, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

print(f"   Train: {len(X_train):,} games")
print(f"   Test:  {len(X_test):,} games")
print()

# Train model
print(f"[3/10] Training model on all {len(feature_cols)} features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Get predictions
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
overfit = (test_mae - train_mae) / train_mae * 100 if train_mae > 0 else 0

baseline_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
edge = (baseline_mae - test_mae) / baseline_mae * 100

print(f"✅ Model trained")
print(f"   Train MAE: {train_mae:.3f}")
print(f"   Test MAE:  {test_mae:.3f}")
print(f"   Overfit:   {overfit:.1f}%")
print(f"   Edge:      {edge:.1f}%")
print()

# Add predictions to test data
errors = np.abs(test_pred - y_test)
signed_errors = test_pred - y_test

df_test['prediction'] = test_pred
df_test['actual'] = y_test
df_test['error'] = errors
df_test['signed_error'] = signed_errors

# =============================
# SECTION 1: MODEL SPECIFICATIONS
# =============================
print("=" * 100)
print("📋 SECTION 1: MODEL SPECIFICATIONS")
print("=" * 100)
print()

print("MODEL ARCHITECTURE:")
print(f"  Type: Linear Regression")
print(f"  Features: {len(feature_cols)}")
print(f"  Training: {len(X_train):,} games (80%)")
print(f"  Testing: {len(X_test):,} games (20%)")
print(f"  Scaler: RobustScaler")
print()

print("PERFORMANCE:")
print(f"  Train MAE:    {train_mae:.3f}")
print(f"  Test MAE:     {test_mae:.3f}")
print(f"  Overfitting:  {overfit:.1f}%")
print(f"  Baseline MAE: {baseline_mae:.3f}")
print(f"  Edge:         {edge:.1f}%")
print(f"  Median Error: {np.median(errors):.3f}")
print(f"  Std Dev:      {np.std(errors):.3f}")
print()

# =============================
# SECTION 2: ERROR DISTRIBUTION
# =============================
print("=" * 100)
print("📊 SECTION 2: ERROR DISTRIBUTION")
print("=" * 100)
print()

bins = [0, 2, 5, 8, 12, 15, 20, 100]
bin_labels = ['0-2 pts', '2-5 pts', '5-8 pts', '8-12 pts', '12-15 pts', '15-20 pts', '20+ pts']

df_test['error_bin'] = pd.cut(df_test['error'], bins=bins, labels=bin_labels, include_lowest=True)

print("ERROR DISTRIBUTION:")
cumulative = 0
for label in bin_labels:
    count = (df_test['error_bin'] == label).sum()
    pct = count / len(df_test) * 100
    cumulative += pct
    bar = '█' * int(pct / 2)
    print(f"  {label:12s}: {count:4d} ({pct:5.1f}%) {bar} [Cumulative: {cumulative:5.1f}%]")
print()

# Detailed cumulative
print("CUMULATIVE ACCURACY:")
for threshold in [2, 5, 8, 10, 12, 15, 20]:
    within = (df_test['error'] <= threshold).sum()
    pct = within / len(df_test) * 100
    stars = '⭐' * int(pct / 20)
    print(f"  Within {threshold:2d} points: {within:4d} games ({pct:5.1f}%) {stars}")
print()

# =============================
# SECTION 3: GAME TYPE ANALYSIS
# =============================
print("=" * 100)
print("📊 SECTION 3: PERFORMANCE BY GAME TYPE")
print("=" * 100)
print()

# Use diff_at_2q_6min for categorization
df_test['game_state'] = pd.cut(
    df_test['diff_at_2q_6min'].abs(),
    bins=[0, 3, 7, 12, 20, 100],
    labels=['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)']
)

print("MAE BY GAME STATE (at Q2 6:00):")
print()
for state in ['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)']:
    mask = df_test['game_state'] == state
    if mask.sum() > 0:
        mae = df_test.loc[mask, 'error'].mean()
        median = df_test.loc[mask, 'error'].median()
        count = mask.sum()
        pct = count / len(df_test) * 100
        print(f"  {state:25s}: MAE {mae:6.3f}, Median {median:6.3f} ({count:4d} games, {pct:5.1f}%)")
print()

# Outcome categorization
def categorize_outcome(row):
    current = row['diff_at_2q_6min']
    final = row['actual']
    
    if abs(current) <= 3:
        return 'Was Close'
    elif (current > 3 and final > 0) or (current < -3 and final < 0):
        return 'Lead Held'
    elif (current > 3 and final <= 0) or (current < -3 and final >= 0):
        return 'Comeback/Flip'
    else:
        return 'Tightened'

df_test['outcome_type'] = df_test.apply(categorize_outcome, axis=1)

print("MAE BY OUTCOME TYPE:")
print()
for outcome in df_test['outcome_type'].value_counts().index:
    mask = df_test['outcome_type'] == outcome
    mae = df_test.loc[mask, 'error'].mean()
    median = df_test.loc[mask, 'error'].median()
    count = mask.sum()
    pct = count / len(df_test) * 100
    print(f"  {outcome:25s}: MAE {mae:6.3f}, Median {median:6.3f} ({count:4d} games, {pct:5.1f}%)")
print()

# =============================
# SECTION 4: DIRECTIONAL ACCURACY
# =============================
print("=" * 100)
print("📊 SECTION 4: DIRECTIONAL ACCURACY (Who Wins?)")
print("=" * 100)
print()

df_test['pred_winner'] = np.sign(df_test['prediction'])
df_test['actual_winner'] = np.sign(df_test['actual'])
df_test['direction_correct'] = df_test['pred_winner'] == df_test['actual_winner']

correct = df_test['direction_correct'].sum()
pct_correct = correct / len(df_test) * 100

print(f"WINNER PREDICTION ACCURACY:")
print(f"  Correct: {correct:4d} games ({pct_correct:.1f}%) ✅")
print(f"  Wrong:   {len(df_test) - correct:4d} games ({100 - pct_correct:.1f}%) ❌")
print()

print("MAE WHEN DIRECTION IS:")
correct_mae = df_test[df_test['direction_correct']]['error'].mean()
wrong_mae = df_test[~df_test['direction_correct']]['error'].mean()
print(f"  Correct: {correct_mae:6.3f} MAE (right winner, might be off on margin)")
print(f"  Wrong:   {wrong_mae:6.3f} MAE (picked wrong winner!)")
print()

# =============================
# SECTION 5: BEST & WORST
# =============================
print("=" * 100)
print("📊 SECTION 5: BEST & WORST PREDICTIONS")
print("=" * 100)
print()

print("TOP 10 BEST PREDICTIONS:")
best = df_test.nsmallest(10, 'error')
for idx, (_, row) in enumerate(best.iterrows(), 1):
    current = row['diff_at_2q_6min']
    pred = row['prediction']
    actual = row['actual']
    error = row['error']
    print(f"  {idx:2d}. Current: {current:+6.1f} → Predicted: {pred:+6.1f}, Actual: {actual:+6.1f} | Error: {error:4.1f} pts ⭐")
print()

print("TOP 10 WORST PREDICTIONS:")
worst = df_test.nlargest(10, 'error')
for idx, (_, row) in enumerate(worst.iterrows(), 1):
    current = row['diff_at_2q_6min']
    pred = row['prediction']
    actual = row['actual']
    error = row['error']
    outcome = row['outcome_type']
    print(f"  {idx:2d}. Current: {current:+6.1f} → Predicted: {pred:+6.1f}, Actual: {actual:+6.1f} | Error: {error:4.1f} pts ({outcome}) ❌")
print()

# =============================
# SECTION 6: BIAS ANALYSIS
# =============================
print("=" * 100)
print("📊 SECTION 6: BIAS ANALYSIS")
print("=" * 100)
print()

mean_bias = np.mean(signed_errors)
median_bias = np.median(signed_errors)

print("OVERALL BIAS:")
print(f"  Mean signed error:   {mean_bias:+.3f}")
print(f"  Median signed error: {median_bias:+.3f}")
print()

if abs(mean_bias) < 0.5:
    print("  ✅ EXCELLENT: Nearly perfectly unbiased!")
elif abs(mean_bias) < 1.0:
    print("  ✅ GOOD: Model is well-calibrated")
else:
    bias_direction = "OVER-predicting" if mean_bias > 0 else "UNDER-predicting"
    print(f"  ⚠️  Model is {bias_direction} by {abs(mean_bias):.3f} points")
print()

print("BIAS BY GAME STATE:")
for state in ['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)']:
    mask = df_test['game_state'] == state
    if mask.sum() > 5:
        bias = df_test.loc[mask, 'signed_error'].mean()
        direction = "over" if bias > 0 else "under"
        print(f"  {state:25s}: {bias:+6.3f} pts ({direction}-predicting)")
print()

# =============================
# SECTION 7: QUARTILE ANALYSIS
# =============================
print("=" * 100)
print("📊 SECTION 7: ERROR QUARTILES")
print("=" * 100)
print()

quartiles = [0, 25, 50, 75, 90, 95, 100]
error_percentiles = [np.percentile(errors, q) for q in quartiles]

print("ERROR PERCENTILES:")
for i, q in enumerate(quartiles):
    print(f"  {q:3d}th percentile: {error_percentiles[i]:6.2f} points")
print()

print("INTERPRETATION:")
print(f"  • Best 25% of games:    Error ≤ {error_percentiles[1]:.1f} points ⭐⭐⭐")
print(f"  • Middle 50% of games:  Error {error_percentiles[1]:.1f} - {error_percentiles[3]:.1f} points ⭐⭐")
print(f"  • 75-90th percentile:   Error {error_percentiles[3]:.1f} - {error_percentiles[4]:.1f} points ⭐")
print(f"  • Top 5% worst:         Error > {error_percentiles[5]:.1f} points ❌")
print()

# =============================
# SECTION 8: CONFIDENCE ZONES
# =============================
print("=" * 100)
print("📊 SECTION 8: CONFIDENCE ZONES")
print("=" * 100)
print()

print("HIGH-CONFIDENCE ZONE (Error ≤ 5 points):")
high_conf = df_test[df_test['error'] <= 5]
print(f"  Games: {len(high_conf):,} ({len(high_conf)/len(df_test)*100:.1f}%)")
print(f"  Avg Error: {high_conf['error'].mean():.3f}")
print(f"  Direction Accuracy: {high_conf['direction_correct'].mean()*100:.1f}%")
print(f"  Most common game states:")
for state in high_conf['game_state'].value_counts().head(3).items():
    pct = state[1] / len(high_conf) * 100
    print(f"    {state[0]:25s}: {state[1]:3d} games ({pct:5.1f}%)")
print()

print("MEDIUM ZONE (Error 5-12 points):")
medium = df_test[(df_test['error'] > 5) & (df_test['error'] <= 12)]
print(f"  Games: {len(medium):,} ({len(medium)/len(df_test)*100:.1f}%)")
print(f"  Avg Error: {medium['error'].mean():.3f}")
print(f"  Direction Accuracy: {medium['direction_correct'].mean()*100:.1f}%")
print()

print("RISK ZONE (Error > 15 points):")
risk_zone = df_test[df_test['error'] > 15]
print(f"  Games: {len(risk_zone):,} ({len(risk_zone)/len(df_test)*100:.1f}%)")
if len(risk_zone) > 0:
    print(f"  Avg Error: {risk_zone['error'].mean():.3f}")
    print(f"  Direction Accuracy: {risk_zone['direction_correct'].mean()*100:.1f}%")
    print(f"  Most common outcome:")
    for outcome in risk_zone['outcome_type'].value_counts().head(3).items():
        pct = outcome[1] / len(risk_zone) * 100
        print(f"    {outcome[0]:25s}: {outcome[1]:3d} games ({pct:5.1f}%)")
print()

# =============================
# SECTION 9: WHERE MODEL EXCELS VS STRUGGLES
# =============================
print("=" * 100)
print("🎯 SECTION 9: WHERE MODEL EXCELS VS STRUGGLES")
print("=" * 100)
print()

# Game state performance
game_state_maes = []
for state in df_test['game_state'].dropna().unique():
    mask = df_test['game_state'] == state
    if mask.sum() >= 5:
        mae = df_test.loc[mask, 'error'].mean()
        count = mask.sum()
        game_state_maes.append((state, mae, count))

game_state_maes.sort(key=lambda x: x[1])

print("MODEL EXCELS IN:")
for i, (state, mae, count) in enumerate(game_state_maes[:3], 1):
    print(f"  {i}. {state:25s}: {mae:6.3f} MAE ({count:,} games) ✅")
print()

print("MODEL STRUGGLES IN:")
for i, (state, mae, count) in enumerate(list(reversed(game_state_maes))[:3], 1):
    print(f"  {i}. {state:25s}: {mae:6.3f} MAE ({count:,} games) ⚠️")
print()

# =============================
# SECTION 10: BETTING RECOMMENDATIONS
# =============================
print("=" * 100)
print("💰 SECTION 10: BETTING STRATEGY RECOMMENDATIONS")
print("=" * 100)
print()

# Calculate prediction confidence based on change magnitude
df_test['pred_change'] = abs(df_test['prediction'] - df_test['diff_at_2q_6min'])

print("BETTING SIMULATION BY EDGE THRESHOLD:")
print()

for edge_threshold in [3, 5, 7, 10, 12, 15]:
    bet_games = df_test[df_test['pred_change'] >= edge_threshold]
    
    if len(bet_games) > 0:
        avg_error = bet_games['error'].mean()
        direction_acc = bet_games['direction_correct'].mean() * 100
        within_5 = (bet_games['error'] <= 5).sum()
        within_10 = (bet_games['error'] <= 10).sum()
        
        print(f"Edge ≥ {edge_threshold:2d} points:")
        print(f"  Bet on: {len(bet_games):4d} games ({len(bet_games)/len(df_test)*100:5.1f}% of season)")
        print(f"  Avg Error: {avg_error:6.2f} MAE")
        print(f"  Direction Accuracy: {direction_acc:5.1f}%")
        print(f"  Within 5 pts:  {within_5:4d} ({within_5/len(bet_games)*100:5.1f}%)")
        print(f"  Within 10 pts: {within_10:4d} ({within_10/len(bet_games)*100:5.1f}%)")
        print()

# =============================
# SAVE EVERYTHING
# =============================
print("=" * 100)
print("💾 SAVING COMPLETE BREAKDOWN")
print("=" * 100)
print()

breakdown = {
    'model': model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'test_predictions': df_test,
    'performance': {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'overfit_pct': overfit,
        'baseline_mae': baseline_mae,
        'edge_pct': edge,
        'median_error': np.median(errors),
        'direction_accuracy': pct_correct
    },
    'distributions': {
        'error_bins': df_test['error_bin'].value_counts().to_dict(),
        'error_quartiles': error_percentiles,
        'game_state_mae': {state: df_test[df_test['game_state']==state]['error'].mean() 
                           for state in df_test['game_state'].dropna().unique()},
        'outcome_type_mae': {outcome: df_test[df_test['outcome_type']==outcome]['error'].mean() 
                             for outcome in df_test['outcome_type'].unique()}
    },
    'insights': {
        'high_conf_pct': len(high_conf)/len(df_test)*100,
        'medium_zone_pct': len(medium)/len(df_test)*100,
        'risk_zone_pct': len(risk_zone)/len(df_test)*100,
        'bias': mean_bias,
        'best_game_types': game_state_maes[:3],
        'worst_game_types': list(reversed(game_state_maes))[:3]
    }
}

with open("Action/📊_76_FEATURE_BREAKDOWN.pkl", 'wb') as f:
    pickle.dump(breakdown, f)

# Create markdown report
report = f"""# 📊 76-FEATURE MODEL BREAKDOWN

**Generated:** {datetime.now().strftime('%I:%M %p, %A %B %d, %Y')}  
**Model:** Linear Regression (76 features)  
**Test Games:** {len(df_test):,}

---

## 📋 MODEL SPECIFICATIONS

- **Type:** Linear Regression
- **Features:** {len(feature_cols)}
- **Training:** {len(X_train):,} games (2015-2024, chronological)
- **Testing:** {len(X_test):,} games (2024-2025, chronological)

**Performance:**
- Train MAE: {train_mae:.3f}
- Test MAE: {test_mae:.3f}
- Overfitting: {overfit:.1f}%
- Edge over baseline: {edge:.1f}%

---

## 📊 ERROR DISTRIBUTION

| Range | Games | Percentage | Cumulative |
|-------|-------|------------|------------|
{chr(10).join([f'| {label} | {(df_test["error_bin"] == label).sum():,} | {(df_test["error_bin"] == label).sum()/len(df_test)*100:.1f}% | - |' for label in bin_labels])}

**Cumulative Accuracy:**
{chr(10).join([f'- Within {t:2d} points: {(df_test["error"] <= t).sum():,} games ({(df_test["error"] <= t).sum()/len(df_test)*100:.1f}%)' for t in [5, 8, 10, 12, 15]])}

---

## 🎯 PERFORMANCE BY GAME TYPE

### By Game State (Q2 6:00):

{chr(10).join([f'- **{state}**: {df_test[df_test["game_state"]==state]["error"].mean():.3f} MAE ({(df_test["game_state"]==state).sum():,} games, {(df_test["game_state"]==state).sum()/len(df_test)*100:.1f}%)' for state in ['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)'] if (df_test['game_state']==state).sum() > 0])}

### By Outcome:

{chr(10).join([f'- **{outcome}**: {df_test[df_test["outcome_type"]==outcome]["error"].mean():.3f} MAE ({(df_test["outcome_type"]==outcome).sum():,} games)' for outcome in df_test['outcome_type'].value_counts().index])}

---

## 🧭 DIRECTIONAL ACCURACY

- **Correct winner:** {correct:,} games ({pct_correct:.1f}%)
- **Wrong winner:** {len(df_test) - correct:,} games ({100 - pct_correct:.1f}%)

**MAE by direction:**
- When correct: {correct_mae:.3f} MAE
- When wrong: {wrong_mae:.3f} MAE

---

## ✅ WHERE MODEL EXCELS

{chr(10).join([f'{i}. {state}: {mae:.3f} MAE ({count:,} games)' for i, (state, mae, count) in enumerate(game_state_maes[:3], 1)])}

---

## ⚠️ WHERE MODEL STRUGGLES

{chr(10).join([f'{i}. {state}: {mae:.3f} MAE ({count:,} games)' for i, (state, mae, count) in enumerate(list(reversed(game_state_maes))[:3], 1)])}

---

## 💡 KEY INSIGHTS

1. **High-confidence zone:** {len(high_conf)/len(df_test)*100:.1f}% of games have error ≤ 5 pts
2. **Direction accuracy:** {pct_correct:.1f}% predict the right winner
3. **Median error:** {np.median(errors):.1f} points (50% of predictions better than this)
4. **Risk zone:** {len(risk_zone)/len(df_test)*100:.1f}% of games have error > 15 pts

---

**BREAKDOWN COMPLETE** ✅
"""

with open("Action/📊_76_FEATURE_REPORT.md", 'w') as f:
    f.write(report)

print("✅ Saved breakdown to: Action/📊_76_FEATURE_BREAKDOWN.pkl")
print("✅ Saved report to: Action/📊_76_FEATURE_REPORT.md")
print()

print("=" * 100)
print("✅ COMPLETE BREAKDOWN FINISHED!")
print("=" * 100)
print()

