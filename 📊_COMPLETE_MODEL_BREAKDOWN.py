#!/usr/bin/env python3
"""
COMPLETE MODEL BREAKDOWN & DISTRIBUTION ANALYSIS
Analyze where the model is right, wrong, and why
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

print("=" * 100)
print("📊 COMPLETE MODEL BREAKDOWN & DISTRIBUTION ANALYSIS")
print("=" * 100)
print()

# Load existing model and data
print("[1/8] Loading HYBRID_ULTIMATE_V2_CLEAN model and data...")
with open("Action/HYBRID_ULTIMATE_V2_CLEAN.pkl", 'rb') as f:
    system = pickle.load(f)

# Load the actual data
with open("Action/ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl", 'rb') as f:
    df = pickle.load(f)
    
# Convert to DataFrame if it's a list
if isinstance(df, list):
    df = pd.DataFrame(df)

print(f"✅ Loaded model and {len(df)} games")
print()

# Sort chronologically and split
df = df.sort_values('game_id').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
df_test = df.iloc[split_idx:].copy()

# Get predictions
print("[2/8] Generating predictions...")
feature_cols = system['feature_cols']
X_test = df_test[feature_cols].values
scaler = system['scaler']
X_test_scaled = scaler.transform(X_test)

model = system['model']
predictions = model.predict(X_test_scaled)
actuals = df_test['target'].values

# Calculate errors
errors = np.abs(predictions - actuals)
signed_errors = predictions - actuals

df_test['prediction'] = predictions
df_test['actual'] = actuals
df_test['error'] = errors
df_test['signed_error'] = signed_errors

print(f"✅ Generated {len(predictions)} predictions")
print()

# =============================
# SECTION 1: MODEL SPECIFICATIONS
# =============================
print("=" * 100)
print("📋 SECTION 1: MODEL SPECIFICATIONS")
print("=" * 100)
print()

print("MODEL ARCHITECTURE:")
print(f"  Type: {type(model).__name__}")
print(f"  Features: {len(feature_cols)}")
print(f"  Training samples: {int(len(df) * 0.8):,}")
print(f"  Test samples: {len(df_test):,}")
print()

print("FEATURE LIST:")
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {feat}")
print()

print("PERFORMANCE METRICS:")
print(f"  Test MAE: {mean_absolute_error(actuals, predictions):.3f}")
print(f"  Median Error: {np.median(errors):.3f}")
print(f"  Std Dev: {np.std(errors):.3f}")
print(f"  Min Error: {np.min(errors):.3f}")
print(f"  Max Error: {np.max(errors):.3f}")
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
for label in bin_labels:
    count = (df_test['error_bin'] == label).sum()
    pct = count / len(df_test) * 100
    print(f"  {label:12s}: {count:4d} games ({pct:5.1f}%)")
print()

# Cumulative
print("CUMULATIVE ACCURACY:")
for threshold in [2, 5, 8, 10, 12, 15, 20]:
    within = (df_test['error'] <= threshold).sum()
    pct = within / len(df_test) * 100
    print(f"  Within {threshold:2d} points: {within:4d} games ({pct:5.1f}%)")
print()

# =============================
# SECTION 3: GAME TYPE ANALYSIS
# =============================
print("=" * 100)
print("📊 SECTION 3: PERFORMANCE BY GAME TYPE")
print("=" * 100)
print()

# Categorize games by current differential at Q2 6:00
def categorize_game_state(row):
    diff = abs(row['current_diff']) if 'current_diff' in row else abs(row.get('diff_point_6', 0))
    if diff <= 3:
        return 'Very Close (≤3)'
    elif diff <= 7:
        return 'Close (4-7)'
    elif diff <= 12:
        return 'Moderate (8-12)'
    elif diff <= 20:
        return 'Large Lead (13-20)'
    else:
        return 'Blowout (>20)'

df_test['game_state'] = df_test.apply(categorize_game_state, axis=1)

print("MAE BY GAME STATE (at Q2 6:00):")
for state in ['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)']:
    mask = df_test['game_state'] == state
    if mask.sum() > 0:
        mae = df_test.loc[mask, 'error'].mean()
        count = mask.sum()
        pct = count / len(df_test) * 100
        print(f"  {state:25s}: {mae:6.3f} MAE ({count:4d} games, {pct:5.1f}%)")
print()

# Categorize by final outcome type
def categorize_outcome(row):
    current = row['current_diff'] if 'current_diff' in row else row.get('diff_point_6', 0)
    final = row['actual']
    
    # Did lead hold or flip?
    if abs(current) <= 3:
        return 'Was Close'
    elif (current > 0 and final > 0) or (current < 0 and final < 0):
        return 'Lead Held'
    else:
        return 'Comeback'

df_test['outcome_type'] = df_test.apply(categorize_outcome, axis=1)

print("MAE BY OUTCOME TYPE:")
for outcome in ['Was Close', 'Lead Held', 'Comeback']:
    mask = df_test['outcome_type'] == outcome
    if mask.sum() > 0:
        mae = df_test.loc[mask, 'error'].mean()
        count = mask.sum()
        pct = count / len(df_test) * 100
        print(f"  {outcome:25s}: {mae:6.3f} MAE ({count:4d} games, {pct:5.1f}%)")
print()

# =============================
# SECTION 4: DIRECTION ANALYSIS
# =============================
print("=" * 100)
print("📊 SECTION 4: DIRECTIONAL ACCURACY")
print("=" * 100)
print()

# Did we predict the right direction?
def get_direction_accuracy(row):
    pred_sign = 1 if row['prediction'] > 0 else -1 if row['prediction'] < 0 else 0
    actual_sign = 1 if row['actual'] > 0 else -1 if row['actual'] < 0 else 0
    return pred_sign == actual_sign

df_test['direction_correct'] = df_test.apply(get_direction_accuracy, axis=1)

correct_direction = df_test['direction_correct'].sum()
pct_correct = correct_direction / len(df_test) * 100

print(f"DIRECTION ACCURACY:")
print(f"  Correct: {correct_direction:4d} games ({pct_correct:.1f}%)")
print(f"  Wrong:   {len(df_test) - correct_direction:4d} games ({100 - pct_correct:.1f}%)")
print()

print("MAE BY DIRECTIONAL ACCURACY:")
for correct in [True, False]:
    mask = df_test['direction_correct'] == correct
    if mask.sum() > 0:
        mae = df_test.loc[mask, 'error'].mean()
        count = mask.sum()
        label = "Correct Direction" if correct else "Wrong Direction"
        print(f"  {label:25s}: {mae:6.3f} MAE ({count:4d} games)")
print()

# =============================
# SECTION 5: MAGNITUDE ANALYSIS
# =============================
print("=" * 100)
print("📊 SECTION 5: PREDICTION MAGNITUDE ANALYSIS")
print("=" * 100)
print()

# Categorize by predicted magnitude of change
def categorize_prediction(row):
    current = row['current_diff'] if 'current_diff' in row else row.get('diff_point_6', 0)
    predicted = row['prediction']
    change = predicted - current
    
    if abs(change) <= 2:
        return 'Stable (±2)'
    elif abs(change) <= 5:
        return 'Small Change (2-5)'
    elif abs(change) <= 10:
        return 'Moderate Change (5-10)'
    else:
        return 'Large Change (>10)'

df_test['prediction_type'] = df_test.apply(categorize_prediction, axis=1)

print("MAE BY PREDICTED CHANGE:")
for pred_type in ['Stable (±2)', 'Small Change (2-5)', 'Moderate Change (5-10)', 'Large Change (>10)']:
    mask = df_test['prediction_type'] == pred_type
    if mask.sum() > 0:
        mae = df_test.loc[mask, 'error'].mean()
        count = mask.sum()
        pct = count / len(df_test) * 100
        print(f"  {pred_type:25s}: {mae:6.3f} MAE ({count:4d} games, {pct:5.1f}%)")
print()

# =============================
# SECTION 6: BEST vs WORST GAMES
# =============================
print("=" * 100)
print("📊 SECTION 6: BEST & WORST PREDICTIONS")
print("=" * 100)
print()

# Best predictions
print("TOP 10 BEST PREDICTIONS (lowest error):")
best = df_test.nsmallest(10, 'error')
for idx, (i, row) in enumerate(best.iterrows(), 1):
    print(f"  {idx:2d}. Game {row['game_id']}: Predicted {row['prediction']:+6.1f}, Actual {row['actual']:+6.1f}, Error {row['error']:.1f}")
print()

# Worst predictions
print("TOP 10 WORST PREDICTIONS (highest error):")
worst = df_test.nlargest(10, 'error')
for idx, (i, row) in enumerate(worst.iterrows(), 1):
    current = row['current_diff'] if 'current_diff' in row else row.get('diff_point_6', 0)
    print(f"  {idx:2d}. Game {row['game_id']}: Current {current:+6.1f}, Predicted {row['prediction']:+6.1f}, Actual {row['actual']:+6.1f}, Error {row['error']:.1f}")
print()

# =============================
# SECTION 7: BIAS ANALYSIS
# =============================
print("=" * 100)
print("📊 SECTION 7: BIAS ANALYSIS")
print("=" * 100)
print()

print("PREDICTION BIAS:")
print(f"  Mean signed error: {np.mean(signed_errors):+.3f}")
print(f"  Median signed error: {np.median(signed_errors):+.3f}")
print()

if abs(np.mean(signed_errors)) < 1.0:
    print("  ✅ UNBIASED: Model doesn't systematically over/under-predict")
else:
    bias_direction = "over-predicting" if np.mean(signed_errors) > 0 else "under-predicting"
    print(f"  ⚠️  BIAS: Model is {bias_direction} by {abs(np.mean(signed_errors)):.3f} points on average")
print()

# Over/under prediction by game state
print("BIAS BY GAME STATE:")
for state in ['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)']:
    mask = df_test['game_state'] == state
    if mask.sum() > 0:
        bias = df_test.loc[mask, 'signed_error'].mean()
        bias_label = "over" if bias > 0 else "under"
        print(f"  {state:25s}: {bias:+6.3f} ({bias_label}-predicting by {abs(bias):.1f} pts)")
print()

# =============================
# SECTION 8: ACTIONABLE INSIGHTS
# =============================
print("=" * 100)
print("🎯 SECTION 8: ACTIONABLE INSIGHTS")
print("=" * 100)
print()

print("WHERE THE MODEL EXCELS:")
# Find game types with lowest MAE
game_state_maes = []
for state in ['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)']:
    mask = df_test['game_state'] == state
    if mask.sum() > 5:  # At least 5 games
        mae = df_test.loc[mask, 'error'].mean()
        count = mask.sum()
        game_state_maes.append((state, mae, count))

game_state_maes.sort(key=lambda x: x[1])
for i, (state, mae, count) in enumerate(game_state_maes[:3], 1):
    print(f"  {i}. {state:25s}: {mae:.3f} MAE ({count} games)")
print()

print("WHERE THE MODEL STRUGGLES:")
for i, (state, mae, count) in enumerate(reversed(game_state_maes[-3:]), 1):
    print(f"  {i}. {state:25s}: {mae:.3f} MAE ({count} games)")
print()

# High-confidence predictions
print("HIGH-CONFIDENCE ZONE (Error ≤ 5 points):")
high_conf = df_test[df_test['error'] <= 5]
if len(high_conf) > 0:
    print(f"  Total: {len(high_conf)} games ({len(high_conf)/len(df_test)*100:.1f}%)")
    print(f"  Avg Error: {high_conf['error'].mean():.3f}")
    
    # What characterizes high-confidence predictions?
    print()
    print("  Characteristics of accurate predictions:")
    for state in ['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)']:
        mask = high_conf['game_state'] == state
        if mask.sum() > 0:
            pct = mask.sum() / len(high_conf) * 100
            print(f"    {state:25s}: {mask.sum():4d} games ({pct:5.1f}% of accurate predictions)")
print()

# Risk zone
print("RISK ZONE (Error > 15 points):")
risk_zone = df_test[df_test['error'] > 15]
if len(risk_zone) > 0:
    print(f"  Total: {len(risk_zone)} games ({len(risk_zone)/len(df_test)*100:.1f}%)")
    print(f"  Avg Error: {risk_zone['error'].mean():.3f}")
    
    print()
    print("  Characteristics of big misses:")
    for state in df_test['game_state'].unique():
        mask = risk_zone['game_state'] == state
        if mask.sum() > 0:
            pct = mask.sum() / len(risk_zone) * 100
            print(f"    {state:25s}: {mask.sum():4d} games ({pct:5.1f}% of big misses)")
    
    print()
    print("  Outcome types in risk zone:")
    for outcome in df_test['outcome_type'].unique():
        mask = risk_zone['outcome_type'] == outcome
        if mask.sum() > 0:
            pct = mask.sum() / len(risk_zone) * 100
            print(f"    {outcome:25s}: {mask.sum():4d} games ({pct:5.1f}% of big misses)")
print()

# =============================
# SECTION 9: QUARTILE ANALYSIS
# =============================
print("=" * 100)
print("📊 SECTION 9: PERFORMANCE BY ERROR QUARTILES")
print("=" * 100)
print()

quartiles = [0, 25, 50, 75, 100]
error_quartiles = [np.percentile(errors, q) for q in quartiles]

print("ERROR QUARTILES:")
for i, q in enumerate(quartiles):
    print(f"  {q:3d}th percentile: {error_quartiles[i]:6.3f} points")
print()

print("INTERPRETATION:")
print(f"  • Best 25% of predictions: Error ≤ {error_quartiles[1]:.1f} points")
print(f"  • Middle 50% of predictions: Error between {error_quartiles[1]:.1f} and {error_quartiles[3]:.1f} points")
print(f"  • Worst 25% of predictions: Error > {error_quartiles[3]:.1f} points")
print()

# =============================
# SECTION 10: BETTING EDGE ANALYSIS
# =============================
print("=" * 100)
print("💰 SECTION 10: BETTING EDGE ANALYSIS")
print("=" * 100)
print()

# Simulate different confidence thresholds
print("WIN RATE BY CONFIDENCE THRESHOLD:")
print("(Only bet when our edge is above threshold)")
print()

for threshold in [3, 5, 7, 10, 12, 15]:
    # Filter predictions where we have strong edge
    # Edge = difference between current diff and predicted diff
    df_test['edge'] = abs(df_test['prediction'] - df_test.get('current_diff', 0))
    high_edge = df_test[df_test['edge'] >= threshold]
    
    if len(high_edge) > 0:
        # How many would we have won?
        # Win if error < edge (we were right by enough margin)
        wins = (high_edge['error'] < threshold).sum()
        win_rate = wins / len(high_edge) * 100
        avg_error = high_edge['error'].mean()
        
        print(f"  Edge ≥ {threshold:2d} pts: {len(high_edge):4d} games, {wins:4d} wins ({win_rate:5.1f}%), Avg Error: {avg_error:.2f}")

print()

# =============================
# SAVE FULL BREAKDOWN
# =============================
print("=" * 100)
print("💾 SAVING DETAILED BREAKDOWN")
print("=" * 100)
print()

breakdown = {
    'model': model,
    'predictions': predictions,
    'actuals': actuals,
    'errors': errors,
    'test_data': df_test,
    'error_distribution': df_test['error_bin'].value_counts().to_dict(),
    'game_state_performance': df_test.groupby('game_state')['error'].agg(['mean', 'count']).to_dict(),
    'outcome_type_performance': df_test.groupby('outcome_type')['error'].agg(['mean', 'count']).to_dict(),
    'direction_accuracy_pct': pct_correct,
    'error_quartiles': error_quartiles,
    'mae': mean_absolute_error(actuals, predictions)
}

with open("Action/📊_MODEL_BREAKDOWN.pkl", 'wb') as f:
    pickle.dump(breakdown, f)

print("✅ Saved detailed breakdown to: Action/📊_MODEL_BREAKDOWN.pkl")
print()

# Create summary markdown
print("Creating summary document...")

summary = f"""# 📊 HYBRID_V2_CLEAN MODEL BREAKDOWN

**Generated:** {datetime.now().strftime('%I:%M %p, %A %B %d, %Y')}  
**Model:** HYBRID_V2_CLEAN  
**Test Games:** {len(df_test):,}

---

## 📋 MODEL SPECIFICATIONS

**Architecture:**
- Type: {type(model).__name__}
- Features: {len(feature_cols)}
- Training: {int(len(df) * 0.8):,} games (2015-2024)
- Testing: {len(df_test):,} games (2024-2025)

**Features Used:**
{chr(10).join([f'{i}. {feat}' for i, feat in enumerate(feature_cols, 1)])}

---

## 📊 OVERALL PERFORMANCE

```
Test MAE:      {mean_absolute_error(actuals, predictions):.3f}
Median Error:  {np.median(errors):.3f}
Std Dev:       {np.std(errors):.3f}
Min Error:     {np.min(errors):.3f}
Max Error:     {np.max(errors):.3f}
```

---

## 📈 ERROR DISTRIBUTION

| Range | Games | Percentage |
|-------|-------|------------|
{chr(10).join([f'| {label} | {(df_test["error_bin"] == label).sum():,} | {(df_test["error_bin"] == label).sum()/len(df_test)*100:.1f}% |' for label in bin_labels])}

**Cumulative Accuracy:**
{chr(10).join([f'- Within {t} points: {(df_test["error"] <= t).sum():,} games ({(df_test["error"] <= t).sum()/len(df_test)*100:.1f}%)' for t in [5, 8, 10, 12, 15]])}

---

## 🎯 PERFORMANCE BY GAME TYPE

### By Game State (at Q2 6:00):

| Game State | MAE | Games | % of Total |
|------------|-----|-------|------------|
{chr(10).join([f"| {state} | {df_test[df_test['game_state'] == state]['error'].mean():.3f} | {(df_test['game_state'] == state).sum():,} | {(df_test['game_state'] == state).sum()/len(df_test)*100:.1f}% |" for state in ['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)'] if (df_test['game_state'] == state).sum() > 0])}

### By Outcome Type:

| Outcome Type | MAE | Games | % of Total |
|--------------|-----|-------|------------|
{chr(10).join([f"| {outcome} | {df_test[df_test['outcome_type'] == outcome]['error'].mean():.3f} | {(df_test['outcome_type'] == outcome).sum():,} | {(df_test['outcome_type'] == outcome).sum()/len(df_test)*100:.1f}% |" for outcome in ['Was Close', 'Lead Held', 'Comeback'] if (df_test['outcome_type'] == outcome).sum() > 0])}

---

## 🧭 DIRECTIONAL ACCURACY

```
Correct Direction: {correct_direction:,} games ({pct_correct:.1f}%)
Wrong Direction:   {len(df_test) - correct_direction:,} games ({100 - pct_correct:.1f}%)
```

**MAE by Direction:**
- Correct Direction: {df_test[df_test['direction_correct'] == True]['error'].mean():.3f} MAE
- Wrong Direction: {df_test[df_test['direction_correct'] == False]['error'].mean():.3f} MAE

---

## ✅ WHERE MODEL EXCELS

**Best Performance:**
{chr(10).join([f'{i}. {state}: {mae:.3f} MAE ({count} games)' for i, (state, mae, count) in enumerate(game_state_maes[:3], 1)])}

**High-Confidence Zone (Error ≤ 5 pts):**
- {len(high_conf):,} games ({len(high_conf)/len(df_test)*100:.1f}%)
- Average error: {high_conf['error'].mean():.3f}

---

## ⚠️ WHERE MODEL STRUGGLES

**Worst Performance:**
{chr(10).join([f'{i}. {state}: {mae:.3f} MAE ({count} games)' for i, (state, mae, count) in enumerate(reversed(game_state_maes[-3:]), 1)])}

**Risk Zone (Error > 15 pts):**
- {len(risk_zone):,} games ({len(risk_zone)/len(df_test)*100:.1f}%)
- Average error: {risk_zone['error'].mean():.3f}
- Most common: {risk_zone['outcome_type'].value_counts().index[0] if len(risk_zone) > 0 else 'N/A'}

---

## 💡 KEY INSIGHTS

1. **Direction accuracy:** {pct_correct:.1f}% of predictions get the direction right
2. **Median error:** {np.median(errors):.1f} points (half of predictions within this)
3. **High confidence:** {len(high_conf)/len(df_test)*100:.1f}% of games have error ≤ 5 points
4. **Risk games:** {len(risk_zone)/len(df_test)*100:.1f}% of games have error > 15 points

---

## 🚀 BETTING STRATEGY RECOMMENDATIONS

**AGGRESSIVE (High Volume):**
- Bet when edge ≥ 5 points
- Expected: ~{(df_test['edge'] >= 5).sum() if 'edge' in df_test.columns else 0} games per season
- Risk: Moderate

**BALANCED (Recommended):**
- Bet when edge ≥ 7 points  
- Expected: ~{(df_test['edge'] >= 7).sum() if 'edge' in df_test.columns else 0} games per season
- Risk: Low-Moderate

**CONSERVATIVE (High Confidence):**
- Bet when edge ≥ 10 points
- Expected: ~{(df_test['edge'] >= 10).sum() if 'edge' in df_test.columns else 0} games per season
- Risk: Low

---

**MODEL BREAKDOWN COMPLETE**  
**All performance dimensions analyzed** ✅
"""

with open("Action/📊_MODEL_BREAKDOWN_REPORT.md", 'w') as f:
    f.write(summary)

print("✅ Saved summary to: Action/📊_MODEL_BREAKDOWN_REPORT.md")
print()

print("=" * 100)
print("✅ COMPLETE MODEL BREAKDOWN FINISHED!")
print("=" * 100)
print()
print("Generated:")
print("  • Action/📊_MODEL_BREAKDOWN.pkl (detailed data)")
print("  • Action/📊_MODEL_BREAKDOWN_REPORT.md (summary report)")
print()

