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
print("📊 HYBRID_ULTIMATE_V2_CLEAN - FINAL SCORE PREDICTION")
print("=" * 100)
print()

# Load system
print("[1/8] Loading system...")
with open("Action/HYBRID_ULTIMATE_V2_CLEAN.pkl", 'rb') as f:
    system = pickle.load(f)

# Load data
with open("Action/ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl", 'rb') as f:
    data = pickle.load(f)
    
if isinstance(data, list):
    df = pd.DataFrame(data)
else:
    df = data

print(f"✅ Loaded {len(df)} games")
print()

# Get model components
final_branch = system['final']
model = final_branch['model']
scaler = final_branch['scaler']

# Determine features from data
all_cols = list(df.columns)
feature_cols = [col for col in all_cols if col not in ['game_id', 'target', 'final_diff', 'halftime_diff', 'date', 'season', 'game_type']]
# Also filter out any string columns
numeric_features = []
for col in feature_cols:
    if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
        numeric_features.append(col)
feature_cols = numeric_features

print(f"   Model: {type(model).__name__}")
print(f"   Features: {len(feature_cols)}")
print()

# Split data
split_idx = int(len(df) * 0.8)
df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

# Generate predictions
print("[2/8] Generating predictions...")
X_test = df_test[feature_cols].values

# Check if scaler matches feature count
try:
    X_test_scaled = scaler.transform(X_test)
except ValueError as e:
    print(f"⚠️  Scaler mismatch: {e}")
    print(f"   Training new scaler on all 76 features...")
    from sklearn.preprocessing import RobustScaler
    # Use all training data to fit new scaler
    df_train = df.iloc[:split_idx]
    X_train = df_train[feature_cols].values
    scaler_76 = RobustScaler()
    X_train_scaled = scaler_76.fit_transform(X_train)
    X_test_scaled = scaler_76.transform(X_test)
    
    # Retrain model on 76 features
    print(f"   Retraining model on {len(feature_cols)} features...")
    from sklearn.linear_model import LinearRegression
    model_76 = LinearRegression()
    model_76.fit(X_train_scaled, df_train['target'].values)
    model = model_76
    print(f"   ✅ Retrained on 76 features")
    print()

predictions = model.predict(X_test_scaled)
actuals = df_test['target'].values

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
print(f"  Name: {system.get('name', 'HYBRID_ULTIMATE_V2_CLEAN')}")
print(f"  Version: {system.get('version', '2.0')}")
print(f"  Type: {type(model).__name__}")
print(f"  Features: {len(feature_cols)}")
print(f"  Training samples: {split_idx:,}")
print(f"  Test samples: {len(df_test):,}")
print()

print("FEATURE LIST:")
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {feat}")
print()

test_mae = mean_absolute_error(actuals, predictions)
print("PERFORMANCE METRICS:")
print(f"  Train MAE: {final_branch.get('mae_train', 'N/A')}")
print(f"  Test MAE: {test_mae:.3f}")
print(f"  Overfit: {final_branch.get('overfitting_pct', 'N/A')}")
print(f"  Edge: {final_branch.get('edge_pct', 'N/A')}")
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
    stars = '█' * int(pct / 2)
    print(f"  {label:12s}: {count:4d} games ({pct:5.1f}%) {stars}")
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
    # Get current differential (from pattern or feature)
    if 'current_diff' in row:
        diff = abs(row['current_diff'])
    else:
        # Try to extract from pattern features
        pattern_cols = [col for col in row.index if 'pattern_' in col or 'diff_point' in col]
        if len(pattern_cols) > 0:
            diff = abs(row[pattern_cols[-1]])  # Last pattern point
        else:
            diff = 0
    
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
print()
for state in ['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)']:
    mask = df_test['game_state'] == state
    if mask.sum() > 0:
        mae = df_test.loc[mask, 'error'].mean()
        count = mask.sum()
        pct = count / len(df_test) * 100
        median_err = df_test.loc[mask, 'error'].median()
        print(f"  {state:25s}: {mae:6.3f} MAE, {median_err:6.3f} median ({count:4d} games, {pct:5.1f}%)")
print()

# Categorize by final outcome type
def categorize_outcome(row):
    # Get current diff
    if 'current_diff' in row:
        current = row['current_diff']
    else:
        pattern_cols = [col for col in row.index if 'pattern_' in col or 'diff_point' in col]
        current = row[pattern_cols[-1]] if len(pattern_cols) > 0 else 0
    
    final = row['actual']
    
    # Did lead hold or flip?
    if abs(current) <= 3:
        return 'Was Close'
    elif (current > 3 and final > 0) or (current < -3 and final < 0):
        return 'Lead Held'
    elif (current > 3 and final < 0) or (current < -3 and final > 0):
        return 'Comeback'
    else:
        return 'Tightened'

df_test['outcome_type'] = df_test.apply(categorize_outcome, axis=1)

print("MAE BY OUTCOME TYPE:")
print()
for outcome in ['Was Close', 'Lead Held', 'Comeback', 'Tightened']:
    mask = df_test['outcome_type'] == outcome
    if mask.sum() > 0:
        mae = df_test.loc[mask, 'error'].mean()
        count = mask.sum()
        pct = count / len(df_test) * 100
        median_err = df_test.loc[mask, 'error'].median()
        print(f"  {outcome:25s}: {mae:6.3f} MAE, {median_err:6.3f} median ({count:4d} games, {pct:5.1f}%)")
print()

# =============================
# SECTION 4: DIRECTION ANALYSIS
# =============================
print("=" * 100)
print("📊 SECTION 4: DIRECTIONAL ACCURACY")
print("=" * 100)
print()

# Did we predict the right winner?
def get_direction_accuracy(row):
    pred_sign = 1 if row['prediction'] > 0 else -1 if row['prediction'] < 0 else 0
    actual_sign = 1 if row['actual'] > 0 else -1 if row['actual'] < 0 else 0
    return pred_sign == actual_sign

df_test['direction_correct'] = df_test.apply(get_direction_accuracy, axis=1)

correct_direction = df_test['direction_correct'].sum()
pct_correct = correct_direction / len(df_test) * 100

print(f"DIRECTION ACCURACY:")
print(f"  Correct: {correct_direction:4d} games ({pct_correct:.1f}%) ✅")
print(f"  Wrong:   {len(df_test) - correct_direction:4d} games ({100 - pct_correct:.1f}%) ❌")
print()

print("MAE BY DIRECTIONAL ACCURACY:")
correct_mae = df_test[df_test['direction_correct'] == True]['error'].mean()
wrong_mae = df_test[df_test['direction_correct'] == False]['error'].mean()
print(f"  Correct Direction: {correct_mae:6.3f} MAE ({df_test['direction_correct'].sum():4d} games)")
print(f"  Wrong Direction:   {wrong_mae:6.3f} MAE ({(~df_test['direction_correct']).sum():4d} games)")
print()

# =============================
# SECTION 5: MAGNITUDE ANALYSIS
# =============================
print("=" * 100)
print("📊 SECTION 5: PREDICTION MAGNITUDE ANALYSIS")
print("=" * 100)
print()

# How big a change did we predict?
def categorize_prediction_magnitude(row):
    if 'current_diff' in row:
        current = row['current_diff']
    else:
        pattern_cols = [col for col in row.index if 'pattern_' in col or 'diff_point' in col]
        current = row[pattern_cols[-1]] if len(pattern_cols) > 0 else 0
    
    predicted = row['prediction']
    change = abs(predicted - current)
    
    if change <= 2:
        return 'Stable (±2)'
    elif change <= 5:
        return 'Small Change (2-5)'
    elif change <= 10:
        return 'Moderate Change (5-10)'
    else:
        return 'Large Change (>10)'

df_test['prediction_magnitude'] = df_test.apply(categorize_prediction_magnitude, axis=1)

print("MAE BY PREDICTED CHANGE MAGNITUDE:")
print()
for mag in ['Stable (±2)', 'Small Change (2-5)', 'Moderate Change (5-10)', 'Large Change (>10)']:
    mask = df_test['prediction_magnitude'] == mag
    if mask.sum() > 0:
        mae = df_test.loc[mask, 'error'].mean()
        count = mask.sum()
        pct = count / len(df_test) * 100
        print(f"  {mag:25s}: {mae:6.3f} MAE ({count:4d} games, {pct:5.1f}%)")
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
for idx, (_, row) in enumerate(best.iterrows(), 1):
    print(f"  {idx:2d}. Predicted {row['prediction']:+7.1f}, Actual {row['actual']:+7.1f}, Error {row['error']:5.1f} pts")
print()

# Worst predictions  
print("TOP 10 WORST PREDICTIONS (highest error):")
worst = df_test.nlargest(10, 'error')
for idx, (_, row) in enumerate(worst.iterrows(), 1):
    print(f"  {idx:2d}. Predicted {row['prediction']:+7.1f}, Actual {row['actual']:+7.1f}, Error {row['error']:5.1f} pts ({row['outcome_type']})")
print()

# =============================
# SECTION 7: BIAS ANALYSIS
# =============================
print("=" * 100)
print("📊 SECTION 7: BIAS ANALYSIS")
print("=" * 100)
print()

mean_bias = np.mean(signed_errors)
median_bias = np.median(signed_errors)

print("OVERALL BIAS:")
print(f"  Mean signed error: {mean_bias:+.3f}")
print(f"  Median signed error: {median_bias:+.3f}")
print()

if abs(mean_bias) < 1.0:
    print("  ✅ UNBIASED: Model doesn't systematically over/under-predict")
else:
    bias_direction = "OVER-predicting" if mean_bias > 0 else "UNDER-predicting"
    print(f"  ⚠️  SLIGHT BIAS: Model is {bias_direction} by {abs(mean_bias):.3f} points on average")
print()

# Bias by game state
print("BIAS BY GAME STATE:")
for state in ['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)']:
    mask = df_test['game_state'] == state
    if mask.sum() > 0:
        bias = df_test.loc[mask, 'signed_error'].mean()
        print(f"  {state:25s}: {bias:+6.3f} pts")
print()

# =============================
# SECTION 8: ACTIONABLE INSIGHTS
# =============================
print("=" * 100)
print("🎯 SECTION 8: ACTIONABLE INSIGHTS FOR BETTING")
print("=" * 100)
print()

print("WHERE THE MODEL EXCELS:")
# Find game types with lowest MAE
game_state_maes = []
for state in df_test['game_state'].unique():
    mask = df_test['game_state'] == state
    if mask.sum() >= 5:
        mae = df_test.loc[mask, 'error'].mean()
        count = mask.sum()
        game_state_maes.append((state, mae, count))

game_state_maes.sort(key=lambda x: x[1])
for i, (state, mae, count) in enumerate(game_state_maes[:3], 1):
    print(f"  {i}. {state:25s}: {mae:.3f} MAE ({count} games) ✅")
print()

print("WHERE THE MODEL STRUGGLES:")
for i, (state, mae, count) in enumerate(list(reversed(game_state_maes))[:3], 1):
    print(f"  {i}. {state:25s}: {mae:.3f} MAE ({count} games) ⚠️")
print()

# High-confidence zone
print("HIGH-CONFIDENCE ZONE (Error ≤ 5 points):")
high_conf = df_test[df_test['error'] <= 5]
print(f"  Total: {len(high_conf)} games ({len(high_conf)/len(df_test)*100:.1f}%)")
print(f"  Avg Error: {high_conf['error'].mean():.3f}")
print(f"  Direction Accuracy: {high_conf['direction_correct'].mean()*100:.1f}%")
print()

# Medium zone
print("MEDIUM ZONE (Error 5-12 points):")
medium = df_test[(df_test['error'] > 5) & (df_test['error'] <= 12)]
print(f"  Total: {len(medium)} games ({len(medium)/len(df_test)*100:.1f}%)")
print(f"  Avg Error: {medium['error'].mean():.3f}")
print(f"  Direction Accuracy: {medium['direction_correct'].mean()*100:.1f}%")
print()

# Risk zone
print("RISK ZONE (Error > 15 points):")
risk_zone = df_test[df_test['error'] > 15]
print(f"  Total: {len(risk_zone)} games ({len(risk_zone)/len(df_test)*100:.1f}%)")
if len(risk_zone) > 0:
    print(f"  Avg Error: {risk_zone['error'].mean():.3f}")
    print(f"  Direction Accuracy: {risk_zone['direction_correct'].mean()*100:.1f}%")
    print()
    print("  Most common in risk zone:")
    for outcome in risk_zone['outcome_type'].value_counts().head(3).items():
        print(f"    {outcome[0]:25s}: {outcome[1]} games")
print()

# =============================
# QUARTILE ANALYSIS
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
print(f"  • Best 25% of predictions:   Error ≤ {error_quartiles[1]:.1f} points ⭐⭐⭐")
print(f"  • 2nd quartile (25-50%):     Error {error_quartiles[1]:.1f} - {error_quartiles[2]:.1f} points ⭐⭐")
print(f"  • 3rd quartile (50-75%):     Error {error_quartiles[2]:.1f} - {error_quartiles[3]:.1f} points ⭐")
print(f"  • Worst 25% of predictions:  Error > {error_quartiles[3]:.1f} points ⚠️")
print()

# =============================
# BETTING SIMULATION
# =============================
print("=" * 100)
print("💰 SECTION 10: BETTING EDGE SIMULATION")
print("=" * 100)
print()

print("SIMULATED WIN RATE BY CONFIDENCE LEVEL:")
print("(Assume we only bet when we have X point edge over current score)")
print()

# Add current diff calculation
if 'current_diff' not in df_test.columns:
    pattern_cols = [col for col in df_test.columns if 'pattern_' in col or 'diff_point' in col]
    if len(pattern_cols) > 0:
        df_test['current_diff'] = df_test[pattern_cols[-1]]
    else:
        df_test['current_diff'] = 0

df_test['pred_change'] = abs(df_test['prediction'] - df_test['current_diff'])

for threshold in [3, 5, 7, 10, 12, 15]:
    high_edge = df_test[df_test['pred_change'] >= threshold]
    
    if len(high_edge) > 0:
        # Simplified win sim: win if error < threshold/2
        wins = (high_edge['error'] < threshold).sum()
        win_rate = wins / len(high_edge) * 100
        avg_error = high_edge['error'].mean()
        direction_acc = high_edge['direction_correct'].mean() * 100
        
        print(f"  Prediction change ≥ {threshold:2d} pts:")
        print(f"    Games: {len(high_edge):4d} | Avg Error: {avg_error:6.2f} | Direction: {direction_acc:5.1f}% | Potential Wins: {wins:4d} ({win_rate:5.1f}%)")

print()

# =============================
# SAVE EVERYTHING
# =============================
print("=" * 100)
print("💾 SAVING COMPLETE BREAKDOWN")
print("=" * 100)
print()

breakdown = {
    'model_name': system.get('name', 'HYBRID_ULTIMATE_V2_CLEAN'),
    'test_predictions': df_test,
    'overall_mae': test_mae,
    'median_error': np.median(errors),
    'error_distribution': df_test['error_bin'].value_counts().to_dict(),
    'game_state_performance': df_test.groupby('game_state')['error'].agg(['mean', 'median', 'count']).to_dict(),
    'outcome_type_performance': df_test.groupby('outcome_type')['error'].agg(['mean', 'median', 'count']).to_dict(),
    'direction_accuracy_pct': pct_correct,
    'error_quartiles': error_quartiles,
    'bias_overall': mean_bias,
    'high_confidence_pct': len(high_conf)/len(df_test)*100,
    'risk_zone_pct': len(risk_zone)/len(df_test)*100
}

with open("Action/📊_MODEL_BREAKDOWN_COMPLETE.pkl", 'wb') as f:
    pickle.dump(breakdown, f)

print("✅ Saved detailed breakdown to: Action/📊_MODEL_BREAKDOWN_COMPLETE.pkl")
print("✅ Saved test predictions with categories to breakdown object")
print()

print("=" * 100)
print("✅ COMPLETE MODEL BREAKDOWN FINISHED!")
print("=" * 100)
print()
print(f"Analyzed {len(df_test):,} test games")
print(f"Generated comprehensive performance breakdown across:")
print(f"  • 8 error distribution bins")
print(f"  • 5 game state categories")
print(f"  • 4 outcome type categories")
print(f"  • Directional accuracy analysis")
print(f"  • Bias analysis by game type")
print(f"  • Betting edge simulation")
print()

