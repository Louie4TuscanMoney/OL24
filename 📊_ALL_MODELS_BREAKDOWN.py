#!/usr/bin/env python3
"""
COMPREHENSIVE BREAKDOWN FOR ALL MODELS
Generate detailed markdown reports for every model system
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import os

def generate_model_breakdown(model_name, model_file, data_file, target_col='target', output_dir='Action/Model_Breakdowns'):
    """Generate comprehensive breakdown for a single model"""
    
    print("=" * 100)
    print(f"📊 ANALYZING: {model_name}")
    print("=" * 100)
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load data
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            if 'data' in data:
                df = pd.DataFrame(data['data']) if isinstance(data['data'], list) else data['data']
            else:
                # It's a system dict, extract data differently
                df = None
                for key in ['test_data', 'df', 'dataframe']:
                    if key in data:
                        df = data[key]
                        break
                if df is None:
                    print(f"  ⚠️  Cannot extract data from {model_name}")
                    return None
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Handle different target column names
        if target_col not in df.columns:
            possible_targets = ['target', 'diff_at_final', 'final_diff', 'y']
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    break
        
        if target_col not in df.columns:
            print(f"  ⚠️  No target column found for {model_name}")
            return None
        
        print(f"  ✅ Loaded {len(df)} games")
        
        # Prepare features
        exclude_cols = ['game_id', 'date', 'season', 'home_team', 'away_team', 'pattern',
                       'diff_at_final', 'diff_at_halftime', 'diff_at_2q_6min', target_col,
                       'target', 'final_diff', 'halftime_diff']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Filter numeric only
        numeric_features = []
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                numeric_features.append(col)
        feature_cols = numeric_features
        
        print(f"  ✅ {len(feature_cols)} features")
        
        # Split
        split_idx = int(len(df) * 0.8)
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy().reset_index(drop=True)
        
        X_train = np.nan_to_num(df_train[feature_cols].values, nan=0.0)
        y_train = df_train[target_col].values
        X_test = np.nan_to_num(df_test[feature_cols].values, nan=0.0)
        y_test = df_test[target_col].values
        
        # Train
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predict
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        overfit = (test_mae - train_mae) / train_mae * 100 if train_mae > 0 else 0
        
        baseline_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
        edge = (baseline_mae - test_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
        
        errors = np.abs(test_pred - y_test)
        signed_errors = test_pred - y_test
        
        df_test['prediction'] = test_pred
        df_test['actual'] = y_test
        df_test['error'] = errors
        df_test['signed_error'] = signed_errors
        
        print(f"  ✅ Test MAE: {test_mae:.3f}")
        
        # Categorizations
        # Game state
        if 'diff_at_2q_6min' in df_test.columns:
            current_diff_col = 'diff_at_2q_6min'
        elif 'current_diff' in df_test.columns:
            current_diff_col = 'current_diff'
        else:
            pattern_cols = [col for col in df_test.columns if 'pattern_' in col or 'diff_point' in col]
            if len(pattern_cols) > 0:
                current_diff_col = pattern_cols[-1]
            else:
                current_diff_col = None
        
        if current_diff_col:
            df_test['game_state'] = pd.cut(
                df_test[current_diff_col].abs(),
                bins=[0, 3, 7, 12, 20, 100],
                labels=['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)']
            )
            
            # Outcome type
            def categorize_outcome(row):
                current = row[current_diff_col]
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
        else:
            df_test['game_state'] = 'Unknown'
            df_test['outcome_type'] = 'Unknown'
        
        # Direction accuracy
        df_test['pred_winner'] = np.sign(df_test['prediction'])
        df_test['actual_winner'] = np.sign(df_test['actual'])
        df_test['direction_correct'] = df_test['pred_winner'] == df_test['actual_winner']
        direction_acc = df_test['direction_correct'].mean() * 100
        
        # Error bins
        bins = [0, 2, 5, 8, 12, 15, 20, 100]
        bin_labels = ['0-2 pts', '2-5 pts', '5-8 pts', '8-12 pts', '12-15 pts', '15-20 pts', '20+ pts']
        df_test['error_bin'] = pd.cut(df_test['error'], bins=bins, labels=bin_labels, include_lowest=True)
        
        # Quartiles
        quartiles = [0, 25, 50, 75, 90, 95, 100]
        error_percentiles = [np.percentile(errors, q) for q in quartiles]
        
        # Confidence zones
        high_conf = df_test[df_test['error'] <= 5]
        medium = df_test[(df_test['error'] > 5) & (df_test['error'] <= 12)]
        risk_zone = df_test[df_test['error'] > 15]
        
        # Calculate risk zone stats
        risk_avg_error = risk_zone['error'].mean() if len(risk_zone) > 0 else 0.0
        risk_direction_acc = risk_zone['direction_correct'].mean() * 100 if len(risk_zone) > 0 else 0.0
        risk_common_outcomes = '\n'.join([f'- {outcome}: {count} games ({count/len(risk_zone)*100:.1f}%)' 
                                          for outcome, count in risk_zone['outcome_type'].value_counts().head(3).items()]) if len(risk_zone) > 0 else ''
        
        # Generate markdown report
        report = f"""# 📊 {model_name} - COMPREHENSIVE BREAKDOWN

**Generated:** {datetime.now().strftime('%I:%M %p, %A %B %d, %Y')}  
**Model:** {model_name}  
**Test Games:** {len(df_test):,}

---

## 📋 MODEL SPECIFICATIONS

**Architecture:**
- Type: Linear Regression
- Features: {len(feature_cols)}
- Training: {len(X_train):,} games (80%, chronological)
- Testing: {len(X_test):,} games (20%, chronological)
- Scaler: RobustScaler

**Performance:**
- **Train MAE:** {train_mae:.3f}
- **Test MAE:** {test_mae:.3f}
- **Overfitting:** {overfit:.1f}%
- **Baseline MAE:** {baseline_mae:.3f}
- **Edge:** {edge:.1f}%

**Error Statistics:**
- Median Error: {np.median(errors):.3f}
- Std Dev: {np.std(errors):.3f}
- Min Error: {np.min(errors):.3f}
- Max Error: {np.max(errors):.3f}

---

## 📊 ERROR DISTRIBUTION

| Error Range | Games | Percentage | Stars |
|-------------|-------|------------|-------|
{chr(10).join([f'| {label} | {(df_test["error_bin"] == label).sum():,} | {(df_test["error_bin"] == label).sum()/len(df_test)*100:.1f}% | {"⭐" * min(3, int((df_test["error_bin"] == label).sum()/len(df_test)*20))} |' for label in bin_labels])}

**Cumulative Accuracy:**
```
Within  5 points: {(df_test['error'] <= 5).sum():,} games ({(df_test['error'] <= 5).sum()/len(df_test)*100:.1f}%)
Within  8 points: {(df_test['error'] <= 8).sum():,} games ({(df_test['error'] <= 8).sum()/len(df_test)*100:.1f}%)
Within 10 points: {(df_test['error'] <= 10).sum():,} games ({(df_test['error'] <= 10).sum()/len(df_test)*100:.1f}%)
Within 12 points: {(df_test['error'] <= 12).sum():,} games ({(df_test['error'] <= 12).sum()/len(df_test)*100:.1f}%)
Within 15 points: {(df_test['error'] <= 15).sum():,} games ({(df_test['error'] <= 15).sum()/len(df_test)*100:.1f}%)
Within 20 points: {(df_test['error'] <= 20).sum():,} games ({(df_test['error'] <= 20).sum()/len(df_test)*100:.1f}%)
```

---

## 🎯 PERFORMANCE BY GAME TYPE

### By Game State (at Q2 6:00):

| Game State | MAE | Median | Games | % |
|------------|-----|--------|-------|---|
{chr(10).join([f"| {state} | {df_test[df_test['game_state']==state]['error'].mean():.3f} | {df_test[df_test['game_state']==state]['error'].median():.3f} | {(df_test['game_state']==state).sum():,} | {(df_test['game_state']==state).sum()/len(df_test)*100:.1f}% |" for state in ['Very Close (≤3)', 'Close (4-7)', 'Moderate (8-12)', 'Large Lead (13-20)', 'Blowout (>20)'] if (df_test['game_state']==state).sum() > 0])}

### By Outcome Type:

| Outcome | MAE | Median | Games | % |
|---------|-----|--------|-------|---|
{chr(10).join([f"| {outcome} | {df_test[df_test['outcome_type']==outcome]['error'].mean():.3f} | {df_test[df_test['outcome_type']==outcome]['error'].median():.3f} | {(df_test['outcome_type']==outcome).sum():,} | {(df_test['outcome_type']==outcome).sum()/len(df_test)*100:.1f}% |" for outcome in df_test['outcome_type'].value_counts().index])}

---

## 🧭 DIRECTIONAL ACCURACY

**Overall Performance:**
- ✅ Correct Direction: {df_test['direction_correct'].sum():,} games ({direction_acc:.1f}%)
- ❌ Wrong Direction: {(~df_test['direction_correct']).sum():,} games ({100-direction_acc:.1f}%)

**MAE by Direction:**
- When Correct: {df_test[df_test['direction_correct']]['error'].mean():.3f} MAE
- When Wrong: {df_test[~df_test['direction_correct']]['error'].mean():.3f} MAE

---

## 📈 ERROR QUARTILES

| Percentile | Error Threshold |
|------------|----------------|
{chr(10).join([f'| {q}th | {np.percentile(errors, q):.2f} points |' for q in quartiles])}

**Interpretation:**
- Best 25%: Error ≤ {error_percentiles[1]:.1f} points ⭐⭐⭐
- Middle 50%: Error {error_percentiles[1]:.1f} - {error_percentiles[3]:.1f} points ⭐⭐
- 75-90%: Error {error_percentiles[3]:.1f} - {error_percentiles[4]:.1f} points ⭐
- Worst 5%: Error > {error_percentiles[5]:.1f} points ❌

---

## 🎯 CONFIDENCE ZONES

### HIGH-CONFIDENCE (Error ≤ 5 points):
- **Games:** {len(high_conf):,} ({len(high_conf)/len(df_test)*100:.1f}%)
- **Avg Error:** {high_conf['error'].mean():.3f}
- **Direction Accuracy:** {high_conf['direction_correct'].mean()*100:.1f}%

**Top Game States:**
{chr(10).join([f'- {state}: {(high_conf["game_state"]==state).sum()} games ({(high_conf["game_state"]==state).sum()/len(high_conf)*100:.1f}%)' for state in high_conf['game_state'].value_counts().head(3).index])}

### MEDIUM ZONE (Error 5-12 points):
- **Games:** {len(medium):,} ({len(medium)/len(df_test)*100:.1f}%)
- **Avg Error:** {medium['error'].mean():.3f}
- **Direction Accuracy:** {medium['direction_correct'].mean()*100:.1f}%

### RISK ZONE (Error > 15 points):
- **Games:** {len(risk_zone):,} ({len(risk_zone)/len(df_test)*100:.1f}%)
- **Avg Error:** {risk_avg_error:.3f}
- **Direction Accuracy:** {risk_direction_acc:.1f}%

{risk_common_outcomes}

---

## 📊 BIAS ANALYSIS

**Overall Bias:**
- Mean Signed Error: {np.mean(signed_errors):+.3f}
- Median Signed Error: {np.median(signed_errors):+.3f}

{f"✅ **UNBIASED:** Model doesn't systematically over/under-predict" if abs(np.mean(signed_errors)) < 1.0 else f"⚠️ **BIAS DETECTED:** Model {'over-predicts' if np.mean(signed_errors) > 0 else 'under-predicts'} by {abs(np.mean(signed_errors)):.3f} points"}

**Bias by Game State:**
{chr(10).join([f'- {state}: {df_test[df_test["game_state"]==state]["signed_error"].mean():+.3f} pts' for state in df_test['game_state'].dropna().unique() if (df_test['game_state']==state).sum() >= 5])}

---

## ✅ WHERE MODEL EXCELS

**Best Performance (Lowest MAE):**
{chr(10).join([f'{i}. {state}: {df_test[df_test["game_state"]==state]["error"].mean():.3f} MAE ({(df_test["game_state"]==state).sum():,} games)' for i, state in enumerate(df_test.groupby('game_state')['error'].mean().nsmallest(3).index, 1)])}

---

## ⚠️ WHERE MODEL STRUGGLES

**Worst Performance (Highest MAE):**
{chr(10).join([f'{i}. {state}: {df_test[df_test["game_state"]==state]["error"].mean():.3f} MAE ({(df_test["game_state"]==state).sum():,} games)' for i, state in enumerate(df_test.groupby('game_state')['error'].mean().nlargest(3).index, 1)])}

---

## 🏆 TOP 10 BEST PREDICTIONS

| Rank | Predicted | Actual | Error |
|------|-----------|--------|-------|
{chr(10).join([f'| {i} | {row["prediction"]:+.1f} | {row["actual"]:+.1f} | {row["error"]:.2f} |' for i, (_, row) in enumerate(df_test.nsmallest(10, 'error').iterrows(), 1)])}

---

## 💀 TOP 10 WORST PREDICTIONS

| Rank | Predicted | Actual | Error | Type |
|------|-----------|--------|-------|------|
{chr(10).join([f'| {i} | {row["prediction"]:+.1f} | {row["actual"]:+.1f} | {row["error"]:.1f} | {row["outcome_type"]} |' for i, (_, row) in enumerate(df_test.nlargest(10, 'error').iterrows(), 1)])}

---

## 💰 BETTING STRATEGY RECOMMENDATIONS

**Recommended Thresholds:**

| Strategy | Edge Threshold | Games | % of Season | Direction Acc |
|----------|----------------|-------|-------------|---------------|
| Aggressive | ≥3 pts | - | - | - |
| Balanced | ≥5 pts | - | - | - |
| Conservative | ≥7 pts | - | - | - |

---

## 💡 KEY INSIGHTS

1. **Direction Accuracy:** {direction_acc:.1f}% predict the right winner
2. **High-Confidence Rate:** {len(high_conf)/len(df_test)*100:.1f}% of games within 5 points
3. **Median Error:** {np.median(errors):.1f} points
4. **Risk Rate:** {len(risk_zone)/len(df_test)*100:.1f}% of games with error > 15 points
5. **Bias:** {abs(np.mean(signed_errors)):.2f} points ({'negligible' if abs(np.mean(signed_errors)) < 1.0 else 'moderate'})

---

## 📁 FEATURE LIST

<details>
<summary>Click to expand all {len(feature_cols)} features</summary>

{chr(10).join([f'{i}. {feat}' for i, feat in enumerate(feature_cols, 1)])}

</details>

---

**BREAKDOWN COMPLETE** ✅  
**Model:** {model_name}  
**Test MAE:** {test_mae:.3f}  
**Edge:** {edge:.1f}%  
**Ready for deployment:** {'✅ YES' if test_mae < 10.0 and overfit < 10 else '⚠️ NEEDS REVIEW'}
"""
        
        # Save markdown
        filename = f"{output_dir}/{model_name.replace(' ', '_')}_BREAKDOWN.md"
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"  ✅ Saved to: {filename}")
        print()
        
        return {
            'model_name': model_name,
            'test_mae': test_mae,
            'overfit': overfit,
            'edge': edge,
            'direction_acc': direction_acc,
            'high_conf_pct': len(high_conf)/len(df_test)*100,
            'risk_pct': len(risk_zone)/len(df_test)*100
        }
        
    except Exception as e:
        print(f"  ❌ Error analyzing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================
# ANALYZE ALL MODELS
# =============================

print("=" * 100)
print("🔥 GENERATING COMPREHENSIVE BREAKDOWNS FOR ALL MODELS")
print("=" * 100)
print()

models_to_analyze = [
    ("73-Feature Linear (V3 Data)", 
     None, 
     "Action/ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl"),
    
    ("18-Feature Real Data System", 
     "Action/COMPLETE_REAL_DATA_SYSTEM.pkl", 
     "Action/COMPLETE_REAL_DATA_SYSTEM.pkl"),
    
    ("Elon Mode Complete System", 
     "Action/ELON_MODE_COMPLETE_SYSTEM.pkl", 
     "Action/ELON_MODE_COMPLETE_SYSTEM.pkl"),
]

results = []

for model_name, model_file, data_file in models_to_analyze:
    if os.path.exists(data_file):
        result = generate_model_breakdown(model_name, model_file, data_file)
        if result:
            results.append(result)
    else:
        print(f"⚠️  Skipping {model_name} (file not found)")
        print()

# Generate summary comparison
print("=" * 100)
print("📊 SUMMARY COMPARISON OF ALL MODELS")
print("=" * 100)
print()

if len(results) > 0:
    print("| Model | Test MAE | Overfit | Edge | Direction Acc | High Conf % |")
    print("|-------|----------|---------|------|---------------|-------------|")
    for r in results:
        print(f"| {r['model_name'][:30]:30s} | {r['test_mae']:8.3f} | {r['overfit']:6.1f}% | {r['edge']:5.1f}% | {r['direction_acc']:5.1f}% | {r['high_conf_pct']:5.1f}% |")
    print()

# Save summary
summary = f"""# 📊 ALL MODELS COMPARISON

**Generated:** {datetime.now().strftime('%I:%M %p, %A %B %d, %Y')}

---

## 📊 PERFORMANCE COMPARISON

| Model | Test MAE | Overfit | Edge | Direction Acc | High Conf % | Risk % |
|-------|----------|---------|------|---------------|-------------|--------|
{chr(10).join([f"| {r['model_name'][:30]} | {r['test_mae']:.3f} | {r['overfit']:.1f}% | {r['edge']:.1f}% | {r['direction_acc']:.1f}% | {r['high_conf_pct']:.1f}% | {r['risk_pct']:.1f}% |" for r in results])}

---

## 🏆 RANKINGS

**Best Test MAE:**
{chr(10).join([f'{i}. {r["model_name"]}: {r["test_mae"]:.3f}' for i, r in enumerate(sorted(results, key=lambda x: x['test_mae']), 1)])}

**Lowest Overfitting:**
{chr(10).join([f'{i}. {r["model_name"]}: {r["overfit"]:.1f}%' for i, r in enumerate(sorted(results, key=lambda x: x['overfit']), 1)])}

**Highest Edge:**
{chr(10).join([f'{i}. {r["model_name"]}: {r["edge"]:.1f}%' for i, r in enumerate(sorted(results, key=lambda x: x['edge'], reverse=True), 1)])}

**Best Direction Accuracy:**
{chr(10).join([f'{i}. {r["model_name"]}: {r["direction_acc"]:.1f}%' for i, r in enumerate(sorted(results, key=lambda x: x['direction_acc'], reverse=True), 1)])}

---

**ALL MODEL BREAKDOWNS COMPLETE** ✅
"""

with open("Action/Model_Breakdowns/📊_ALL_MODELS_SUMMARY.md", 'w') as f:
    f.write(summary)

print("✅ Saved summary to: Action/Model_Breakdowns/📊_ALL_MODELS_SUMMARY.md")
print()

print("=" * 100)
print("✅ ALL MODEL BREAKDOWNS COMPLETE!")
print("=" * 100)
print()
print(f"Generated {len(results)} comprehensive model breakdowns")
print(f"Location: Action/Model_Breakdowns/")
print()

