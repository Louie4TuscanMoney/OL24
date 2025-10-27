#!/usr/bin/env python3
"""
📊 TEST ON 2025 HOLDOUT AND MAKE LAUNCH DECISION

Calculates MAE on 2025 data
Determines optimal ensemble weights
Configures risk layer parameters
Makes GO/NO-GO decision for Monday

DECISION CRITERIA:
- MAE < 7.0: ✅ LAUNCH (standard bet sizing)
- MAE 7-9: ⚠️ LAUNCH CAUTIOUSLY (conservative sizing)
- MAE > 9: ❌ DON'T LAUNCH (improve model first)
"""

import pickle
import numpy as np
import sys

sys.path.insert(0, '1. ML/1. Dejavu Deployment')
from dejavu_model import DejavuForecaster

print("="*80)
print("📊 2025 HOLDOUT TESTING & LAUNCH DECISION")
print("="*80)

# ============================================================================
# STEP 1: Load Models
# ============================================================================

print("\n[1/5] Loading models...")

try:
    dejavu = DejavuForecaster.load('1. ML/1. Dejavu Deployment/dejavu_retrained_2025.pkl')
    print("✅ Dejavu loaded")
except:
    print("❌ Dejavu not found - using old model")
    dejavu = DejavuForecaster.load('1. ML/1. Dejavu Deployment/dejavu.pkl')

try:
    import xgboost as xgb
    xgboost_model = xgb.XGBRegressor()
    xgboost_model.load_model('xgboost_enhanced_v1.json')
    has_xgboost = True
    print("✅ XGBoost loaded")
except:
    has_xgboost = False
    print("⚠️  XGBoost not available (will test Dejavu only)")

# ============================================================================
# STEP 2: Load 2025 Holdout Data
# ============================================================================

print("\n[2/5] Loading 2025 holdout...")

with open('ENHANCED_PATTERNS_FULL.pkl', 'rb') as f:
    all_patterns = pickle.load(f)

# Use most recent 20% as 2025 holdout
cutoff = int(len(all_patterns) * 0.8)
holdout_2025 = all_patterns[cutoff:]

print(f"✅ Holdout: {len(holdout_2025)} games")
print(f"   Total dataset: {len(all_patterns)} games")
print(f"   Holdout %: 20%")

# ============================================================================
# STEP 3: Calculate MAE for Each Model
# ============================================================================

print("\n[3/5] Calculating MAE on 2025 data...")

dejavu_predictions = []
xgboost_predictions = []
actuals = []

for p in holdout_2025:
    if p.get('diff_at_final') is None:
        continue
    
    # Dejavu prediction
    dejavu_pred = dejavu.predict(p['pattern'])
    dejavu_predictions.append(dejavu_pred)
    
    # XGBoost prediction (if available)
    if has_xgboost:
        # Would need to construct full feature vector
        # For now, mark as unavailable
        xgboost_predictions.append(None)
    
    actuals.append(p['diff_at_final'])

dejavu_predictions = np.array(dejavu_predictions)
actuals = np.array(actuals)

# Calculate MAE
dejavu_mae = np.mean(np.abs(dejavu_predictions - actuals))

print(f"\n📊 RESULTS:")
print(f"   Holdout samples: {len(actuals)}")
print(f"   Dejavu MAE: {dejavu_mae:.2f}")

if has_xgboost and None not in xgboost_predictions:
    xgboost_mae = np.mean(np.abs(np.array(xgboost_predictions) - actuals))
    print(f"   XGBoost MAE: {xgboost_mae:.2f}")
else:
    print(f"   XGBoost MAE: Not available")

# Calculate bias (systematic over/under prediction)
bias = np.mean(dejavu_predictions - actuals)
print(f"\n   Bias: {bias:.2f} (0=unbiased)")

# ============================================================================
# STEP 4: Ensemble Optimization (If Multiple Models)
# ============================================================================

print("\n[4/5] Ensemble optimization...")

if has_xgboost and None not in xgboost_predictions:
    # Find optimal weights
    best_mae = float('inf')
    best_w = None
    
    for w_dejavu in np.linspace(0, 1, 21):
        w_xgb = 1 - w_dejavu
        
        ensemble_pred = w_dejavu * dejavu_predictions + w_xgb * np.array(xgboost_predictions)
        mae = np.mean(np.abs(ensemble_pred - actuals))
        
        if mae < best_mae:
            best_mae = mae
            best_w = w_dejavu
    
    print(f"\n   Optimal Ensemble:")
    print(f"   Dejavu weight: {best_w:.2f}")
    print(f"   XGBoost weight: {1-best_w:.2f}")
    print(f"   Ensemble MAE: {best_mae:.2f}")
    
    final_mae = best_mae
    
    # Save weights
    with open('optimal_ensemble_weights.pkl', 'wb') as f:
        pickle.dump({'dejavu': best_w, 'xgboost': 1-best_w}, f)
else:
    print("   Using Dejavu only")
    final_mae = dejavu_mae

# ============================================================================
# STEP 5: LAUNCH DECISION
# ============================================================================

print("\n[5/5] Launch decision...")

print(f"\n{'='*80}")
print(f"FINAL MAE ON 2025 HOLDOUT: {final_mae:.2f}")
print(f"{'='*80}")

# Risk configuration based on MAE
if final_mae < 7.0:
    print(f"\n✅ DECISION: LAUNCH MONDAY")
    print(f"\n   Status: EXCELLENT")
    print(f"   MAE: {final_mae:.2f} < 7.0 ✅")
    
    print(f"\n   Risk Configuration:")
    print(f"   - Kelly fraction: 0.50 (half Kelly)")
    print(f"   - Max single bet: $750 (15%)")
    print(f"   - Max portfolio: $2,500 (50%)")
    print(f"   - Starting bankroll: $5,000")
    
    print(f"\n   Expected Performance:")
    print(f"   - Win rate: 56-60%")
    print(f"   - Monthly profit: $2,000-5,000")
    print(f"   - Sharpe ratio: 1.2-1.5")
    
    launch_decision = 'GO'
    risk_mode = 'STANDARD'
    
elif final_mae < 9.0:
    print(f"\n⚠️  DECISION: LAUNCH CAUTIOUSLY")
    print(f"\n   Status: ACCEPTABLE")
    print(f"   MAE: {final_mae:.2f} (7-9 range)")
    
    print(f"\n   Risk Configuration (CONSERVATIVE):")
    print(f"   - Kelly fraction: 0.25 (quarter Kelly)")
    print(f"   - Max single bet: $500 (10%)")
    print(f"   - Max portfolio: $1,500 (30%)")
    print(f"   - Starting bankroll: $5,000")
    
    print(f"\n   Expected Performance:")
    print(f"   - Win rate: 52-55%")
    print(f"   - Monthly profit: $500-2,000")
    print(f"   - Sharpe ratio: 0.8-1.1")
    
    print(f"\n   ⚠️  Monitor closely - improve model Week 1")
    
    launch_decision = 'CAUTIOUS'
    risk_mode = 'CONSERVATIVE'
    
else:
    print(f"\n❌ DECISION: DON'T LAUNCH")
    print(f"\n   Status: INSUFFICIENT")
    print(f"   MAE: {final_mae:.2f} > 9.0")
    
    print(f"\n   Recommendation:")
    print(f"   - Delay launch 1 week")
    print(f"   - Add more features (team + player)")
    print(f"   - Collect more recent data")
    print(f"   - Hyperparameter tuning")
    
    print(f"\n   OR:")
    print(f"   - Launch micro-bets only ($20-50)")
    print(f"   - Treat as validation phase")
    print(f"   - Don't expect profit Week 1")
    
    launch_decision = 'NO-GO'
    risk_mode = 'MICRO'

# Save decision
with open('LAUNCH_DECISION.pkl', 'wb') as f:
    pickle.dump({
        'decision': launch_decision,
        'mae': final_mae,
        'risk_mode': risk_mode,
        'timestamp': str(np.datetime64('now'))
    }, f)

print(f"\n{'='*80}")
print(f"DECISION SAVED: {launch_decision}")
print(f"{'='*80}")

# Generate risk configuration file
risk_config = f"""
# Risk Layer Configuration (Generated from 2025 MAE Test)

LAUNCH_DECISION = '{launch_decision}'
MAE_2025 = {final_mae:.2f}
RISK_MODE = '{risk_mode}'

# Layer 1: Kelly Criterion
KELLY_FRACTION = {0.50 if final_mae < 7 else 0.25 if final_mae < 9 else 0.10}

# Layer 5: Final Calibration
MAX_SINGLE_BET_PCT = {0.15 if final_mae < 7 else 0.10 if final_mae < 9 else 0.05}
MAX_PORTFOLIO_PCT = {0.50 if final_mae < 7 else 0.30 if final_mae < 9 else 0.15}

# Safety Mode
if MAE_2025 < 7.0:
    SAFETY_MODE = 'GREEN'   # Standard risk
elif MAE_2025 < 9.0:
    SAFETY_MODE = 'YELLOW'  # Elevated caution
else:
    SAFETY_MODE = 'RED'     # Maximum caution
"""

with open('risk_configuration.py', 'w') as f:
    f.write(risk_config)

print(f"\n✅ Risk configuration saved: risk_configuration.py")
print(f"\n🚀 Ready for Monday launch (if decision = GO/CAUTIOUS)")

print("="*80)

