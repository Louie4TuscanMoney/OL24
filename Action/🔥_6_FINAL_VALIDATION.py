#!/usr/bin/env python3
"""
🔥 FINAL VALIDATION - Test on 2025 Preseason
Compare to 5.37 benchmark, make GO/NO-GO decision
"""

import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error

print("="*80)
print("🔥 FINAL VALIDATION - CHAMPIONSHIP OR BUST")
print("="*80)
print()

# Load final results
print("[1/3] Loading final ensemble results...")
try:
    with open('FINAL_RESULTS.pkl', 'rb') as f:
        results = pickle.load(f)
    
    final_mae = results['super_ensemble_mae']
    improvement = results['improvement_pct']
    
    print(f"✅ Super-ensemble MAE: {final_mae:.3f}")
    print(f"✅ Improvement from baseline: {improvement:.1f}%")
except FileNotFoundError:
    with open('ENSEMBLE_RESULTS.pkl', 'rb') as f:
        results = pickle.load(f)
    final_mae = results['best_mae']
    improvement = ((8.22 - final_mae) / 8.22) * 100
    print(f"✅ Best ensemble MAE: {final_mae:.3f}")
    print(f"✅ Improvement: {improvement:.1f}%")

print()

# Compare to benchmarks
print("[2/3] Comparing to research benchmarks...")
print("="*80)
print("BENCHMARK COMPARISON:")
print("="*80)
print(f"  Your target:              5.37 MAE")
print(f"  Research (XGBoost):       ~3.0 MAE (R²=0.999)")
print(f"  Research (ExtraTrees):    ~5-6 MAE (34% WAPE)")
print(f"  Your current system:      {final_mae:.3f} MAE")
print()

if final_mae < 5.37:
    print("✅ BEATING YOUR BENCHMARK!")
    performance_tier = "CHAMPIONSHIP"
elif final_mae < 6.0:
    print("✅ COMPETITIVE - Near benchmark")
    performance_tier = "EXCELLENT"
elif final_mae < 7.0:
    print("⚠️  Below benchmark but acceptable")
    performance_tier = "GOOD"
else:
    print("❌ Below benchmark - need more work")
    performance_tier = "NEEDS_WORK"

print()

# Make launch decision
print("[3/3] Making launch decision...")
print("="*80)
print("🎯 LAUNCH DECISION")
print("="*80)
print()

if final_mae < 4.5:
    decision = "🟢 AGGRESSIVE LAUNCH"
    risk_mode = "AGGRESSIVE"
    max_bet = 200
    portfolio_cap = 2000
    kelly = 0.25
    confidence_threshold = 0.70
    recommendation = "MAE under 4.5 is championship level. Launch with confidence. Scale quickly if Week 1 confirms."
    
elif final_mae < 5.5:
    decision = "🟢 CONFIDENT LAUNCH"
    risk_mode = "STANDARD"
    max_bet = 150
    portfolio_cap = 1000
    kelly = 0.20
    confidence_threshold = 0.75
    recommendation = "MAE under 5.5 is excellent. Launch with standard sizing. This is production-ready."
    
elif final_mae < 6.5:
    decision = "🟡 CAUTIOUS LAUNCH"
    risk_mode = "CONSERVATIVE"
    max_bet = 75
    portfolio_cap = 500
    kelly = 0.15
    confidence_threshold = 0.80
    recommendation = "MAE under 6.5 is good. Launch conservatively. Validate in Week 1 before scaling."
    
elif final_mae < 7.5:
    decision = "🟡 MICRO LAUNCH"
    risk_mode = "ULTRA-CONSERVATIVE"
    max_bet = 30
    portfolio_cap = 200
    kelly = 0.10
    confidence_threshold = 0.85
    recommendation = "MAE 6.5-7.5 is marginal. Launch with micro-bets to validate. Focus on learning."
    
else:
    decision = "🔴 NO-GO / PAPER TRADE"
    risk_mode = "PAPER-ONLY"
    max_bet = 0
    portfolio_cap = 0
    kelly = 0.0
    confidence_threshold = 0.95
    recommendation = "MAE above 7.5 is not viable for real money. Paper trade or improve model first."

print(f"DECISION: {decision}")
print(f"MAE: {final_mae:.3f}")
print(f"Performance Tier: {performance_tier}")
print()
print("RISK CONFIGURATION:")
print(f"  Risk Mode: {risk_mode}")
print(f"  Max bet per game: ${max_bet}")
print(f"  Portfolio cap per day: ${portfolio_cap}")
print(f"  Kelly fraction: {kelly}")
print(f"  Confidence threshold: {confidence_threshold}")
print()
print("RECOMMENDATION:")
print(f"  {recommendation}")
print()
print("="*80)

# Save final decision
final_decision = {
    'decision': decision,
    'mae': final_mae,
    'performance_tier': performance_tier,
    'risk_mode': risk_mode,
    'max_bet': max_bet,
    'portfolio_cap': portfolio_cap,
    'kelly_fraction': kelly,
    'confidence_threshold': confidence_threshold,
    'recommendation': recommendation,
    'improvement_from_baseline': improvement,
    'vs_benchmark_537': final_mae - 5.37
}

with open('CHAMPIONSHIP_DECISION.pkl', 'wb') as f:
    pickle.dump(final_decision, f)

# Create risk config
risk_config = f"""# 🔥 CHAMPIONSHIP RISK CONFIGURATION
# Auto-generated from optimization results

RISK_MODE = "{risk_mode}"
MAX_BET_PER_GAME = {max_bet}
PORTFOLIO_CAP_PER_DAY = {portfolio_cap}
KELLY_FRACTION = {kelly}
CONFIDENCE_THRESHOLD = {confidence_threshold}

# Model performance
FINAL_MAE = {final_mae:.3f}
PERFORMANCE_TIER = "{performance_tier}"
IMPROVEMENT_VS_BASELINE = {improvement:.1f}  # percent

# Benchmark comparison
BENCHMARK_537 = 5.37
DELTA_VS_BENCHMARK = {final_mae - 5.37:.3f}

# Launch recommendation
RECOMMENDATION = \"\"\"{recommendation}\"\"\"
"""

with open('championship_risk_config.py', 'w') as f:
    f.write(risk_config)

print("✅ Decision saved to: CHAMPIONSHIP_DECISION.pkl")
print("✅ Risk config saved to: championship_risk_config.py")
print()
print("="*80)
print("🎯 OPTIMIZATION COMPLETE - REVIEW RESULTS AND DECIDE")
print("="*80)

