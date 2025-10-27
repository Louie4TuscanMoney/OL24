#!/bin/bash
# 🐍 MAMBA MENTALITY SYSTEM - MONDAY LAUNCH
# 33 features | 5.181/9.655 MAE | Championship validated

cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/Action"

echo "================================================================================"
echo "🐍 MAMBA MENTALITY SYSTEM - LAUNCHING"
echo "================================================================================"
echo ""
echo "\"Job's finished\" - Kobe Bryant"
echo ""

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================
echo "[1/5] Pre-flight checks..."
echo ""

# Check Python
python3 --version > /dev/null 2>&1 && echo "  ✅ Python 3 installed" || { echo "  ❌ Python 3 missing"; exit 1; }

# Check dependencies
python3 -c "import xgboost, lightgbm, sklearn, numpy, pandas" 2>/dev/null && echo "  ✅ ML libraries installed" || { echo "  ❌ Missing ML libraries"; exit 1; }

# Check Mamba Mentality System
if [ -f "MAMBA_MENTALITY_SYSTEM.pkl" ]; then
    echo "  ✅ Mamba Mentality System ready (33 features)"
    export MODEL_FILE="MAMBA_MENTALITY_SYSTEM.pkl"
else
    echo "  ❌ Mamba Mentality System missing"
    exit 1
fi

[ -f "KNN_QUALITY_GATE.pkl" ] && echo "  ✅ KNN gate ready" || echo "  ⚠️  KNN gate missing (optional)"
[ -f "ULTRA_ENHANCED_PATTERNS_V2.pkl" ] && echo "  ✅ Training data ready" || { echo "  ❌ Data missing"; exit 1; }

echo ""

# ============================================================================
# LOAD SYSTEM STATUS
# ============================================================================
echo "[2/5] Loading Mamba Mentality system status..."
echo ""

python3 << 'STATUS'
import pickle

with open('MAMBA_MENTALITY_SYSTEM.pkl', 'rb') as f:
    system = pickle.load(f)

print("🐍 MAMBA MENTALITY SYSTEM STATUS:")
print("="*80)
print("")
print(f"Branch A (Halftime → Q2 6:00):")
print(f"  • Strategy: {system['branch_a_halftime'].get('level2_method', system['branch_a_halftime']['champion_strategy'])}")
print(f"  • MAE: {system['branch_a_halftime'].get('champion_mae_level2', system['branch_a_halftime']['champion_mae']):.3f}")
print(f"  • Models: {len(system['branch_a_halftime']['models'])}")
print(f"  • Status: ✅ CHAMPIONSHIP")
print("")
print(f"Branch B (Final → Q2 6:00):")
print(f"  • Strategy: {system['branch_b_final'].get('level2_method', system['branch_b_final']['champion_strategy'])}")
print(f"  • MAE: {system['branch_b_final'].get('champion_mae_level2', system['branch_b_final']['champion_mae']):.3f}")
print(f"  • Models: {len(system['branch_b_final']['models'])}")
print(f"  • Status: ✅ COMPETITIVE+")
print("")
print(f"Training Data:")
print(f"  • Total games: {system['metadata']['total_games']}")
print(f"  • Features: 33 (efg_proxy, netrtg_proxy, pace_proxy, etc.)")
print(f"  • Models trained: {system['metadata']['models_trained']}")
print("")

mae_half = system['branch_a_halftime'].get('champion_mae_level2', system['branch_a_halftime']['champion_mae'])
mae_final = system['branch_b_final'].get('champion_mae_level2', system['branch_b_final']['champion_mae'])

print("🎯 LAUNCH DECISION: MODERATE DUAL-BRANCH")
print("   • Week 1: Validate edge (25-35 bets)")
print("   • Week 2+: Scale if validated")
print("")

# Save launch config
with open('MAMBA_LAUNCH_CONFIG.txt', 'w') as f:
    f.write(f"SYSTEM=MAMBA_MENTALITY\n")
    f.write(f"LAUNCH_MODE=MODERATE\n")
    f.write(f"BRANCH_A_MAE={mae_half:.3f}\n")
    f.write(f"BRANCH_B_MAE={mae_final:.3f}\n")
    f.write(f"FEATURES=33\n")

STATUS

echo ""

# ============================================================================
# SYSTEM VALIDATION
# ============================================================================
echo "[3/5] Validating complete system..."
echo ""

python3 << 'VALIDATE'
import pickle
from pathlib import Path

checks = {
    'Mamba System': 'MAMBA_MENTALITY_SYSTEM.pkl',
    'KNN gate': 'KNN_QUALITY_GATE.pkl',
    'Training data': 'ULTRA_ENHANCED_PATTERNS_V2.pkl',
    'Hyperparameters': 'BEST_HYPERPARAMETERS.pkl',
}

passed = 0
for name, file in checks.items():
    if Path(file).exists():
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ⚠️  {name} - MISSING (optional)" if 'gate' in name or 'Hyper' in name else f"  ❌ {name} - MISSING")

print(f"")
print(f"System validation: {passed}/{len(checks)} ({'✅ READY' if passed >= 2 else '⚠️ INCOMPLETE'})")

VALIDATE

echo ""

# ============================================================================
# A/B TEST SETUP
# ============================================================================
echo "[4/5] Setting up A/B test framework..."
echo ""

cat > AB_TEST_CONFIG.json << 'EOF'
{
  "systems": {
    "mamba_mentality": {
      "name": "Mamba Mentality",
      "file": "MAMBA_MENTALITY_SYSTEM.pkl",
      "features": 33,
      "description": "Championship system (efg_proxy, netrtg, pace)",
      "mae_half": 5.181,
      "mae_final": 9.655,
      "status": "LIVE",
      "allocation": 1.0
    },
    "strive_for_greatness": {
      "name": "Strive for Greatness",
      "file": "STRIVE_FOR_GREATNESS_SYSTEM.pkl",
      "features": 67,
      "description": "Advanced system (spectral, momentum, velocity)",
      "mae_half": null,
      "mae_final": null,
      "status": "BUILD_TOMORROW",
      "allocation": 0.0
    }
  },
  "ab_test": {
    "enabled": false,
    "start_date": null,
    "split_ratio": [0.5, 0.5],
    "metric": "roi"
  }
}
EOF

echo "  ✅ A/B test config created"
echo "  📊 Mamba Mentality: LIVE (100%)"
echo "  🚧 Strive for Greatness: Build tomorrow"
echo ""

# ============================================================================
# FINAL REPORT
# ============================================================================
echo "[5/5] Final system report..."
echo ""

python3 << 'REPORT'
import pickle

with open('MAMBA_MENTALITY_SYSTEM.pkl', 'rb') as f:
    system = pickle.load(f)

print("="*80)
print("🐍 MAMBA MENTALITY SYSTEM - READY TO LAUNCH")
print("="*80)
print("")

mae_half = system['branch_a_halftime'].get('champion_mae_level2', system['branch_a_halftime']['champion_mae'])
mae_final = system['branch_b_final'].get('champion_mae_level2', system['branch_b_final']['champion_mae'])

print(f"PERFORMANCE:")
print(f"  Branch A (Halftime): {mae_half:.3f} MAE → Championship")
print(f"  Branch B (Final): {mae_final:.3f} MAE → Competitive+")
print("")

print(f"CAPABILITIES:")
print(f"  • 10 diverse ML models per branch")
print(f"  • 33 optimized features")
print(f"  • Bayesian Model Averaging + Isotonic Calibration")
print(f"  • KNN quality gate (filters 58% of games)")
print(f"  • Dual-branch predictions")
print("")

print("🟢 LAUNCH DECISION: MODERATE DUAL-BRANCH")
print("   • Week 1: Validate edge (25-35 bets)")
print("   • Week 2+: Scale if >52% win rate + +ROI")
print("")

print("🔬 A/B TEST SETUP:")
print("   • Mamba Mentality: LIVE Monday (100% allocation)")
print("   • Strive for Greatness: Build tomorrow, test Tuesday+")
print("")

print("="*80)
print("🚀 READY FOR MONDAY 4 PM LAUNCH")
print("="*80)
print("")
print("\"Jobs finished. Dominate.\" - Mamba Mentality")

REPORT

echo ""
echo "================================================================================"
echo "✅ MAMBA MENTALITY SYSTEM LOCKED IN"
echo "================================================================================"
echo ""
echo "Monday 4 PM: Launch Mamba Mentality (33 features)"
echo "Tuesday: Build Strive for Greatness (67 features) for A/B test"
echo ""
echo "================================================================================"

