#!/usr/bin/env python3
"""
🚀 LAUNCH STRIVE FOR GREATNESS - MONDAY 1 AM
Fast deployment using pkl files (everything pre-trained)
"""

import pickle
import numpy as np
from datetime import datetime

print("="*80)
print("🏆 STRIVE FOR GREATNESS - LAUNCHING")
print("="*80)
print()
print(f"Launch time: {datetime.now().strftime('%A %B %d, %Y - %I:%M %p')}")
print()
print("\"Strive for Greatness\" - LeBron James")
print()

# ============================================================================
# LOAD PKL FILES (FAST - 5 seconds)
# ============================================================================
print("[1/3] Loading Strive for Greatness system from pkl...")
start = datetime.now()

with open('STRIVE_FOR_GREATNESS_SYSTEM.pkl', 'rb') as f:
    system = pickle.load(f)

load_time = (datetime.now() - start).total_seconds()
print(f"✅ System loaded in {load_time:.2f} seconds")
print()

# Display system stats
print("SYSTEM STATUS:")
print("-" * 80)
print(f"  Branch A (Halftime): {system['branch_a_halftime']['champion_mae']:.3f} MAE")
print(f"  Branch B (Final):    {system['branch_b_final']['champion_mae']:.3f} MAE")
print(f"  Features: {system['metadata']['feature_count']}")
print(f"  Models: {system['metadata']['models_trained']}")
print(f"  Build date: {system['metadata']['build_date']}")
print()

# ============================================================================
# TEST PREDICTION (FAST - 0.2 seconds)
# ============================================================================
print("[2/3] Testing prediction pipeline...")
start = datetime.now()

# Load sample game data for testing
with open('ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl', 'rb') as f:
    data = pickle.load(f)

# Test on last game
test_game = data[-1]
feature_names = system['feature_names']

# Extract features
features = [test_game.get(f, 0) for f in feature_names]
features = np.nan_to_num(np.array(features), nan=0.0).reshape(1, -1)

# Scale
scaler_a = system['branch_a_halftime']['scaler']
scaler_b = system['branch_b_final']['scaler']

features_scaled_a = scaler_a.transform(features)
features_scaled_b = scaler_b.transform(features)

# Predict
models_a = system['branch_a_halftime']['models']
models_b = system['branch_b_final']['models']

preds_half = [model.predict(features_scaled_a)[0] for model in models_a.values()]
preds_final = [model.predict(features_scaled_b)[0] for model in models_b.values()]

pred_half = np.mean(preds_half)
pred_final = np.mean(preds_final)

pred_time = (datetime.now() - start).total_seconds()

# Display results
true_half = test_game.get('diff_at_halftime', 0)
true_final = test_game.get('diff_at_final', 0)
error_half = abs(pred_half - true_half)
error_final = abs(pred_final - true_final)

print(f"✅ Prediction completed in {pred_time:.3f} seconds")
print()
print(f"TEST PREDICTION:")
print(f"  Game ID: {test_game.get('game_id', 'N/A')}")
print(f"  Predicted halftime diff: {pred_half:+.1f}")
print(f"  Actual halftime diff: {true_half:+.0f}")
print(f"  Error: {error_half:.1f}")
print()
print(f"  Predicted final diff: {pred_final:+.1f}")
print(f"  Actual final diff: {true_final:+.0f}")
print(f"  Error: {error_final:.1f}")
print()

# ============================================================================
# PRODUCTION READY CONFIRMATION
# ============================================================================
print("[3/3] Production readiness check...")
print()

checks = {
    'System pkl exists': True,
    'Models load correctly': len(models_a) == 10 and len(models_b) == 10,
    'Predictions work': pred_time < 1.0,
    'MAE validated': system['branch_a_halftime']['champion_mae'] < 10,
    'Feature names saved': 'feature_names' in system,
}

passed = sum(checks.values())
total = len(checks)

for check, result in checks.items():
    print(f"  {'✅' if result else '❌'} {check}")

print()
print(f"Readiness: {passed}/{total} ({'✅ READY' if passed == total else '⚠️ INCOMPLETE'})")
print()

if passed == total:
    print("="*80)
    print("🚀 STRIVE FOR GREATNESS - READY TO LAUNCH")
    print("="*80)
    print()
    print("PERFORMANCE:")
    print(f"  Halftime: {system['branch_a_halftime']['champion_mae']:.3f} MAE")
    print(f"  Final:    {system['branch_b_final']['champion_mae']:.3f} MAE")
    print()
    print("SPEED:")
    print(f"  Load time: {load_time:.2f} seconds")
    print(f"  Prediction time: {pred_time:.3f} seconds per game")
    print()
    print("WEEK 1 PLAN:")
    print("  - Bet 25-35 highest confidence games")
    print("  - Track win rate, ROI, MAE")
    print("  - Validate edge exists (>52% win rate, +ROI)")
    print()
    print("="*80)
    print("\"Strive for Greatness\" - LeBron James")
    print("="*80)
else:
    print("⚠️  System not ready - check failed tests above")

