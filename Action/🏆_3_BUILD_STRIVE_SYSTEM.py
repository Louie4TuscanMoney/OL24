#!/usr/bin/env python3
"""
🏆 STRIVE FOR GREATNESS - PHASE 3: BUILD FINAL SYSTEM
Package everything into final Strive for Greatness system
"""

import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression

print("="*80)
print("🏆 STRIVE FOR GREATNESS - FINAL SYSTEM BUILD")
print("="*80)
print()

# Load branches
print("[1/4] Loading trained models...")
with open('STRIVE_BRANCH_A.pkl', 'rb') as f:
    branch_a = pickle.load(f)

with open('STRIVE_BRANCH_B.pkl', 'rb') as f:
    branch_b = pickle.load(f)

print(f"✅ Branch A: {len(branch_a['models'])} models, {branch_a['best_mae']:.3f} MAE")
print(f"✅ Branch B: {len(branch_b['models'])} models, {branch_b['best_mae']:.3f} MAE")
print()

# Build final system
print("[2/4] Building Strive for Greatness system...")

strive_system = {
    'branch_a_halftime': {
        'models': branch_a['models'],
        'maes': branch_a['maes'],
        'scaler': branch_a['scaler'],
        'champion_mae': branch_a['best_mae'],
        'champion_strategy': 'Simple Average (quick build)',
        'level2_method': 'ElasticNet (best single model)'
    },
    'branch_b_final': {
        'models': branch_b['models'],
        'maes': branch_b['maes'],
        'scaler': branch_b['scaler'],
        'champion_mae': branch_b['best_mae'],
        'champion_strategy': 'Simple Average (quick build)',
        'level2_method': 'Ridge (best ensemble)'
    },
    'metadata': {
        'total_games': 6912,
        'feature_count': 73,
        'models_trained': 20,
        'build_date': '2025-10-19',
        'philosophy': 'Strive for Greatness - LeBron James'
    },
    'features': {
        'count': 73,
        'type': 'pattern-based',
        'includes': [
            '18 raw pattern values',
            '8 advanced momentum (jerk, etc.)',
            '6 quarterly breakdowns',
            '4 extreme values',
            '4 pattern complexity',
            'Plus all 33 Mamba features'
        ]
    }
}

print("✅ System packaged")
print()

# Save
print("[3/4] Saving Strive for Greatness system...")
with open('STRIVE_FOR_GREATNESS_SYSTEM.pkl', 'wb') as f:
    pickle.dump(strive_system, f)

print("✅ Saved to: STRIVE_FOR_GREATNESS_SYSTEM.pkl")
print()

# Update A/B test config
print("[4/4] Updating A/B test config...")
import json

ab_config = {
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
            "features": 73,
            "description": "Advanced system (spectral, momentum, velocity)",
            "mae_half": round(branch_a['best_mae'], 3),
            "mae_final": round(branch_b['best_mae'], 3),
            "status": "READY",
            "allocation": 0.0
        }
    },
    "ab_test": {
        "enabled": False,
        "start_time": "Tuesday 1 AM (24-hour window)",
        "split_ratio": [0.5, 0.5],
        "metric": "roi",
        "launch_schedule": {
            "day_1_mon_1am_to_tue_1am": "Mamba only (100%)",
            "day_2_tue_1am_to_wed_1am": "A/B test begins (50/50)",
            "after_50_bets_each": "Optimize allocation based on performance"
        }
    }
}

with open('AB_TEST_CONFIG.json', 'w') as f:
    json.dump(ab_config, f, indent=2)

print("✅ A/B test config updated")
print()

# Final summary
print("="*80)
print("🏆 STRIVE FOR GREATNESS SYSTEM COMPLETE")
print("="*80)
print()
print("PERFORMANCE:")
print(f"  Branch A (Halftime): {branch_a['best_mae']:.3f} MAE")
print(f"  Branch B (Final):    {branch_b['best_mae']:.3f} MAE")
print()
print("vs MAMBA MENTALITY:")
print(f"  Halftime: Strive {branch_a['best_mae']:.3f} vs Mamba 5.181 ({((branch_a['best_mae'] - 5.181)/5.181*100):+.1f}%)")
print(f"  Final:    Strive {branch_b['best_mae']:.3f} vs Mamba 9.655 ({((branch_b['best_mae'] - 9.655)/9.655*100):+.1f}%)")
print()
print("VERDICT:")
if branch_a['best_mae'] < 5.181 and branch_b['best_mae'] < 9.655:
    print("  🏆 STRIVE WINS BOTH BRANCHES - A/B test with 60/40 split")
elif branch_a['best_mae'] < 5.181 or branch_b['best_mae'] < 9.655:
    print("  🔬 MIXED RESULTS - A/B test with 50/50 split")
else:
    print("  🐍 MAMBA WINS - A/B test with 50/50 for learning")
print()
print("STATUS: READY FOR TUESDAY 1 AM A/B TEST")
print()
print("="*80)
print("\"Strive for Greatness\" - LeBron James")
print("="*80)

