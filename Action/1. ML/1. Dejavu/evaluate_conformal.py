"""
Evaluate Conformal Coverage on Test Set
Target: 95% coverage
"""

import pickle
import numpy as np
import json
from ensemble_model import EnsembleForecaster
from conformal_wrapper import ConformalPredictor

print("="*80)
print("CONFORMAL PREDICTION EVALUATION")
print("="*80)

# Load models
print("\nLoading models...")
ensemble = EnsembleForecaster(dejavu_weight=0.4, lstm_weight=0.6)
ensemble.load_models()

conformal = ConformalPredictor.load('conformal_predictor.pkl')
print(f"✅ Models loaded")
print(f"   Conformal quantile: ±{conformal.quantile:.2f} points")

# Load test data
print("\nLoading test data...")
with open('splits/test.pkl', 'rb') as f:
    test_df = pickle.load(f)
print(f"✅ {len(test_df)} test games")

# Evaluate coverage
coverage_metrics = conformal.evaluate_coverage(test_df, ensemble)

# Print results
print("\n" + "="*80)
print("COVERAGE EVALUATION RESULTS")
print("="*80)

print(f"\n📊 Coverage Performance:")
print(f"   Target coverage:     {coverage_metrics['target_coverage']:.1%}")
print(f"   Empirical coverage:  {coverage_metrics['empirical_coverage']:.1%}")
print(f"   Coverage gap:        {coverage_metrics['coverage_gap']:.3f}")

if coverage_metrics['coverage_gap'] < 0.05:
    print(f"   ✅ Coverage within acceptable range (<5% gap)")
else:
    print(f"   ⚠️  Coverage gap large ({coverage_metrics['coverage_gap']:.1%})")

print(f"\n📏 Interval Efficiency:")
print(f"   Quantile:            ±{coverage_metrics['quantile']:.2f} points")
print(f"   Average width:       {coverage_metrics['avg_interval_width']:.2f} points")

print(f"\n🎯 Point Forecast Accuracy:")
print(f"   MAE:                 {coverage_metrics['mae']:.2f} points")

# Compare with research expectations
print(f"\n🔍 Comparison with Research:")
print(f"   Expected quantile: ±3.8 points (MODELSYNERGY.md)")
print(f"   Actual quantile:   ±{coverage_metrics['quantile']:.2f} points")
print(f"   Difference:        {abs(coverage_metrics['quantile'] - 3.8):.2f} points")

if abs(coverage_metrics['quantile'] - 3.8) > 2.0:
    print(f"   ⚠️  Wider than expected (reflects higher model uncertainty)")
else:
    print(f"   ✅ Close to expected")

# Save metrics
print(f"\n💾 Saving results...")
with open('conformal_test_results.json', 'w') as f:
    json.dump(coverage_metrics, f, indent=2)
print(f"   ✅ Results saved to conformal_test_results.json")

# Final summary
print(f"\n" + "="*80)
print("COMPLETE SYSTEM PERFORMANCE")
print("="*80)

print(f"\n📊 Full Stack:")
print(f"  1. Dejavu (pattern matching):    MAE {coverage_metrics['mae']:.2f}")
print(f"  2. LSTM (pattern learning):      Contributing {60}%")
print(f"  3. Ensemble (weighted):          MAE {coverage_metrics['mae']:.2f}")
print(f"  4. Conformal (uncertainty):      ±{coverage_metrics['quantile']:.2f} at {coverage_metrics['empirical_coverage']:.1%} coverage")

print(f"\n🎯 Example Prediction:")
with open('splits/test.pkl', 'rb') as f:
    test_df_sample = pickle.load(f)
sample_game = test_df_sample.iloc[0]
sample_pred, sample_interval, sample_comp = conformal.predict(sample_game['pattern'], ensemble)
sample_actual = sample_game['diff_at_halftime']

print(f"  Game: {sample_game['away_team']} @ {sample_game['home_team']}")
print(f"  Prediction: {sample_pred:+.1f} points")
print(f"  95% Interval: [{sample_interval[0]:+.1f}, {sample_interval[1]:+.1f}]")
print(f"  Actual: {sample_actual:+.1f} points")
print(f"  Covered: {'✅ YES' if sample_interval[0] <= sample_actual <= sample_interval[1] else '❌ NO'}")

print(f"\n" + "="*80)
print("✅ CONFORMAL EVALUATION COMPLETE")
print("="*80)
print(f"\nComplete system ready:")
print(f"  • Dejavu (interpretable)")
print(f"  • LSTM (accurate)")  
print(f"  • Ensemble (combined)")
print(f"  • Conformal (uncertainty)")

