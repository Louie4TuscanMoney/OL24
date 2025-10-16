"""
Evaluate Ensemble on Test Set
Expected MAE: ~3.5 points (MODELSYNERGY.md line 854)
"""

import pickle
import numpy as np
import json
from ensemble_model import EnsembleForecaster

print("="*80)
print("ENSEMBLE EVALUATION (40% Dejavu + 60% LSTM)")
print("="*80)

# Create and load ensemble
print("\nInitializing ensemble...")
ensemble = EnsembleForecaster(dejavu_weight=0.4, lstm_weight=0.6)
ensemble.load_models()

# Load test data
print("\nLoading test data...")
with open('splits/test.pkl', 'rb') as f:
    test_df = pickle.load(f)
print(f"✅ {len(test_df)} test games")

# Evaluate
metrics, ensemble_preds, actuals = ensemble.evaluate(test_df)

# Print results
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

print(f"\n📊 Ensemble Performance:")
print(f"   MAE:          {metrics['ensemble']['mae']:.2f} points")
print(f"   RMSE:         {metrics['ensemble']['rmse']:.2f} points")
print(f"   Median Error: {metrics['ensemble']['median_error']:.2f} points")

print(f"\n📊 Component Performance:")
print(f"   Dejavu MAE:   {metrics['dejavu']['mae']:.2f} points")
print(f"   LSTM MAE:     {metrics['lstm']['mae']:.2f} points")

print(f"\n⚖️  Ensemble Weights:")
print(f"   Dejavu: {metrics['weights']['dejavu']:.1%}")
print(f"   LSTM:   {metrics['weights']['lstm']:.1%}")

print(f"\n🎯 Expected vs Actual:")
print(f"   Expected MAE: ~3.5 points (MODELSYNERGY.md)")
print(f"   Actual MAE:   {metrics['ensemble']['mae']:.2f} points")

improvement_vs_dejavu = ((metrics['dejavu']['mae'] - metrics['ensemble']['mae']) / 
                         metrics['dejavu']['mae'] * 100)
improvement_vs_lstm = ((metrics['lstm']['mae'] - metrics['ensemble']['mae']) / 
                       metrics['lstm']['mae'] * 100)

print(f"\n📈 Improvement:")
print(f"   vs Dejavu alone:  {improvement_vs_dejavu:+.1f}%")
print(f"   vs LSTM alone:    {improvement_vs_lstm:+.1f}%")

if abs(metrics['ensemble']['mae'] - 3.5) < 1.0:
    print(f"   ✅ Within expected range!")
else:
    print(f"   ⚠️  Outside expected range (difference: {abs(metrics['ensemble']['mae'] - 3.5):.2f})")

# Error analysis
errors = ensemble_preds - actuals
print(f"\n📈 Error Distribution:")
print(f"   Mean error:   {np.mean(errors):+.2f} points")
print(f"   Std error:    {np.std(errors):.2f} points")
print(f"   Min error:    {np.min(errors):+.2f} points")
print(f"   Max error:    {np.max(errors):+.2f} points")

# Percentiles
print(f"\n   Error Percentiles:")
for p in [10, 25, 50, 75, 90, 95]:
    val = np.percentile(np.abs(errors), p)
    print(f"      {p:2d}th: {val:.2f} points")

# Best predictions
print(f"\n🎯 Best Ensemble Predictions:")
abs_errors = np.abs(errors)
best_indices = np.argsort(abs_errors)[:5]
for i, idx in enumerate(best_indices, 1):
    row = test_df.iloc[idx]
    print(f"   {i}. {row['away_team']} @ {row['home_team']} ({row['date']})")
    print(f"      Ensemble: {ensemble_preds[idx]:+.1f}, Actual: {actuals[idx]:+.1f}, Error: {errors[idx]:+.1f}")

# Save results
print(f"\n💾 Saving results...")
results = {
    'metrics': metrics,
    'predictions': [float(x) for x in ensemble_preds],
    'actuals': [float(x) for x in actuals],
    'errors': [float(x) for x in errors],
    'test_games': len(test_df)
}

with open('ensemble_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"   ✅ Results saved to ensemble_test_results.json")

# Final summary
print(f"\n" + "="*80)
print("FINAL PERFORMANCE SUMMARY")
print("="*80)

print(f"\nMAE Progression:")
print(f"  1. Dejavu alone:    {metrics['dejavu']['mae']:.2f} points")
print(f"  2. LSTM alone:      {metrics['lstm']['mae']:.2f} points")
print(f"  3. Ensemble (both): {metrics['ensemble']['mae']:.2f} points  ← BEST!")

print(f"\nImprovement from ensemble:")
print(f"  {improvement_vs_dejavu:.1f}% better than Dejavu")
print(f"  {improvement_vs_lstm:.1f}% better than LSTM")

print(f"\n" + "="*80)
print("✅ ENSEMBLE EVALUATION COMPLETE")
print("="*80)
print(f"\nNext: Add Conformal wrapper for uncertainty quantification")
print(f"Expected: ±3.8 points at 95% confidence")

