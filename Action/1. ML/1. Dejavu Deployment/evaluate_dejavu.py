"""
Evaluate Dejavu on Test Set
Expected MAE: ~6.0 points (from MODELSYNERGY.md)
"""

import pickle
import numpy as np
from dejavu_model import DejavuForecaster

print("="*80)
print("DEJAVU EVALUATION ON TEST SET")
print("="*80)

# Load model
print("\nLoading Dejavu model...")
dejavu = DejavuForecaster.load('dejavu_k500.pkl')
print(f"✅ Loaded model with {len(dejavu.database)} patterns, k={dejavu.k}")

# Load test data
print("\nLoading test data...")
with open('splits/test.pkl', 'rb') as f:
    test_df = pickle.load(f)
print(f"✅ Loaded {len(test_df)} test games")

# Evaluate
metrics, predictions, actuals = dejavu.evaluate(test_df)

# Print results
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

print(f"\n📊 Performance Metrics:")
print(f"   MAE:          {metrics['mae']:.2f} points")
print(f"   RMSE:         {metrics['rmse']:.2f} points")
print(f"   MAPE:         {metrics['mape']:.1f}%")
print(f"   Median Error: {metrics['median_error']:.2f} points")
print(f"   Max Error:    {metrics['max_error']:.2f} points")

print(f"\n⚡ Speed:")
print(f"   Total time:   {metrics['inference_time_total']:.2f} seconds")
print(f"   Per game:     {metrics['inference_time_per_game']:.2f} ms")

print(f"\n🎯 Expected vs Actual:")
print(f"   Expected MAE: ~6.0 points (MODELSYNERGY.md)")
print(f"   Actual MAE:   {metrics['mae']:.2f} points")

if abs(metrics['mae'] - 6.0) < 1.0:
    print(f"   ✅ Within expected range!")
else:
    print(f"   ⚠️  Outside expected range (difference: {abs(metrics['mae'] - 6.0):.2f})")

# Error distribution
errors = predictions - actuals
print(f"\n📈 Error Distribution:")
print(f"   Mean error:   {np.mean(errors):+.2f} points")
print(f"   Std error:    {np.std(errors):.2f} points")
print(f"   Min error:    {np.min(errors):+.2f} points")
print(f"   Max error:    {np.max(errors):+.2f} points")

# Percentiles
print(f"\n   Error Percentiles:")
for p in [10, 25, 50, 75, 90]:
    val = np.percentile(np.abs(errors), p)
    print(f"      {p}th: {val:.2f} points")

# Best and worst predictions
print(f"\n🎯 Best Predictions (smallest errors):")
abs_errors = np.abs(errors)
best_indices = np.argsort(abs_errors)[:5]
for i, idx in enumerate(best_indices, 1):
    row = test_df.iloc[idx]
    print(f"   {i}. {row['away_team']} @ {row['home_team']} ({row['date']})")
    print(f"      Predicted: {predictions[idx]:+.1f}, Actual: {actuals[idx]:+.1f}, Error: {errors[idx]:+.1f}")

print(f"\n❌ Worst Predictions (largest errors):")
worst_indices = np.argsort(abs_errors)[-5:][::-1]
for i, idx in enumerate(worst_indices, 1):
    row = test_df.iloc[idx]
    print(f"   {i}. {row['away_team']} @ {row['home_team']} ({row['date']})")
    print(f"      Predicted: {predictions[idx]:+.1f}, Actual: {actuals[idx]:+.1f}, Error: {errors[idx]:+.1f}")

# Save results
print(f"\n💾 Saving results...")
results = {
    'metrics': metrics,
    'predictions': predictions.tolist(),
    'actuals': actuals.tolist(),
    'errors': errors.tolist(),
    'test_games': len(test_df),
    'k': dejavu.k,
    'pattern_length': dejavu.pattern_length,
    'database_size': len(dejavu.database)
}

import json
with open('dejavu_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"   ✅ Results saved to dejavu_test_results.json")

# Summary
print(f"\n" + "="*80)
print("✅ EVALUATION COMPLETE")
print("="*80)
print(f"\nDejavu K-NN Forecaster:")
print(f"  Database:     {len(dejavu.database)} patterns")
print(f"  k-neighbors:  {dejavu.k}")
print(f"  Test games:   {len(test_df)}")
print(f"  MAE:          {metrics['mae']:.2f} points")
print(f"  Speed:        {metrics['inference_time_per_game']:.2f} ms/game")

print(f"\n{'='*80}")
print("NEXT STEP: Build LSTM model")
print("="*80)

