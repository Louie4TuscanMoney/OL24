"""
Hyperparameter Optimization: Find Optimal Ensemble Weights
ML Engineering approach: Grid search on validation set

Research default: 40% Dejavu + 60% LSTM
Let's find the ACTUAL optimal weights for our data!
"""

import pickle
import numpy as np
import json
from ensemble_model import EnsembleForecaster
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("="*80)
print("ENSEMBLE WEIGHT OPTIMIZATION")
print("="*80)

# Load validation data (for hyperparameter tuning)
print("\nLoading validation data...")
with open('splits/validation.pkl', 'rb') as f:
    val_df = pickle.load(f)
print(f"✅ {len(val_df)} validation games")

# Grid search over weights
print("\n" + "="*80)
print("GRID SEARCH FOR OPTIMAL WEIGHTS")
print("="*80)

# Test weights from 0.0 to 1.0 in steps of 0.05
dejavu_weights = np.arange(0.0, 1.05, 0.05)
results = []

print(f"\nTesting {len(dejavu_weights)} weight combinations...")

for dejavu_w in dejavu_weights:
    lstm_w = 1.0 - dejavu_w
    
    # Create ensemble with this weight
    ensemble = EnsembleForecaster(dejavu_weight=dejavu_w, lstm_weight=lstm_w)
    ensemble.load_models()
    
    # Evaluate on validation set
    preds = []
    actuals = []
    
    for idx, row in val_df.iterrows():
        ens_pred, _, _ = ensemble.predict(row['pattern'])
        preds.append(ens_pred)
        actuals.append(row['diff_at_halftime'])
    
    # Calculate MAE
    mae = np.mean(np.abs(np.array(preds) - np.array(actuals)))
    
    results.append({
        'dejavu_weight': float(dejavu_w),
        'lstm_weight': float(lstm_w),
        'val_mae': float(mae)
    })
    
    print(f"  Dejavu {dejavu_w:.2f} / LSTM {lstm_w:.2f} → MAE: {mae:.3f}")

# Find optimal weights
results_sorted = sorted(results, key=lambda x: x['val_mae'])
best = results_sorted[0]

print(f"\n" + "="*80)
print("OPTIMIZATION RESULTS")
print("="*80)

print(f"\n🏆 OPTIMAL WEIGHTS (Validation Set):")
print(f"   Dejavu: {best['dejavu_weight']:.2f} ({best['dejavu_weight']*100:.0f}%)")
print(f"   LSTM:   {best['lstm_weight']:.2f} ({best['lstm_weight']*100:.0f}%)")
print(f"   Val MAE: {best['val_mae']:.3f} points")

print(f"\n📊 Comparison with Research Default:")
research_result = [r for r in results if abs(r['dejavu_weight'] - 0.4) < 0.01][0]
print(f"   Research (40/60): MAE {research_result['val_mae']:.3f}")
print(f"   Optimal ({best['dejavu_weight']*100:.0f}/{best['lstm_weight']*100:.0f}): MAE {best['val_mae']:.3f}")
print(f"   Improvement: {(research_result['val_mae'] - best['val_mae']) / research_result['val_mae'] * 100:.2f}%")

# Show top 5 weight combinations
print(f"\n📈 Top 5 Weight Combinations:")
for i, r in enumerate(results_sorted[:5], 1):
    print(f"   {i}. Dejavu {r['dejavu_weight']:.2f} / LSTM {r['lstm_weight']:.2f} → MAE {r['val_mae']:.3f}")

# Show worst combinations (sanity check)
print(f"\n📉 Worst 3 Weight Combinations:")
for i, r in enumerate(results_sorted[-3:], 1):
    print(f"   {i}. Dejavu {r['dejavu_weight']:.2f} / LSTM {r['lstm_weight']:.2f} → MAE {r['val_mae']:.3f}")

# Save results
print(f"\n💾 Saving optimization results...")
optimization_results = {
    'all_results': results,
    'optimal_weights': best,
    'research_default': research_result,
    'n_combinations_tested': len(results)
}

with open('weight_optimization_results.json', 'w') as f:
    json.dump(optimization_results, f, indent=2)
print(f"   ✅ Saved to weight_optimization_results.json")

# Plot results
print(f"\n📊 Creating visualization...")
fig, ax = plt.subplots(figsize=(10, 6))

dejavu_ws = [r['dejavu_weight'] for r in results]
maes = [r['val_mae'] for r in results]

ax.plot(dejavu_ws, maes, 'b-', linewidth=2, label='Validation MAE')
ax.scatter([best['dejavu_weight']], [best['val_mae']], 
           color='red', s=200, zorder=5, label=f'Optimal ({best["dejavu_weight"]:.2f})')
ax.scatter([0.4], [research_result['val_mae']], 
           color='green', s=200, marker='s', zorder=5, label='Research default (0.40)')
ax.axvline(x=best['dejavu_weight'], color='red', linestyle='--', alpha=0.3)
ax.axvline(x=0.4, color='green', linestyle='--', alpha=0.3)

ax.set_xlabel('Dejavu Weight', fontsize=12)
ax.set_ylabel('Validation MAE (points)', fontsize=12)
ax.set_title('Ensemble Weight Optimization', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('weight_optimization_plot.png', dpi=150)
print(f"   ✅ Plot saved to weight_optimization_plot.png")

# Now evaluate optimal weights on TEST set
print(f"\n" + "="*80)
print("EVALUATING OPTIMAL WEIGHTS ON TEST SET")
print("="*80)

print(f"\nLoading test data...")
with open('splits/test.pkl', 'rb') as f:
    test_df = pickle.load(f)
print(f"✅ {len(test_df)} test games")

print(f"\nCreating ensemble with OPTIMAL weights...")
optimal_ensemble = EnsembleForecaster(
    dejavu_weight=best['dejavu_weight'],
    lstm_weight=best['lstm_weight']
)
optimal_ensemble.load_models()

print(f"\nEvaluating on test set...")
optimal_metrics, optimal_preds, test_actuals = optimal_ensemble.evaluate(test_df)

print(f"\n" + "="*80)
print("FINAL TEST SET RESULTS")
print("="*80)

print(f"\n📊 Optimal Weights Performance:")
print(f"   Weights: Dejavu {best['dejavu_weight']:.2f} / LSTM {best['lstm_weight']:.2f}")
print(f"   Test MAE: {optimal_metrics['ensemble']['mae']:.3f} points")

# Load research default results for comparison
with open('ensemble_test_results.json', 'r') as f:
    research_ensemble = json.load(f)

print(f"\n📊 Comparison:")
print(f"   Research default (40/60): {research_ensemble['metrics']['ensemble']['mae']:.3f} points")
print(f"   Optimized ({best['dejavu_weight']*100:.0f}/{best['lstm_weight']*100:.0f}): {optimal_metrics['ensemble']['mae']:.3f} points")
print(f"   Improvement: {(research_ensemble['metrics']['ensemble']['mae'] - optimal_metrics['ensemble']['mae']) / research_ensemble['metrics']['ensemble']['mae'] * 100:+.2f}%")

# Save final optimized results
final_results = {
    'optimal_weights': best,
    'test_performance': optimal_metrics,
    'research_default_test_mae': research_ensemble['metrics']['ensemble']['mae'],
    'optimized_test_mae': optimal_metrics['ensemble']['mae'],
    'improvement_percent': float((research_ensemble['metrics']['ensemble']['mae'] - optimal_metrics['ensemble']['mae']) / research_ensemble['metrics']['ensemble']['mae'] * 100)
}

with open('optimized_ensemble_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\n{'='*80}")
print("✅ OPTIMIZATION COMPLETE")
print("="*80)

print(f"\n🎯 BEST ENSEMBLE:")
print(f"   Dejavu weight: {best['dejavu_weight']:.2f}")
print(f"   LSTM weight:   {best['lstm_weight']:.2f}")
print(f"   Test MAE:      {optimal_metrics['ensemble']['mae']:.3f} points")

if optimal_metrics['ensemble']['mae'] <= 5.0:
    print(f"   ✅ ACHIEVED TARGET: MAE ≤ 5.0 points!")
else:
    print(f"   Current: {optimal_metrics['ensemble']['mae']:.3f}, Target: 5.0")
    print(f"   Gap: {optimal_metrics['ensemble']['mae'] - 5.0:.3f} points")

