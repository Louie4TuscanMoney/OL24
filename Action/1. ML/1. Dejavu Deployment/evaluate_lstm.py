"""
Evaluate LSTM on Test Set
Expected MAE: ~4.0 points (from MODELSYNERGY.md line 853)
"""

import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from lstm_model import LSTMForecaster

# Use same dataset class
from train_lstm import NBADataset

print("="*80)
print("LSTM EVALUATION ON TEST SET")
print("="*80)

# Load model
print("\nLoading LSTM model...")
model = LSTMForecaster(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    dropout=0.1,
    forecast_horizon=7
)
model.load_state_dict(torch.load('lstm_best.pth'))
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print(f"✅ Model loaded")
print(f"   Device: {device}")

# Load normalization parameters
print("\nLoading normalization parameters...")
with open('lstm_normalization.pkl', 'rb') as f:
    norm_params = pickle.load(f)
print(f"✅ Normalization loaded")

# Load test data
print("\nLoading test data...")
with open('splits/test.pkl', 'rb') as f:
    test_df = pickle.load(f)

test_dataset = NBADataset(
    test_df,
    pattern_mean=norm_params['pattern_mean'],
    pattern_std=norm_params['pattern_std'],
    outcome_mean=norm_params['outcome_mean'],
    outcome_std=norm_params['outcome_std']
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"✅ Test set: {len(test_df)} games")

# Evaluate
print(f"\nEvaluating...")
all_preds = []
all_actuals = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        
        # Predict (normalized)
        predictions = model(batch_X)
        
        # Denormalize predictions and actuals
        preds_denorm = predictions.cpu().numpy() * norm_params['outcome_std'] + norm_params['outcome_mean']
        actuals_denorm = batch_y.numpy() * norm_params['outcome_std'] + norm_params['outcome_mean']
        
        # We want halftime prediction (last step = minute 24 = HALFTIME)
        all_preds.extend(preds_denorm[:, -1])
        all_actuals.extend(actuals_denorm[:, -1])

all_preds = np.array(all_preds)
all_actuals = np.array(all_actuals)

# Calculate metrics
errors = all_preds - all_actuals
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))
median_error = np.median(np.abs(errors))
max_error = np.max(np.abs(errors))

# Print results
print(f"\n{'='*80}")
print("EVALUATION RESULTS")
print("="*80)

print(f"\n📊 Performance Metrics:")
print(f"   MAE:          {mae:.2f} points")
print(f"   RMSE:         {rmse:.2f} points")
print(f"   Median Error: {median_error:.2f} points")
print(f"   Max Error:    {max_error:.2f} points")

print(f"\n🎯 Expected vs Actual:")
print(f"   Expected MAE: ~4.0 points (MODELSYNERGY.md)")
print(f"   Actual MAE:   {mae:.2f} points")

if abs(mae - 4.0) < 1.0:
    print(f"   ✅ Within expected range!")
else:
    print(f"   ⚠️  Outside expected range (difference: {abs(mae - 4.0):.2f})")

# Error distribution
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
    print(f"      Predicted: {all_preds[idx]:+.1f}, Actual: {all_actuals[idx]:+.1f}, Error: {errors[idx]:+.1f}")

print(f"\n❌ Worst Predictions (largest errors):")
worst_indices = np.argsort(abs_errors)[-5:][::-1]
for i, idx in enumerate(worst_indices, 1):
    row = test_df.iloc[idx]
    print(f"   {i}. {row['away_team']} @ {row['home_team']} ({row['date']})")
    print(f"      Predicted: {all_preds[idx]:+.1f}, Actual: {all_actuals[idx]:+.1f}, Error: {errors[idx]:+.1f}")

# Save results
print(f"\n💾 Saving results...")
results = {
    'mae': float(mae),
    'rmse': float(rmse),
    'median_error': float(median_error),
    'max_error': float(max_error),
    'predictions': [float(x) for x in all_preds],
    'actuals': [float(x) for x in all_actuals],
    'errors': [float(x) for x in errors],
    'test_games': len(test_df),
    'hidden_size': 64,
    'num_layers': 2,
    'forecast_horizon': 7
}

import json
with open('lstm_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"   ✅ Results saved to lstm_test_results.json")

# Compare with Dejavu
print(f"\n{'='*80}")
print("COMPARISON: LSTM vs DEJAVU")
print("="*80)

with open('dejavu_test_results.json', 'r') as f:
    dejavu_results = json.load(f)

print(f"\nMAE Comparison:")
print(f"  Dejavu:      {dejavu_results['metrics']['mae']:.2f} points")
print(f"  LSTM:        {mae:.2f} points")
print(f"  Improvement: {((dejavu_results['metrics']['mae'] - mae) / dejavu_results['metrics']['mae'] * 100):.1f}%")

print(f"\n{'='*80}")
print("✅ LSTM EVALUATION COMPLETE")
print("="*80)
print(f"\nNext: Build ensemble (40% Dejavu + 60% LSTM)")
print(f"Expected ensemble MAE: ~3.5 points")

