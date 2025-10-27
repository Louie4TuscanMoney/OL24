#!/usr/bin/env python3
"""
🧠 FEEDBACK LOOP ENGINE - Adaptive ML System

PROBLEM: Static model with 10.75 MAE on 2025 data
SOLUTION: Learn from EVERY prediction to improve dynamically

NEW ARCHITECTURE:
┌─────────────────────────────────────────────────┐
│  GAME @ 6:00 2Q (18 minutes)                   │
│  Current Score: LAL -7.5 live spread           │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │  ML ENSEMBLE    │
        │  - Dejavu (w₁)  │
        │  - LSTM (w₂)    │
        │  - Trend (w₃)   │
        └────────┬────────┘
                 │
        ┌────────▼────────────┐
        │ PREDICTION:         │
        │ Final Spread: -9.5  │
        │ (LAL wins by 9.5)   │
        └────────┬────────────┘
                 │
        ┌────────▼──────────────────┐
        │ COMPARE TO BETONLINE:     │
        │ BetOnline: LAL -8.5       │
        │ Edge: 1.0 pts → BET LAL!  │
        └────────┬──────────────────┘
                 │
        ┌────────▼────────────────┐
        │ GAME FINISHES:          │
        │ Actual: LAL wins by 12  │
        │ Error: 2.5 points       │
        └────────┬────────────────┘
                 │
        ┌────────▼────────────────────┐
        │ FEEDBACK LOOP:              │
        │ 1. Store (pattern, actual)  │
        │ 2. Calculate model errors   │
        │ 3. Update ensemble weights  │
        │ 4. Improve for next game!   │
        └─────────────────────────────┘

DYNAMIC LEARNING:
- Week 1: Start with equal weights [Dejavu: 0.5, LSTM: 0.5]
- After 10 games: Reweight based on performance
- After 30 games: Fully adapted to 2025
- Continuous improvement!
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle
import json

class FeedbackLoopEngine:
    """
    Adaptive ML system with continuous learning from predictions
    """
    
    def __init__(self, models_config):
        """
        Initialize feedback loop engine
        
        Args:
            models_config: {
                'dejavu': {'model': model_obj, 'initial_weight': 0.5},
                'lstm': {'model': model_obj, 'initial_weight': 0.5},
            }
        """
        self.models = models_config
        self.prediction_history = []
        self.performance_window = 20  # Last N games for reweighting
        
        # Initialize weights
        self.current_weights = {
            name: config['initial_weight'] 
            for name, config in models_config.items()
        }
        
        # Normalize weights
        total = sum(self.current_weights.values())
        self.current_weights = {k: v/total for k, v in self.current_weights.items()}
        
        print(f"🧠 Feedback Loop Engine Initialized")
        print(f"   Models: {list(self.models.keys())}")
        print(f"   Initial weights: {self.current_weights}")
        print(f"   Performance window: {self.performance_window} games")
    
    def predict(self, pattern, return_details=False):
        """
        Make ensemble prediction with current weights
        
        Args:
            pattern: 18-minute differential pattern
            return_details: Return individual model predictions
        
        Returns:
            prediction: Weighted ensemble prediction for FINAL score
            details: (optional) Individual model outputs
        """
        individual_predictions = {}
        
        # Get prediction from each model
        for model_name, model_config in self.models.items():
            model = model_config['model']
            
            try:
                pred = model.predict(pattern)
                individual_predictions[model_name] = pred
            except Exception as e:
                print(f"⚠️  {model_name} failed: {e}")
                individual_predictions[model_name] = None
        
        # Weighted ensemble
        weighted_sum = 0
        weight_sum = 0
        
        for model_name, pred in individual_predictions.items():
            if pred is not None:
                weight = self.current_weights[model_name]
                weighted_sum += weight * pred
                weight_sum += weight
        
        ensemble_prediction = weighted_sum / weight_sum if weight_sum > 0 else 0
        
        if return_details:
            return ensemble_prediction, {
                'individual': individual_predictions,
                'weights': self.current_weights.copy(),
                'ensemble': ensemble_prediction
            }
        
        return ensemble_prediction
    
    def record_outcome(self, pattern, prediction, actual, bet_placed=False):
        """
        Record prediction outcome and learn from it
        
        Args:
            pattern: Input pattern used
            prediction: What ensemble predicted
            actual: Actual final differential
            bet_placed: Whether we bet on this game
        
        Returns:
            feedback: Analysis of this prediction
        """
        error = abs(prediction - actual)
        
        # Get individual model predictions
        _, details = self.predict(pattern, return_details=True)
        individual_preds = details['individual']
        
        # Calculate individual errors
        individual_errors = {}
        for model_name, pred in individual_preds.items():
            if pred is not None:
                individual_errors[model_name] = abs(pred - actual)
        
        # Store in history
        record = {
            'timestamp': datetime.now(),
            'pattern': pattern,
            'prediction': prediction,
            'actual': actual,
            'error': error,
            'individual_predictions': individual_preds,
            'individual_errors': individual_errors,
            'weights_used': details['weights'].copy(),
            'bet_placed': bet_placed
        }
        
        self.prediction_history.append(record)
        
        print(f"\n📝 Recorded: Pred {prediction:+.1f}, Actual {actual:+.1f}, Error {error:.1f}")
        print(f"   Individual errors: {individual_errors}")
        
        # Update weights if we have enough history
        if len(self.prediction_history) >= 10:
            self.update_weights()
        
        return record
    
    def update_weights(self):
        """
        Update ensemble weights based on recent performance
        
        Uses exponentially weighted moving average of errors
        Better models get higher weights
        """
        # Get recent predictions
        recent = self.prediction_history[-self.performance_window:]
        
        # Calculate average error for each model
        model_performance = {}
        
        for model_name in self.models.keys():
            errors = []
            for record in recent:
                if model_name in record['individual_errors']:
                    errors.append(record['individual_errors'][model_name])
            
            if errors:
                avg_error = np.mean(errors)
                model_performance[model_name] = avg_error
            else:
                model_performance[model_name] = 999  # Penalty if no predictions
        
        # Convert errors to weights (inverse error)
        # Lower error → higher weight
        inverse_errors = {
            name: 1.0 / (error + 1.0)  # +1 to avoid division by zero
            for name, error in model_performance.items()
        }
        
        # Normalize to sum to 1
        total = sum(inverse_errors.values())
        new_weights = {name: inv_err / total for name, inv_err in inverse_errors.items()}
        
        # Check if weights changed significantly
        weight_changes = {
            name: new_weights[name] - self.current_weights[name]
            for name in self.models.keys()
        }
        
        significant_change = any(abs(change) > 0.05 for change in weight_changes.values())
        
        if significant_change:
            print(f"\n🔄 UPDATING ENSEMBLE WEIGHTS (after {len(self.prediction_history)} games):")
            print(f"\n   Model       Old Weight  New Weight  Recent MAE  Change")
            print(f"   " + "-"*60)
            
            for name in self.models.keys():
                old_w = self.current_weights[name]
                new_w = new_weights[name]
                mae = model_performance[name]
                change = weight_changes[name]
                arrow = "↑" if change > 0 else "↓" if change < 0 else "="
                
                print(f"   {name:12} {old_w:8.2%}    {new_w:8.2%}    {mae:7.2f}    {change:+.2%} {arrow}")
            
            self.current_weights = new_weights
        
        return self.current_weights
    
    def get_performance_report(self):
        """Get current system performance"""
        if len(self.prediction_history) == 0:
            return {"status": "No predictions yet"}
        
        errors = [r['error'] for r in self.prediction_history]
        
        report = {
            'total_predictions': len(self.prediction_history),
            'mae': np.mean(errors),
            'median_error': np.median(errors),
            'recent_10_mae': np.mean([r['error'] for r in self.prediction_history[-10:]]) if len(self.prediction_history) >= 10 else np.mean(errors),
            'current_weights': self.current_weights.copy(),
            'individual_performance': {}
        }
        
        # Individual model performance
        for model_name in self.models.keys():
            model_errors = []
            for record in self.prediction_history:
                if model_name in record['individual_errors']:
                    model_errors.append(record['individual_errors'][model_name])
            
            if model_errors:
                report['individual_performance'][model_name] = {
                    'mae': np.mean(model_errors),
                    'count': len(model_errors)
                }
        
        return report
    
    def save_history(self, filepath):
        """Save prediction history for analysis"""
        filepath = Path(filepath)
        
        # Convert to DataFrame for easy analysis
        history_df = pd.DataFrame([
            {
                'timestamp': r['timestamp'],
                'prediction': r['prediction'],
                'actual': r['actual'],
                'error': r['error'],
                'bet_placed': r['bet_placed'],
                **{f'{name}_pred': r['individual_predictions'][name] 
                   for name in self.models.keys() if name in r['individual_predictions']},
                **{f'{name}_weight': r['weights_used'][name]
                   for name in self.models.keys() if name in r['weights_used']}
            }
            for r in self.prediction_history
        ])
        
        history_df.to_csv(filepath, index=False)
        print(f"✅ History saved: {filepath}")
        
        return history_df


# Demo usage
if __name__ == "__main__":
    print("="*80)
    print("🧠 FEEDBACK LOOP ENGINE - Demo")
    print("="*80)
    
    print("\nThis engine will:")
    print("  1. Start with initial model weights")
    print("  2. Make predictions using ensemble")
    print("  3. Record actual outcomes")
    print("  4. Learn which models are best")
    print("  5. Adjust weights dynamically")
    print("  6. Improve over time!")
    
    print("\n💡 ADAPTIVE LEARNING:")
    print("  - If Dejavu performs better → increase its weight")
    print("  - If LSTM performs better → increase its weight")
    print("  - Continuously adapt to 2025 game dynamics")
    print("  - Week 1: Learn, Week 2+: Profit from learning!")
    
    print("\n🎯 This solves the drift problem through ADAPTATION!")
    print("   Instead of static weights, weights evolve with data")
    
    print("\n" + "="*80)
    print("TO USE:")
    print("="*80)
    print("""
# Initialize
from dejavu_model import DejavuForecaster
# from lstm_model import LSTMForecaster  # If available

models = {
    'dejavu': {
        'model': dejavu_model,
        'initial_weight': 1.0  # Start with Dejavu only
    },
    # 'lstm': {
    #     'model': lstm_model,
    #     'initial_weight': 0.0  # Add later if available
    # }
}

engine = FeedbackLoopEngine(models)

# Make prediction
prediction = engine.predict(pattern_18min)

# After game finishes
engine.record_outcome(pattern_18min, prediction, actual_final)

# Automatically adapts weights based on performance!
    """)

