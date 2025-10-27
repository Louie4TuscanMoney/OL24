#!/usr/bin/env python3
"""
GAME ENGINE - Your Monday Launch System

DUAL BRANCH PREDICTIONS:
- Branch A: Halftime spread (6 min ahead)
- Branch B: Final spread (30 min ahead)

FEEDBACK LOOP:
- Records every prediction
- Learns from outcomes
- Adapts weights dynamically

USAGE:
    python3 game_engine.py
"""

import sys
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent / "1. ML/1. Dejavu"))

from dejavu_model import DejavuForecaster

class GameEngine:
    """
    Complete prediction engine with dual branches and feedback
    """
    
    def __init__(self):
        """Initialize game engine"""
        print("="*80)
        print("🎮 GAME ENGINE INITIALIZING")
        print("="*80)
        
        # Load model
        print("\nLoading Dejavu model...")
        model_path = Path(__file__).parent / "1. ML/1. Dejavu Deployment/dejavu_k500.pkl"
        sys.modules['__main__'].DejavuForecaster = DejavuForecaster
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✅ Model loaded: {len(self.model.database)} patterns")
        
        # Dual branch parameters
        self.halftime_to_final_ratio = 1.4  # Initial estimate
        
        # Feedback loop storage
        self.prediction_history = []
        self.load_history()
        
        print(f"✅ Game Engine ready!")
        print(f"   - Branch A: Halftime predictions")
        print(f"   - Branch B: Final score predictions")
        print(f"   - Feedback loop: {len(self.prediction_history)} games in history")
    
    def predict(self, pattern_18min, return_details=False):
        """
        Make dual branch prediction
        
        Args:
            pattern_18min: 18-minute differential pattern
            return_details: Return neighbor info
        
        Returns:
            {
                'halftime': prediction for halftime,
                'final': prediction for final,
                'confidence': quality metrics,
                'neighbors': similar games (if return_details=True)
            }
        """
        # Get prediction from Dejavu with neighbors
        halftime_pred, neighbors = self.model.predict(pattern_18min, return_neighbors=True)
        
        # Calculate quality metrics
        distances = [n['distance'] for n in neighbors]
        outcomes = [n['outcome'] for n in neighbors]
        
        avg_distance = np.mean(distances)
        outcome_std = np.std(outcomes)
        
        # Branch B: Final score using ratio
        final_pred = halftime_pred * self.halftime_to_final_ratio
        
        # Confidence assessment
        if avg_distance < 2.5:
            confidence = "HIGH"
        elif avg_distance < 3.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Betting filter (multiple criteria)
        should_bet = (
            avg_distance < 3.5 and          # Good pattern match
            outcome_std < 12 and             # Low neighbor variance
            abs(halftime_pred) < 15 and      # Not extreme prediction
            abs(final_pred) < 20             # Not extreme final prediction
        )
        
        result = {
            'halftime': halftime_pred,
            'final': final_pred,
            'confidence': confidence,
            'avg_neighbor_distance': avg_distance,
            'neighbor_std': outcome_std,
            'should_bet': should_bet
        }
        
        if return_details:
            result['neighbors'] = neighbors[:5]  # Top 5
        
        return result
    
    def record_outcome(self, pattern, halftime_actual, final_actual, bets_placed):
        """
        Record game outcome and update feedback loop
        
        Args:
            pattern: 18-min pattern used
            halftime_actual: Actual halftime differential  
            final_actual: Actual final differential
            bets_placed: {'halftime': True/False, 'final': True/False}
        """
        # Make prediction (what we would have predicted)
        pred = self.predict(pattern)
        
        # Calculate errors
        halftime_error = abs(pred['halftime'] - halftime_actual)
        final_error = abs(pred['final'] - final_actual)
        
        # Record
        record = {
            'timestamp': datetime.now().isoformat(),
            'pred_halftime': pred['halftime'],
            'pred_final': pred['final'],
            'actual_halftime': halftime_actual,
            'actual_final': final_actual,
            'error_halftime': halftime_error,
            'error_final': final_error,
            'confidence': pred['confidence'],
            'bets_placed': bets_placed,
            'current_ratio': self.halftime_to_final_ratio
        }
        
        self.prediction_history.append(record)
        
        # Update halftime→final ratio (FEEDBACK LOOP!)
        if halftime_actual != 0:
            observed_ratio = final_actual / halftime_actual
            
            # Exponential moving average (learn slowly)
            alpha = 0.1
            old_ratio = self.halftime_to_final_ratio
            self.halftime_to_final_ratio = (
                alpha * observed_ratio + 
                (1 - alpha) * self.halftime_to_final_ratio
            )
            
            ratio_change = self.halftime_to_final_ratio - old_ratio
            
            if abs(ratio_change) > 0.05:
                print(f"\n🔄 Ratio updated: {old_ratio:.2f} → {self.halftime_to_final_ratio:.2f}")
        
        # Save history
        self.save_history()
        
        print(f"\n📝 Game recorded:")
        print(f"   Halftime: Pred {pred['halftime']:+.1f}, Actual {halftime_actual:+.1f}, Error {halftime_error:.1f}")
        print(f"   Final: Pred {pred['final']:+.1f}, Actual {final_actual:+.1f}, Error {final_error:.1f}")
        
        return record
    
    def get_performance(self):
        """Get current performance stats"""
        if len(self.prediction_history) == 0:
            return None
        
        recent = self.prediction_history[-20:]  # Last 20 games
        
        halftime_errors = [r['error_halftime'] for r in recent]
        final_errors = [r['error_final'] for r in recent]
        
        return {
            'total_games': len(self.prediction_history),
            'halftime_mae': np.mean(halftime_errors),
            'final_mae': np.mean(final_errors),
            'current_ratio': self.halftime_to_final_ratio
        }
    
    def save_history(self):
        """Save prediction history"""
        history_file = Path(__file__).parent / "prediction_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.prediction_history, f, indent=2)
    
    def load_history(self):
        """Load previous prediction history"""
        history_file = Path(__file__).parent / "prediction_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.prediction_history = json.load(f)
                
                # Recalculate ratio from history
                if len(self.prediction_history) > 10:
                    ratios = []
                    for r in self.prediction_history[-20:]:
                        if r['actual_halftime'] != 0:
                            ratios.append(r['actual_final'] / r['actual_halftime'])
                    
                    if ratios:
                        self.halftime_to_final_ratio = np.mean(ratios)


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("🎮 GAME ENGINE - Quick Test")
    print("="*80)
    
    # Initialize
    engine = GameEngine()
    
    # Test prediction with dummy pattern
    print("\n📊 Test Prediction:")
    pattern = np.array([0, -2, -1, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 7, 8, 9, 10, 8])
    
    result = engine.predict(pattern, return_details=True)
    
    print(f"\nAt 6:00 2Q (18 minutes):")
    print(f"  Current differential: {pattern[-1]:+d}")
    print(f"\n🎯 Predictions:")
    print(f"  Halftime (6 min):  {result['halftime']:+.1f}")
    print(f"  Final (30 min):    {result['final']:+.1f}")
    print(f"  Confidence:        {result['confidence']}")
    print(f"  Should bet:        {result['should_bet']}")
    
    if 'neighbors' in result:
        print(f"\n📋 Top 5 similar games:")
        for n in result['neighbors']:
            print(f"  {n['rank']}. {n['away_team']} @ {n['home_team']} → {n['outcome']:+.1f}")
    
    print("\n✅ Engine ready for Monday!")
    print("\nUsage:")
    print("  engine = GameEngine()")
    print("  preds = engine.predict(pattern_18min)")
    print("  # After game: engine.record_outcome(...)")



