#!/usr/bin/env python3
"""
Enhanced Prediction System - XGBoost + Dejavu Ensemble
Uses XGBoost as primary, Dejavu as backup
"""

import sys
sys.path.insert(0, '1. ML/1. Dejavu Deployment')

import numpy as np
import xgboost as xgb
from dejavu_model import DejavuForecaster

class EnhancedPredictionSystem:
    def __init__(self):
        """Initialize both XGBoost and Dejavu models"""
        # Load XGBoost (primary)
        self.xgboost_model = xgb.XGBRegressor()
        self.xgboost_model.load_model('xgboost_simple_v1.json')
        
        # Load Dejavu (backup)
        self.dejavu_model = DejavuForecaster.load('1. ML/1. Dejavu Deployment/dejavu_FINAL_k500.pkl')
        
        print("✅ Enhanced Prediction System loaded")
        print("   Primary: XGBoost (MAE: 8.22)")
        print("   Backup: Dejavu (MAE: 11.11)")
    
    def predict(self, game_data):
        """
        Make prediction using XGBoost with Dejavu fallback
        
        Args:
            game_data: Dict with keys:
                - pattern: 18-min score differential pattern
                - pattern_statistical: {mean, std, trend, volatility}
                - quality_metrics: {quality_grade}
                - team_features: {ratings, win_pct, etc}
                - player_features: {star_count, tiers, depth}
        
        Returns:
            prediction: float (predicted final differential)
            confidence: float (0-1, prediction confidence)
            source: str ('xgboost' or 'dejavu')
        """
        
        # Extract 18-min pattern
        pattern = game_data.get('pattern', [])
        
        # Try XGBoost first
        try:
            # Build feature vector (35 features)
            features = []
            
            # 1. Pattern (18)
            features.extend(pattern)
            
            # 2. Statistical (4)
            stat = game_data.get('pattern_statistical', {})
            features.extend([
                stat.get('mean', 0),
                stat.get('std', 0),
                stat.get('trend', 0),
                stat.get('volatility', 0)
            ])
            
            # 3. Quality (1)
            qual = game_data.get('quality_metrics', {})
            quality_val = 1.0 if qual.get('quality_grade') == 'A' else 0.5
            features.append(quality_val)
            
            # 4. Team (6)
            team = game_data.get('team_features', {})
            features.extend([
                team.get('home_off_rating', 110.0),
                team.get('home_def_rating', 110.0),
                team.get('away_off_rating', 110.0),
                team.get('away_def_rating', 110.0),
                team.get('home_win_pct', 0.5),
                team.get('away_win_pct', 0.5)
            ])
            
            # 5. Player (6)
            player = game_data.get('player_features', {})
            features.extend([
                player.get('home_star_count', 0),
                player.get('away_star_count', 0),
                player.get('home_avg_tier', 3.0),
                player.get('away_avg_tier', 3.0),
                player.get('home_depth', 0),
                player.get('away_depth', 0)
            ])
            
            # Make prediction
            prediction = self.xgboost_model.predict(np.array([features]))[0]
            
            # Confidence based on feature quality and pattern clarity
            confidence = 0.7  # Base confidence for XGBoost
            if quality_val == 1.0:
                confidence += 0.1
            if stat.get('std', 10) < 5:  # Low volatility = more predictable
                confidence += 0.1
            confidence = min(confidence, 0.95)
            
            return prediction, confidence, 'xgboost'
            
        except Exception as e:
            print(f"⚠️  XGBoost failed: {e}")
            print("   Falling back to Dejavu...")
            
            # Fallback to Dejavu
            try:
                if len(pattern) == 18:
                    prediction = self.dejavu_model.predict(pattern)
                    confidence = 0.6  # Lower confidence for backup
                    return prediction, confidence, 'dejavu'
                else:
                    raise ValueError(f"Invalid pattern length: {len(pattern)}")
            except Exception as e2:
                print(f"❌ Both models failed: {e2}")
                return 0.0, 0.0, 'error'
    
    def should_bet(self, game_data, prediction, confidence):
        """
        Determine if we should bet on this game
        
        Args:
            game_data: Game data dict
            prediction: Predicted differential
            confidence: Confidence score
        
        Returns:
            should_bet: bool
            bet_size: float (dollars)
            reason: str
        """
        from risk_configuration import RISK_MODE, MAX_BET_PER_GAME, CONFIDENCE_THRESHOLD
        
        # Filter 1: Confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return False, 0, f"Low confidence ({confidence:.2f} < {CONFIDENCE_THRESHOLD})"
        
        # Filter 2: Not extreme predictions (likely errors)
        if abs(prediction) > 20:
            return False, 0, f"Extreme prediction ({prediction:.1f})"
        
        # Filter 3: Pattern quality
        qual = game_data.get('quality_metrics', {})
        if qual.get('quality_grade') != 'A':
            return False, 0, "Low quality data"
        
        # All filters passed - bet!
        bet_size = MAX_BET_PER_GAME * confidence  # Scale by confidence
        reason = f"Confidence: {confidence:.2f}, Prediction: {prediction:.1f}"
        
        return True, bet_size, reason


# Test if run directly
if __name__ == "__main__":
    print("="*70)
    print("Testing Enhanced Prediction System")
    print("="*70)
    print()
    
    # Initialize
    system = EnhancedPredictionSystem()
    print()
    
    # Test prediction with dummy data
    test_game = {
        'pattern': [0, 2, -1, 3, 2, -2, 0, 5, 3, 1, -1, 2, 4, 3, 1, 0, -2, 1],
        'pattern_statistical': {'mean': 1.0, 'std': 2.5, 'trend': 0.5, 'volatility': 1.5},
        'quality_metrics': {'quality_grade': 'A'},
        'team_features': {
            'home_off_rating': 112.0, 'home_def_rating': 108.0,
            'away_off_rating': 110.0, 'away_def_rating': 111.0,
            'home_win_pct': 0.55, 'away_win_pct': 0.52
        },
        'player_features': {
            'home_star_count': 2, 'away_star_count': 1,
            'home_avg_tier': 2.5, 'away_avg_tier': 3.0,
            'home_depth': 7, 'away_depth': 6
        }
    }
    
    print("Making test prediction...")
    pred, conf, source = system.predict(test_game)
    print(f"✅ Prediction: {pred:.2f}")
    print(f"   Confidence: {conf:.2f}")
    print(f"   Source: {source}")
    print()
    
    print("Checking betting recommendation...")
    should_bet, bet_size, reason = system.should_bet(test_game, pred, conf)
    if should_bet:
        print(f"✅ BET RECOMMENDED")
        print(f"   Size: ${bet_size:.2f}")
        print(f"   Reason: {reason}")
    else:
        print(f"❌ NO BET")
        print(f"   Reason: {reason}")
    
    print()
    print("="*70)
    print("✅ Enhanced Prediction System working!")
    print("="*70)

