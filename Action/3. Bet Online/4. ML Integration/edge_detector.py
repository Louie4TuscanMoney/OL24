"""
Edge Detection System
Compare ML predictions with BetOnline market odds

Following: BETONLINE/EDGE_DETECTION_SYSTEM.md
"""

from typing import Dict, Optional

class EdgeDetector:
    """
    Detect betting edges by comparing ML vs Market
    """
    
    def __init__(self, min_edge_threshold: float = 2.0):
        """
        Args:
            min_edge_threshold: Minimum edge to consider (points)
        """
        self.min_edge_threshold = min_edge_threshold
    
    def calculate_edge(self, ml_prediction: Dict, market_odds: Dict) -> Optional[Dict]:
        """
        Calculate betting edge
        
        Args:
            ml_prediction: From ML model
                {
                    'point_forecast': 15.1,
                    'interval_lower': 11.3,
                    'interval_upper': 18.9
                }
            market_odds: From BetOnline
                {
                    'spread': {'away': +7.5, 'home': -7.5},
                    'home_team': 'LAL'
                }
        
        Returns:
            Edge analysis dict or None if no edge
        """
        ml_forecast = ml_prediction['point_forecast']
        ml_lower = ml_prediction['interval_lower']
        ml_upper = ml_prediction['interval_upper']
        
        market_spread = market_odds['spread']['home']  # -7.5 means home favored by 7.5
        
        # Edge = ML forecast vs market line
        edge = ml_forecast - market_spread
        
        # Check if edge exceeds threshold
        if abs(edge) < self.min_edge_threshold:
            return None  # No significant edge
        
        # Determine direction
        if edge > 0:
            # ML predicts home team to outperform market
            side = 'home'
            confidence = 'high' if ml_lower > market_spread else 'medium'
        else:
            # ML predicts away team to outperform market
            side = 'away'
            confidence = 'high' if ml_upper < market_spread else 'medium'
        
        return {
            'has_edge': True,
            'edge_size': abs(edge),
            'direction': side,
            'confidence': confidence,
            'ml_forecast': ml_forecast,
            'ml_interval': [ml_lower, ml_upper],
            'market_spread': market_spread,
            'recommended_bet': f"{side.upper()} {market_spread:+.1f}"
        }
    
    def analyze_game(self, game_id: str, ml_prediction: Dict, market_odds: Dict) -> Dict:
        """
        Complete edge analysis for a game
        
        Returns:
            Full analysis with recommendation
        """
        edge = self.calculate_edge(ml_prediction, market_odds)
        
        if edge is None:
            return {
                'game_id': game_id,
                'has_edge': False,
                'reason': f'Edge {abs(ml_prediction["point_forecast"] - market_odds["spread"]["home"]):.1f} < threshold {self.min_edge_threshold}'
            }
        
        return {
            'game_id': game_id,
            **edge,
            'analysis': f"ML predicts {edge['direction'].upper()} outperforms by {edge['edge_size']:.1f} points"
        }


# Demo
if __name__ == "__main__":
    print("="*80)
    print("EDGE DETECTION TEST")
    print("="*80)
    
    detector = EdgeDetector(min_edge_threshold=2.0)
    
    # Example 1: ML predicts Lakers stronger
    ml_pred = {
        'point_forecast': 15.1,
        'interval_lower': 11.3,
        'interval_upper': 18.9
    }
    
    market = {
        'spread': {'home': -7.5, 'away': +7.5},
        'home_team': 'LAL'
    }
    
    edge = detector.calculate_edge(ml_pred, market)
    
    print("\nExample: Lakers vs Celtics")
    print(f"  ML predicts: LAL {ml_pred['point_forecast']:+.1f} at halftime")
    print(f"  Market spread: LAL {market['spread']['home']:+.1f}")
    
    if edge:
        print(f"\n  ✅ EDGE DETECTED!")
        print(f"     Size: {edge['edge_size']:.1f} points")
        print(f"     Direction: {edge['direction'].upper()}")
        print(f"     Confidence: {edge['confidence']}")
        print(f"     Recommended: {edge['recommended_bet']}")
    else:
        print(f"\n  ❌ No significant edge")
    
    print("\n" + "="*80)
    print("✅ EDGE DETECTOR READY")
    print("="*80)

