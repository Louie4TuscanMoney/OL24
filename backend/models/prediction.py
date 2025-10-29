"""
ML Prediction Data Model
Type-safe contract for Mamba ML predictions
"""

from typing import TypedDict, List, Optional
from enum import Enum


class PredictionType(str, Enum):
    """Type of ML prediction"""
    Q2_6MIN = "q2_6min"        # Official Q2 6:00 trade signal
    CONTINUOUS = "continuous"   # 30-second continuous predictions


class MLPrediction(TypedDict):
    """
    Type-safe ML prediction contract
    
    This matches what the Mamba model outputs and what frontend expects
    """
    # Game reference
    game_id: str
    
    # Prediction values
    point_forecast: float          # Predicted final spread
    interval_lower: float          # 90% CI lower bound
    interval_upper: float          # 90% CI upper bound
    confidence_interval_90: List[float]  # [lower, upper]
    
    # Probabilities
    win_probability: float         # Home team win probability (0-1)
    model_confidence: float        # Model confidence (0-1)
    coverage_probability: float    # Usually 0.90
    
    # Edge detection
    edge_detected: bool
    edge_magnitude: Optional[float]
    
    # Prediction metadata
    prediction_type: str           # q2_6min or continuous
    is_q2_6min: bool              # True if official Q2 6:00 signal
    is_trade_signal: bool         # True if edge detected
    
    # Game context
    quarter: int
    time_remaining: str
    
    # Features used
    features_extracted: Optional[int]  # Number of features (should be 33)
    feature_importance: Optional[dict]
    
    # Timestamp
    timestamp: str
    predicted_at: str


def create_empty_prediction(game_id: str = "") -> MLPrediction:
    """Create empty prediction with default values"""
    from datetime import datetime
    
    return {
        'game_id': game_id,
        'point_forecast': 0.0,
        'interval_lower': 0.0,
        'interval_upper': 0.0,
        'confidence_interval_90': [0.0, 0.0],
        'win_probability': 0.5,
        'model_confidence': 0.0,
        'coverage_probability': 0.90,
        'edge_detected': False,
        'edge_magnitude': None,
        'prediction_type': PredictionType.CONTINUOUS,
        'is_q2_6min': False,
        'is_trade_signal': False,
        'quarter': 0,
        'time_remaining': '',
        'features_extracted': None,
        'feature_importance': None,
        'timestamp': datetime.now().isoformat(),
        'predicted_at': datetime.now().isoformat()
    }

