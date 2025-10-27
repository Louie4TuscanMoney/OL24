"""
WEBSOCKET MESSAGE SCHEMA

Purpose: Define the complete data package sent from backend to frontend
Author: Ontologic XYZ
Date: October 27, 2025

Philosophy: Backend does ALL heavy work, frontend just displays
"""

from typing import Dict, List, Optional, TypedDict
from datetime import datetime


class GameData(TypedDict):
    """Live game information"""
    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    period: int
    clock: str
    status_text: str
    current_diff: int
    can_predict: bool
    is_q2_6min: bool


class MambaPrediction(TypedDict):
    """Mamba ML prediction with full context"""
    game_id: str
    matchup: str
    period: int
    clock: str
    current_score: str
    current_diff: int
    
    # Prediction
    mamba_prediction: float  # Final score differential
    branch: str  # "B" (final score)
    mae: float  # 9.655
    
    # Features used
    features_used: int  # 33
    feature_breakdown: Dict[str, float]  # Optional: show which features contributed
    
    # Metadata
    timestamp: str  # ISO format
    model: str  # "MAMBA_MENTALITY"
    
    # Confidence
    confidence_interval: List[float]  # [lower, upper] based on MAE


class BetOnlineOdds(TypedDict):
    """Real-time BetOnline odds"""
    spread: float  # e.g., -6.0 (Lakers -6)
    spread_odds: int  # e.g., -110
    total: float  # e.g., 225.5
    over_odds: int  # e.g., -110
    under_odds: int  # e.g., -110
    moneyline_home: int  # e.g., -240
    moneyline_away: int  # e.g., +200
    
    # Implied probabilities
    spread_implied_prob: float  # Convert -110 to probability
    no_vig_prob: float  # Remove vig
    vig_percentage: float  # Calculate house edge
    
    # Team context
    favorite: str  # "Lakers" or "Warriors"
    favorite_spread: str  # "Lakers -6" (human readable)
    
    # Timestamp
    last_updated: str  # When odds were fetched


class OntoRiskAnalysis(TypedDict):
    """Complete OntoRisk risk management analysis"""
    
    # Probability calibration
    calibrated_prob: float  # Mamba prediction → probability
    market_prob: float  # BetOnline odds → probability
    edge: float  # calibrated_prob - market_prob
    
    # Kelly Criterion
    kelly_fraction: float  # Optimal bet size
    recommended_stake: float  # $ amount
    kelly_percentage: float  # % of bankroll
    
    # Risk validation
    can_bet: bool  # Pass all risk checks?
    risk_score: float  # 0-100 (higher = riskier)
    risk_alerts: List[str]  # ["High volatility", "Exceeds max bet"]
    
    # Bankroll management
    current_bankroll: float
    peak_bankroll: float
    current_drawdown: float  # % from peak
    daily_loss: float
    open_positions: int
    
    # Game archetype
    archetype: str  # "Blowout", "Close", "Comeback", etc.
    archetype_confidence: float


class BettingOpportunity(TypedDict):
    """Complete betting opportunity with all analysis"""
    game: GameData
    mamba: MambaPrediction
    odds: BetOnlineOdds
    risk: OntoRiskAnalysis
    
    # Decision
    should_bet: bool
    bet_side: str  # "HOME" or "AWAY"
    bet_line: str  # "Lakers -6 @ -110"
    expected_value: float  # EV calculation
    
    # Metadata
    opportunity_id: str  # Unique ID
    created_at: str  # ISO timestamp


class SystemStatus(TypedDict):
    """Overall system health"""
    mamba_loaded: bool
    nba_api_connected: bool
    betonline_scraper_active: bool
    ontorisk_enabled: bool
    
    # Performance
    total_predictions_today: int
    avg_mae_today: float
    win_rate_today: float
    
    # Bankroll
    starting_bankroll: float
    current_bankroll: float
    total_profit: float
    roi: float
    
    # Errors
    last_error: Optional[str]
    error_count_today: int


class WebSocketMessage(TypedDict):
    """Complete message sent to frontend via WebSocket"""
    type: str  # "update", "opportunity", "error", "status"
    timestamp: str  # ISO format
    
    # Data
    live_games: List[GameData]
    opportunities: List[BettingOpportunity]
    system_status: SystemStatus
    
    # Logs (optional, for debugging)
    recent_logs: Optional[List[str]]


# ============================================================================
# EXAMPLE WEBSOCKET MESSAGE
# ============================================================================

EXAMPLE_MESSAGE = {
    "type": "update",
    "timestamp": "2025-10-27T19:06:23Z",
    
    "live_games": [
        {
            "game_id": "0022500123",
            "home_team": "GSW",
            "away_team": "LAL",
            "home_score": 52,
            "away_score": 48,
            "period": 2,
            "clock": "6:00",
            "status_text": "Q2 6:00 - PREDICTION WINDOW OPEN",
            "current_diff": 4,
            "can_predict": True,
            "is_q2_6min": True
        }
    ],
    
    "opportunities": [
        {
            "game": {
                "game_id": "0022500123",
                "home_team": "GSW",
                "away_team": "LAL",
                "home_score": 52,
                "away_score": 48,
                "period": 2,
                "clock": "6:00",
                "status_text": "Q2 6:00",
                "current_diff": 4,
                "can_predict": True,
                "is_q2_6min": True
            },
            
            "mamba": {
                "game_id": "0022500123",
                "matchup": "LAL @ GSW",
                "period": 2,
                "clock": "6:00",
                "current_score": "48-52",
                "current_diff": 4,
                "mamba_prediction": 8.3,
                "branch": "B",
                "mae": 9.655,
                "features_used": 33,
                "feature_breakdown": {
                    "pattern_mean": -2.1,
                    "pattern_std": 3.4,
                    "spectral_energy": 12.5,
                    "team_form_10games": 5.2
                },
                "timestamp": "2025-10-27T19:06:20Z",
                "model": "MAMBA_MENTALITY",
                "confidence_interval": [-1.4, 18.0]
            },
            
            "odds": {
                "spread": -6.0,
                "spread_odds": -110,
                "total": 225.5,
                "over_odds": -110,
                "under_odds": -110,
                "moneyline_home": -240,
                "moneyline_away": 200,
                "spread_implied_prob": 0.524,
                "no_vig_prob": 0.512,
                "vig_percentage": 4.5,
                "favorite": "GSW",
                "favorite_spread": "Warriors -6",
                "last_updated": "2025-10-27T19:06:15Z"
            },
            
            "risk": {
                "calibrated_prob": 0.68,
                "market_prob": 0.512,
                "edge": 0.168,
                "kelly_fraction": 0.15,
                "recommended_stake": 150.0,
                "kelly_percentage": 15.0,
                "can_bet": True,
                "risk_score": 35.0,
                "risk_alerts": [],
                "current_bankroll": 1000.0,
                "peak_bankroll": 1200.0,
                "current_drawdown": 16.7,
                "daily_loss": 0.0,
                "open_positions": 0,
                "archetype": "Close Game",
                "archetype_confidence": 0.85
            },
            
            "should_bet": True,
            "bet_side": "HOME",
            "bet_line": "Warriors -6 @ -110",
            "expected_value": 25.2,
            "opportunity_id": "0022500123_q2_6min",
            "created_at": "2025-10-27T19:06:23Z"
        }
    ],
    
    "system_status": {
        "mamba_loaded": True,
        "nba_api_connected": True,
        "betonline_scraper_active": False,
        "ontorisk_enabled": True,
        "total_predictions_today": 5,
        "avg_mae_today": 8.2,
        "win_rate_today": 0.6,
        "starting_bankroll": 1000.0,
        "current_bankroll": 1000.0,
        "total_profit": 0.0,
        "roi": 0.0,
        "last_error": None,
        "error_count_today": 0
    },
    
    "recent_logs": [
        "🔍 EXTRACTING REAL MAMBA FEATURES (33)...",
        "✅ EXTRACTED 33 REAL MAMBA FEATURES",
        "🔮 Running Mamba model...",
        "✅ MAMBA PREDICTION: +8.3 points",
        "🎯 OntoRisk: Edge = 16.8%, Recommended stake = $150"
    ]
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_websocket_message(
    live_games: List[Dict],
    opportunities: List[Dict],
    system_status: Dict
) -> Dict:
    """
    Create a complete WebSocket message
    
    Args:
        live_games: List of live game dicts
        opportunities: List of betting opportunity dicts
        system_status: System status dict
        
    Returns:
        WebSocket message dict
    """
    return {
        "type": "update",
        "timestamp": datetime.now().isoformat(),
        "live_games": live_games,
        "opportunities": opportunities,
        "system_status": system_status,
        "recent_logs": None  # Optional
    }


def create_error_message(error: str) -> Dict:
    """Create error WebSocket message"""
    return {
        "type": "error",
        "timestamp": datetime.now().isoformat(),
        "error": error,
        "live_games": [],
        "opportunities": [],
        "system_status": {}
    }


def create_status_message(status: Dict) -> Dict:
    """Create status-only WebSocket message"""
    return {
        "type": "status",
        "timestamp": datetime.now().isoformat(),
        "system_status": status,
        "live_games": [],
        "opportunities": []
    }

