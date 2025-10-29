"""
Game Data Model
Type-safe contract for NBA game data throughout the system
"""

from typing import TypedDict
from enum import IntEnum


class GameStatus(IntEnum):
    """Game status codes"""
    SCHEDULED = 1
    LIVE = 2
    FINAL = 3


class GameData(TypedDict):
    """
    Type-safe game data contract
    
    This is the SINGLE SOURCE OF TRUTH for game data structure.
    Backend and Frontend MUST match this exactly!
    """
    # Identifiers
    game_id: str
    
    # Teams
    home_team: str
    away_team: str
    
    # Scores
    score_home: int
    score_away: int
    
    # Game state
    quarter: int
    time_remaining: str
    clock: str
    is_live: bool
    status: int
    status_text: str
    
    # Scheduling
    game_time: str  # "07:00 PM ET"
    game_date: str  # "Oct 28, 2025"
    
    # ML flags
    is_q2_6min: bool
    can_predict: bool
    
    # Metadata
    timestamp: str


def create_empty_game(game_id: str = "") -> GameData:
    """Create an empty game object with default values"""
    from datetime import datetime
    
    return {
        'game_id': game_id,
        'home_team': '',
        'away_team': '',
        'score_home': 0,
        'score_away': 0,
        'quarter': 0,
        'time_remaining': '',
        'clock': '',
        'is_live': False,
        'status': GameStatus.SCHEDULED,
        'status_text': 'SCHEDULED',
        'game_time': '',
        'game_date': '',
        'is_q2_6min': False,
        'can_predict': False,
        'timestamp': datetime.now().isoformat()
    }


def validate_game_data(game: dict) -> bool:
    """
    Validate that a dict matches the GameData contract
    
    Args:
        game: Dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'game_id', 'home_team', 'away_team', 'score_home', 'score_away',
        'quarter', 'time_remaining', 'clock', 'is_live', 'status',
        'status_text', 'game_time', 'game_date', 'is_q2_6min',
        'can_predict', 'timestamp'
    ]
    
    for field in required_fields:
        if field not in game:
            print(f"❌ Missing required field: {field}")
            return False
    
    return True

