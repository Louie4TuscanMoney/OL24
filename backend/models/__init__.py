"""
Data Models Package
Type-safe data contracts for the entire system
"""

from .game import GameData, GameStatus
from .prediction import MLPrediction, PredictionType
from .bet import BetEntry, BetOutcome

__all__ = [
    'GameData',
    'GameStatus',
    'MLPrediction',
    'PredictionType',
    'BetEntry',
    'BetOutcome'
]

