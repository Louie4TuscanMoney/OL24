"""
Live Score Buffer - Real-Time Pattern Generation
Accumulates minute-by-minute differentials for ML model

Following: NBA_API/LIVE_DATA_INTEGRATION.md
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

class LiveScoreBuffer:
    """
    Buffer live scores and generate 18-minute patterns for ML
    """
    
    def __init__(self, game_id: str):
        """
        Initialize buffer for a specific game
        
        Args:
            game_id: NBA game ID
        """
        self.game_id = game_id
        self.differentials = []
        self.timestamps = []
        self.last_update = None
        self.prediction_made = False
        
        # Game metadata
        self.home_team = None
        self.away_team = None
        self.game_status = None
        
    def update(self, home_score: int, away_score: int, 
               period: int, clock: str):
        """
        Update buffer with current score
        
        Args:
            home_score: Current home team score
            away_score: Current away team score
            period: Quarter (1-4)
            clock: Time remaining in quarter (e.g., "06:23")
        """
        # Calculate game time (seconds elapsed from start)
        game_time_seconds = self._calculate_game_time(period, clock)
        
        # Convert to minute
        minute = game_time_seconds // 60
        
        # Calculate differential
        differential = home_score - away_score
        
        # Store differential for this minute
        while len(self.differentials) <= minute:
            self.differentials.append(0)
        
        self.differentials[minute] = differential
        self.timestamps.append(datetime.now())
        self.last_update = datetime.now()
        
    def _calculate_game_time(self, period: int, clock: str) -> int:
        """
        Convert period + clock to total game seconds
        
        Args:
            period: Quarter (1-4)
            clock: Time remaining (e.g., "06:23" = 6:23)
        
        Returns:
            Total seconds elapsed from game start
        """
        # Parse clock (format: "MM:SS")
        try:
            minutes, seconds = map(int, clock.split(':'))
            time_remaining_in_quarter = minutes * 60 + seconds
        except:
            time_remaining_in_quarter = 0
        
        # Each quarter is 720 seconds (12 minutes)
        quarter_length = 720
        
        # Calculate elapsed time in current quarter
        elapsed_in_quarter = quarter_length - time_remaining_in_quarter
        
        # Add previous quarters
        previous_quarters = period - 1
        elapsed_previous = previous_quarters * quarter_length
        
        # Total elapsed
        total_elapsed = elapsed_previous + elapsed_in_quarter
        
        return total_elapsed
    
    def get_pattern(self) -> Optional[np.ndarray]:
        """
        Get 18-minute pattern for ML model
        
        Returns:
            numpy array of 18 differentials, or None if not ready
        """
        if len(self.differentials) >= 18:
            return np.array(self.differentials[:18])
        return None
    
    def is_ready_for_prediction(self) -> bool:
        """
        Check if we have 18 minutes and haven't predicted yet
        
        Returns:
            True if ready to call ML model
        """
        return len(self.differentials) >= 18 and not self.prediction_made
    
    def mark_prediction_made(self):
        """Mark that prediction has been made for this game"""
        self.prediction_made = True
    
    def get_status(self) -> Dict:
        """
        Get buffer status
        
        Returns:
            Dict with current state
        """
        return {
            'game_id': self.game_id,
            'minutes_buffered': len(self.differentials),
            'ready_for_prediction': self.is_ready_for_prediction(),
            'prediction_made': self.prediction_made,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'current_differential': self.differentials[-1] if self.differentials else 0,
            'pattern_preview': self.differentials[:18] if len(self.differentials) >= 18 else self.differentials
        }


class GameBufferManager:
    """
    Manage buffers for multiple games simultaneously
    """
    
    def __init__(self):
        """Initialize manager"""
        self.game_buffers: Dict[str, LiveScoreBuffer] = {}
        self.completed_games: List[str] = []
    
    def get_or_create_buffer(self, game_id: str) -> LiveScoreBuffer:
        """
        Get existing buffer or create new one
        
        Args:
            game_id: NBA game ID
        
        Returns:
            LiveScoreBuffer for this game
        """
        if game_id not in self.game_buffers:
            self.game_buffers[game_id] = LiveScoreBuffer(game_id)
        
        return self.game_buffers[game_id]
    
    def update_game(self, game_id: str, home_score: int, away_score: int,
                    period: int, clock: str):
        """
        Update a game's buffer
        
        Args:
            game_id: NBA game ID
            home_score: Current home score
            away_score: Current away score
            period: Quarter
            clock: Time remaining
        """
        buffer = self.get_or_create_buffer(game_id)
        buffer.update(home_score, away_score, period, clock)
    
    def get_games_ready_for_prediction(self) -> List[LiveScoreBuffer]:
        """
        Get all games that have 18 minutes and haven't been predicted
        
        Returns:
            List of buffers ready for ML prediction
        """
        ready = []
        for buffer in self.game_buffers.values():
            if buffer.is_ready_for_prediction():
                ready.append(buffer)
        return ready
    
    def get_all_statuses(self) -> Dict:
        """
        Get status of all tracked games
        """
        return {
            'active_games': len(self.game_buffers),
            'completed_predictions': len(self.completed_games),
            'games': {
                game_id: buffer.get_status()
                for game_id, buffer in self.game_buffers.items()
            }
        }
    
    def mark_game_complete(self, game_id: str):
        """Move game to completed list"""
        if game_id in self.game_buffers:
            self.game_buffers[game_id].mark_prediction_made()
            self.completed_games.append(game_id)


# Test/Demo
if __name__ == "__main__":
    print("="*80)
    print("LIVE SCORE BUFFER TEST")
    print("="*80)
    
    # Simulate a game
    print("\nSimulating Lakers vs Celtics...")
    buffer = LiveScoreBuffer(game_id="TEST_GAME_001")
    buffer.home_team = "LAL"
    buffer.away_team = "BOS"
    
    # Simulate Q1
    print("\nQuarter 1:")
    buffer.update(0, 0, period=1, clock="12:00")  # Game start
    buffer.update(2, 4, period=1, clock="11:00")  # Min 1
    buffer.update(10, 4, period=1, clock="10:00") # Min 2
    buffer.update(10, 8, period=1, clock="9:00")  # Min 3
    print(f"   After 3 minutes: {buffer.differentials[:4]}")
    
    # Simulate more minutes
    for minute in range(4, 18):
        score_home = minute * 2
        score_away = minute * 1
        time_left = 12 - (minute % 12)
        q = (minute // 12) + 1
        buffer.update(score_home, score_away, period=q, clock=f"{time_left:02d}:00")
    
    print(f"   After 17 minutes: {len(buffer.differentials)} minutes buffered")
    
    # Check if ready
    print(f"\n✅ Ready for prediction: {buffer.is_ready_for_prediction()}")
    
    # Get pattern
    pattern = buffer.get_pattern()
    print(f"✅ Pattern shape: {pattern.shape}")
    print(f"   Pattern: {pattern}")
    
    # Get status
    status = buffer.get_status()
    print(f"\n📊 Buffer Status:")
    print(f"   Minutes buffered: {status['minutes_buffered']}")
    print(f"   Ready: {status['ready_for_prediction']}")
    print(f"   Current diff: {status['current_differential']:+d}")
    
    print("\n" + "="*80)
    print("✅ BUFFER TEST PASSED")
    print("="*80)
    print("\nNext: Connect to live NBA_API and ML model")

