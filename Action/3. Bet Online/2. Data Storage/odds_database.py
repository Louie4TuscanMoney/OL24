"""
Odds Database - Store BetOnline odds time series
In-memory storage with optional persistence
"""

from datetime import datetime
from typing import Dict, List
import json

class OddsDatabase:
    """
    Store and query odds data
    """
    
    def __init__(self):
        """Initialize database"""
        self.odds_history: Dict[str, List[Dict]] = {}  # game_id -> list of odds snapshots
        self.latest_odds: Dict[str, Dict] = {}  # game_id -> latest odds
        
    def store_odds(self, game_id: str, odds_data: Dict):
        """
        Store odds snapshot
        
        Args:
            game_id: Game identifier
            odds_data: Odds dict with timestamp
        """
        # Add timestamp if not present
        if 'timestamp' not in odds_data:
            odds_data['timestamp'] = datetime.now().isoformat()
        
        # Initialize history for game
        if game_id not in self.odds_history:
            self.odds_history[game_id] = []
        
        # Append to history
        self.odds_history[game_id].append(odds_data)
        
        # Update latest
        self.latest_odds[game_id] = odds_data
    
    def get_latest_odds(self, game_id: str) -> Optional[Dict]:
        """Get most recent odds for game"""
        return self.latest_odds.get(game_id)
    
    def get_odds_history(self, game_id: str) -> List[Dict]:
        """Get full odds history for game"""
        return self.odds_history.get(game_id, [])
    
    def get_line_movement(self, game_id: str) -> Dict:
        """
        Calculate line movement for game
        """
        history = self.get_odds_history(game_id)
        
        if len(history) < 2:
            return {'movement': None}
        
        first = history[0]
        latest = history[-1]
        
        # Calculate movement (simplified)
        return {
            'initial': first,
            'current': latest,
            'snapshots': len(history),
            'time_span_seconds': (datetime.fromisoformat(latest['timestamp']) - 
                                 datetime.fromisoformat(first['timestamp'])).total_seconds()
        }
    
    def save_to_file(self, filepath: str):
        """Save all data to JSON file"""
        data = {
            'odds_history': self.odds_history,
            'latest_odds': self.latest_odds,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Saved {len(self.odds_history)} games to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load data from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.odds_history = data['odds_history']
        self.latest_odds = data['latest_odds']
        
        print(f"✅ Loaded {len(self.odds_history)} games from {filepath}")

