"""
GAME DATA LOGGER - SAVES EVERYTHING FOR ML ANALYSIS
Logs scores + odds + predictions every update
"""

import json
import os
from datetime import datetime
from typing import Dict, List


class GameDataLogger:
    """
    Logs all game data to JSON for ML analysis
    """
    
    def __init__(self, log_dir: str = "data/game_logs"):
        """Initialize logger"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Today's log file
        today = datetime.now().strftime("%Y%m%d")
        self.log_file = os.path.join(log_dir, f"games_{today}.json")
        
        # Load existing data if any
        self.data = self._load_existing()
        
        print(f"✅ Game Data Logger: {self.log_file}")
    
    def _load_existing(self) -> Dict:
        """Load existing log file"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "games": {},
            "total_updates": 0
        }
    
    def log_game_state(
        self,
        game_id: str,
        game_data: Dict,
        odds_data: Dict = None,
        prediction: Dict = None
    ):
        """
        Log game state
        
        Args:
            game_id: Game ID
            game_data: NBA game data
            odds_data: BetOnline odds (optional)
            prediction: ML prediction (optional)
        """
        timestamp = datetime.now().isoformat()
        
        # Initialize game if not exists
        if game_id not in self.data["games"]:
            self.data["games"][game_id] = {
                "game_id": game_id,
                "home_team": game_data.get("home_team"),
                "away_team": game_data.get("away_team"),
                "updates": []
            }
        
        # Create update entry
        update = {
            "timestamp": timestamp,
            "period": game_data.get("period"),
            "clock": game_data.get("clock"),
            "home_score": game_data.get("home_score"),
            "away_score": game_data.get("away_score"),
            "current_diff": game_data.get("current_diff"),
            "status": game_data.get("status_text"),
            "is_q2_6min": game_data.get("is_q2_6min", False)
        }
        
        # Add odds if available
        if odds_data:
            update["odds"] = {
                "spread": odds_data.get("spread"),
                "total": odds_data.get("total"),
                "home_ml": odds_data.get("moneyline_home"),
                "away_ml": odds_data.get("moneyline_away"),
                "locked": odds_data.get("locked", False),
                "source": odds_data.get("source")
            }
        
        # Add prediction if available
        if prediction:
            update["prediction"] = {
                "pred_diff": prediction.get("predicted_diff"),
                "confidence": prediction.get("confidence"),
                "edge": prediction.get("edge"),
                "bet_recommendation": prediction.get("should_bet")
            }
        
        # Append update
        self.data["games"][game_id]["updates"].append(update)
        self.data["total_updates"] += 1
        
        # Save to disk (fast write)
        self._save()
    
    def _save(self):
        """Save data to disk (optimized)"""
        try:
            # Write to temp file first (atomic)
            temp_file = self.log_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            
            # Move to actual file (atomic)
            os.replace(temp_file, self.log_file)
        except Exception as e:
            print(f"⚠️ Logger write error: {e}")
    
    def get_game_history(self, game_id: str) -> List[Dict]:
        """Get all updates for a game"""
        if game_id in self.data["games"]:
            return self.data["games"][game_id]["updates"]
        return []
    
    def get_summary(self) -> Dict:
        """Get logging summary"""
        return {
            "total_games": len(self.data["games"]),
            "total_updates": self.data["total_updates"],
            "log_file": self.log_file
        }


# Global logger instance
_logger = None

def get_logger():
    """Get global logger"""
    global _logger
    if _logger is None:
        _logger = GameDataLogger()
    return _logger

