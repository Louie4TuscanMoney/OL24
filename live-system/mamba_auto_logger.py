"""
MAMBA AUTO-LOGGER - STORE EVERY INTERACTION WITH LIVE DATA

Purpose: Automatically log all Mamba predictions with live data for daily review
Author: Ontologic XYZ
Date: October 27, 2025

This stores:
- Every prediction made
- Live game state at prediction time
- BetOnline odds at prediction time
- Actual final outcome
- Prediction accuracy (MAE)
- Profit/loss if bet was placed
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class MambaAutoLogger:
    """
    Automatic logging system for all Mamba interactions
    """
    
    def __init__(self, log_dir: str = "mamba_logs"):
        """
        Initialize auto-logger
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.log_dir / "daily").mkdir(exist_ok=True)
        (self.log_dir / "games").mkdir(exist_ok=True)
        (self.log_dir / "summaries").mkdir(exist_ok=True)
        
        print(f"✅ Mamba Auto-Logger initialized: {self.log_dir}")
    
    def log_prediction(
        self,
        game_id: str,
        game_data: Dict,
        prediction: float,
        odds: Dict,
        features: Dict = None,
        metadata: Dict = None
    ):
        """
        Log a Mamba prediction with full context
        
        Args:
            game_id: NBA game ID
            game_data: Live game state
            prediction: Mamba prediction
            odds: BetOnline odds at time of prediction
            features: Extracted features (optional)
            metadata: Additional metadata (optional)
        """
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H:%M:%S")
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "date": date_str,
            "time": time_str,
            "game_id": game_id,
            "game_data": {
                "home_team": game_data.get("home_team"),
                "away_team": game_data.get("away_team"),
                "home_score": game_data.get("home_score"),
                "away_score": game_data.get("away_score"),
                "period": game_data.get("period"),
                "clock": game_data.get("clock"),
                "current_diff": game_data.get("current_diff"),
                "game_status": game_data.get("game_status")
            },
            "prediction": {
                "mamba_prediction": prediction,
                "spread_line": odds.get("spread"),
                "edge": abs(prediction - odds.get("spread", 0)),
                "prediction_type": "live_q2_6min"
            },
            "odds": {
                "spread": odds.get("spread"),
                "total": odds.get("total"),
                "home_ml": odds.get("home_ml"),
                "away_ml": odds.get("away_ml"),
                "home_implied_prob": odds.get("home_implied_prob"),
                "away_implied_prob": odds.get("away_implied_prob"),
                "vig_percentage": odds.get("vig_percentage"),
                "source": odds.get("source", "betonline")
            },
            "features": features if features else {},
            "metadata": metadata if metadata else {},
            "outcome": None,  # Will be updated post-game
            "accuracy": None,  # Will be calculated post-game
            "profit_loss": None  # Will be calculated if bet placed
        }
        
        # Save to daily log
        daily_file = self.log_dir / "daily" / f"{date_str}.json"
        self._append_to_daily_log(daily_file, log_entry)
        
        # Save to game-specific log
        game_file = self.log_dir / "games" / f"{game_id}.json"
        self._append_to_game_log(game_file, log_entry)
        
        print(f"📝 Logged prediction: {game_data['away_team']}@{game_data['home_team']} - Mamba: {prediction:+.1f}")
        
        return log_entry
    
    def _append_to_daily_log(self, file_path: Path, entry: Dict):
        """Append entry to daily log file"""
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            data = {"date": entry["date"], "predictions": []}
        
        data["predictions"].append(entry)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _append_to_game_log(self, file_path: Path, entry: Dict):
        """Append entry to game-specific log file"""
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            data = {"game_id": entry["game_id"], "predictions": []}
        
        data["predictions"].append(entry)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_outcome(
        self,
        game_id: str,
        final_home_score: int,
        final_away_score: int,
        bet_placed: bool = False,
        bet_amount: float = 0,
        bet_result: str = None
    ):
        """
        Update log entry with final outcome
        
        Args:
            game_id: NBA game ID
            final_home_score: Final home score
            final_away_score: Final away score
            bet_placed: Whether a bet was placed
            bet_amount: Bet amount if placed
            bet_result: "win", "loss", or "push"
        """
        game_file = self.log_dir / "games" / f"{game_id}.json"
        
        if not game_file.exists():
            print(f"⚠️ No log found for game {game_id}")
            return
        
        with open(game_file, 'r') as f:
            data = json.load(f)
        
        # Calculate actual outcome
        actual_diff = final_home_score - final_away_score
        
        # Update all predictions for this game
        for pred in data["predictions"]:
            mamba_pred = pred["prediction"]["mamba_prediction"]
            spread_line = pred["prediction"]["spread_line"]
            
            # Calculate MAE
            mae = abs(mamba_pred - actual_diff)
            
            # Update prediction
            pred["outcome"] = {
                "final_home_score": final_home_score,
                "final_away_score": final_away_score,
                "actual_diff": actual_diff,
                "updated_at": datetime.now().isoformat()
            }
            pred["accuracy"] = {
                "mae": mae,
                "beat_spread": abs(mamba_pred - spread_line) > 5.0,
                "within_10pts": mae <= 10.0,
                "within_5pts": mae <= 5.0
            }
            
            if bet_placed:
                # Calculate profit/loss
                if bet_result == "win":
                    profit = bet_amount * 0.91  # -110 odds = 0.91 profit
                elif bet_result == "loss":
                    profit = -bet_amount
                else:  # push
                    profit = 0
                
                pred["profit_loss"] = {
                    "bet_placed": True,
                    "bet_amount": bet_amount,
                    "bet_result": bet_result,
                    "profit": profit,
                    "roi": (profit / bet_amount) if bet_amount > 0 else 0
                }
        
        # Save updated data
        with open(game_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Updated outcome for {game_id}: Actual: {actual_diff:+.1f}, MAE: {mae:.2f}")
    
    def get_daily_summary(self, date: str = None) -> Dict:
        """
        Get summary of predictions for a specific date
        
        Args:
            date: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            Summary dictionary
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        daily_file = self.log_dir / "daily" / f"{date}.json"
        
        if not daily_file.exists():
            return {"date": date, "total_predictions": 0}
        
        with open(daily_file, 'r') as f:
            data = json.load(f)
        
        predictions = data["predictions"]
        
        # Calculate summary stats
        total = len(predictions)
        with_outcomes = [p for p in predictions if p["outcome"] is not None]
        
        if not with_outcomes:
            return {
                "date": date,
                "total_predictions": total,
                "pending_outcomes": total
            }
        
        maes = [p["accuracy"]["mae"] for p in with_outcomes]
        avg_mae = sum(maes) / len(maes)
        
        within_5 = sum(1 for p in with_outcomes if p["accuracy"]["within_5pts"])
        within_10 = sum(1 for p in with_outcomes if p["accuracy"]["within_10pts"])
        
        # Betting summary
        bets_placed = [p for p in with_outcomes if p.get("profit_loss", {}).get("bet_placed")]
        if bets_placed:
            total_profit = sum(p["profit_loss"]["profit"] for p in bets_placed)
            total_wagered = sum(p["profit_loss"]["bet_amount"] for p in bets_placed)
            roi = (total_profit / total_wagered) if total_wagered > 0 else 0
            wins = sum(1 for p in bets_placed if p["profit_loss"]["bet_result"] == "win")
            losses = sum(1 for p in bets_placed if p["profit_loss"]["bet_result"] == "loss")
            win_rate = wins / len(bets_placed) if bets_placed else 0
        else:
            total_profit = 0
            total_wagered = 0
            roi = 0
            wins = 0
            losses = 0
            win_rate = 0
        
        summary = {
            "date": date,
            "total_predictions": total,
            "completed": len(with_outcomes),
            "pending": total - len(with_outcomes),
            "accuracy": {
                "average_mae": avg_mae,
                "within_5pts": within_5,
                "within_10pts": within_10,
                "within_5pts_pct": within_5 / len(with_outcomes),
                "within_10pts_pct": within_10 / len(with_outcomes)
            },
            "betting": {
                "bets_placed": len(bets_placed),
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "total_wagered": total_wagered,
                "total_profit": total_profit,
                "roi": roi
            }
        }
        
        # Save summary
        summary_file = self.log_dir / "summaries" / f"{date}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def print_daily_report(self, date: str = None):
        """
        Print a formatted daily report
        
        Args:
            date: Date string (YYYY-MM-DD), defaults to today
        """
        summary = self.get_daily_summary(date)
        date = summary["date"]
        
        print("\n" + "="*80)
        print(f"📊 MAMBA DAILY REPORT - {date}")
        print("="*80 + "\n")
        
        print(f"Total Predictions: {summary['total_predictions']}")
        print(f"Completed: {summary['completed']}")
        print(f"Pending: {summary['pending']}")
        
        if summary['completed'] > 0:
            print(f"\n📈 Accuracy:")
            print(f"  Average MAE: {summary['accuracy']['average_mae']:.2f} points")
            print(f"  Within 5pts: {summary['accuracy']['within_5pts']} ({summary['accuracy']['within_5pts_pct']:.1%})")
            print(f"  Within 10pts: {summary['accuracy']['within_10pts']} ({summary['accuracy']['within_10pts_pct']:.1%})")
        
        if summary['betting']['bets_placed'] > 0:
            print(f"\n💰 Betting:")
            print(f"  Bets Placed: {summary['betting']['bets_placed']}")
            print(f"  Wins: {summary['betting']['wins']}")
            print(f"  Losses: {summary['betting']['losses']}")
            print(f"  Win Rate: {summary['betting']['win_rate']:.1%}")
            print(f"  Total Wagered: ${summary['betting']['total_wagered']:,.2f}")
            print(f"  Total Profit: ${summary['betting']['total_profit']:,.2f}")
            print(f"  ROI: {summary['betting']['roi']:.1%}")
        
        print("\n" + "="*80)


def test_auto_logger():
    """Test the auto-logger"""
    logger = MambaAutoLogger()
    
    # Example: Log a prediction
    game_data = {
        "game_id": "0022500123",
        "home_team": "LAL",
        "away_team": "GSW",
        "home_score": 55,
        "away_score": 48,
        "period": 2,
        "clock": "6:00",
        "current_diff": 7,
        "game_status": "live"
    }
    
    odds = {
        "spread": -6.0,
        "total": 215.5,
        "home_ml": -250,
        "away_ml": +210,
        "home_implied_prob": 0.714,
        "away_implied_prob": 0.323,
        "vig_percentage": 3.7
    }
    
    logger.log_prediction(
        game_id="0022500123",
        game_data=game_data,
        prediction=-5.2,
        odds=odds
    )
    
    # Example: Update outcome
    logger.update_outcome(
        game_id="0022500123",
        final_home_score=112,
        final_away_score=108,
        bet_placed=True,
        bet_amount=100,
        bet_result="win"
    )
    
    # Print daily report
    logger.print_daily_report()


if __name__ == "__main__":
    test_auto_logger()

