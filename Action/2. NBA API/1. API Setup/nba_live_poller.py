"""
NBA Live Score Poller
Fetches live NBA games every 10 seconds and feeds to buffer

Following: NBA_API/LIVE_DATA_INTEGRATION.md
Connects to: ML model from Action/1. ML/X. MVP Model/
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
import sys
from pathlib import Path

from live_score_buffer import GameBufferManager

# Import nba_api
try:
    from nba_api.live.nba.endpoints import scoreboard
    print("✅ NBA_API imported")
except ImportError:
    print("❌ NBA_API not installed. Run: pip install nba-api")
    sys.exit(1)


class NBALivePoller:
    """
    Poll NBA live scores and generate patterns for ML
    """
    
    def __init__(self, poll_interval=10):
        """
        Args:
            poll_interval: Seconds between polls (default 10)
        """
        self.poll_interval = poll_interval
        self.buffer_manager = GameBufferManager()
        self.poll_count = 0
        self.error_count = 0
        self.start_time = None
        
    def fetch_live_games(self) -> List[Dict]:
        """
        Fetch current live games from NBA_API
        
        Returns:
            List of game dicts
        """
        try:
            board = scoreboard.ScoreBoard()
            games = board.games.get_dict()
            return games
        except Exception as e:
            self.error_count += 1
            print(f"❌ Error fetching scoreboard: {e}")
            return []
    
    def process_game_update(self, game_data: Dict):
        """
        Process a single game update
        
        Args:
            game_data: Game dict from NBA_API
        """
        game_id = game_data['gameId']
        
        # Extract scores
        home_score = game_data['homeTeam']['score']
        away_score = game_data['awayTeam']['score']
        period = game_data['period']
        clock = game_data['gameClock']
        
        # Update buffer
        self.buffer_manager.update_game(
            game_id=game_id,
            home_score=home_score,
            away_score=away_score,
            period=period,
            clock=clock
        )
    
    def check_for_predictions(self) -> List[Dict]:
        """
        Check if any games are ready for ML prediction
        
        Returns:
            List of games ready for prediction with their patterns
        """
        ready_games = []
        
        for buffer in self.buffer_manager.get_games_ready_for_prediction():
            pattern = buffer.get_pattern()
            
            if pattern is not None:
                ready_games.append({
                    'game_id': buffer.game_id,
                    'pattern': pattern,
                    'home_team': buffer.home_team,
                    'away_team': buffer.away_team,
                    'buffer': buffer
                })
        
        return ready_games
    
    def poll_once(self):
        """
        Single poll iteration
        """
        self.poll_count += 1
        
        print(f"\n[Poll #{self.poll_count}] {datetime.now().strftime('%H:%M:%S')}")
        
        # Fetch live games
        games = self.fetch_live_games()
        
        if not games:
            print("  No live games currently")
            return
        
        print(f"  Found {len(games)} live games")
        
        # Process each game
        for game in games:
            game_status = game['gameStatusText']
            
            # Only process games in progress
            if game_status not in ['Final', 'Halftime']:
                self.process_game_update(game)
                
                away = game['awayTeam']['teamTricode']
                home = game['homeTeam']['teamTricode']
                away_score = game['awayTeam']['score']
                home_score = game['homeTeam']['score']
                
                print(f"  {away} @ {home}: {away_score}-{home_score} (Q{game['period']} {game['gameClock']})")
        
        # Check for prediction opportunities
        ready = self.check_for_predictions()
        
        if ready:
            print(f"\n  🎯 {len(ready)} games ready for ML prediction!")
            for game_info in ready:
                print(f"     Game {game_info['game_id']}: Pattern ready ({len(game_info['pattern'])} minutes)")
    
    def start(self, duration_minutes: Optional[int] = None):
        """
        Start polling loop
        
        Args:
            duration_minutes: How long to run (None = forever)
        """
        print("="*80)
        print("NBA LIVE POLLER STARTED")
        print("="*80)
        print(f"Poll interval: {self.poll_interval} seconds")
        if duration_minutes:
            print(f"Duration: {duration_minutes} minutes")
        else:
            print("Duration: Continuous (Ctrl+C to stop)")
        
        self.start_time = time.time()
        end_time = self.start_time + (duration_minutes * 60) if duration_minutes else None
        
        try:
            while True:
                # Poll
                self.poll_once()
                
                # Check if done
                if end_time and time.time() >= end_time:
                    print(f"\n⏱️  Duration limit reached ({duration_minutes} minutes)")
                    break
                
                # Wait for next poll
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")
        
        # Summary
        elapsed = time.time() - self.start_time
        print(f"\n" + "="*80)
        print("POLLING SUMMARY")
        print("="*80)
        print(f"Runtime: {elapsed/60:.1f} minutes")
        print(f"Polls: {self.poll_count}")
        print(f"Errors: {self.error_count}")
        print(f"Games tracked: {len(self.buffer_manager.game_buffers)}")
        print(f"Predictions made: {len(self.buffer_manager.completed_games)}")


# Demo/Test
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Live Score Poller')
    parser.add_argument('--duration', type=int, default=None,
                       help='Run for N minutes (default: continuous)')
    parser.add_argument('--interval', type=int, default=10,
                       help='Poll interval in seconds (default: 10)')
    
    args = parser.parse_args()
    
    # Create poller
    poller = NBALivePoller(poll_interval=args.interval)
    
    # Start
    poller.start(duration_minutes=args.duration)

