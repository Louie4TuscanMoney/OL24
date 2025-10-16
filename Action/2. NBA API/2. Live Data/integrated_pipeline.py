"""
Integrated Pipeline: NBA_API → Buffer → ML → WebSocket
Complete end-to-end system

Flow: Live scores → Pattern → Prediction → Broadcast
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add paths
api_setup_path = Path(__file__).parent.parent / "1. API Setup"
sys.path.insert(0, str(api_setup_path))

from live_score_buffer import GameBufferManager
from websocket_server import LiveDataBroadcaster

class IntegratedPipeline:
    """
    Complete NBA → ML → Dashboard pipeline
    """
    
    def __init__(self):
        """Initialize all components"""
        self.buffer_manager = GameBufferManager()
        self.broadcaster = LiveDataBroadcaster(host='0.0.0.0', port=8765)
        self.ml_predictions = {}
        self.is_running = False
        
    async def fetch_and_update_scores(self):
        """
        Fetch live scores and update buffers
        """
        try:
            from nba_api.live.nba.endpoints import scoreboard
            
            # Get live games
            board = scoreboard.ScoreBoard()
            games = board.games.get_dict()
            
            if not games:
                return
            
            # Process each game
            for game in games:
                game_id = game['gameId']
                
                # Skip if finished
                if game['gameStatusText'] in ['Final', 'Final/OT']:
                    continue
                
                # Extract data
                home_score = game['homeTeam']['score']
                away_score = game['awayTeam']['score']
                home_tricode = game['homeTeam']['teamTricode']
                away_tricode = game['awayTeam']['teamTricode']
                period = game['period']
                clock = game['gameClock']
                
                # Update buffer
                self.buffer_manager.update_game(
                    game_id=game_id,
                    home_score=home_score,
                    away_score=away_score,
                    period=period,
                    clock=clock
                )
                
                # Update home/away names
                buffer = self.buffer_manager.get_or_create_buffer(game_id)
                buffer.home_team = home_tricode
                buffer.away_team = away_tricode
                buffer.game_status = game['gameStatusText']
                
                # Broadcast score update
                await self.broadcaster.broadcast_score_update(game_id, {
                    'game_id': game_id,
                    'away_team': away_tricode,
                    'home_team': home_tricode,
                    'away_score': away_score,
                    'home_score': home_score,
                    'differential': home_score - away_score,
                    'period': period,
                    'clock': clock,
                    'status': game['gameStatusText']
                })
                
                # Broadcast pattern progress
                minutes_collected = len(buffer.differentials)
                if minutes_collected < 18:
                    await self.broadcaster.broadcast_pattern_progress(game_id, minutes_collected)
            
        except Exception as e:
            print(f"❌ Error in fetch_and_update: {e}")
    
    async def check_and_make_predictions(self):
        """
        Check for games ready for prediction
        """
        ready_buffers = self.buffer_manager.get_games_ready_for_prediction()
        
        if not ready_buffers:
            return
        
        for buffer in ready_buffers:
            game_id = buffer.game_id
            pattern = buffer.get_pattern()
            
            if pattern is None:
                continue
            
            print(f"\n🎯 Making prediction for {buffer.away_team} @ {buffer.home_team}")
            print(f"   Pattern: {pattern}")
            
            # TODO: Call ML model here
            # For now, create mock prediction
            prediction = {
                'game_id': game_id,
                'away_team': buffer.away_team,
                'home_team': buffer.home_team,
                'point_forecast': 15.0,  # Mock
                'interval_lower': 11.3,
                'interval_upper': 18.9,
                'coverage_probability': 0.95,
                'components': {
                    'dejavu_prediction': 14.1,
                    'lstm_prediction': 15.8,
                    'ensemble_forecast': 15.0
                },
                'status': 'success'
            }
            
            # Store
            self.ml_predictions[game_id] = prediction
            
            # Broadcast to dashboard
            await self.broadcaster.broadcast_prediction(game_id, prediction)
            
            # Mark done
            buffer.mark_prediction_made()
            
            print(f"   ✅ Prediction: {prediction['point_forecast']:+.1f}")
            print(f"   ✅ Broadcast to {len(self.broadcaster.clients)} clients")
    
    async def polling_loop(self):
        """
        Main polling loop
        """
        poll_count = 0
        
        while self.is_running:
            poll_count += 1
            print(f"\n[Poll #{poll_count}] {datetime.now().strftime('%H:%M:%S')}")
            
            # Fetch and update
            await self.fetch_and_update_scores()
            
            # Check for predictions
            await self.check_and_make_predictions()
            
            # Wait 10 seconds
            await asyncio.sleep(10)
    
    async def start(self):
        """
        Start integrated pipeline
        """
        print("="*80)
        print("INTEGRATED PIPELINE STARTING")
        print("="*80)
        print(f"WebSocket: ws://localhost:8765")
        print(f"NBA polling: Every 10 seconds")
        print(f"ML predictions: Auto-trigger at 18 minutes")
        print("\nPress Ctrl+C to stop")
        print("="*80)
        
        self.is_running = True
        
        # Start WebSocket server in background
        websocket_task = asyncio.create_task(self.broadcaster.start())
        
        # Give WebSocket time to start
        await asyncio.sleep(1)
        
        # Start polling
        try:
            await self.polling_loop()
        except KeyboardInterrupt:
            print("\n\n⏹️  Shutting down...")
            self.is_running = False
        finally:
            websocket_task.cancel()


# Run
if __name__ == "__main__":
    pipeline = IntegratedPipeline()
    asyncio.run(pipeline.start())

