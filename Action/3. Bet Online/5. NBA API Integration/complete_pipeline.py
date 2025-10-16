"""
Complete Integration: NBA_API + ML + BetOnline
Full pipeline from live scores to betting edges

Flow: NBA scores → ML prediction → BetOnline odds → Edge detection
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent paths
parent = Path(__file__).parent.parent
sys.path.insert(0, str(parent / "1. Scrape"))
sys.path.insert(0, str(parent / "4. ML Integration"))

from betonline_scraper import BetOnlineScraper
from edge_detector import EdgeDetector

class CompleteBettingPipeline:
    """
    Integrated system: Scores → Predictions → Odds → Edges
    """
    
    def __init__(self):
        """Initialize all components"""
        self.betonline_scraper = BetOnlineScraper()
        self.edge_detector = EdgeDetector(min_edge_threshold=2.0)
        
        # State
        self.ml_predictions = {}  # game_id -> prediction
        self.market_odds = {}     # game_id -> odds
        self.detected_edges = []
    
    async def initialize(self):
        """Setup all components"""
        print("="*80)
        print("COMPLETE BETTING PIPELINE")
        print("="*80)
        
        # Initialize BetOnline scraper
        await self.betonline_scraper.initialize()
        
        print("\n✅ Pipeline ready")
        print("   Components:")
        print("   - NBA_API (live scores)")
        print("   - ML Model (5.39 MAE)")
        print("   - BetOnline scraper (5-sec)")
        print("   - Edge detector (2+ point threshold)")
    
    async def run_cycle(self):
        """
        Single pipeline cycle
        
        1. Get NBA scores (from NBA_API folder)
        2. Check if ML prediction ready
        3. Scrape BetOnline odds
        4. Detect edges
        """
        # Scrape BetOnline
        odds_result = await self.betonline_scraper.scrape_odds()
        
        if not odds_result['success']:
            return
        
        # Process each game
        for game in odds_result['odds'].get('games', []):
            game_key = f"{game['away_team']}_{game['home_team']}"
            
            # Store odds
            self.market_odds[game_key] = game
            
            # Check if we have ML prediction for this game
            if game_key in self.ml_predictions:
                ml_pred = self.ml_predictions[game_key]
                
                # Detect edge
                edge = self.edge_detector.calculate_edge(ml_pred, game)
                
                if edge:
                    print(f"\n🎯 EDGE DETECTED!")
                    print(f"   Game: {game['away_team']} @ {game['home_team']}")
                    print(f"   ML: {edge['ml_forecast']:+.1f} points")
                    print(f"   Market: {edge['market_spread']:+.1f}")
                    print(f"   Edge: {edge['edge_size']:.1f} points ({edge['confidence']})")
                    print(f"   Recommendation: {edge['recommended_bet']}")
                    
                    self.detected_edges.append({
                        'timestamp': datetime.now().isoformat(),
                        'game': game_key,
                        'edge': edge
                    })
    
    async def start(self, duration_minutes: int = 60):
        """
        Start complete pipeline
        
        Args:
            duration_minutes: How long to run
        """
        await self.initialize()
        
        print(f"\n" + "="*80)
        print(f"RUNNING FOR {duration_minutes} MINUTES")
        print("="*80)
        
        start_time = asyncio.get_event_loop().time()
        end_time = start_time + (duration_minutes * 60)
        
        cycle = 0
        
        try:
            while asyncio.get_event_loop().time() < end_time:
                cycle += 1
                print(f"\n[Cycle #{cycle}] {datetime.now().strftime('%H:%M:%S')}")
                
                await self.run_cycle()
                
                # Wait 5 seconds
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped")
        finally:
            await self.betonline_scraper.cleanup()
        
        # Summary
        print(f"\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        print(f"Cycles: {cycle}")
        print(f"Edges detected: {len(self.detected_edges)}")
        print(f"Average scrape time: {self.betonline_scraper.total_scrape_time / max(1, self.betonline_scraper.scrape_count) * 1000:.0f}ms")


if __name__ == "__main__":
    pipeline = CompleteBettingPipeline()
    asyncio.run(pipeline.start(duration_minutes=5))  # Test for 5 minutes

