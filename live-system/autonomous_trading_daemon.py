"""
AUTONOMOUS TRADING DAEMON

Purpose: Fully autonomous background service that runs 24/7
Author: Ontologic XYZ
Date: October 20, 2025

This daemon:
1. Monitors NBA games continuously
2. Fetches live lines from BetOnline
3. Makes predictions when games hit Q2 6:00
4. Identifies betting opportunities
5. Enforces risk management
6. Logs all activity
7. Serves data to dashboard via API
8. RUNS FOREVER (auto-restart on crash)

Usage:
    python autonomous_trading_daemon.py
    
Or as background service:
    nohup python autonomous_trading_daemon.py > logs/daemon.log 2>&1 &
"""

import time
import asyncio
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import logging
from typing import Dict, List
import traceback

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'daemon_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import components
sys.path.append('../4. Risk')

try:
    from nba_live_scores import NBALiveScores
    from betonline_live_lines import BetOnlineScraper
    from live_trading_engine import LiveTradingEngine
    from ontorisk_phase1_probability_calibration import ProbabilityCalibrator
    from ontorisk_phase4_risk_management import RiskManager, AdaptiveKellyManager
    
    logger.info("✅ All components loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    sys.exit(1)


class AutonomousTradingDaemon:
    """
    Fully autonomous trading daemon
    """
    
    def __init__(
        self,
        check_interval: int = 3,  # Check every 3 seconds (MAXIMUM SPEED!)
        model_path: str = "../Action/HYBRID_ULTIMATE_V2_CLEAN.pkl",
        mae: float = 9.029,
        starting_bankroll: float = 1000
    ):
        """
        Initialize daemon
        
        Args:
            check_interval: Seconds between checks
            model_path: Path to ML model
            mae: Model MAE
            starting_bankroll: Initial capital
        """
        self.check_interval = check_interval
        self.model_path = model_path
        self.mae = mae
        self.starting_bankroll = starting_bankroll
        
        # State file (persistent across restarts)
        self.state_file = Path("state/daemon_state.json")
        self.state_file.parent.mkdir(exist_ok=True)
        
        # Opportunities cache (for dashboard)
        self.opportunities_file = Path("state/current_opportunities.json")
        
        # Initialize components
        self.trading_engine = None
        self.running = False
        self.cycle_count = 0
        
        # Load state
        self.state = self._load_state()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🔥 Autonomous Trading Daemon initialized")
    
    def _load_state(self) -> Dict:
        """Load daemon state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"✅ Loaded state from {self.state_file}")
                return state
            except:
                logger.warning("⚠️ Could not load state, using defaults")
        
        return {
            'current_bankroll': self.starting_bankroll,
            'total_bets_placed': 0,
            'total_profit': 0,
            'last_restart': datetime.now().isoformat(),
            'opportunities_found_today': 0
        }
    
    def _save_state(self):
        """Save daemon state to file"""
        self.state['last_save'] = datetime.now().isoformat()
        
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
        self._save_state()
        sys.exit(0)
    
    def initialize_components(self):
        """Initialize trading components"""
        logger.info("📂 Initializing trading engine...")
        
        try:
            self.trading_engine = LiveTradingEngine(
                model_path=self.model_path,
                mae=self.mae,
                starting_bankroll=self.state['current_bankroll']
            )
            
            # Restore bankroll if restarting
            if self.trading_engine.ontorisk_enabled:
                self.trading_engine.risk_manager.state.current_bankroll = self.state['current_bankroll']
            
            logger.info("✅ Trading engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def monitoring_cycle(self):
        """
        Single monitoring cycle
        
        This runs every check_interval seconds
        """
        self.cycle_count += 1
        
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 Monitoring Cycle #{self.cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*80}")
            
            # Get live games
            logger.info("🏀 Fetching live NBA games...")
            games = self.trading_engine.nba_api.get_todays_games()
            logger.info(f"   Found {len(games)} games")
            
            live_games = [g for g in games if g['status_text'] == 'LIVE']
            logger.info(f"   {len(live_games)} are LIVE")
            
            # Get live lines
            logger.info("💰 Fetching live betting lines...")
            lines = self.trading_engine.line_scraper.get_live_lines()
            logger.info(f"   Found {len(lines)} lines")
            
            # Scan for opportunities
            logger.info("🔍 Scanning for betting opportunities...")
            opportunities = self.trading_engine.scan_live_opportunities()
            logger.info(f"   Found {len(opportunities)} opportunities")
            
            # Update state
            self.state['opportunities_found_today'] = len(opportunities)
            self.state['last_scan'] = datetime.now().isoformat()
            
            # Save opportunities for dashboard
            self._save_opportunities(opportunities)
            
            # Log opportunities
            if opportunities:
                logger.info(f"\n🎯 BETTING OPPORTUNITIES:")
                for opp in opportunities:
                    logger.info(f"   • {opp['matchup']}: {opp['bet_line']}")
                    logger.info(f"     Edge: {opp['edge']:.1f} pts, P(Win): {opp['p_win']:.1%}, Stake: ${opp['recommended_stake']:.0f}")
            
            # Check risk status
            if self.trading_engine.ontorisk_enabled:
                status = self.trading_engine.risk_manager.get_status()
                logger.info(f"\n🛡️ Risk Status:")
                logger.info(f"   Bankroll: ${status['current_bankroll']:,.0f}")
                logger.info(f"   Drawdown: {status['current_drawdown']:.1%}")
                logger.info(f"   Can Bet: {status['can_bet']}")
                
                if not status['can_bet']:
                    logger.warning("🚨 RISK LIMITS EXCEEDED - Not betting")
                    for alert in status['alerts']:
                        logger.warning(f"   {alert}")
            
            # Save state periodically
            if self.cycle_count % 10 == 0:
                self._save_state()
                logger.info("💾 State saved")
            
            logger.info(f"✅ Cycle #{self.cycle_count} complete\n")
            
        except Exception as e:
            logger.error(f"❌ Error in monitoring cycle: {e}")
            logger.error(traceback.format_exc())
    
    def _save_opportunities(self, opportunities: List[Dict]):
        """Save opportunities for dashboard"""
        try:
            with open(self.opportunities_file, 'w') as f:
                json.dump({
                    'opportunities': opportunities,
                    'timestamp': datetime.now().isoformat(),
                    'count': len(opportunities)
                }, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Could not save opportunities: {e}")
    
    def run(self):
        """
        Main daemon loop - runs forever
        """
        logger.info("\n" + "="*80)
        logger.info("🔥 AUTONOMOUS TRADING DAEMON STARTING")
        logger.info("="*80)
        logger.info(f"\nConfiguration:")
        logger.info(f"  Check Interval: {self.check_interval}s")
        logger.info(f"  Model MAE: {self.mae}")
        logger.info(f"  Bankroll: ${self.state['current_bankroll']:,.0f}")
        logger.info(f"  Last Restart: {self.state['last_restart']}")
        logger.info("")
        
        # Initialize
        if not self.initialize_components():
            logger.error("❌ Failed to initialize, exiting")
            return
        
        self.running = True
        
        logger.info("✅ Daemon is running")
        logger.info(f"   PID: {Path('/proc/self').resolve().name if Path('/proc/self').exists() else 'N/A'}")
        log_filename = f"daemon_{datetime.now().strftime('%Y%m%d')}.log"
        logger.info(f"   Logs: {log_dir / log_filename}")
        logger.info("\n🔄 Starting monitoring loop...")
        logger.info("   Press Ctrl+C to stop gracefully\n")
        
        # Main loop
        while self.running:
            try:
                # Run monitoring cycle
                self.monitoring_cycle()
                
                # Wait for next cycle
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Keyboard interrupt received")
                break
                
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                logger.error(traceback.format_exc())
                logger.info(f"⏳ Waiting {self.check_interval}s before retry...")
                time.sleep(self.check_interval)
        
        # Cleanup
        logger.info("\n🛑 Daemon stopping...")
        self._save_state()
        logger.info("✅ State saved")
        logger.info("👋 Daemon stopped\n")


def main():
    """
    Main entry point
    """
    print("\n" + "="*80)
    print("🔥 ONTOLOGIC XYZ - AUTONOMOUS TRADING DAEMON")
    print("="*80)
    print("\nThis daemon runs continuously in the background, monitoring:")
    print("  • NBA live games")
    print("  • BetOnline betting lines")
    print("  • Betting opportunities")
    print("  • Risk management")
    print("\nIt serves data to the dashboard and logs all activity.")
    print("\nTo stop: Press Ctrl+C or send SIGTERM")
    print("="*80)
    
    # Create daemon
    daemon = AutonomousTradingDaemon(
        check_interval=3,  # Check every 3 seconds (MAXIMUM SPEED!)
        mae=9.029,
        starting_bankroll=1000
    )
    
    # Run forever
    daemon.run()


if __name__ == "__main__":
    main()

