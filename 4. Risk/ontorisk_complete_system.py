"""
ONTORISK COMPLETE SYSTEM

Purpose: Full integration of ML models, OntoRisk, and backtesting
Author: Ontologic XYZ
Date: October 20, 2025

This is the COMPLETE production system that ties everything together.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime

# Import our OntoRisk components
from ontorisk_phase1_probability_calibration import ProbabilityCalibrator, PredictionProbability
from ontorisk_phase2_model_integration import OntoRiskIntegration, BacktestEngine, BetResult
from ontorisk_phase3_historical_spreads import HistoricalSpreadScraper


@dataclass
class BacktestResults:
    """Complete backtest results"""
    total_games: int
    games_bet: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    total_staked: float
    total_profit: float
    roi: float
    sharpe_ratio: float
    max_drawdown: float
    starting_bankroll: float
    ending_bankroll: float
    avg_bet_size: float
    avg_profit_per_bet: float
    profitable_game_types: Dict
    bet_history: List[BetResult]


class OntoRiskCompleteSystem:
    """
    Complete OntoRisk system integrating all components
    """
    
    def __init__(
        self,
        model_path: str = "../Action/HYBRID_ULTIMATE_V2_CLEAN.pkl",
        data_path: str = "../Action/ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl",
        mae: float = 9.029,
        starting_bankroll: float = 10000,
        kelly_fraction: float = 0.25,
        min_edge: float = 5.0,
        min_p_win: float = 0.55
    ):
        """
        Initialize complete OntoRisk system
        
        Args:
            model_path: Path to trained ML model
            data_path: Path to game data
            mae: Model MAE for calibration
            starting_bankroll: Initial capital
            kelly_fraction: Fraction of Kelly to use
            min_edge: Minimum edge to bet (points)
            min_p_win: Minimum P(win) to bet
        """
        self.model_path = model_path
        self.data_path = data_path
        self.mae = mae
        self.starting_bankroll = starting_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.min_p_win = min_p_win
        
        # Initialize components
        print("\n" + "="*80)
        print("🔥 INITIALIZING ONTORISK COMPLETE SYSTEM")
        print("="*80 + "\n")
        
        # Load ML model
        self.model = self._load_model()
        
        # Load data
        self.data = self._load_data()
        
        # Initialize OntoRisk components
        self.calibrator = ProbabilityCalibrator(mae=mae, method='gaussian')
        self.spread_scraper = HistoricalSpreadScraper()
        self.backtest_engine = BacktestEngine(
            starting_bankroll=starting_bankroll,
            kelly_fraction=kelly_fraction
        )
        
        print("✅ System initialized successfully\n")
    
    def _load_model(self):
        """Load ML model from pickle"""
        print(f"📂 Loading model: {self.model_path}")
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                model = model_data.get('model')
                scaler = model_data.get('scaler')
                print(f"✅ Model loaded successfully")
                return {'model': model, 'scaler': scaler}
            else:
                print(f"⚠️ Model file has unexpected format")
                return None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def _load_data(self) -> List[Dict]:
        """Load game data from pickle"""
        print(f"📂 Loading data: {self.data_path}")
        
        try:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, list):
                print(f"✅ Loaded {len(data)} games")
                return data
            else:
                print(f"⚠️ Data file has unexpected format")
                return []
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return []
    
    def predict_game(
        self,
        game: Dict,
        spread_line: float
    ) -> Optional[PredictionProbability]:
        """
        Make prediction for a single game
        
        Args:
            game: Game dict with features
            spread_line: Market spread
            
        Returns:
            PredictionProbability or None
        """
        if self.model is None:
            return None
        
        # Extract features
        if 'pattern' in game and isinstance(game['pattern'], (list, np.ndarray)):
            features = np.array(game['pattern'])
        else:
            return None
        
        # Scale features
        if self.model.get('scaler') is not None:
            features_scaled = self.model['scaler'].transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # Predict
        try:
            prediction = self.model['model'].predict(features_scaled)[0]
        except:
            return None
        
        # Convert to probability
        prob = self.calibrator.calculate_probability(
            prediction=prediction,
            spread_line=spread_line,
            home_team=game.get('home_team', 'Home'),
            away_team=game.get('away_team', 'Away')
        )
        
        return prob
    
    def run_backtest(
        self,
        test_data: Optional[List[Dict]] = None,
        use_synthetic_spreads: bool = True
    ) -> BacktestResults:
        """
        Run complete backtest on historical data
        
        Args:
            test_data: Test dataset (if None, uses loaded data)
            use_synthetic_spreads: Use synthetic spreads for testing
            
        Returns:
            BacktestResults object
        """
        if test_data is None:
            test_data = self.data
        
        print("\n" + "="*80)
        print("🎲 RUNNING BACKTEST")
        print("="*80 + "\n")
        
        print(f"Test Games: {len(test_data)}")
        print(f"Min Edge: {self.min_edge} points")
        print(f"Min P(Win): {self.min_p_win:.1%}")
        print(f"Starting Bankroll: ${self.starting_bankroll:,.0f}")
        print(f"Kelly Fraction: {self.kelly_fraction:.0%}")
        print()
        
        # Get spreads
        print("📊 Loading spreads...")
        if use_synthetic_spreads:
            spreads = self.spread_scraper.match_spreads_to_games(
                test_data,
                use_synthetic=True
            )
        else:
            # Load from cache or scrape
            spreads = self.spread_scraper.load_from_cache()
        
        print(f"✅ Spreads available for {len(spreads)} games\n")
        
        # Find betting opportunities
        opportunities = []
        
        print("🔍 Analyzing games...")
        for game in test_data:
            game_id = game.get('game_id', '')
            
            # Skip if no spread available
            if game_id not in spreads:
                continue
            
            spread_line = spreads[game_id]
            
            # Make prediction
            prob = self.predict_game(game, spread_line)
            if prob is None:
                continue
            
            # Calculate edge
            edge = abs(prob.prediction - prob.spread_line)
            
            # Filter by criteria
            if edge >= self.min_edge and prob.p_win >= self.min_p_win:
                opportunities.append({
                    'game': game,
                    'prob': prob,
                    'spread': spread_line
                })
        
        print(f"✅ Found {len(opportunities)} betting opportunities\n")
        
        if len(opportunities) == 0:
            print("⚠️ No betting opportunities found")
            print("   Try lowering min_edge or min_p_win")
            return None
        
        # Run backtest
        print("💰 Simulating bets...\n")
        
        for opp in opportunities:
            game = opp['game']
            prob = opp['prob']
            
            # Calculate stake
            stake = self.backtest_engine.calculate_stake(
                p_win=prob.p_win,
                kelly_edge=prob.kelly_edge
            )
            
            # Get actual result
            actual_result = game.get('diff_at_final', 0)
            
            # Settle bet
            result = self.backtest_engine.settle_bet(
                prob=prob,
                actual_result=actual_result,
                stake=stake,
                game_id=game.get('game_id', ''),
                date=game.get('date', ''),
                home_team=game.get('home_team', 'Home'),
                away_team=game.get('away_team', 'Away')
            )
        
        # Calculate statistics
        stats = self.backtest_engine.run_backtest([], {}, {})
        
        # Analyze performance by game type
        profitable_types = self._analyze_by_game_type()
        
        # Create results object
        results = BacktestResults(
            total_games=len(test_data),
            games_bet=stats['total_games'],
            wins=stats['wins'],
            losses=stats['losses'],
            pushes=stats['pushes'],
            win_rate=stats['win_rate'],
            total_staked=stats['total_staked'],
            total_profit=stats['total_profit'],
            roi=stats['roi'],
            sharpe_ratio=stats['sharpe_ratio'],
            max_drawdown=stats['max_drawdown'],
            starting_bankroll=stats['starting_bankroll'],
            ending_bankroll=stats['ending_bankroll'],
            avg_bet_size=stats['total_staked'] / stats['total_games'] if stats['total_games'] > 0 else 0,
            avg_profit_per_bet=stats['total_profit'] / stats['total_games'] if stats['total_games'] > 0 else 0,
            profitable_game_types=profitable_types,
            bet_history=self.backtest_engine.bet_history
        )
        
        # Print results
        self._print_backtest_results(results)
        
        return results
    
    def _analyze_by_game_type(self) -> Dict:
        """Analyze profitability by game type"""
        # Simplified version - would need more game context
        return {
            'all_games': {
                'count': len(self.backtest_engine.bet_history),
                'profit': sum(bet.profit for bet in self.backtest_engine.bet_history)
            }
        }
    
    def _print_backtest_results(self, results: BacktestResults):
        """Print formatted backtest results"""
        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS")
        print("="*80 + "\n")
        
        print(f"Total Games: {results.total_games}")
        print(f"Games Bet: {results.games_bet} ({results.games_bet/results.total_games*100:.1f}%)")
        print()
        
        print(f"Wins: {results.wins}")
        print(f"Losses: {results.losses}")
        print(f"Pushes: {results.pushes}")
        print(f"Win Rate: {results.win_rate:.1%}")
        print()
        
        print(f"Total Staked: ${results.total_staked:,.0f}")
        print(f"Total Profit: ${results.total_profit:+,.0f}")
        print(f"ROI: {results.roi:.1%}")
        print()
        
        print(f"Avg Bet Size: ${results.avg_bet_size:,.0f}")
        print(f"Avg Profit/Bet: ${results.avg_profit_per_bet:+,.2f}")
        print()
        
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.1%}")
        print()
        
        print(f"Starting Bankroll: ${results.starting_bankroll:,.0f}")
        print(f"Ending Bankroll: ${results.ending_bankroll:,.0f}")
        print(f"Net Change: ${results.ending_bankroll - results.starting_bankroll:+,.0f}")
        
        print("\n" + "="*80)
        
        # Estimate season performance
        if results.games_bet > 0:
            games_per_season = 1230
            season_bets = (results.games_bet / results.total_games) * games_per_season
            season_profit = (results.total_profit / results.games_bet) * season_bets
            
            print("\n📈 SEASON PROJECTION:")
            print(f"   Expected Bets: {season_bets:.0f}")
            print(f"   Expected Profit: ${season_profit:+,.0f}")
            print("="*80)
    
    def save_backtest_results(
        self,
        results: BacktestResults,
        filepath: str = "backtest_results.json"
    ):
        """Save backtest results to file"""
        # Convert to dict (excluding bet_history for size)
        results_dict = asdict(results)
        results_dict.pop('bet_history')  # Too large for JSON
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"✅ Results saved to {filepath}")
    
    def predict_live_game(
        self,
        game_features: np.ndarray,
        spread_line: float,
        home_team: str,
        away_team: str
    ) -> Dict:
        """
        Make prediction for a live game (API interface)
        
        Args:
            game_features: Feature vector
            spread_line: Current spread line
            home_team, away_team: Team names
            
        Returns:
            Dict with prediction and betting info
        """
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        # Scale features
        if self.model.get('scaler') is not None:
            features_scaled = self.model['scaler'].transform(game_features.reshape(1, -1))
        else:
            features_scaled = game_features.reshape(1, -1)
        
        # Predict
        try:
            prediction = self.model['model'].predict(features_scaled)[0]
        except Exception as e:
            return {'error': f'Prediction failed: {e}'}
        
        # Convert to probability
        prob = self.calibrator.calculate_probability(
            prediction=prediction,
            spread_line=spread_line,
            home_team=home_team,
            away_team=away_team
        )
        
        # Calculate edge
        edge = abs(prob.prediction - prob.spread_line)
        
        # Determine if bet is recommended
        is_bet = edge >= self.min_edge and prob.p_win >= self.min_p_win
        
        # Calculate stake if betting
        if is_bet:
            stake = self.backtest_engine.calculate_stake(
                p_win=prob.p_win,
                kelly_edge=prob.kelly_edge
            )
        else:
            stake = 0
        
        # Return formatted response
        return {
            'prediction': float(prob.prediction),
            'spread_line': float(prob.spread_line),
            'edge': float(edge),
            'p_win': float(prob.p_win),
            'kelly_edge': float(prob.kelly_edge),
            'confidence_interval': [float(prob.confidence_interval[0]), float(prob.confidence_interval[1])],
            'bet_recommended': is_bet,
            'bet_side': prob.bet_side,
            'bet_line': prob.bet_line,
            'recommended_stake': float(stake),
            'home_team': home_team,
            'away_team': away_team
        }


def main():
    """
    Main function: Run complete OntoRisk system
    """
    print("\n" + "="*80)
    print("🔥 ONTORISK COMPLETE SYSTEM")
    print("="*80 + "\n")
    
    # Initialize system
    system = OntoRiskCompleteSystem(
        model_path="../Action/HYBRID_ULTIMATE_V2_CLEAN.pkl",
        data_path="../Action/ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl",
        mae=9.029,
        starting_bankroll=10000,
        kelly_fraction=0.25,
        min_edge=5.0,
        min_p_win=0.55
    )
    
    # Run backtest
    results = system.run_backtest(use_synthetic_spreads=True)
    
    if results is not None:
        # Save results
        system.save_backtest_results(results)
        
        print("\n" + "="*80)
        print("✅ ONTORISK COMPLETE SYSTEM READY")
        print("="*80)
        print("\n📋 Next Steps:")
        print("   1. Review backtest results")
        print("   2. If results look good, proceed to paper trading")
        print("   3. Use predict_live_game() for live predictions")
        print("\n" + "="*80)


if __name__ == "__main__":
    main()

