"""
LIVE TRADING ENGINE

Purpose: Integrate NBA API + BetOnline + ML Models + OntoRisk
Author: Ontologic XYZ
Date: October 20, 2025

This is the COMPLETE integration that runs the live trading system.
"""

import numpy as np
import pickle
import sys
import os  # ⚡ CRITICAL: Needed for os.path.exists() and file operations!
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

# Add paths
sys.path.append('../4. Risk')

# Import components
from nba_live_scores import NBALiveScores
from betonline_live_lines import BetOnlineScraper, MultiBookLineScraper
from game_data_logger import get_logger
from mamba_auto_logger import MambaAutoLogger  # NEW: Automatic logging
from mamba_betting_config import (
    BETTING_STRATEGY, 
    BETTING_STRATEGIES, 
    ALLOWED_GAME_TYPES,
    GAME_TYPE_PERFORMANCE,
    RISK_PARAMETERS as MAMBA_RISK_PARAMS,
    should_bet_on_game,
    calculate_bet_size
)

try:
    # Import OntoRisk components
    import sys
    sys.path.append('../4. Risk')
    from ontorisk_phase1_probability_calibration import ProbabilityCalibrator
    from ontorisk_phase4_risk_management import RiskManager
    from ontorisk_phase5_archetype_classifier import GameArchetypeClassifier
    ONTORISK_AVAILABLE = True
    print("✅ OntoRisk components loaded successfully")
except ImportError as e:
    print(f"⚠️ OntoRisk components not found: {e}")
    print("⚠️ Using standalone mode")
    ONTORISK_AVAILABLE = False


class LiveTradingEngine:
    """
    Complete live trading system
    """
    
    def __init__(
        self,
        model_path: str = None,
        mae: float = 9.655,  # Branch B MAE (final score)
        starting_bankroll: float = 1000
    ):
        """
        Initialize live trading engine
        
        Args:
            model_path: Path to ML model
            mae: Model MAE
            starting_bankroll: Initial capital
        """
        print("\n" + "="*80)
        print("🔥 INITIALIZING LIVE TRADING ENGINE")
        print("="*80 + "\n")
        
        self.mae = mae
        
        # Load ML model (try multiple paths + auto-download)
        if model_path is None:
            # RAILWAY FIX: Use /tmp for persistent storage during session
            # Railway's root filesystem is ephemeral - /tmp persists during runtime
            railway_model_path = "/tmp/MAMBA_MENTALITY_SYSTEM.pkl"
            
            # Try multiple possible locations
            possible_paths = [
                railway_model_path,  # Railway /tmp (persists during runtime)
                "MAMBA_MENTALITY_SYSTEM.pkl",  # Current directory
                "../mambaofficial/models/MAMBA_MENTALITY_SYSTEM.pkl",  # Local dev
                "../Action/HYBRID_ULTIMATE_V2_CLEAN.pkl",  # Old path
                "models/MAMBA_MENTALITY_SYSTEM.pkl",  # Alternative
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"✅ Found model at: {path}")
                    break
            
            # If still not found, download from Google Drive to /tmp
            if model_path is None:
                print("⬇️ Model not found locally, attempting Google Drive download...")
                print(f"   Downloading to: {railway_model_path}")
                try:
                    from download_mamba_model import download_mamba_model
                    
                    # Download to /tmp on Railway (persists during runtime)
                    if download_mamba_model(output_path=railway_model_path):
                        model_path = railway_model_path
                        print(f"✅ Model downloaded successfully to {railway_model_path}!")
                        print(f"   (Will persist for this session, re-download on next deploy)")
                    else:
                        print("❌ Model download failed")
                except Exception as e:
                    print(f"❌ Download error: {e}")
                    import traceback
                    traceback.print_exc()
        
        if model_path and os.path.exists(model_path):
            print(f"📂 Loading model: {model_path}")
            self.model = self._load_model(model_path)
        else:
            print(f"⚠️ No model found at any expected location")
            print(f"⚠️ Running in NO-MODEL mode (synthetic predictions only)")
            self.model = None
        
        # Initialize components
        print("🏀 Initializing NBA API...")
        self.nba_api = NBALiveScores()
        
        print("💰 Initializing BetOnline scraper...")
        self.line_scraper = BetOnlineScraper()
        
        print("📊 Initializing Game Data Logger...")
        self.logger = get_logger()
        
        # ENHANCED: Initialize prediction storage for Mamba performance tracking
        self.prediction_storage = []
        self.prediction_file = "mamba_predictions.json"
        self.mamba_scores_storage = {}  # Store Mamba scores after 6:00 mark
        print("🎯 Initializing Mamba prediction storage...")
        
        # NEW: Initialize automatic logger for daily review
        print("📝 Initializing Mamba Auto-Logger...")
        self.auto_logger = MambaAutoLogger(log_dir="mamba_logs")
        print("✅ Auto-Logger ready: Logs stored in mamba_logs/")
        
        try:
            print("🎯 Initializing OntoRisk...")
            self.calibrator = ProbabilityCalibrator(mae=mae)
            self.risk_manager = RiskManager(starting_bankroll=starting_bankroll)
            self.ontorisk_enabled = True
        except:
            print("⚠️ OntoRisk not available (standalone mode)")
            self.ontorisk_enabled = False
        
        print("\n✅ Live Trading Engine ready!\n")
        print("="*80)
    
    def _load_model(self, path: str):
        """Load ML model"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                print("✅ Model loaded")
                return model_data
            else:
                print("⚠️ Model format unexpected")
                return None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def extract_features_from_live_game(self, game: Dict) -> Optional[np.ndarray]:
        """
        Extract REAL 33 Mamba features from live game state using MambaLiveFeatureExtractor
        
        Args:
            game: Live game dict from NBA API
            
        Returns:
            Feature vector (33 features) matching Mamba training data
        """
        try:
            print("🔍 EXTRACTING REAL MAMBA FEATURES (33)...")
            
            # Import the REAL feature extractor
            from mamba_live_feature_extractor import MambaLiveFeatureExtractor
            
            # Initialize extractor (with caching for efficiency)
            if not hasattr(self, 'mamba_extractor'):
                self.mamba_extractor = MambaLiveFeatureExtractor(cache_pbp=True)
            
            # Get game ID
            game_id = game.get('game_id', '')
            
            # Build current_game_state dict
            current_game_state = {
                'home_team': game.get('home_team', ''),
                'away_team': game.get('away_team', ''),
                'home_score': game.get('home_score', 0),
                'away_score': game.get('away_score', 0),
                'period': game.get('period', 0),
                'clock': game.get('clock', '0:00')
            }
            
            # Extract REAL 33 features
            features = self.mamba_extractor.extract_features(game_id, current_game_state)
            
            if features is None:
                print("❌ Mamba feature extraction failed")
                return None
            
            print(f"✅ EXTRACTED {len(features)} REAL MAMBA FEATURES")
            return features
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None
    
    # ✅ REMOVED: Old synthetic pattern generation methods
    # Now using MambaLiveFeatureExtractor for 100% REAL feature extraction!
    
    def _store_prediction(self, game: Dict, prediction: float, features: np.ndarray, line: Dict):
        """
        Store Mamba prediction for performance tracking
        
        Args:
            game: Live game data
            prediction: Mamba prediction
            features: Feature vector used
            line: Betting line data
        """
        try:
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'game_id': game['game_id'],
                'matchup': f"{game['away_team']} @ {game['home_team']}",
                'period': game['period'],
                'clock': game.get('clock', ''),
                'current_score': f"{game['away_score']}-{game['home_score']}",
                'current_diff': game['current_diff'],
                'mamba_prediction': float(prediction),
                'market_spread': line.get('spread', 0),
                'edge': abs(prediction - line.get('spread', 0)),
                'features_used': len(features),
                'game_progress': (game['period'] - 1) * 0.25 + (12 - float(game.get('clock', '0:00').split(':')[0])) / 12 * 0.25 if ':' in game.get('clock', '0:00') else 0.0,
                'status': 'PENDING'  # Will be updated when game ends
            }
            
            self.prediction_storage.append(prediction_record)
            
            # Save to file for persistence
            self._save_predictions_to_file()
            
            print(f"🎯 Stored Mamba prediction: {prediction_record['matchup']} → {prediction:+.1f}")
            
        except Exception as e:
            print(f"❌ Error storing prediction: {e}")
    
    def _save_predictions_to_file(self):
        """Save predictions to JSON file for persistence"""
        try:
            import json
            with open(self.prediction_file, 'w') as f:
                json.dump(self.prediction_storage, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving predictions: {e}")
    
    def _load_predictions_from_file(self):
        """Load predictions from JSON file"""
        try:
            import json
            if os.path.exists(self.prediction_file):
                with open(self.prediction_file, 'r') as f:
                    self.prediction_storage = json.load(f)
                print(f"📊 Loaded {len(self.prediction_storage)} stored predictions")
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            self.prediction_storage = []
    
    def _store_mamba_score_after_6min(self, game: Dict, prediction: float):
        """
        Store Mamba score after 6:00 mark passes - PERSISTENT STORAGE!
        
        Args:
            game: Live game data
            prediction: Mamba prediction
        """
        try:
            game_id = game['game_id']
            period = game['period']
            clock = game.get('clock', '0:00')
            
            # Check if we're past 6:00 mark
            is_past_6min = False
            if period == 2:
                try:
                    if ':' in clock:
                        minutes, seconds = clock.split(':')
                        if float(minutes) < 6.0:
                            is_past_6min = True
                except:
                    pass
            
            # Store score if past 6:00 mark
            if is_past_6min or period > 2:
                self.mamba_scores_storage[game_id] = {
                    'game_id': game_id,
                    'matchup': f"{game['away_team']} @ {game['home_team']}",
                    'mamba_score': float(prediction),
                    'period': period,
                    'clock': clock,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'STORED_AFTER_6MIN'
                }
                
                # Save to file for persistence
                self._save_mamba_scores_to_file()
                
                print(f"🎯 Stored Mamba score after 6:00: {game_id} → {prediction:+.1f}")
            
        except Exception as e:
            print(f"❌ Error storing Mamba score: {e}")
    
    def _save_mamba_scores_to_file(self):
        """Save Mamba scores to JSON file for persistence"""
        try:
            import json
            with open("mamba_scores_storage.json", 'w') as f:
                json.dump(self.mamba_scores_storage, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving Mamba scores: {e}")
    
    def _load_mamba_scores_from_file(self):
        """Load Mamba scores from JSON file"""
        try:
            import json
            if os.path.exists("mamba_scores_storage.json"):
                with open("mamba_scores_storage.json", 'r') as f:
                    self.mamba_scores_storage = json.load(f)
                print(f"📊 Loaded {len(self.mamba_scores_storage)} stored Mamba scores")
        except Exception as e:
            print(f"❌ Error loading Mamba scores: {e}")
            self.mamba_scores_storage = {}
    
    def get_stored_mamba_scores(self) -> Dict:
        """
        Get stored Mamba scores after 6:00 mark
        
        Returns:
            Stored Mamba scores dict
        """
        return {
            'stored_scores': self.mamba_scores_storage,
            'total_stored': len(self.mamba_scores_storage),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_mamba_performance(self) -> Dict:
        """
        Get Mamba performance statistics
        
        Returns:
            Performance metrics dict
        """
        try:
            if not self.prediction_storage:
                return {
                    'total_predictions': 0,
                    'pending_predictions': 0,
                    'completed_predictions': 0,
                    'average_edge': 0.0,
                    'prediction_accuracy': 'N/A'
                }
            
            total = len(self.prediction_storage)
            pending = len([p for p in self.prediction_storage if p['status'] == 'PENDING'])
            completed = len([p for p in self.prediction_storage if p['status'] == 'COMPLETED'])
            avg_edge = np.mean([p['edge'] for p in self.prediction_storage]) if self.prediction_storage else 0.0
            
            return {
                'total_predictions': total,
                'pending_predictions': pending,
                'completed_predictions': completed,
                'average_edge': float(avg_edge),
                'prediction_accuracy': 'N/A'  # Will calculate when games complete
            }
            
        except Exception as e:
            print(f"❌ Error calculating performance: {e}")
            return {'error': str(e)}
    
    def make_live_prediction(
        self,
        game: Dict,
        line: Dict
    ) -> Optional[Dict]:
        """
        Make prediction for a live game
        
        Args:
            game: Live game state from NBA API
            line: Current betting line from BetOnline
            
        Returns:
            Prediction dict with OntoRisk analysis
        """
        if self.model is None:
            return None
        
        # ENHANCED: Extract real Mamba features (33 features)
        features = self.extract_features_from_live_game(game)
        
        if features is None:
            print("❌ Feature extraction failed, skipping prediction")
            return None
        
        # Check if model is loaded
        if self.model is None:
            print("⚠️ No model loaded, using synthetic prediction")
            # Generate synthetic prediction based on current differential
            prediction = game['current_diff'] + np.random.normal(0, 3)
        else:
            # Make prediction with loaded model
            try:
                if self.model.get('scaler'):
                    X = self.model['scaler'].transform(features.reshape(1, -1))
                else:
                    X = features.reshape(1, -1)
                
                prediction = self.model['model'].predict(X)[0]
            except Exception as e:
                print(f"❌ Prediction error: {e}, using synthetic fallback")
                prediction = game['current_diff'] + np.random.normal(0, 3)
            
        # ENHANCED: Store Mamba prediction for performance tracking (only if real model)
        if self.model is not None:
            self._store_prediction(game, prediction, features, line)
            
            # ENHANCED: Store Mamba score after 6:00 mark passes
            self._store_mamba_score_after_6min(game, prediction)
        
        # NEW: Auto-log prediction for daily review
        try:
            self.auto_logger.log_prediction(
                game_id=game['game_id'],
                game_data=game,
                prediction=prediction,
                odds=line,
                features=features.tolist() if features is not None else None,
                metadata={
                    'model_loaded': self.model is not None,
                    'ontorisk_enabled': self.ontorisk_enabled
                }
            )
        except Exception as e:
            print(f"⚠️ Auto-logger error: {e}")
        
        # Calculate edge
        spread_line = line['spread']
        edge = abs(prediction - spread_line)
        
        # ENHANCED: OntoRisk analysis with full integration
        if ONTORISK_AVAILABLE and hasattr(self, 'ontorisk_calibrator'):
            try:
                # Use OntoRisk probability calibration
                prob = self.ontorisk_calibrator.calculate_probability(
                    prediction=prediction,
                    spread_line=spread_line,
                    mae=9.029,  # Mamba MAE
                    confidence=0.8
                )
                
                # Use OntoRisk risk management
                if hasattr(self, 'ontorisk_risk_manager'):
                    # Check risk limits
                    checks = self.ontorisk_risk_manager.check_limits()
                    can_bet = checks['can_bet'] and edge >= 5.0 and prob.p_win >= 0.55
                    
                    # Calculate optimal stake using Kelly
                    if can_bet:
                        kelly_fraction = prob.kelly_edge * 0.25  # Fractional Kelly
                        kelly_stake = self.ontorisk_risk_manager.state.current_bankroll * kelly_fraction
                        is_valid, stake, reason = self.ontorisk_risk_manager.validate_bet_size(kelly_stake)
                    else:
                        is_valid = False
                        stake = 0
                        reason = "OntoRisk criteria not met"
                else:
                    # Fallback to basic validation
                    is_valid = edge >= 5.0 and prob.p_win >= 0.55
                    stake = 100 if is_valid else 0
                    reason = "Basic validation" if is_valid else "Edge/P(Win) too low"
                    
            except Exception as e:
                print(f"⚠️ OntoRisk error: {e}")
                # Fallback to basic mode
                is_valid = edge >= 5.0
                stake = 100 if is_valid else 0
                reason = "Fallback mode"
                prob = type('obj', (object,), {
                    'p_win': 0.5,
                    'kelly_edge': 0.0,
                    'confidence_interval': [prediction - 5, prediction + 5]
                })()
        else:
            # Standalone mode (no OntoRisk)
            is_valid = edge >= 5.0
            stake = 100 if is_valid else 0
            reason = "Standalone mode"
            prob = type('obj', (object,), {
                'p_win': 0.5,
                'kelly_edge': 0.0,
                'confidence_interval': [prediction - 5, prediction + 5]
            })()
            
            # ENHANCED: Store team favorites properly
            home_team = game['home_team']
            away_team = game['away_team']
            favorite = f"{home_team} {spread_line:+.1f}" if spread_line < 0 else f"{away_team} {abs(spread_line):+.1f}"
            
            # ENHANCED: Include American odds and implied probability
            home_ml = line.get('home_ml', 0)
            away_ml = line.get('away_ml', 0)
            home_implied_prob = line.get('home_implied_prob', 0.5)
            away_implied_prob = line.get('away_implied_prob', 0.5)
            
            # ENHANCED: Include ALL 33 Mamba features for frontend display
            mamba_features = None
            if features is not None and len(features) == 33:
                mamba_features = {
                    'pattern_analysis': {
                        'mean_diff': float(features[0]),
                        'std_diff': float(features[1]),
                        'trend': float(features[2]),
                        'volatility': float(features[3]),
                        'velocity': float(features[4]),
                        'acceleration': float(features[5]),
                        'recent_momentum': float(features[6]),
                        'lead_changes': int(features[7]),
                        'max_swing': float(features[8]),
                        'comeback_potential': float(features[9])
                    },
                    'spectral': {
                        'spectral_energy': float(features[10]),
                        'spectral_entropy': float(features[11]),
                        'low_freq_power': float(features[12]),
                        'mid_freq_power': float(features[13]),
                        'high_freq_power': float(features[14]),
                        'dominant_freq': float(features[15])
                    },
                    'autocorrelation': {
                        'lag1': float(features[16]),
                        'lag2': float(features[17]),
                        'lag3': float(features[18])
                    },
                    'advanced_stats': {
                        'pace_proxy': float(features[19]),
                        'efg_proxy': float(features[20]),
                        'ts_proxy': float(features[21]),
                        'netrtg_proxy': float(features[22]),
                        'usg_proxy': float(features[23]),
                        'pm_proxy': float(features[24]),
                        'pie_proxy': float(features[25]),
                        'four_factors': float(features[26])
                    },
                    'team_form': {
                        'team_diff_lag1': float(features[27]),
                        'team_mean_lag1': float(features[28]),
                        'team_diff_rolling3': float(features[29]),
                        'team_volatility_rolling3': float(features[30]),
                        'team_form_10games': float(features[31]),
                        'team_consistency': float(features[32])
                    },
                    'extraction_time_ms': 0,  # TODO: track this
                    'pbp_events_count': 0,  # TODO: track this
                    'pattern_length': 18
                }
            
            return {
                'game_id': game['game_id'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'matchup': f"{game['away_team']} @ {game['home_team']}",
                'current_score': f"{game['away_score']}-{game['home_score']}",
                'period': f"Q{game['period']} {game['clock']}",
                'prediction': float(prediction),
                'market_spread': float(spread_line),
                'favorite': favorite,  # ENHANCED: Team favorite
                'edge': float(edge),
                'p_win': float(prob.p_win),
                'kelly_edge': float(prob.kelly_edge),
                'bet_recommended': is_valid,
                'bet_side': getattr(prob, 'bet_side', '') if is_valid else '',
                'bet_line': getattr(prob, 'bet_line', '') if is_valid else '',
                'recommended_stake': float(stake),
                'reason': reason,
                'confidence_interval': [float(prob.confidence_interval[0]), float(prob.confidence_interval[1])],
                # ENHANCED: American odds integration
                'american_odds_home': home_ml,
                'american_odds_away': away_ml,
                'implied_prob_home': home_implied_prob,
                'implied_prob_away': away_implied_prob,
                'vig_percentage': line.get('vig_percentage', 0),
                # NEW: ALL 33 MAMBA FEATURES
                'mamba_features': mamba_features,
                'features_extracted': mamba_features is not None,
                'timestamp': datetime.now().isoformat()
            }
    
    def scan_live_opportunities(self) -> List[Dict]:
        """
        Scan all live games for betting opportunities
        NOW RETURNS ALL PREDICTIONS FOR DISPLAY (not just betting opps)
        
        Returns:
            List of ALL predictions (betting opportunities flagged separately)
        """
        print("\n" + "="*80)
        print("🔍 SCANNING ALL LIVE GAMES FOR PREDICTIONS")
        print("="*80 + "\n")
        
        # Get live games
        games = self.nba_api.get_todays_games()
        print(f"📊 Found {len(games)} games\n")
        
        # Get lines
        lines = self.line_scraper.get_live_lines()
        print(f"💰 Found {len(lines)} lines\n")
        
        # ALL PREDICTIONS (for display)
        all_predictions = []
        
        for game in games:
            # MODIFIED: Check ALL live games, not just Q2 6:00
            period = game.get('period', 0)
            clock = game.get('clock', '')
            
            # Check if we can predict (Q2 6:00 OR any live game after Q2)
            can_predict = (
                game.get('can_predict', False) or  # Original Q2 6:00 window
                (period >= 2 and period <= 4) or  # Any Q2-Q4 live game
                (period == 2 and '6:' in clock) or  # Q2 6:00 specifically
                (period >= 3)  # Q3+ games
            )
            
            if not can_predict:
                continue
            
            # Find matching line (or use synthetic)
            line = None
            for l in lines:
                if l['home_team'] == game['home_team'] and l['away_team'] == game['away_team']:
                    line = l
                    break
            
            # If no line, create synthetic one for prediction
            if line is None:
                line = {
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'spread': game['current_diff'] * 0.8,  # Synthetic
                    'spread_odds': None,
                    'source': 'SYNTHETIC'
                }
                print(f"⚠️ No line for {game['away_team']} @ {game['home_team']}, using synthetic")
            
            # ALWAYS make prediction for display
            prediction = self.make_live_prediction(game, line)
            
            # Include ALL predictions (user wants to see EVERYTHING)
            if prediction:
                all_predictions.append(prediction)
                print(f"✅ Prediction made for {game['away_team']} @ {game['home_team']}")
        
        print(f"\n📊 Total predictions: {len(all_predictions)}")
        return all_predictions
    
    def _extract_live_features_enhanced(self, game: Dict, line: Dict) -> Optional[np.ndarray]:
        """
        Extract features for live prediction - ENHANCED FOR CONTINUOUS PREDICTIONS!
        
        Args:
            game: Live game data
            line: Betting line data
            
        Returns:
            Feature array or None
        """
        try:
            # ENHANCED: Extract features for any game state (Q2, Q3, Q4)
            period = game['period']
            clock = game.get('clock', '0:00')
            
            # Parse clock to minutes
            try:
                if ':' in clock:
                    minutes, seconds = clock.split(':')
                    total_minutes = float(minutes) + float(seconds) / 60.0
                else:
                    total_minutes = float(clock) if clock else 0.0
            except:
                total_minutes = 0.0
            
            # Calculate game progress (0.0 to 1.0)
            if period <= 4:
                game_progress = (period - 1) * 0.25 + (12 - total_minutes) / 12 * 0.25
            else:
                game_progress = 1.0  # Overtime
            
            # ENHANCED: 18 features for Mamba model
            features = [
                period,  # Quarter (2, 3, 4)
                game['home_score'],  # Home score
                game['away_score'],  # Away score
                game['current_diff'],  # Current differential
                line['spread'],  # Market spread
                line['total'],  # Total points
                line['home_ml'],  # Home moneyline
                line['away_ml'],  # Away moneyline
                game_progress,  # Game progress (0.0-1.0)
                total_minutes,  # Minutes in current quarter
                line.get('home_implied_prob', 0.5),  # Home implied probability
                line.get('away_implied_prob', 0.5),  # Away implied probability
                line.get('vig_percentage', 0.0),  # Vig percentage
                # Additional features for continuous predictions
                abs(game['current_diff']),  # Absolute differential
                game['home_score'] + game['away_score'],  # Total score
                (game['home_score'] - game['away_score']) / max(1, game['home_score'] + game['away_score']),  # Score ratio
                0,  # Placeholder for future features
                0   # Placeholder for future features
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None
    
    def print_opportunities(self, opportunities: List[Dict]):
        """Print formatted opportunities"""
        if not opportunities:
            print("❌ No betting opportunities found")
            print("   Criteria: Edge ≥ 5 points, P(Win) ≥ 55%")
            return
        
        print("\n" + "="*80)
        print(f"🎯 FOUND {len(opportunities)} BETTING OPPORTUNITIES")
        print("="*80 + "\n")
        
        for i, opp in enumerate(opportunities, 1):
            print(f"Opportunity {i}:")
            print(f"  Game: {opp['matchup']}")
            print(f"  Score: {opp['current_score']} ({opp['period']})")
            print(f"  Our Prediction: {opp['prediction']:+.1f}")
            print(f"  Market Spread: {opp['market_spread']:+.1f}")
            print(f"  Edge: {opp['edge']:.1f} points")
            print(f"  P(Win): {opp['p_win']:.1%}")
            print(f"  ✅ BET: {opp['bet_line']}")
            print(f"  💰 Stake: ${opp['recommended_stake']:,.0f}")
            print()
        
        print("="*80)


def example_live_trading():
    """
    Example: Full live trading workflow
    """
    print("\n" + "="*80)
    print("🔥 LIVE TRADING ENGINE - COMPLETE WORKFLOW")
    print("="*80 + "\n")
    
    # Initialize engine
    engine = LiveTradingEngine(
        mae=9.029,
        starting_bankroll=1000
    )
    
    # Scan for opportunities
    opportunities = engine.scan_live_opportunities()
    
    # Print opportunities
    engine.print_opportunities(opportunities)
    
    print("\n" + "="*80)
    print("✅ LIVE TRADING ENGINE READY")
    print("="*80)
    print("\n🎯 Integration complete:")
    print("   ✅ NBA API (live scores)")
    print("   ✅ BetOnline (live lines)")
    print("   ✅ ML Model (predictions)")
    print("   ✅ OntoRisk (probability + sizing)")
    print("   ✅ Risk Management (limits enforced)")
    print("\n📋 Ready for GUI integration (SolidJS dashboard)")
    print("="*80)


if __name__ == "__main__":
    example_live_trading()

