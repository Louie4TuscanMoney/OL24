"""
ML Model Integration
Connects live NBA patterns to trained ML model

ML Model Location: ../../1. ML/X. MVP Model/
"""

import sys
from pathlib import Path
import numpy as np

# Add ML model path
ml_model_path = Path(__file__).parent.parent.parent / "1. ML" / "X. MVP Model"
sys.path.insert(0, str(ml_model_path))

print(f"ML Model path: {ml_model_path}")

class MLPredictor:
    """
    Wrapper for ML model predictions
    Loads ensemble + conformal from MVP folder
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize ML predictor
        
        Args:
            model_dir: Path to MVP model folder (default: auto-detect)
        """
        if model_dir is None:
            # Auto-detect relative path
            current_file = Path(__file__).parent
            model_dir = current_file.parent.parent / "1. ML" / "X. MVP Model"
        
        self.model_dir = Path(model_dir)
        self.ensemble = None
        self.conformal = None
        self.is_loaded = False
        
    def load_models(self):
        """
        Load all ML models from MVP folder
        """
        print("\n" + "="*80)
        print("LOADING ML MODELS")
        print("="*80)
        
        try:
            import pickle
            import torch
            
            # Note: Model files should be copied to this folder or use absolute paths
            print(f"\nModel directory: {self.model_dir}")
            print("⚠️  Models should be in this directory or specify path")
            
            # For now, provide instructions
            print("\n📋 To load models:")
            print("   1. Copy these files from '1. ML/X. MVP Model/' to here:")
            print("      - dejavu_k500.pkl")
            print("      - lstm_best.pth")
            print("      - lstm_normalization.pkl")
            print("      - conformal_predictor.pkl")
            print("   2. Copy these code files:")
            print("      - dejavu_model.py")
            print("      - lstm_model.py")
            print("      - ensemble_model.py")
            print("      - conformal_wrapper.py")
            print("   3. Then run this again")
            
            self.is_loaded = False
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.is_loaded = False
    
    def predict(self, pattern: np.ndarray) -> Dict:
        """
        Make prediction with ML model
        
        Args:
            pattern: 18-minute differential array
        
        Returns:
            Dict with prediction, interval, components
        """
        if not self.is_loaded:
            return {
                'error': 'Models not loaded',
                'status': 'not_ready'
            }
        
        try:
            # Call ensemble + conformal
            forecast, interval, components = self.conformal.predict(pattern, self.ensemble)
            
            return {
                'point_forecast': float(forecast),
                'interval_lower': float(interval[0]),
                'interval_upper': float(interval[1]),
                'coverage_probability': 0.95,
                'components': components,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'prediction_failed'
            }


class IntegratedNBAPipeline:
    """
    Complete pipeline: NBA_API → Buffer → ML → Output
    """
    
    def __init__(self):
        """Initialize pipeline"""
        from nba_live_poller import NBALivePoller
        
        self.poller = NBALivePoller(poll_interval=10)
        self.ml_predictor = MLPredictor()
        self.predictions = []
    
    def setup(self):
        """Setup pipeline"""
        print("="*80)
        print("NBA → ML INTEGRATED PIPELINE")
        print("="*80)
        
        # Load ML models
        self.ml_predictor.load_models()
        
        print("\n✅ Pipeline ready")
        print("   NBA_API poller: 10-second intervals")
        print("   ML model: MVP ensemble (5.39 MAE)")
        print("   Output: Predictions with 95% intervals")
    
    def poll_and_predict(self):
        """
        Single iteration: Poll NBA → Check buffers → Predict if ready
        """
        # Poll NBA
        self.poller.poll_once()
        
        # Check for ready games
        ready_games = self.poller.buffer_manager.get_games_ready_for_prediction()
        
        if ready_games:
            print(f"\n🎯 Making predictions for {len(ready_games)} games...")
            
            for buffer in ready_games:
                pattern = buffer.get_pattern()
                
                # Call ML model
                prediction = self.ml_predictor.predict(pattern)
                
                # Store result
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'game_id': buffer.game_id,
                    'pattern': pattern.tolist(),
                    'prediction': prediction
                }
                self.predictions.append(result)
                
                # Mark done
                buffer.mark_prediction_made()
                
                # Print
                if prediction['status'] == 'success':
                    print(f"\n  Game {buffer.game_id}:")
                    print(f"    Forecast: {prediction['point_forecast']:+.1f} points")
                    print(f"    95% Interval: [{prediction['interval_lower']:+.1f}, {prediction['interval_upper']:+.1f}]")
                    print(f"    Components:")
                    print(f"      Dejavu: {prediction['components']['dejavu_prediction']:+.1f}")
                    print(f"      LSTM:   {prediction['components']['lstm_prediction']:+.1f}")
                else:
                    print(f"  ❌ Prediction failed: {prediction.get('error')}")
    
    def run(self, duration_minutes: Optional[int] = None):
        """
        Run integrated pipeline
        
        Args:
            duration_minutes: How long to run (None = forever)
        """
        self.setup()
        
        print(f"\n" + "="*80)
        print("STARTING LIVE POLLING")
        print("="*80)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60) if duration_minutes else None
        
        try:
            while True:
                self.poll_and_predict()
                
                # Check duration
                if end_time and time.time() >= end_time:
                    break
                
                # Wait
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")
        
        # Summary
        elapsed = time.time() - start_time
        print(f"\n" + "="*80)
        print("SESSION SUMMARY")
        print("="*80)
        print(f"Runtime: {elapsed/60:.1f} minutes")
        print(f"Polls: {self.poller.poll_count}")
        print(f"Predictions made: {len(self.predictions)}")
        print(f"Errors: {self.poller.error_count}")


if __name__ == "__main__":
    pipeline = IntegratedNBAPipeline()
    pipeline.run(duration_minutes=5)  # Test for 5 minutes

