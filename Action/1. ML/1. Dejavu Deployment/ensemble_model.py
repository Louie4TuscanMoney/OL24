"""
Ensemble Model: Dejavu + LSTM
Following: ML/Action Steps Folder/07_ENSEMBLE_AND_PRODUCTION_API.md

EXACT WEIGHTS (lines 35-36):
- Dejavu: 0.4 (40%)
- LSTM: 0.6 (60%)

Expected MAE: ~3.5 points (MODELSYNERGY.md line 854)
"""

import torch
import pickle
import numpy as np
from pathlib import Path

from dejavu_model import DejavuForecaster
from lstm_model import LSTMForecaster

class EnsembleForecaster:
    """
    Ensemble of Dejavu (40%) + LSTM (60%)
    
    From 07_ENSEMBLE_AND_PRODUCTION_API.md lines 33-130
    """
    
    def __init__(self, dejavu_weight=0.4, lstm_weight=0.6):
        """
        Args:
            dejavu_weight: Weight for Dejavu predictions (0.4 from research)
            lstm_weight: Weight for LSTM predictions (0.6 from research)
        """
        self.dejavu_weight = dejavu_weight
        self.lstm_weight = lstm_weight
        
        # Verify weights sum to 1.0
        assert abs(dejavu_weight + lstm_weight - 1.0) < 1e-6, "Weights must sum to 1.0"
        
        self.dejavu_model = None
        self.lstm_model = None
        self.lstm_normalization = None
        self.device = None
    
    def load_models(self):
        """Load both models"""
        print("Loading models...")
        
        # Load Dejavu
        print("  Loading Dejavu (k=500)...")
        self.dejavu_model = DejavuForecaster.load('dejavu_k500.pkl')
        print(f"    ✅ Database: {len(self.dejavu_model.database)} patterns")
        
        # Load LSTM
        print("  Loading LSTM...")
        self.lstm_model = LSTMForecaster(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            forecast_horizon=7
        )
        self.lstm_model.load_state_dict(torch.load('lstm_best.pth'))
        self.lstm_model.eval()
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lstm_model = self.lstm_model.to(self.device)
        print(f"    ✅ Device: {self.device}")
        
        # Load LSTM normalization
        with open('lstm_normalization.pkl', 'rb') as f:
            self.lstm_normalization = pickle.load(f)
        print(f"    ✅ Normalization loaded")
        
        print("✅ All models loaded")
    
    def predict(self, query_pattern):
        """
        Ensemble prediction: 0.4 × Dejavu + 0.6 × LSTM
        
        From 07_ENSEMBLE_AND_PRODUCTION_API.md lines 82-130
        
        Args:
            query_pattern: 18-minute differential array
        
        Returns:
            ensemble_forecast: Weighted combination
            dejavu_pred: Dejavu component prediction
            lstm_pred: LSTM component prediction
        """
        if self.dejavu_model is None or self.lstm_model is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Dejavu prediction (lines 94-95)
        dejavu_pred = self.dejavu_model.predict(query_pattern)
        
        # LSTM prediction (lines 97-111)
        # Normalize pattern
        pattern_norm = (query_pattern - self.lstm_normalization['pattern_mean']) / \
                       self.lstm_normalization['pattern_std']
        
        # Predict
        with torch.no_grad():
            pattern_tensor = torch.FloatTensor(pattern_norm).unsqueeze(0).unsqueeze(-1)  # (1, 18, 1)
            pattern_tensor = pattern_tensor.to(self.device)
            lstm_output = self.lstm_model(pattern_tensor)
            lstm_pred_norm = lstm_output[0, -1].cpu().numpy()  # Last step = halftime (minute 24)
        
        # Denormalize LSTM prediction
        lstm_pred = float(lstm_pred_norm * self.lstm_normalization['outcome_std'][-1] + \
                          self.lstm_normalization['outcome_mean'][-1])
        
        # Ensemble (line 114-115)
        ensemble_forecast = (self.dejavu_weight * dejavu_pred + 
                           self.lstm_weight * lstm_pred)
        
        return ensemble_forecast, dejavu_pred, lstm_pred
    
    def evaluate(self, test_df):
        """
        Evaluate ensemble on test set
        
        Returns:
            metrics: Performance metrics
        """
        print(f"\nEvaluating ensemble on {len(test_df)} test games...")
        
        ensemble_preds = []
        dejavu_preds = []
        lstm_preds = []
        actuals = []
        
        for idx, row in test_df.iterrows():
            # Predict
            ens_pred, dej_pred, lstm_pred = self.predict(row['pattern'])
            actual = row['diff_at_halftime']
            
            ensemble_preds.append(ens_pred)
            dejavu_preds.append(dej_pred)
            lstm_preds.append(lstm_pred)
            actuals.append(actual)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(test_df)} games...")
        
        # Convert to numpy
        ensemble_preds = np.array(ensemble_preds)
        dejavu_preds = np.array(dejavu_preds)
        lstm_preds = np.array(lstm_preds)
        actuals = np.array(actuals)
        
        # Calculate metrics
        ensemble_errors = ensemble_preds - actuals
        dejavu_errors = dejavu_preds - actuals
        lstm_errors = lstm_preds - actuals
        
        metrics = {
            'ensemble': {
                'mae': float(np.mean(np.abs(ensemble_errors))),
                'rmse': float(np.sqrt(np.mean(ensemble_errors ** 2))),
                'median_error': float(np.median(np.abs(ensemble_errors)))
            },
            'dejavu': {
                'mae': float(np.mean(np.abs(dejavu_errors))),
                'rmse': float(np.sqrt(np.mean(dejavu_errors ** 2)))
            },
            'lstm': {
                'mae': float(np.mean(np.abs(lstm_errors))),
                'rmse': float(np.sqrt(np.mean(lstm_errors ** 2)))
            },
            'weights': {
                'dejavu': self.dejavu_weight,
                'lstm': self.lstm_weight
            }
        }
        
        return metrics, ensemble_preds, actuals


if __name__ == "__main__":
    print("="*80)
    print("ENSEMBLE MODEL (40% Dejavu + 60% LSTM)")
    print("="*80)
    
    # Create ensemble
    ensemble = EnsembleForecaster(dejavu_weight=0.4, lstm_weight=0.6)
    ensemble.load_models()
    
    # Load test data
    print("\nLoading test data...")
    with open('splits/test.pkl', 'rb') as f:
        test_df = pickle.load(f)
    print(f"✅ {len(test_df)} test games")
    
    # Test single prediction
    print("\n" + "="*80)
    print("TEST PREDICTION (Single Game)")
    print("="*80)
    
    test_game = test_df.iloc[0]
    print(f"\nGame: {test_game['away_team']} @ {test_game['home_team']} ({test_game['date']})")
    print(f"Pattern: {test_game['pattern']}")
    
    ens_pred, dej_pred, lstm_pred = ensemble.predict(test_game['pattern'])
    actual = test_game['diff_at_halftime']
    
    print(f"\nPredictions:")
    print(f"  Dejavu (40%):  {dej_pred:+.2f} points")
    print(f"  LSTM (60%):    {lstm_pred:+.2f} points")
    print(f"  Ensemble:      {ens_pred:+.2f} points")
    print(f"  Actual:        {actual:+.2f} points")
    print(f"  Ensemble error: {abs(ens_pred - actual):.2f} points")
    
    print(f"\n" + "="*80)
    print("✅ ENSEMBLE READY FOR FULL EVALUATION")
    print("="*80)
    print(f"\nRun: python3 evaluate_ensemble.py")

