"""
Conformal Prediction Wrapper
Following: ML/Action Steps Folder/05_CONFORMAL_WRAPPER.md

Wraps ensemble to provide 95% confidence intervals
Expected interval width: ±3.8 points (from MODELSYNERGY.md)
"""

import numpy as np
import pickle

class ConformalPredictor:
    """
    Adaptive Conformal Prediction for NBA
    
    From 05_CONFORMAL_WRAPPER.md lines 41-184
    """
    
    def __init__(self, alpha=0.05):
        """
        Args:
            alpha: Significance level (0.05 = 95% coverage)
        """
        self.alpha = alpha
        self.quantile = None
        self.scores = []
        self.is_fitted = False
    
    def fit(self, calibration_df, ensemble_model):
        """
        Calibrate on held-out calibration set
        
        Args:
            calibration_df: 932 calibration games
            ensemble_model: Trained ensemble (Dejavu + LSTM)
        """
        print(f"Calibrating conformal predictor...")
        print(f"  Calibration games: {len(calibration_df)}")
        print(f"  Target coverage: {1-self.alpha:.1%}")
        
        scores = []
        
        for idx, row in calibration_df.iterrows():
            # Get ensemble prediction
            ensemble_pred, _, _ = ensemble_model.predict(row['pattern'])
            
            # True value
            actual = row['diff_at_halftime']
            
            # Nonconformity score (absolute error)
            score = abs(ensemble_pred - actual)
            scores.append(score)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(calibration_df)} games...")
        
        self.scores = np.array(scores)
        
        # Compute (1-alpha) quantile
        # From CONFORMAL_MATH_BREAKDOWN.txt: Use ceiling for finite-sample guarantee
        n = len(self.scores)
        k = int(np.ceil((1 - self.alpha) * (n + 1)))
        k = min(k, n)
        
        sorted_scores = np.sort(self.scores)
        self.quantile = sorted_scores[k-1] if k > 0 else sorted_scores[0]
        
        self.is_fitted = True
        
        print(f"✅ Calibration complete")
        print(f"   Quantile (α={self.alpha}): ±{self.quantile:.2f} points")
        print(f"   Expected coverage: {1-self.alpha:.1%}")
        
        return self
    
    def predict(self, pattern, ensemble_model):
        """
        Generate prediction with conformal interval
        
        Args:
            pattern: 18-minute differential pattern
            ensemble_model: Ensemble model
        
        Returns:
            point_forecast: Ensemble prediction
            interval: (lower, upper) bounds with (1-alpha) coverage
            components: Dict with Dejavu/LSTM predictions
        """
        if not self.is_fitted:
            raise ValueError("Conformal predictor not calibrated. Call fit() first.")
        
        # Get ensemble prediction
        ensemble_pred, dejavu_pred, lstm_pred = ensemble_model.predict(pattern)
        
        # Construct conformal interval
        lower = ensemble_pred - self.quantile
        upper = ensemble_pred + self.quantile
        
        components = {
            'dejavu_prediction': float(dejavu_pred),
            'lstm_prediction': float(lstm_pred),
            'ensemble_forecast': float(ensemble_pred)
        }
        
        return ensemble_pred, (lower, upper), components
    
    def evaluate_coverage(self, test_df, ensemble_model):
        """
        Evaluate empirical coverage on test set
        
        Returns:
            coverage_metrics: Dict with coverage rate, interval width, etc.
        """
        print(f"\nEvaluating coverage on {len(test_df)} test games...")
        
        covered = []
        widths = []
        forecasts = []
        actuals = []
        
        for idx, row in test_df.iterrows():
            forecast, interval, _ = self.predict(row['pattern'], ensemble_model)
            lower, upper = interval
            actual = row['diff_at_halftime']
            
            # Check if actual falls within interval
            is_covered = (lower <= actual <= upper)
            covered.append(is_covered)
            widths.append(upper - lower)
            forecasts.append(forecast)
            actuals.append(actual)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(test_df)} games...")
        
        # Calculate metrics
        empirical_coverage = np.mean(covered)
        avg_width = np.mean(widths)
        mae = np.mean(np.abs(np.array(forecasts) - np.array(actuals)))
        
        metrics = {
            'empirical_coverage': float(empirical_coverage),
            'target_coverage': 1 - self.alpha,
            'coverage_gap': float(abs(empirical_coverage - (1 - self.alpha))),
            'avg_interval_width': float(avg_width),
            'quantile': float(self.quantile),
            'mae': float(mae),
            'n_samples': len(test_df)
        }
        
        return metrics
    
    def save(self, filepath):
        """Save conformal predictor"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Conformal predictor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load conformal predictor"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    from ensemble_model import EnsembleForecaster
    
    print("="*80)
    print("CONFORMAL WRAPPER CALIBRATION")
    print("="*80)
    
    # Load ensemble
    print("\nLoading ensemble...")
    ensemble = EnsembleForecaster(dejavu_weight=0.4, lstm_weight=0.6)
    ensemble.load_models()
    
    # Load calibration data
    print("\nLoading calibration data...")
    with open('splits/calibration.pkl', 'rb') as f:
        cal_df = pickle.load(f)
    print(f"✅ {len(cal_df)} calibration games")
    
    # Create and fit conformal
    print("\n" + "="*80)
    print("CALIBRATING CONFORMAL PREDICTOR")
    print("="*80)
    
    conformal = ConformalPredictor(alpha=0.05)  # 95% coverage
    conformal.fit(cal_df, ensemble)
    
    # Save
    conformal.save('conformal_predictor.pkl')
    
    print(f"\n" + "="*80)
    print("✅ CONFORMAL PREDICTOR READY")
    print("="*80)
    print(f"\nNext: Evaluate coverage on test set")

