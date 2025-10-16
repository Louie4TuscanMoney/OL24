"""
Dejavu K-NN Forecaster for NBA Halftime Prediction
Research-verified implementation following Kang et al. 2020

EXACT SPECIFICATIONS:
- k = 500 (paper optimal, MATH_BREAKDOWN.txt line 83)
- Median aggregation (paper method, line 81)
- Euclidean distance (L2, fastest with good performance)
- Z-score normalization (per pattern)
"""

import numpy as np
import pickle
from pathlib import Path
import time

class DejavuForecaster:
    """
    Pattern-matching forecaster using K-Nearest Neighbors
    
    Paper: Kang et al., "Déjà vu: A data-centric forecasting approach 
           through time series cross-similarity", arXiv 2020
    """
    
    def __init__(self, k=500):
        """
        Initialize Dejavu forecaster
        
        Args:
            k: Number of nearest neighbors (paper optimal: 500)
        """
        self.k = k
        self.database = []
        self.pattern_length = None
    
    def fit(self, train_df):
        """
        Build pattern database from training games
        
        Args:
            train_df: Training DataFrame with 'pattern' and 'diff_at_halftime' columns
        """
        print(f"Building Dejavu database...")
        print(f"  k = {self.k} (paper-verified optimal)")
        
        for idx, row in train_df.iterrows():
            pattern = row['pattern']
            
            # Validate pattern
            if self.pattern_length is None:
                self.pattern_length = len(pattern)
            else:
                assert len(pattern) == self.pattern_length, \
                    f"Pattern length mismatch: expected {self.pattern_length}, got {len(pattern)}"
            
            self.database.append({
                'pattern': pattern,
                'outcome': row['diff_at_halftime'],  # What we're predicting
                'game_id': row['game_id'],
                'date': row['date'],
                'season': row['season'],
                'home_team': row['home_team'],
                'away_team': row['away_team']
            })
        
        print(f"✅ Database built: {len(self.database)} patterns")
        print(f"   Pattern length: {self.pattern_length}")
    
    def _normalize(self, pattern):
        """
        Z-score normalization (research-verified)
        
        From 04_DEJAVU_DEPLOYMENT.md lines 100-101
        """
        mean = np.mean(pattern)
        std = np.std(pattern)
        
        # Avoid division by zero
        if std < 1e-10:
            return pattern - mean
        
        return (pattern - mean) / std
    
    def _euclidean_distance(self, pattern1, pattern2):
        """
        Compute Euclidean (L2) distance between normalized patterns
        
        From MATH_BREAKDOWN.txt line 76:
        d_L2 = √(Σ(ỹ_t - Q̃^(i)_t)²)
        """
        p1_norm = self._normalize(pattern1)
        p2_norm = self._normalize(pattern2)
        
        return np.sqrt(np.sum((p1_norm - p2_norm) ** 2))
    
    def predict(self, query_pattern, return_neighbors=False):
        """
        Forecast using k-NN with median aggregation
        
        From MATH_BREAKDOWN.txt line 81:
        "Forecast: median of their future paths"
        
        Args:
            query_pattern: 18-minute differential pattern
            return_neighbors: If True, return k nearest neighbors info
        
        Returns:
            forecast: Predicted halftime differential (median of k=500)
            neighbors: (optional) List of k nearest neighbors with metadata
        """
        if len(query_pattern) != self.pattern_length:
            raise ValueError(f"Query pattern length {len(query_pattern)} != {self.pattern_length}")
        
        # Compute distances to all database patterns
        distances = []
        for entry in self.database:
            dist = self._euclidean_distance(query_pattern, entry['pattern'])
            distances.append((dist, entry))
        
        # Sort by distance and select k nearest
        distances.sort(key=lambda x: x[0])
        top_k = distances[:self.k]
        
        # Extract outcomes
        outcomes = [entry['outcome'] for (dist, entry) in top_k]
        
        # MEDIAN aggregation (paper-verified)
        forecast = np.median(outcomes)
        
        if return_neighbors:
            neighbors = []
            for rank, (dist, entry) in enumerate(top_k, 1):
                neighbors.append({
                    'rank': rank,
                    'distance': dist,
                    'outcome': entry['outcome'],
                    'game_id': entry['game_id'],
                    'date': entry['date'],
                    'home_team': entry['home_team'],
                    'away_team': entry['away_team']
                })
            return forecast, neighbors
        
        return forecast
    
    def evaluate(self, test_df):
        """
        Evaluate forecaster on test set
        
        Returns:
            metrics: Dict with MAE, RMSE, etc.
            predictions: List of (actual, predicted) tuples
        """
        print(f"\nEvaluating Dejavu on {len(test_df)} test games...")
        
        predictions = []
        actuals = []
        
        start_time = time.time()
        
        for idx, row in test_df.iterrows():
            # Predict
            forecast = self.predict(row['pattern'])
            actual = row['diff_at_halftime']
            
            predictions.append(forecast)
            actuals.append(actual)
            
            # Progress
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(test_df)} games...")
        
        elapsed = time.time() - start_time
        
        # Convert to numpy
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        errors = predictions - actuals
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        mape = np.mean(np.abs(errors / (actuals + 1e-10))) * 100
        
        # Additional metrics
        median_error = np.median(np.abs(errors))
        max_error = np.max(np.abs(errors))
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'median_error': median_error,
            'max_error': max_error,
            'n_samples': len(test_df),
            'inference_time_total': elapsed,
            'inference_time_per_game': elapsed / len(test_df) * 1000  # ms
        }
        
        return metrics, predictions, actuals
    
    def save(self, filepath):
        """Save forecaster to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"✅ Dejavu forecaster saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load forecaster from disk"""
        import sys
        sys.modules['__main__'].DejavuForecaster = cls
        with open(filepath, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    print("="*80)
    print("DEJAVU MODEL (Research-Verified Implementation)")
    print("="*80)
    
    # Load training data
    print("\nLoading training data...")
    with open('splits/train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    print(f"✅ Loaded {len(train_df)} training games")
    
    # Create and fit Dejavu
    print("\nCreating Dejavu forecaster...")
    dejavu = DejavuForecaster(k=500)
    dejavu.fit(train_df)
    
    # Test on one game
    print("\n" + "="*80)
    print("TEST PREDICTION (Single Game)")
    print("="*80)
    
    test_pattern = train_df.iloc[0]['pattern']
    test_actual = train_df.iloc[0]['diff_at_halftime']
    
    print(f"\nTest game: {train_df.iloc[0]['away_team']} @ {train_df.iloc[0]['home_team']}")
    print(f"Date: {train_df.iloc[0]['date']}")
    print(f"Pattern (18 min): {test_pattern}")
    
    forecast, neighbors = dejavu.predict(test_pattern, return_neighbors=True)
    
    print(f"\nPrediction:")
    print(f"  Forecast: {forecast:.2f} points")
    print(f"  Actual:   {test_actual:.2f} points")
    print(f"  Error:    {abs(forecast - test_actual):.2f} points")
    
    print(f"\nTop 5 similar games (out of k={dejavu.k}):")
    for neighbor in neighbors[:5]:
        print(f"  {neighbor['rank']:3d}. {neighbor['away_team']} @ {neighbor['home_team']} "
              f"({neighbor['date']}) → {neighbor['outcome']:+.1f} pts (dist: {neighbor['distance']:.3f})")
    
    # Save model
    print(f"\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    dejavu.save('dejavu_k500.pkl')
    
    print(f"\n" + "="*80)
    print("✅ DEJAVU MODEL READY")
    print("="*80)
    print(f"\nNext: Run evaluation on test set (893 games)")
    print(f"Expected MAE: ~6.0 points (MODELSYNERGY.md)")

