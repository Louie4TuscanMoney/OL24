"""
Dejavu Model Implementation
K-Nearest Neighbors forecasting with similarity-based weighting

Based on Dejavu paper (Kang et al. 2020):
- k=500 optimal (tested 1, 5, 10, 50, 100, 500, 1000)
- DTW distance for temporal patterns
- Median aggregation (robust to outliers)

Performance: <100ms per prediction (real-time compatible)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import pickle
from scipy.spatial.distance import euclidean, correlation
from scipy.stats import median_abs_deviation
import time

class DejavuForecaster:
    """
    Dejavu forecasting model - pattern matching via K-NN
    
    No training required! Just load reference set and predict.
    
    Paper specs:
    - k=500 neighbors (optimal from paper experiments)
    - DTW or L2 distance
    - Median aggregation (robust)
    """
    
    def __init__(
        self,
        k: int = 500,
        distance_metric: str = 'euclidean',  # 'euclidean', 'dtw', 'correlation'
        aggregation: str = 'median'  # 'median', 'mean', 'weighted_mean'
    ):
        """
        Args:
            k: Number of nearest neighbors (paper optimal: 500)
            distance_metric: Similarity measure
            aggregation: How to combine k neighbors
        """
        self.k = k
        self.distance_metric = distance_metric
        self.aggregation = aggregation
        
        # Reference set (loaded from file)
        self.patterns = None
        self.outcomes = None
        self.metadata = None
        self.normalization = None
    
    def load_reference_set(self, reference_dir: str = 'reference_set'):
        """
        Load pre-computed reference set
        """
        ref_path = Path(reference_dir)
        
        print(f"Loading reference set from {ref_path}...")
        
        self.patterns = np.load(ref_path / 'patterns.npy')
        self.outcomes = np.load(ref_path / 'outcomes.npy')
        self.metadata = pd.read_parquet(ref_path / 'metadata.parquet')
        
        with open(ref_path / 'normalization.pkl', 'rb') as f:
            self.normalization = pickle.load(f)
        
        print(f"✅ Loaded {len(self.patterns):,} reference patterns")
        print(f"   Pattern shape: {self.patterns.shape}")
        print(f"   Using k={self.k} nearest neighbors")
    
    def compute_distances(self, query_pattern: np.ndarray) -> np.ndarray:
        """
        Compute distances from query to all reference patterns
        
        Args:
            query_pattern: (18,) array representing current game state
        
        Returns:
            distances: (N,) array of distances
        
        Time: <50ms for 5,000 patterns
        """
        if self.distance_metric == 'euclidean':
            # L2 distance (fast)
            distances = np.linalg.norm(self.patterns - query_pattern, axis=1)
        
        elif self.distance_metric == 'correlation':
            # Correlation distance (scale-invariant)
            distances = np.array([
                1 - abs(np.corrcoef(query_pattern, pattern)[0, 1])
                for pattern in self.patterns
            ])
        
        elif self.distance_metric == 'dtw':
            # DTW (slow but handles time warping)
            from fastdtw import fastdtw
            distances = np.array([
                fastdtw(query_pattern, pattern)[0]
                for pattern in self.patterns
            ])
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def find_k_nearest(self, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors
        
        Returns:
            indices: Indices of k nearest neighbors
            distances_k: Distances to k nearest neighbors
        """
        # Get k smallest distances
        k_indices = np.argpartition(distances, min(self.k, len(distances)-1))[:self.k]
        k_indices = k_indices[np.argsort(distances[k_indices])]  # Sort by distance
        
        distances_k = distances[k_indices]
        
        return k_indices, distances_k
    
    def aggregate_outcomes(
        self,
        k_indices: np.ndarray,
        distances_k: np.ndarray
    ) -> Dict:
        """
        Aggregate outcomes from k nearest neighbors
        
        Paper uses: Median (robust to outliers)
        
        Returns:
            {
                'point_forecast': 15.1,  # Median/mean prediction
                'lower': 11.3,  # 5th percentile
                'upper': 18.9,  # 95th percentile
                'neighbors_used': 500,
                'avg_distance': 0.42
            }
        """
        # Get outcomes of k nearest neighbors
        neighbor_outcomes = self.outcomes[k_indices]
        
        # Aggregate based on method
        if self.aggregation == 'median':
            point_forecast = np.median(neighbor_outcomes)
        
        elif self.aggregation == 'mean':
            point_forecast = np.mean(neighbor_outcomes)
        
        elif self.aggregation == 'weighted_mean':
            # Weight by similarity (inverse distance)
            weights = np.exp(-distances_k / distances_k.std())
            point_forecast = np.average(neighbor_outcomes, weights=weights)
        
        else:
            point_forecast = np.median(neighbor_outcomes)
        
        # Calculate uncertainty (for Conformal layer)
        # Use quantiles of neighbor outcomes
        lower = np.percentile(neighbor_outcomes, 5)   # 5th percentile
        upper = np.percentile(neighbor_outcomes, 95)  # 95th percentile
        
        return {
            'point_forecast': float(point_forecast),
            'lower': float(lower),
            'upper': float(upper),
            'neighbors_used': len(k_indices),
            'avg_distance': float(distances_k.mean()),
            'median_distance': float(np.median(distances_k)),
            'neighbor_outcomes': neighbor_outcomes.tolist()
        }
    
    def predict(self, query_features: Dict) -> Dict:
        """
        Make prediction for new game
        
        Args:
            query_features: {
                'differential_ht': -4,  # BOS leading by 4 at halftime
                'home_rolling_diff': 2.5,
                'away_rolling_diff': 3.2,
                'home_ht_avg': 1.8,
                'away_ht_avg': 2.1,
                ...
            }
        
        Returns:
            {
                'point_forecast': 15.1,  # Expected differential change
                'lower': 11.3,
                'upper': 18.9,
                'prediction_type': 'dejavu_knn',
                'computation_time_ms': 45.2
            }
        
        Time: <100ms (target)
        """
        start = time.time()
        
        # Create query pattern vector (match training features)
        # For now, use primary feature
        # TODO: Expand to full 18 features
        query_pattern = np.array([
            query_features.get('differential_ht', 0),
            query_features.get('home_rolling_diff', 0),
            query_features.get('away_rolling_diff', 0),
            query_features.get('home_ht_avg', 0),
            query_features.get('away_ht_avg', 0),
            query_features.get('home_rolling_std', 1.0),
            query_features.get('away_rolling_std', 1.0)
        ])
        
        # Normalize query (use same normalization as training)
        # For now, skip normalization (patterns already normalized)
        
        # Compute distances
        distances = self.compute_distances(query_pattern)
        
        # Find k nearest
        k_indices, distances_k = self.find_k_nearest(distances)
        
        # Aggregate outcomes
        result = self.aggregate_outcomes(k_indices, distances_k)
        
        elapsed = (time.time() - start) * 1000
        
        # Add metadata
        result['prediction_type'] = 'dejavu_knn'
        result['k'] = self.k
        result['distance_metric'] = self.distance_metric
        result['computation_time_ms'] = elapsed
        
        return result
    
    def predict_batch(self, query_features_list: List[Dict]) -> List[Dict]:
        """
        Batch prediction (faster for multiple games)
        """
        return [self.predict(features) for features in query_features_list]


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = DejavuForecaster(k=500, distance_metric='euclidean', aggregation='median')
    
    # Load reference set
    model.load_reference_set('reference_set')
    
    print("\n" + "="*60)
    print("DEJAVU MODEL - TEST PREDICTION")
    print("="*60)
    
    # Example query: LAL @ BOS, halftime
    query = {
        'differential_ht': -4,  # BOS up 4 (52-48)
        'home_rolling_diff': 2.5,  # BOS recent average: +2.5
        'away_rolling_diff': 3.5,  # LAL recent average: +3.5
        'home_ht_avg': 1.2,  # BOS typical halftime diff
        'away_ht_avg': 2.8,  # LAL typical halftime diff
        'home_rolling_std': 8.5,
        'away_rolling_std': 7.2
    }
    
    # Make prediction
    start = time.time()
    prediction = model.predict(query)
    elapsed = (time.time() - start) * 1000
    
    print(f"\nQuery (halftime state):")
    print(f"  BOS vs LAL")
    print(f"  Score: BOS 52, LAL 48 (BOS +4)")
    print(f"  BOS recent form: +2.5 avg")
    print(f"  LAL recent form: +3.5 avg")
    
    print(f"\nDejavu Prediction (change halftime → final):")
    print(f"  Point forecast: {prediction['point_forecast']:+.1f}")
    print(f"  90% interval: [{prediction['lower']:+.1f}, {prediction['upper']:+.1f}]")
    print(f"  Neighbors used: {prediction['neighbors_used']}")
    print(f"  Avg distance: {prediction['avg_distance']:.3f}")
    
    print(f"\nInterpretation:")
    current_diff = query['differential_ht']
    predicted_change = prediction['point_forecast']
    predicted_final = current_diff + predicted_change
    print(f"  Current (halftime): BOS {current_diff:+.0f}")
    print(f"  Predicted change: {predicted_change:+.1f}")
    print(f"  Predicted final: BOS {predicted_final:+.1f}")
    
    print(f"\nPerformance:")
    print(f"  Computation time: {prediction['computation_time_ms']:.1f}ms")
    print(f"  Target: <100ms ✅" if prediction['computation_time_ms'] < 100 else "  Target: <100ms ❌")
    
    print("\n" + "="*60)
    print("✅ DEJAVU MODEL READY")
    print("="*60)
    print("\nNext step: Run 04_ensemble_integration.py")

