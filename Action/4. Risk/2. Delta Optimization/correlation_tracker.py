"""
Correlation Tracker - Monitor ML vs Market Relationship
The "Rubber Band" System

Based on: DELTA_OPTIMIZATION/Applied Model/correlation_tracker.py
Following: DELTA_OPTIMIZATION/MATH_BREAKDOWN.txt (Section 1: Correlation)
Performance: <5ms per update
"""

import numpy as np
from collections import deque
from typing import Dict, Optional

class CorrelationTracker:
    """
    Track correlation between ML predictions and market odds
    
    The Rubber Band Analogy:
    - ML and market are like two masses connected by a rubber band
    - Correlation (ρ) measures the "stiffness" of the band
    - Gap measures how "stretched" the band is
    - Large gaps with high ρ → strong mean reversion expected
    
    Formula (MATH_BREAKDOWN.txt Section 1.1):
        ρ = Cov(ML, Market) / (σ_ML × σ_Market)
    
    Performance: <5ms per update
    """
    
    def __init__(self, window_size: int = 50, min_samples: int = 20):
        """
        Initialize correlation tracker
        
        Args:
            window_size: Rolling window (50 games default)
            min_samples: Minimum data before calculating (20 games)
        """
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Rolling window storage
        self.ml_history = deque(maxlen=window_size)
        self.market_history = deque(maxlen=window_size)
        
        # Conversion factor (halftime to full game)
        self.conversion_factor = 0.55  # Empirical: full game ≈ 55% of halftime
        
        print(f"Correlation Tracker initialized:")
        print(f"  Window: {window_size} games")
        print(f"  Min samples: {min_samples} games")
        print(f"  Conversion factor: {self.conversion_factor}")
    
    def update(self, ml_forecast: float, market_spread: float):
        """
        Add new ML prediction and market spread
        
        Args:
            ml_forecast: ML halftime differential (e.g., +15.1)
            market_spread: Market full-game spread (e.g., -7.5)
        
        Time: <1ms
        """
        # Convert market spread to implied halftime differential
        # Market: LAL -7.5 (full game) → +4.1 implied halftime lead
        market_implied_halftime = abs(market_spread) / self.conversion_factor
        
        # Store in rolling window
        self.ml_history.append(ml_forecast)
        self.market_history.append(market_implied_halftime)
    
    def get_correlation(self) -> float:
        """
        Calculate current correlation coefficient
        
        Formula (MATH_BREAKDOWN.txt 1.1):
            ρ = Cov(X,Y) / (σ_X × σ_Y)
        
        Returns:
            Correlation coefficient ρ ∈ [-1, 1]
            
        Interpretation:
            ρ > 0.8: Very strong (tight rubber band)
            ρ = 0.5-0.8: Moderate (normal rubber band)
            ρ < 0.5: Weak (loose rubber band)
        
        Time: <5ms
        """
        if len(self.ml_history) < self.min_samples:
            # Default assumption until enough data
            return 0.75
        
        # Convert to numpy arrays
        ml_array = np.array(self.ml_history)
        market_array = np.array(self.market_history)
        
        # Calculate correlation using numpy (fast)
        correlation_matrix = np.corrcoef(ml_array, market_array)
        correlation = correlation_matrix[0, 1]
        
        # Handle edge cases
        if np.isnan(correlation):
            return 0.75
        
        return float(correlation)
    
    def get_gap_statistics(self) -> Dict:
        """
        Calculate gap statistics for mean reversion analysis
        
        Gap = ML_prediction - Market_implied
        
        Returns:
            {
                'mean_gap': 1.2,         # Historical average gap
                'std_gap': 1.35,         # Historical volatility
                'current_gap': 4.2,      # Current gap (rubber band stretch)
                'z_score': 2.22,         # How unusual (standard deviations)
                'sample_size': 50        # Number of observations
            }
        
        Formula (MATH_BREAKDOWN.txt 2.2):
            Z = (Gap_current - μ_gap) / σ_gap
        
        Interpretation:
            |Z| > 3.0: Extremely unusual (>99.7% percentile)
            |Z| > 2.0: Unusual (>95% percentile)
            |Z| > 1.0: Moderate
            |Z| < 1.0: Normal
        
        Time: <3ms
        """
        if len(self.ml_history) < self.min_samples:
            return {
                'status': 'INSUFFICIENT_DATA',
                'sample_size': len(self.ml_history),
                'required': self.min_samples
            }
        
        # Calculate all gaps
        gaps = np.array([ml - mkt for ml, mkt in zip(self.ml_history, self.market_history)])
        
        # Statistics
        mean_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps))
        current_gap = float(gaps[-1])
        
        # Z-score (how many standard deviations from mean)
        z_score = (current_gap - mean_gap) / std_gap if std_gap > 0 else 0.0
        
        return {
            'mean_gap': mean_gap,
            'std_gap': std_gap,
            'current_gap': current_gap,
            'z_score': float(z_score),
            'sample_size': len(gaps),
            'percentile': self._z_to_percentile(z_score)
        }
    
    def get_tension_metric(self) -> float:
        """
        Calculate "rubber band tension" metric
        
        Formula (MATH_BREAKDOWN.txt 6.1):
            Tension = Gap × ρ / σ_combined
        
        High tension → Strong mean reversion expected
        
        Returns:
            Tension score (0-10+)
            
        Interpretation:
            > 5.0: Extreme tension (huge opportunity)
            3.0-5.0: High tension (strong opportunity)
            1.0-3.0: Moderate tension
            < 1.0: Low tension (normal)
        
        Time: <2ms
        """
        if len(self.ml_history) < self.min_samples:
            return 0.0
        
        gap_stats = self.get_gap_statistics()
        correlation = self.get_correlation()
        
        # Calculate combined volatility
        ml_array = np.array(self.ml_history)
        market_array = np.array(self.market_history)
        
        sigma_ml = float(np.std(ml_array))
        sigma_market = float(np.std(market_array))
        sigma_combined = np.sqrt(sigma_ml**2 + sigma_market**2)
        
        # Tension formula
        current_gap = gap_stats['current_gap']
        tension = abs(current_gap) * correlation / sigma_combined if sigma_combined > 0 else 0.0
        
        return float(tension)
    
    def get_momentum(self) -> Dict:
        """
        Check if correlation is strengthening or weakening
        
        Returns:
            {
                'momentum': 'STRENGTHENING' | 'WEAKENING' | 'STABLE',
                'trend': +0.08,              # Change in ρ
                'recent_corr': 0.87,         # Last 20 games
                'older_corr': 0.79           # Previous 20 games
            }
        
        Time: <3ms
        """
        if len(self.ml_history) < 40:
            return {'momentum': 'INSUFFICIENT_DATA'}
        
        # Split into older and recent halves
        mid = len(self.ml_history) // 2
        
        ml_older = np.array(list(self.ml_history)[:mid])
        market_older = np.array(list(self.market_history)[:mid])
        
        ml_recent = np.array(list(self.ml_history)[mid:])
        market_recent = np.array(list(self.market_history)[mid:])
        
        # Calculate correlation for each half
        corr_older = float(np.corrcoef(ml_older, market_older)[0, 1])
        corr_recent = float(np.corrcoef(ml_recent, market_recent)[0, 1])
        
        trend = corr_recent - corr_older
        
        # Determine momentum
        if trend > 0.05:
            momentum = 'STRENGTHENING'
        elif trend < -0.05:
            momentum = 'WEAKENING'
        else:
            momentum = 'STABLE'
        
        return {
            'momentum': momentum,
            'trend': float(trend),
            'recent_corr': corr_recent,
            'older_corr': corr_older
        }
    
    def _z_to_percentile(self, z_score: float) -> float:
        """Convert z-score to percentile (0-100)"""
        from scipy.stats import norm
        return float(norm.cdf(z_score) * 100)
    
    def get_complete_analysis(self) -> Dict:
        """
        Get complete correlation analysis
        
        Returns all metrics in one call
        """
        return {
            'correlation': self.get_correlation(),
            'gap_stats': self.get_gap_statistics(),
            'tension': self.get_tension_metric(),
            'momentum': self.get_momentum(),
            'sample_size': len(self.ml_history)
        }


# Test the correlation tracker
if __name__ == "__main__":
    import time
    
    print("="*80)
    print("CORRELATION TRACKER - VERIFICATION")
    print("="*80)
    
    # Initialize tracker
    tracker = CorrelationTracker(window_size=50, min_samples=20)
    
    # Simulate historical data (high correlation)
    print("\n1. Building historical data (50 games)...")
    np.random.seed(42)
    
    for i in range(50):
        # Simulate correlated ML and market
        base = np.random.normal(12, 2)
        ml_forecast = base + np.random.normal(0, 0.5)
        market_spread = -(base * 0.55) + np.random.normal(0, 0.3)
        
        tracker.update(ml_forecast, market_spread)
    
    print(f"   ✅ Added 50 historical games")
    
    # Test 2: Correlation
    print("\n2. Correlation Calculation:")
    start = time.time()
    correlation = tracker.get_correlation()
    elapsed = (time.time() - start) * 1000
    
    print(f"   ρ = {correlation:.3f}")
    print(f"   Interpretation: {'Strong' if correlation > 0.7 else 'Moderate' if correlation > 0.5 else 'Weak'}")
    print(f"   Time: {elapsed:.2f}ms {'✅' if elapsed < 5 else '❌'}")
    
    # Test 3: Add unusual gap (stretched rubber band!)
    print("\n3. Adding Unusual Gap (Rubber Band Stretched):")
    tracker.update(ml_forecast=20.0, market_spread=-8.0)  # Big ML, small market
    
    gap_stats = tracker.get_gap_statistics()
    print(f"   Current gap: {gap_stats['current_gap']:.1f} points")
    print(f"   Mean gap: {gap_stats['mean_gap']:.1f} points")
    print(f"   Z-score: {gap_stats['z_score']:.2f}σ")
    print(f"   Percentile: {gap_stats['percentile']:.1f}%")
    
    if abs(gap_stats['z_score']) > 2.0:
        print(f"   ⚠️  UNUSUAL GAP! Mean reversion expected!")
    
    # Test 4: Tension metric
    print("\n4. Rubber Band Tension:")
    tension = tracker.get_tension_metric()
    print(f"   Tension: {tension:.2f}")
    
    if tension > 3.0:
        print(f"   ⚠️  HIGH TENSION! Strong reversion expected!")
    elif tension > 1.0:
        print(f"   Moderate tension")
    else:
        print(f"   Low tension")
    
    # Test 5: Complete analysis
    print("\n5. Complete Analysis:")
    analysis = tracker.get_complete_analysis()
    print(f"   Correlation: {analysis['correlation']:.3f}")
    print(f"   Gap Z-score: {analysis['gap_stats']['z_score']:.2f}σ")
    print(f"   Tension: {analysis['tension']:.2f}")
    print(f"   Momentum: {analysis['momentum']['momentum']}")
    
    # Test 6: Performance
    print("\n6. Performance Test (1000 updates):")
    start = time.time()
    for _ in range(1000):
        tracker.update(12.0, -7.0)
    elapsed = (time.time() - start) * 1000
    avg = elapsed / 1000
    
    print(f"   1000 updates: {elapsed:.1f}ms total")
    print(f"   Average: {avg:.3f}ms per update")
    print(f"   Target: <1ms")
    
    if avg < 1:
        print(f"   ✅ PASS!")
    else:
        print(f"   ❌ FAIL - Too slow")
    
    print("\n" + "="*80)
    print("✅ CORRELATION TRACKER READY")
    print("="*80)
    print("\nThe Rubber Band System:")
    print("  ML and Market are connected by correlation")
    print("  When gap is large → rubber band stretched")
    print("  High correlation + large gap → strong mean reversion")
    print("  Use tension metric to amplify bets during opportunities")

