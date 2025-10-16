"""
TEST SUITE 2: Delta Optimization
Comprehensive testing of correlation-based hedging (The Rubber Band)
"""

import sys
from pathlib import Path
import numpy as np

# Add Delta path
delta_path = Path(__file__).parent.parent / "4. RISK" / "2. Delta Optimization"
sys.path.insert(0, str(delta_path))

from correlation_tracker import CorrelationTracker
from delta_calculator import DeltaCalculator
from hedge_optimizer import HedgeOptimizer

def test_delta_optimization():
    """Test Delta Optimization system"""
    print("="*80)
    print("TEST SUITE 2: DELTA OPTIMIZATION (The Rubber Band)")
    print("="*80)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Correlation tracking
    print("\n[Test 2.1] Correlation Tracker")
    tests_total += 1
    
    tracker = CorrelationTracker(window_size=50, min_samples=20)
    
    # Add simulated historical data
    np.random.seed(42)
    for i in range(50):
        base = np.random.normal(12, 2)
        ml_forecast = base + np.random.normal(0, 0.5)
        market_spread = -(base * 0.55) + np.random.normal(0, 0.3)
        tracker.update(ml_forecast, market_spread)
    
    correlation = tracker.get_correlation()
    assert 0.5 < correlation < 0.95, "Correlation should be in reasonable range"
    
    print(f"  ✅ Correlation: {correlation:.3f} (simulated correlated data)")
    tests_passed += 1
    
    # Test 2: Gap z-score analysis
    print("\n[Test 2.2] Gap Z-Score (Rubber Band Stretch)")
    tests_total += 1
    
    # Add unusual gap (stretched rubber band!)
    tracker.update(ml_forecast=20.0, market_spread=-8.0)
    
    gap_stats = tracker.get_gap_statistics()
    assert 'z_score' in gap_stats, "Should calculate z-score"
    
    print(f"  ✅ Gap: {gap_stats['current_gap']:.1f} points")
    print(f"     Z-score: {gap_stats['z_score']:.2f}σ")
    
    if abs(gap_stats['z_score']) > 2.0:
        print(f"     ⚠️  UNUSUAL GAP - Mean reversion expected!")
    
    tests_passed += 1
    
    # Test 3: Tension metric
    print("\n[Test 2.3] Rubber Band Tension")
    tests_total += 1
    
    tension = tracker.get_tension_metric()
    assert tension >= 0, "Tension should be non-negative"
    
    print(f"  ✅ Tension: {tension:.2f}")
    
    if tension > 3.0:
        print(f"     ⚠️  HIGH TENSION - Amplify bet!")
    
    tests_passed += 1
    
    # Test 4: Delta calculation
    print("\n[Test 2.4] Delta Calculator (Sensitivity)")
    tests_total += 1
    
    delta_calc = DeltaCalculator()
    
    ml_delta = delta_calc.calculate_ml_delta(15.1, 11.3, 18.9)
    market_delta = delta_calc.calculate_market_delta(-7.5)
    
    assert ml_delta > 0, "ML delta should be positive"
    assert market_delta > 0, "Market delta should be positive"
    
    print(f"  ✅ ML Delta: {ml_delta:.4f}")
    print(f"     Market Delta: {market_delta:.4f}")
    print(f"     Ratio: {ml_delta/market_delta:.3f}")
    tests_passed += 1
    
    # Test 5: Hedge optimization
    print("\n[Test 2.5] Hedge Optimizer (Strategy Selection)")
    tests_total += 1
    
    optimizer = HedgeOptimizer(max_amplification=2.0)
    
    position = optimizer.optimize_position(
        base_bet=272.50,
        ml_prediction={'point_forecast': 20.0, 'interval_lower': 17.0, 'interval_upper': 23.0},
        market_odds={'spread': -8.0, 'odds': -110},
        correlation=0.85,
        gap_z_score=3.19,
        ml_confidence=0.85
    )
    
    assert 'strategy' in position, "Should return strategy"
    assert position['primary_bet'] >= 0, "Bet should be non-negative"
    
    print(f"  ✅ Strategy: {position['strategy']}")
    print(f"     Primary bet: ${position['primary_bet']:.2f}")
    print(f"     Net exposure: ${position['net_exposure']:.2f}")
    
    if 'amplification' in position:
        print(f"     Amplification: {position['amplification']}x")
    
    tests_passed += 1
    
    # Test 6: Performance
    print("\n[Test 2.6] Performance Test")
    tests_total += 1
    
    import time
    start = time.time()
    for _ in range(100):
        tracker.get_correlation()
        tracker.get_gap_statistics()
        tracker.get_tension_metric()
    elapsed = (time.time() - start) * 1000 / 100
    
    assert elapsed < 15, "Should be under 15ms target"
    print(f"  ✅ Average: {elapsed:.2f}ms (target: <15ms)")
    tests_passed += 1
    
    # Summary
    print("\n" + "="*80)
    print(f"DELTA OPTIMIZATION TESTS: {tests_passed}/{tests_total} PASSED")
    print("="*80)
    
    if tests_passed == tests_total:
        print("✅ ALL TESTS PASSED - The Rubber Band works!")
        return True
    else:
        print(f"❌ {tests_total - tests_passed} TESTS FAILED")
        return False

if __name__ == "__main__":
    success = test_delta_optimization()
    sys.exit(0 if success else 1)

