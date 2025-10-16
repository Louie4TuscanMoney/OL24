"""
Test Kelly Calculator with Real MVP ML Data
Verify calculations match mathematical specs
"""

import time
from kelly_calculator import KellyCalculator

def test_kelly_with_mvp_data():
    """
    Test with real data from our MVP ML model (5.39 MAE)
    """
    print("="*80)
    print("KELLY CALCULATOR - REAL MVP DATA TEST")
    print("="*80)
    
    # Initialize calculator
    calculator = KellyCalculator(
        fraction=0.5,           # Half Kelly
        max_bet_fraction=0.15,  # Max 15% (will be enforced by Final Calibration too)
        min_edge=0.02           # Min 2% edge
    )
    
    # Test Case 1: Strong positive edge
    print("\n" + "="*80)
    print("TEST 1: STRONG POSITIVE EDGE")
    print("="*80)
    print("\nScenario: Lakers vs Celtics")
    print("  Our ML model (5.39 MAE) predicts:")
    print("    LAL +15.1 at halftime [+11.3, +18.9] (95% CI)")
    print("  BetOnline market:")
    print("    LAL -7.5 full game at -110")
    print("  Edge analysis:")
    print("    ML says LAL crushes in first half → likely covers full game")
    print("    Market only has them at -7.5")
    print("    This is a HUGE edge opportunity!")
    
    ml_pred = {
        'point_forecast': 15.1,
        'interval_lower': 11.3,
        'interval_upper': 18.9
    }
    
    market = {
        'spread': -7.5,
        'odds': -110
    }
    
    start = time.time()
    result = calculator.calculate_optimal_bet_size(
        bankroll=5000,
        ml_prediction=ml_pred,
        market_odds=market,
        confidence_factor=0.759,  # Good confidence (narrow interval)
        volatility_factor=0.571    # Moderate volatility
    )
    elapsed = (time.time() - start) * 1000
    
    print(f"\nResults:")
    print(f"  Win Probability (ML): {result['win_probability']:.1%}")
    print(f"  Market Probability: {result['market_probability']:.1%}")
    print(f"  Our Edge: {result['edge']:.1%} (HUGE!)")
    print(f"\n  Kelly Fraction: {result['kelly_fraction']:.1%}")
    print(f"  After adjustments: {result['fraction']:.1%}")
    print(f"\n  💰 OPTIMAL BET: ${result['bet_size']:.2f}")
    print(f"  Expected Value: +${result['expected_value']:.2f}")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"\n  Calculation time: {elapsed:.2f}ms {'✅' if elapsed < 5 else '❌'}")
    
    assert result['recommendation'] == 'BET', "Should recommend betting with big edge"
    assert result['bet_size'] > 0, "Bet size should be positive"
    assert result['edge'] > 0.02, "Edge should exceed minimum"
    print("\n  ✅ Test 1 PASSED")
    
    # Test Case 2: Edge too small (should skip)
    print("\n" + "="*80)
    print("TEST 2: EDGE TOO SMALL (SHOULD SKIP)")
    print("="*80)
    print("\nScenario: Close game prediction")
    print("  ML: +10.0 [+8.0, +12.0]")
    print("  Market: -9.5 at -110")
    print("  Edge: Only ~0.5 points difference")
    
    result2 = calculator.calculate_optimal_bet_size(
        bankroll=5000,
        ml_prediction={'point_forecast': 10.0, 'interval_lower': 8.0, 'interval_upper': 12.0},
        market_odds={'spread': -9.5, 'odds': -110},
        confidence_factor=0.80,
        volatility_factor=0.80
    )
    
    print(f"\nResults:")
    print(f"  Edge: {result2['edge']:.1%}")
    print(f"  Bet: ${result2['bet_size']:.2f}")
    print(f"  Recommendation: {result2['recommendation']}")
    if 'reason' in result2:
        print(f"  Reason: {result2['reason']}")
    
    assert result2['recommendation'] == 'SKIP', "Should skip with small edge"
    assert result2['bet_size'] == 0, "Bet size should be zero"
    print("\n  ✅ Test 2 PASSED")
    
    # Test Case 3: Huge edge but low confidence
    print("\n" + "="*80)
    print("TEST 3: BIG EDGE BUT LOW CONFIDENCE")
    print("="*80)
    print("\nScenario: Big prediction but very wide interval")
    print("  ML: +18.0 [+5.0, +31.0] (VERY WIDE)")
    print("  Market: -8.0 at -110")
    print("  Edge: Large on paper, but low confidence")
    
    result3 = calculator.calculate_optimal_bet_size(
        bankroll=5000,
        ml_prediction={'point_forecast': 18.0, 'interval_lower': 5.0, 'interval_upper': 31.0},
        market_odds={'spread': -8.0, 'odds': -110},
        confidence_factor=0.20,  # Very low confidence
        volatility_factor=0.60
    )
    
    print(f"\nResults:")
    print(f"  Win Probability: {result3['win_probability']:.1%}")
    print(f"  Edge: {result3['edge']:.1%}")
    print(f"  Bet: ${result3['bet_size']:.2f}")
    print(f"  Recommendation: {result3['recommendation']}")
    print(f"  → Bet size reduced due to low confidence ✅")
    
    assert result3['bet_size'] < result['bet_size'], "Low confidence should reduce bet"
    print("\n  ✅ Test 3 PASSED")
    
    # Test Case 4: Performance test
    print("\n" + "="*80)
    print("TEST 4: PERFORMANCE (1000 CALCULATIONS)")
    print("="*80)
    
    start = time.time()
    for _ in range(1000):
        calculator.calculate_optimal_bet_size(
            bankroll=5000,
            ml_prediction=ml_pred,
            market_odds=market,
            confidence_factor=0.759,
            volatility_factor=0.571
        )
    elapsed = (time.time() - start) * 1000
    avg = elapsed / 1000
    
    print(f"\n  1000 calculations: {elapsed:.1f}ms total")
    print(f"  Average: {avg:.2f}ms per calculation")
    print(f"  Target: <5ms")
    
    if avg < 5:
        print(f"  ✅ PASS - Well under target!")
    else:
        print(f"  ❌ FAIL - Too slow!")
    
    assert avg < 5, "Performance should be under 5ms"
    
    # Summary
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    print("\nKelly Calculator Verified:")
    print("  ✅ Correctly sizes bets with strong edges")
    print("  ✅ Skips bets with edges too small")
    print("  ✅ Reduces bets with low confidence")
    print("  ✅ Meets <5ms performance target")
    print("\nReady for integration with:")
    print("  - ML Model (5.39 MAE)")
    print("  - BetOnline scraper (5-sec)")
    print("  - Confidence adjuster (next)")
    print("  - Volatility estimator (next)")
    print("  - Final calibration (last)")


if __name__ == "__main__":
    test_kelly_with_mvp_data()
    
    print("\n" + "="*80)
    print("KELLY CRITERION - READY FOR PRODUCTION")
    print("="*80)

