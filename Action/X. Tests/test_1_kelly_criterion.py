"""
TEST SUITE 1: Kelly Criterion
Comprehensive testing of Kelly calculator
"""

import sys
from pathlib import Path

# Add Kelly path
kelly_path = Path(__file__).parent.parent / "4. RISK" / "1. Kelly Criterion"
sys.path.insert(0, str(kelly_path))

from kelly_calculator import KellyCalculator
from probability_converter import ProbabilityConverter

def test_kelly_criterion():
    """Test Kelly Criterion calculator"""
    print("="*80)
    print("TEST SUITE 1: KELLY CRITERION")
    print("="*80)
    
    calculator = KellyCalculator(fraction=0.5, max_bet_fraction=0.15)
    converter = ProbabilityConverter()
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Probability conversions
    print("\n[Test 1.1] Probability Conversions")
    tests_total += 1
    
    decimal = converter.american_to_decimal_odds(-110)
    implied = converter.american_to_implied_probability(-110)
    
    assert abs(decimal - 1.909) < 0.01, "Decimal odds incorrect"
    assert abs(implied - 0.524) < 0.01, "Implied probability incorrect"
    
    print(f"  ✅ -110 → {decimal:.3f} decimal, {implied:.3f} probability")
    tests_passed += 1
    
    # Test 2: ML interval to probability
    print("\n[Test 1.2] ML Interval to Probability")
    tests_total += 1
    
    prob = converter.ml_interval_to_probability(
        ml_forecast=15.1,
        ml_lower=11.3,
        ml_upper=18.9,
        market_spread=-7.5
    )
    
    # Conservative adjustment ensures prob is between 0.75 and 1.00
    assert 0.70 < prob <= 1.00, f"Probability {prob:.3f} out of reasonable range [0.70, 1.00]"
    print(f"  ✅ ML +15.1 [+11.3, +18.9] vs -7.5 spread → {prob:.3f} win probability")
    tests_passed += 1
    
    # Test 3: Expected value
    print("\n[Test 1.3] Expected Value Calculation")
    tests_total += 1
    
    ev = converter.expected_value(bet_size=400, win_probability=0.65, american_odds=-110)
    
    assert ev > 0, "EV should be positive with edge"
    assert 90 < ev < 105, "EV seems incorrect"
    print(f"  ✅ $400 bet at 65% win prob → ${ev:.2f} EV")
    tests_passed += 1
    
    # Test 4: Kelly calculation with strong edge
    print("\n[Test 1.4] Kelly with Strong Edge")
    tests_total += 1
    
    result = calculator.calculate_optimal_bet_size(
        bankroll=5000,
        ml_prediction={'point_forecast': 15.1, 'interval_lower': 11.3, 'interval_upper': 18.9},
        market_odds={'spread': -7.5, 'odds': -110},
        confidence_factor=0.759,
        volatility_factor=0.571
    )
    
    assert result['recommendation'] == 'BET', "Should recommend betting with strong edge"
    assert result['bet_size'] > 0, "Bet size should be positive"
    assert result['edge'] > 0.02, "Edge should exceed minimum"
    assert result['bet_size'] <= 750, "Should not exceed 15% of $5,000"
    
    print(f"  ✅ Strong edge ({result['edge']:.1%}) → ${result['bet_size']:.2f} bet")
    print(f"     EV: ${result['expected_value']:.2f}")
    tests_passed += 1
    
    # Test 5: Kelly with low confidence (wide interval)
    print("\n[Test 1.5] Kelly with Low Confidence")
    tests_total += 1
    
    result_low_conf = calculator.calculate_optimal_bet_size(
        bankroll=5000,
        ml_prediction={'point_forecast': 10.0, 'interval_lower': 0.0, 'interval_upper': 20.0},
        market_odds={'spread': -9.5, 'odds': -110},
        confidence_factor=0.30,  # Very low confidence
        volatility_factor=0.50   # High volatility
    )
    
    # With very low confidence factors, bet should be small or skip
    assert result_low_conf['bet_size'] < 200, "Bet should be small with low confidence"
    
    print(f"  ✅ Low confidence (wide interval) → ${result_low_conf['bet_size']:.2f} bet (reduced)")
    print(f"     Edge: {result_low_conf['edge']:.1%}, Recommendation: {result_low_conf['recommendation']}")
    tests_passed += 1
    
    # Test 6: Performance test
    print("\n[Test 1.6] Performance Test (1000 calculations)")
    tests_total += 1
    
    import time
    start = time.time()
    for _ in range(1000):
        calculator.calculate_optimal_bet_size(
            bankroll=5000,
            ml_prediction={'point_forecast': 15.1, 'interval_lower': 11.3, 'interval_upper': 18.9},
            market_odds={'spread': -7.5, 'odds': -110},
            confidence_factor=0.759,
            volatility_factor=0.571
        )
    elapsed = (time.time() - start) * 1000
    avg = elapsed / 1000
    
    assert avg < 5, "Performance should be under 5ms"
    print(f"  ✅ Average: {avg:.2f}ms (target: <5ms)")
    tests_passed += 1
    
    # Summary
    print("\n" + "="*80)
    print(f"KELLY CRITERION TESTS: {tests_passed}/{tests_total} PASSED")
    print("="*80)
    
    if tests_passed == tests_total:
        print("✅ ALL TESTS PASSED")
        return True
    else:
        print(f"❌ {tests_total - tests_passed} TESTS FAILED")
        return False

if __name__ == "__main__":
    success = test_kelly_criterion()
    sys.exit(0 if success else 1)

