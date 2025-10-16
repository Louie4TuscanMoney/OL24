"""
Kelly Criterion Calculator
Optimal bet sizing with fractional Kelly and safety limits

Based on: RISK_OPTIMIZATION/Applied Model/kelly_calculator.py
Following: RISK_OPTIMIZATION/MATH_BREAKDOWN.txt (Kelly Formula 1.1, 1.2)
Performance: <5ms per calculation
"""

import numpy as np
from typing import Dict
from probability_converter import ProbabilityConverter


class KellyCalculator:
    """
    Calculate optimal bet size using Kelly Criterion
    
    Formula (from MATH_BREAKDOWN.txt Section 1.1):
        f* = (p(b+1) - 1) / b
    
    Where:
        f* = Optimal fraction of bankroll
        p  = Win probability (from ML)
        b  = Net odds (decimal_odds - 1)
    
    Implements:
        - Basic Kelly formula
        - Fractional Kelly (half Kelly by default)
        - Hard limits (max 20% per bet)
        - Confidence adjustments
        - Volatility adjustments
    
    Performance: <5ms per calculation (REAL-TIME)
    """
    
    def __init__(
        self,
        fraction: float = 0.5,
        max_bet_fraction: float = 0.15,  # Changed from 0.20 to 0.15 (15% max)
        min_edge: float = 0.02  # Minimum 2% edge to bet
    ):
        """
        Initialize Kelly Calculator
        
        Args:
            fraction: Fractional Kelly multiplier
                      0.5 = Half Kelly (recommended, reduces variance by 75%)
                      0.25 = Quarter Kelly (conservative)
                      1.0 = Full Kelly (max growth, high variance)
            max_bet_fraction: Absolute max bet (15% of bankroll)
            min_edge: Minimum edge required (2% = 0.02)
        """
        self.fraction = fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_edge = min_edge
        self.converter = ProbabilityConverter()
        
        print(f"Kelly Calculator initialized:")
        print(f"  Fractional Kelly: {fraction} ({'Half' if fraction == 0.5 else 'Quarter' if fraction == 0.25 else 'Full' if fraction == 1.0 else 'Custom'})")
        print(f"  Max bet: {max_bet_fraction*100:.0f}% of bankroll")
        print(f"  Min edge: {min_edge*100:.0f}%")
    
    def calculate_kelly_fraction(
        self,
        win_probability: float,
        decimal_odds: float
    ) -> float:
        """
        Calculate base Kelly fraction
        
        Formula (MATH_BREAKDOWN.txt 1.1):
            f* = (p × (b+1) - 1) / b
            where b = decimal_odds - 1
        
        Args:
            win_probability: P(win) from ML (0-1)
            decimal_odds: Decimal odds (e.g., 1.909 for -110)
        
        Returns:
            Kelly fraction (0-1)
        
        Example:
            p = 0.65, odds = -110 (decimal 1.909)
            b = 0.909
            f* = (0.65 × 1.909 - 1) / 0.909
               = (1.241 - 1) / 0.909
               = 0.265 (26.5% of bankroll)
        
        Time: <1ms
        """
        b = decimal_odds - 1
        
        if b <= 0:
            return 0.0  # No value
        
        # Kelly formula
        f_kelly = (win_probability * (b + 1) - 1) / b
        
        # Apply fractional Kelly (half Kelly = 0.5)
        f_fractional = f_kelly * self.fraction
        
        # Enforce max bet limit
        return np.clip(f_fractional, 0.0, self.max_bet_fraction)
    
    def calculate_optimal_bet_size(
        self,
        bankroll: float,
        ml_prediction: Dict,
        market_odds: Dict,
        confidence_factor: float = 1.0,
        volatility_factor: float = 1.0
    ) -> Dict:
        """
        Calculate optimal bet size with ALL adjustments
        
        Complete flow:
        1. Convert ML interval → win probability
        2. Get market implied probability
        3. Calculate edge
        4. Check if edge > minimum
        5. Calculate base Kelly
        6. Apply confidence adjustment
        7. Apply volatility adjustment
        8. Apply fractional Kelly
        9. Enforce hard limits
        10. Calculate expected value
        
        Args:
            bankroll: Current bankroll ($5,000)
            ml_prediction: {
                'point_forecast': 15.1,      # Halftime lead
                'interval_lower': 11.3,      # 95% CI lower
                'interval_upper': 18.9       # 95% CI upper
            }
            market_odds: {
                'spread': -7.5,              # Full game spread
                'odds': -110                 # American odds
            }
            confidence_factor: ML confidence (from interval width, 0-1)
            volatility_factor: Market volatility (from Black-Scholes, 0-1)
        
        Returns:
            {
                'bet_size': 272.50,
                'fraction': 0.0545,
                'kelly_fraction': 0.187,
                'win_probability': 0.75,
                'market_probability': 0.524,
                'edge': 0.226,
                'expected_value': 96.36,
                'recommendation': 'BET',
                'adjustments': {...}
            }
        
        Time: <5ms
        """
        # Step 1: Convert ML to probability
        p_win = self.converter.ml_interval_to_probability(
            ml_forecast=ml_prediction['point_forecast'],
            ml_lower=ml_prediction['interval_lower'],
            ml_upper=ml_prediction['interval_upper'],
            market_spread=market_odds['spread']
        )
        
        # Step 2: Get market probability
        decimal_odds = self.converter.american_to_decimal_odds(market_odds['odds'])
        p_market = self.converter.american_to_implied_probability(market_odds['odds'])
        
        # Step 3: Calculate edge
        edge = p_win - p_market
        
        # Step 4: Check minimum edge
        if edge < self.min_edge:
            return {
                'bet_size': 0.0,
                'fraction': 0.0,
                'kelly_fraction': 0.0,
                'win_probability': p_win,
                'market_probability': p_market,
                'edge': edge,
                'expected_value': 0.0,
                'recommendation': 'SKIP',
                'reason': f'Edge too small: {edge:.1%} < {self.min_edge:.1%} minimum'
            }
        
        # Step 5: Calculate base Kelly
        f_kelly = self.calculate_kelly_fraction(p_win, decimal_odds)
        
        # Step 6-8: Apply all adjustments
        f_adjusted = f_kelly * confidence_factor * volatility_factor
        
        # Step 9: Calculate bet size
        bet_size = bankroll * f_adjusted
        
        # Enforce absolute max (15% of original bankroll)
        max_bet = bankroll * self.max_bet_fraction
        bet_size_final = min(bet_size, max_bet)
        
        # Step 10: Calculate expected value
        ev = self.converter.expected_value(
            bet_size_final,
            p_win,
            market_odds['odds']
        )
        
        # Recommendation
        if ev > 0 and bet_size_final >= 10:
            recommendation = 'BET'
        else:
            recommendation = 'SKIP'
        
        return {
            'bet_size': round(bet_size_final, 2),
            'fraction': bet_size_final / bankroll,
            'kelly_fraction': f_kelly,
            'win_probability': p_win,
            'market_probability': p_market,
            'edge': edge,
            'expected_value': round(ev, 2),
            'adjustments': {
                'confidence': confidence_factor,
                'volatility': volatility_factor,
                'fractional': self.fraction,
                'combined': confidence_factor * volatility_factor * self.fraction
            },
            'recommendation': recommendation,
            'limits': {
                'max_bet': max_bet,
                'kelly_raw': f_kelly,
                'kelly_fractional': f_kelly * self.fraction
            }
        }


# Test the Kelly calculator
if __name__ == "__main__":
    import time
    
    print("="*80)
    print("KELLY CRITERION CALCULATOR - VERIFICATION")
    print("="*80)
    
    # Initialize with half Kelly
    calculator = KellyCalculator(fraction=0.5, max_bet_fraction=0.15)
    
    print("\n" + "="*80)
    print("SCENARIO 1: Strong Edge, High Confidence")
    print("="*80)
    
    # Real data from our ML model
    ml_pred = {
        'point_forecast': 15.1,  # LAL leads by 15.1 at halftime
        'interval_lower': 11.3,  # 95% CI
        'interval_upper': 18.9
    }
    
    market = {
        'spread': -7.5,  # LAL favored by 7.5 full game
        'odds': -110     # Standard vig
    }
    
    print(f"\nGame: LAL @ BOS")
    print(f"  ML Forecast: LAL {ml_pred['point_forecast']:+.1f} at halftime")
    print(f"  ML Interval: [{ml_pred['interval_lower']:+.1f}, {ml_pred['interval_upper']:+.1f}]")
    print(f"  Market: LAL {market['spread']:+.1f} full game at {market['odds']}")
    
    # Calculate
    start = time.time()
    result = calculator.calculate_optimal_bet_size(
        bankroll=5000,
        ml_prediction=ml_pred,
        market_odds=market,
        confidence_factor=0.759,  # From interval width
        volatility_factor=0.571    # From Black-Scholes
    )
    elapsed = (time.time() - start) * 1000
    
    print(f"\nProbabilities:")
    print(f"  ML (win): {result['win_probability']:.3f} ({result['win_probability']*100:.1f}%)")
    print(f"  Market (implied): {result['market_probability']:.3f} ({result['market_probability']*100:.1f}%)")
    print(f"  Edge: {result['edge']:.3f} ({result['edge']*100:.1f}%)")
    
    print(f"\nKelly Breakdown:")
    print(f"  Raw Kelly: {result['kelly_fraction']:.3f} ({result['kelly_fraction']*100:.1f}%)")
    print(f"  × Confidence: {result['adjustments']['confidence']:.3f}")
    print(f"  × Volatility: {result['adjustments']['volatility']:.3f}")
    print(f"  × Fractional (0.5): {result['adjustments']['fractional']:.3f}")
    print(f"  = Final: {result['fraction']:.3f} ({result['fraction']*100:.2f}%)")
    
    print(f"\n{'='*80}")
    print(f"💰 OPTIMAL BET: ${result['bet_size']:.2f}")
    print(f"   ({result['fraction']*100:.2f}% of $5,000 bankroll)")
    print(f"   Expected Value: ${result['expected_value']:.2f}")
    print(f"   Recommendation: {result['recommendation']}")
    print(f"{'='*80}")
    
    print(f"\nPerformance: {elapsed:.2f}ms")
    if elapsed < 5:
        print(f"✅ PASS - Under 5ms target")
    else:
        print(f"❌ FAIL - Exceeds 5ms target")
    
    # Additional scenarios
    print(f"\n\n{'='*80}")
    print("ADDITIONAL SCENARIOS")
    print("="*80)
    
    scenarios = [
        {
            'name': 'Huge Edge, Tight Interval',
            'ml': {'point_forecast': 20.0, 'interval_lower': 18.5, 'interval_upper': 21.5},
            'market': {'spread': -5.5, 'odds': -110},
            'conf': 0.95,
            'vol': 0.90
        },
        {
            'name': 'Small Edge (should skip)',
            'ml': {'point_forecast': 10.0, 'interval_lower': 8.0, 'interval_upper': 12.0},
            'market': {'spread': -9.5, 'odds': -110},
            'conf': 0.80,
            'vol': 0.80
        },
        {
            'name': 'Moderate Edge, Wide Interval (low confidence)',
            'ml': {'point_forecast': 15.0, 'interval_lower': 5.0, 'interval_upper': 25.0},
            'market': {'spread': -8.0, 'odds': -110},
            'conf': 0.25,
            'vol': 0.50
        }
    ]
    
    for scenario in scenarios:
        result = calculator.calculate_optimal_bet_size(
            bankroll=5000,
            ml_prediction=scenario['ml'],
            market_odds=scenario['market'],
            confidence_factor=scenario['conf'],
            volatility_factor=scenario['vol']
        )
        
        print(f"\n{scenario['name']}:")
        print(f"  ML: {scenario['ml']['point_forecast']:+.1f} [{scenario['ml']['interval_lower']:+.1f}, {scenario['ml']['interval_upper']:+.1f}]")
        print(f"  Market: {scenario['market']['spread']:+.1f} at {scenario['market']['odds']}")
        print(f"  Edge: {result['edge']:.1%}")
        print(f"  Bet: ${result['bet_size']:.2f} ({result['fraction']*100:.1f}%)")
        print(f"  EV: ${result['expected_value']:.2f}")
        print(f"  Recommendation: {result['recommendation']}")
        if 'reason' in result:
            print(f"  Reason: {result['reason']}")
    
    print("\n" + "="*80)
    print("✅ KELLY CALCULATOR READY")
    print("="*80)

