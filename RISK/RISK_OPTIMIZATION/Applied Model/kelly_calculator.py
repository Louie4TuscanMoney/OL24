"""
Kelly Criterion Calculator
Optimal bet sizing with adjustments
Performance: <5ms per calculation
"""

import numpy as np
from typing import Dict, Optional
from probability_converter import ProbabilityConverter


class KellyCalculator:
    """
    Calculate optimal bet size using Kelly Criterion
    
    Implements:
    - Basic Kelly formula
    - Fractional Kelly (half Kelly default)
    - Confidence adjustments (from ML intervals)
    - Volatility adjustments (Black-Scholes inspired)
    - Hard limits (max 20% per bet)
    
    Performance: <5ms per calculation
    """
    
    def __init__(
        self,
        fraction: float = 0.5,
        max_bet_fraction: float = 0.20,
        min_edge: float = 0.05
    ):
        """
        Initialize Kelly Calculator
        
        Args:
            fraction: Fractional Kelly (0.5 = half Kelly, recommended)
            max_bet_fraction: Max fraction of bankroll per bet (0.20 = 20%)
            min_edge: Minimum edge required to bet (0.05 = 5%)
        """
        self.fraction = fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_edge = min_edge
        self.converter = ProbabilityConverter()
    
    def calculate_kelly_fraction(
        self,
        win_probability: float,
        decimal_odds: float
    ) -> float:
        """
        Calculate Kelly fraction
        
        Formula: f* = (p × b - (1-p)) / b
        Where: b = decimal_odds - 1
        
        Args:
            win_probability: P(win) from 0-1
            decimal_odds: Decimal odds (e.g., 1.909 for -110)
        
        Returns:
            Optimal fraction of bankroll (0-1)
        
        Time: <1ms
        """
        b = decimal_odds - 1
        
        if b <= 0:
            return 0.0  # No value bet
        
        # Kelly formula
        f_kelly = (win_probability * (b + 1) - 1) / b
        
        # Apply fractional Kelly
        f_fractional = f_kelly * self.fraction
        
        # Clip to valid range [0, max_bet_fraction]
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
        Calculate optimal bet size with all adjustments
        
        Complete Kelly calculation including:
        - Base Kelly fraction
        - Confidence adjustment (from ML interval)
        - Volatility adjustment (Black-Scholes)
        - Fractional Kelly (safety)
        - Hard limits (max 20%)
        
        Args:
            bankroll: Current bankroll ($5,000)
            ml_prediction: {
                'point_forecast': 15.1,
                'interval_lower': 11.3,
                'interval_upper': 18.9
            }
            market_odds: {
                'spread': -7.5,
                'odds': -110
            }
            confidence_factor: Adjustment for ML confidence (0-1)
            volatility_factor: Adjustment for volatility (0-1)
        
        Returns:
            {
                'bet_size': 272.50,
                'fraction': 0.0545,
                'kelly_fraction': 0.187,
                'expected_value': 96.36,
                'recommendation': 'BET'
            }
        
        Time: <5ms
        """
        # Convert ML to probability
        p_win = self.converter.ml_interval_to_probability(
            ml_forecast=ml_prediction['point_forecast'],
            ml_lower=ml_prediction['interval_lower'],
            ml_upper=ml_prediction['interval_upper'],
            market_spread=market_odds['spread']
        )
        
        # Convert market odds to decimal
        decimal_odds = self.converter.american_to_decimal_odds(market_odds['odds'])
        
        # Market implied probability
        p_market = self.converter.american_to_implied_probability(market_odds['odds'])
        
        # Calculate edge
        edge = p_win - p_market
        
        # Check minimum edge
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
                'reason': f'Edge too small ({edge:.1%} < {self.min_edge:.1%})'
            }
        
        # Calculate base Kelly
        f_kelly = self.calculate_kelly_fraction(p_win, decimal_odds)
        
        # Apply adjustments
        f_adjusted = f_kelly * confidence_factor * volatility_factor
        
        # Calculate bet size
        bet_size = bankroll * f_adjusted
        
        # Apply hard limits
        max_bet = min(
            bankroll * self.max_bet_fraction,  # Never more than 20%
            bankroll * f_kelly * 0.50  # Never more than 50% of full Kelly
        )
        
        bet_size_final = min(bet_size, max_bet)
        
        # Calculate expected value
        ev = self.converter.expected_value(
            bet_size_final,
            p_win,
            market_odds['odds']
        )
        
        # Recommendation
        recommendation = 'BET' if (ev > 0 and bet_size_final > 10) else 'SKIP'
        
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
                'fractional': self.fraction
            },
            'recommendation': recommendation,
            'limits': {
                'max_single_bet': max_bet,
                'kelly_raw': f_kelly
            }
        }


# Example usage
if __name__ == "__main__":
    import time
    
    calculator = KellyCalculator(fraction=0.5)  # Half Kelly
    
    print("="*60)
    print("KELLY CALCULATOR - COMPLETE EXAMPLE")
    print("="*60)
    
    # Scenario: LAL @ BOS at halftime
    ml_pred = {
        'point_forecast': 15.1,
        'interval_lower': 11.3,
        'interval_upper': 18.9
    }
    
    market = {
        'spread': -7.5,
        'odds': -110
    }
    
    # Calculate optimal bet
    start = time.time()
    
    result = calculator.calculate_optimal_bet_size(
        bankroll=5000,
        ml_prediction=ml_pred,
        market_odds=market,
        confidence_factor=0.759,  # From interval width
        volatility_factor=0.571    # From Black-Scholes
    )
    
    elapsed = (time.time() - start) * 1000
    
    # Display results
    print(f"\nGame: LAL @ BOS (halftime)")
    print(f"\nML Prediction:")
    print(f"  Forecast: {ml_pred['point_forecast']:+.1f}")
    print(f"  Interval: [{ml_pred['interval_lower']:+.1f}, {ml_pred['interval_upper']:+.1f}]")
    
    print(f"\nMarket Odds:")
    print(f"  Spread: {market['spread']:+.1f}")
    print(f"  Odds: {market['odds']}")
    
    print(f"\nProbabilities:")
    print(f"  ML (win): {result['win_probability']:.1%}")
    print(f"  Market (implied): {result['market_probability']:.1%}")
    print(f"  Edge: {result['edge']:.1%}")
    
    print(f"\nKelly Calculation:")
    print(f"  Raw Kelly: {result['kelly_fraction']:.1%}")
    print(f"  × Confidence: {result['adjustments']['confidence']:.3f}")
    print(f"  × Volatility: {result['adjustments']['volatility']:.3f}")
    print(f"  × Fractional: {result['adjustments']['fractional']:.2f}")
    print(f"  = Final: {result['fraction']:.1%}")
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL BET: ${result['bet_size']:.2f}")
    print(f"  ({result['fraction']*100:.2f}% of $5,000 bankroll)")
    print(f"  Expected Value: ${result['expected_value']:.2f}")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"{'='*60}")
    
    print(f"\nCalculation time: {elapsed:.2f}ms")
    
    # Test multiple scenarios
    print(f"\n\n{'='*60}")
    print("ADDITIONAL SCENARIOS")
    print("="*60)
    
    scenarios = [
        {
            'name': 'Strong edge, high confidence',
            'ml': {'point_forecast': 18.2, 'interval_lower': 16.0, 'interval_upper': 20.4},
            'market': {'spread': -6.5, 'odds': -110},
            'conf': 0.90,
            'vol': 0.85
        },
        {
            'name': 'Moderate edge, low confidence',
            'ml': {'point_forecast': 12.5, 'interval_lower': 2.0, 'interval_upper': 23.0},
            'market': {'spread': -8.0, 'odds': -110},
            'conf': 0.35,
            'vol': 0.60
        },
        {
            'name': 'Small edge (should skip)',
            'ml': {'point_forecast': 10.0, 'interval_lower': 8.0, 'interval_upper': 12.0},
            'market': {'spread': -9.5, 'odds': -110},
            'conf': 0.85,
            'vol': 0.80
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
        print(f"  Edge: {result['edge']:.1%}")
        print(f"  Bet: ${result['bet_size']:.2f} ({result['fraction']*100:.1f}%)")
        print(f"  EV: ${result['expected_value']:.2f}")
        print(f"  Recommendation: {result['recommendation']}")
    
    print("\n" + "="*60)

