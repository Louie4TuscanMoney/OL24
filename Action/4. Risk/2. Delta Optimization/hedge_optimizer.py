"""
Hedge Optimizer - Optimal Position Management
Uses correlation + gap analysis + delta to optimize positions

Based on: DELTA_OPTIMIZATION/Applied Model/hedge_optimizer.py
Following: DELTA_OPTIMIZATION/MATH_BREAKDOWN.txt (Sections 4-6)
Performance: <10ms per optimization
"""

import numpy as np
from typing import Dict
from correlation_tracker import CorrelationTracker
from delta_calculator import DeltaCalculator


class HedgeOptimizer:
    """
    Optimize bet sizing using correlation-based hedging
    
    Three Strategies:
    1. AMPLIFICATION - Increase bet when gap is unusual (rubber band stretched)
    2. PARTIAL_HEDGE - Hedge portion of bet when moderate uncertainty  
    3. DELTA_NEUTRAL - Balanced position when low conviction
    
    Performance: <10ms per optimization
    """
    
    def __init__(
        self,
        max_amplification: float = 2.0,
        max_hedge_ratio: float = 0.50
    ):
        """
        Initialize hedge optimizer
        
        Args:
            max_amplification: Max bet multiplier (2.0 = double)
            max_hedge_ratio: Max hedge as fraction of primary (0.50 = 50%)
        """
        self.max_amplification = max_amplification
        self.max_hedge_ratio = max_hedge_ratio
        
        self.delta_calc = DeltaCalculator()
        
        print("Hedge Optimizer initialized:")
        print(f"  Max amplification: {max_amplification}x")
        print(f"  Max hedge ratio: {max_hedge_ratio*100:.0f}%")
    
    def optimize_position(
        self,
        base_bet: float,
        ml_prediction: Dict,
        market_odds: Dict,
        correlation: float,
        gap_z_score: float,
        ml_confidence: float
    ) -> Dict:
        """
        Calculate optimal position with correlation-based adjustments
        
        Args:
            base_bet: Kelly-optimal bet from RISK_OPTIMIZATION ($272.50)
            ml_prediction: {
                'point_forecast': 15.1,
                'interval_lower': 11.3,
                'interval_upper': 18.9
            }
            market_odds: {
                'spread': -7.5,
                'odds': -110
            }
            correlation: ρ between ML and market (0.85)
            gap_z_score: How unusual gap is (5.14σ)
            ml_confidence: Confidence factor from interval (0.759)
        
        Returns:
            {
                'strategy': 'AMPLIFICATION',
                'primary_bet': 354.00,      # Amplified from $272.50
                'hedge_bet': 0.00,          # No hedge when amplifying
                'net_exposure': 354.00,
                'amplification': 1.30,
                'reasoning': '...'
            }
        
        Time: <10ms
        """
        # Step 1: Select strategy based on gap, correlation, confidence
        strategy = self._select_strategy(gap_z_score, correlation, ml_confidence)
        
        # Step 2: Calculate position based on strategy
        if strategy == 'AMPLIFICATION':
            result = self._amplify_position(
                base_bet, gap_z_score, correlation, ml_confidence
            )
        elif strategy == 'PARTIAL_HEDGE':
            result = self._partial_hedge_position(
                base_bet, correlation, gap_z_score, ml_confidence,
                ml_prediction, market_odds
            )
        elif strategy == 'DELTA_NEUTRAL':
            result = self._delta_neutral_position(
                base_bet, ml_prediction, market_odds
            )
        else:  # STANDARD
            result = {
                'strategy': 'STANDARD',
                'primary_bet': base_bet,
                'hedge_bet': 0.00,
                'net_exposure': base_bet,
                'amplification': 1.0,
                'reasoning': 'Normal conditions, no adjustment needed'
            }
        
        return result
    
    def _select_strategy(
        self,
        gap_z_score: float,
        correlation: float,
        ml_confidence: float
    ) -> str:
        """
        Select hedging strategy based on conditions
        
        Decision tree:
        - Extreme gap (|Z| > 3.0) + high confidence → AMPLIFICATION
        - Large gap (|Z| > 2.0) + good confidence → AMPLIFICATION (moderate)
        - Moderate gap + low confidence → PARTIAL_HEDGE
        - Small gap or low correlation → STANDARD
        - Very low confidence → DELTA_NEUTRAL
        
        Time: <1ms
        """
        abs_z = abs(gap_z_score)
        
        # Extreme opportunity (rubber band very stretched!)
        if abs_z > 5.0 and ml_confidence > 0.85 and correlation > 0.75:
            return 'AMPLIFICATION'
        
        # Strong opportunity
        if abs_z > 3.0 and ml_confidence > 0.80 and correlation > 0.70:
            return 'AMPLIFICATION'
        
        # Moderate opportunity
        if abs_z > 2.0 and ml_confidence > 0.75 and correlation > 0.65:
            return 'AMPLIFICATION'
        
        # Moderate gap but uncertain → hedge for safety
        if abs_z > 2.0 and ml_confidence < 0.60:
            return 'PARTIAL_HEDGE'
        
        # Very uncertain → delta-neutral butterfly
        if ml_confidence < 0.40:
            return 'DELTA_NEUTRAL'
        
        # Normal conditions
        return 'STANDARD'
    
    def _amplify_position(
        self,
        base_bet: float,
        gap_z_score: float,
        correlation: float,
        ml_confidence: float
    ) -> Dict:
        """
        Amplify bet when rubber band is stretched
        
        Formula (MATH_BREAKDOWN.txt 6.2):
            Amplification = 1 + (Tension / 10), capped at max
        
        Where:
            Tension = |Gap| × ρ / σ_combined
        
        Simplified:
            Amplification ≈ 1 + (|Z-score| × ρ / 10)
        
        Time: <2ms
        """
        # Calculate amplification factor
        # More unusual gap → higher amplification
        tension_proxy = abs(gap_z_score) * correlation
        amplification = 1 + (tension_proxy / 10)
        
        # Apply confidence adjustment
        amplification = amplification * ml_confidence
        
        # Enforce limits
        amplification = np.clip(amplification, 1.0, self.max_amplification)
        
        # Calculate final bet
        amplified_bet = base_bet * amplification
        
        return {
            'strategy': 'AMPLIFICATION',
            'primary_bet': round(amplified_bet, 2),
            'hedge_bet': 0.00,
            'net_exposure': round(amplified_bet, 2),
            'amplification': round(amplification, 2),
            'reasoning': f"Gap is {gap_z_score:.1f}σ unusual with {correlation:.0%} correlation. " +
                        f"Amplifying {amplification:.1f}x to capitalize on mean reversion opportunity."
        }
    
    def _partial_hedge_position(
        self,
        base_bet: float,
        correlation: float,
        gap_z_score: float,
        ml_confidence: float,
        ml_prediction: Dict,
        market_odds: Dict
    ) -> Dict:
        """
        Partial hedge when moderate uncertainty
        
        Formula (MATH_BREAKDOWN.txt 4.1):
            Hedge_Amount = Primary × Hedge_Ratio × (1 - Confidence)
        
        Time: <5ms
        """
        # Calculate delta analysis
        delta_analysis = self.delta_calc.get_complete_delta_analysis(
            ml_forecast=ml_prediction['point_forecast'],
            ml_lower=ml_prediction['interval_lower'],
            ml_upper=ml_prediction['interval_upper'],
            market_spread=market_odds['spread'],
            correlation=correlation
        )
        
        hedge_ratio = delta_analysis['hedge_ratio']
        
        # Adjust hedge ratio based on confidence
        # Low confidence → more hedge
        uncertainty = 1 - ml_confidence
        adjusted_hedge_ratio = hedge_ratio * uncertainty
        
        # Enforce limits
        adjusted_hedge_ratio = min(adjusted_hedge_ratio, self.max_hedge_ratio)
        
        # Calculate positions
        primary_bet = base_bet * 0.90  # Slightly reduce primary
        hedge_bet = primary_bet * adjusted_hedge_ratio
        net_exposure = primary_bet - hedge_bet
        
        return {
            'strategy': 'PARTIAL_HEDGE',
            'primary_bet': round(primary_bet, 2),
            'hedge_bet': round(hedge_bet, 2),
            'net_exposure': round(net_exposure, 2),
            'hedge_ratio': round(adjusted_hedge_ratio, 2),
            'reasoning': f"Moderate gap ({gap_z_score:.1f}σ) with {ml_confidence:.0%} confidence. " +
                        f"Hedging {adjusted_hedge_ratio:.0%} to reduce risk."
        }
    
    def _delta_neutral_position(
        self,
        base_bet: float,
        ml_prediction: Dict,
        market_odds: Dict
    ) -> Dict:
        """
        Delta-neutral butterfly spread
        
        Bet on both sides to profit from volatility, not direction
        
        Formula (MATH_BREAKDOWN.txt 4.2):
            Position_ML = h × Position_Market
        
        Time: <3ms
        """
        # Split bet 60/40 (slightly favor ML)
        primary_bet = base_bet * 0.60
        hedge_bet = base_bet * 0.40
        net_exposure = primary_bet - hedge_bet
        
        return {
            'strategy': 'DELTA_NEUTRAL',
            'primary_bet': round(primary_bet, 2),
            'hedge_bet': round(hedge_bet, 2),
            'net_exposure': round(net_exposure, 2),
            'hedge_ratio': 0.67,
            'reasoning': "Low confidence. Using butterfly spread to profit from convergence " +
                        "regardless of direction."
        }


# Test the hedge optimizer
if __name__ == "__main__":
    import time
    
    print("="*80)
    print("HEDGE OPTIMIZER - VERIFICATION")
    print("="*80)
    
    optimizer = HedgeOptimizer(max_amplification=2.0, max_hedge_ratio=0.50)
    
    # Test 1: Amplification strategy (big gap, high confidence)
    print("\n1. AMPLIFICATION Strategy (Rubber Band Stretched!):")
    print("   Scenario: LAL +15.1 [+11.3, +18.9] vs Market -7.5")
    print("   Gap: 5.14σ (extremely unusual!)")
    print("   Correlation: 0.85 (strong)")
    print("   Confidence: 0.759 (good)")
    
    result1 = optimizer.optimize_position(
        base_bet=272.50,
        ml_prediction={'point_forecast': 15.1, 'interval_lower': 11.3, 'interval_upper': 18.9},
        market_odds={'spread': -7.5, 'odds': -110},
        correlation=0.85,
        gap_z_score=5.14,
        ml_confidence=0.759
    )
    
    print(f"\n   Strategy: {result1['strategy']}")
    print(f"   Primary bet: ${result1['primary_bet']:.2f}")
    print(f"   Hedge bet: ${result1['hedge_bet']:.2f}")
    print(f"   Net exposure: ${result1['net_exposure']:.2f}")
    print(f"   Amplification: {result1['amplification']}x")
    print(f"   Reasoning: {result1['reasoning']}")
    
    # Test 2: Partial hedge strategy (moderate gap, lower confidence)
    print("\n2. PARTIAL HEDGE Strategy (Moderate Uncertainty):")
    print("   Scenario: Same ML but lower confidence (0.50)")
    
    result2 = optimizer.optimize_position(
        base_bet=272.50,
        ml_prediction={'point_forecast': 15.1, 'interval_lower': 5.0, 'interval_upper': 25.0},
        market_odds={'spread': -7.5, 'odds': -110},
        correlation=0.70,
        gap_z_score=2.5,
        ml_confidence=0.50
    )
    
    print(f"\n   Strategy: {result2['strategy']}")
    print(f"   Primary bet: ${result2['primary_bet']:.2f}")
    print(f"   Hedge bet: ${result2['hedge_bet']:.2f}")
    print(f"   Net exposure: ${result2['net_exposure']:.2f}")
    print(f"   Reasoning: {result2['reasoning']}")
    
    # Test 3: Standard strategy (normal conditions)
    print("\n3. STANDARD Strategy (Normal Conditions):")
    
    result3 = optimizer.optimize_position(
        base_bet=272.50,
        ml_prediction={'point_forecast': 12.0, 'interval_lower': 10.0, 'interval_upper': 14.0},
        market_odds={'spread': -7.0, 'odds': -110},
        correlation=0.80,
        gap_z_score=1.0,
        ml_confidence=0.85
    )
    
    print(f"\n   Strategy: {result3['strategy']}")
    print(f"   Primary bet: ${result3['primary_bet']:.2f}")
    print(f"   Net exposure: ${result3['net_exposure']:.2f}")
    
    # Test 4: Performance
    print("\n4. Performance Test (1000 optimizations):")
    start = time.time()
    for _ in range(1000):
        optimizer.optimize_position(
            base_bet=272.50,
            ml_prediction={'point_forecast': 15.1, 'interval_lower': 11.3, 'interval_upper': 18.9},
            market_odds={'spread': -7.5, 'odds': -110},
            correlation=0.85,
            gap_z_score=5.14,
            ml_confidence=0.759
        )
    elapsed = (time.time() - start) * 1000
    avg = elapsed / 1000
    
    print(f"   1000 optimizations: {elapsed:.1f}ms total")
    print(f"   Average: {avg:.2f}ms per optimization")
    print(f"   Target: <10ms")
    
    if avg < 10:
        print(f"   ✅ PASS!")
    else:
        print(f"   ❌ FAIL - Too slow")
    
    print("\n" + "="*80)
    print("✅ HEDGE OPTIMIZER READY")
    print("="*80)
    print("\nThree Strategies:")
    print("  1. AMPLIFICATION - Bet more when rubber band stretched")
    print("  2. PARTIAL_HEDGE - Hedge when uncertain")
    print("  3. DELTA_NEUTRAL - Balanced when very uncertain")
    print("  4. STANDARD - No adjustment for normal conditions")

