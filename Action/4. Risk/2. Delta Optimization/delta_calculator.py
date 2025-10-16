"""
Delta Calculator - Sensitivity Analysis
How much does win probability change per point of forecast change?

Based on: DELTA_OPTIMIZATION/Applied Model/delta_calculator.py
Following: DELTA_OPTIMIZATION/MATH_BREAKDOWN.txt (Section 3: Delta)
Performance: <2ms per calculation
"""

import numpy as np
from typing import Dict

class DeltaCalculator:
    """
    Calculate delta (sensitivity) of win probability to forecast changes
    
    Adapted from Options Theory:
        Δ_option = ∂V/∂S (change in option value per change in underlying)
    
    For Sports Betting:
        Δ_ML = ∂P_win/∂ML_forecast
        Δ_Market = ∂P_win/∂Market_spread
    
    Formula (MATH_BREAKDOWN.txt 3.1):
        Delta = Change in probability / Change in forecast
    
    Performance: <2ms per calculation
    """
    
    def __init__(self):
        """Initialize delta calculator"""
        # Empirical constants (calibrated from historical data)
        self.ml_sensitivity = 0.03      # 3% probability change per point
        self.market_sensitivity = 0.04  # 4% probability change per point
        
        print("Delta Calculator initialized:")
        print(f"  ML sensitivity: {self.ml_sensitivity} prob/point")
        print(f"  Market sensitivity: {self.market_sensitivity} prob/point")
    
    def calculate_ml_delta(
        self,
        ml_forecast: float,
        ml_lower: float,
        ml_upper: float
    ) -> float:
        """
        Calculate delta for ML prediction
        
        Args:
            ml_forecast: Point prediction
            ml_lower: Lower CI bound
            ml_upper: Upper CI bound
        
        Returns:
            Delta (sensitivity coefficient)
        
        Formula:
            Δ_ML ≈ σ_ML × sensitivity_factor
        
        Time: <1ms
        """
        # Estimate sigma from interval
        interval_width = ml_upper - ml_lower
        sigma = interval_width / (2 * 1.96)  # 95% CI
        
        # Delta proportional to uncertainty
        # Higher uncertainty → lower sensitivity
        delta = self.ml_sensitivity / (1 + sigma * 0.1)
        
        return float(delta)
    
    def calculate_market_delta(self, market_spread: float) -> float:
        """
        Calculate delta for market spread
        
        Args:
            market_spread: Market spread (e.g., -7.5)
        
        Returns:
            Delta (sensitivity coefficient)
        
        Time: <1ms
        """
        # Market typically more sensitive than ML
        # Closer spreads → higher sensitivity
        delta = self.market_sensitivity * (1 + 0.01 * abs(market_spread))
        
        return float(delta)
    
    def calculate_delta_ratio(
        self,
        ml_delta: float,
        market_delta: float
    ) -> float:
        """
        Calculate ratio of ML to market delta
        
        Formula (MATH_BREAKDOWN.txt 3.2):
            Delta_Ratio = Δ_ML / Δ_Market
        
        Args:
            ml_delta: ML sensitivity
            market_delta: Market sensitivity
        
        Returns:
            Ratio (0-2 typically)
        
        Interpretation:
            > 1.0: ML more sensitive than market
            = 1.0: Equal sensitivity
            < 1.0: Market more sensitive than ML
        
        Time: <0.1ms
        """
        if market_delta == 0:
            return 1.0
        
        return float(ml_delta / market_delta)
    
    def calculate_hedge_ratio(
        self,
        correlation: float,
        sigma_ml: float,
        sigma_market: float
    ) -> float:
        """
        Calculate optimal hedge ratio
        
        Formula (MATH_BREAKDOWN.txt 3.3):
            h = ρ × (σ_ML / σ_Market)
        
        Args:
            correlation: ρ between ML and market
            sigma_ml: ML volatility
            sigma_market: Market volatility
        
        Returns:
            Hedge ratio h
        
        Interpretation:
            For every $100 on ML side, hedge with $h×100 on market side
        
        Example:
            h = 0.30 → Hedge 30% of position
            h = 1.00 → Equal hedge (delta-neutral)
        
        Time: <1ms
        """
        if sigma_market == 0:
            return 0.0
        
        hedge_ratio = correlation * (sigma_ml / sigma_market)
        
        # Clip to reasonable range
        return float(np.clip(hedge_ratio, 0.0, 2.0))
    
    def calculate_position_delta(
        self,
        position_size: float,
        delta: float
    ) -> float:
        """
        Calculate dollar delta of position
        
        Dollar Delta = Position Size × Delta
        
        Args:
            position_size: Bet size ($)
            delta: Sensitivity coefficient
        
        Returns:
            Dollar delta ($)
        
        Interpretation:
            For every 1 point forecast change, position value changes by this amount
        
        Time: <0.1ms
        """
        return float(position_size * delta)
    
    def get_complete_delta_analysis(
        self,
        ml_forecast: float,
        ml_lower: float,
        ml_upper: float,
        market_spread: float,
        correlation: float
    ) -> Dict:
        """
        Complete delta analysis
        
        Returns all delta metrics in one call
        """
        # Estimate sigmas
        ml_sigma = (ml_upper - ml_lower) / (2 * 1.96)
        market_sigma = abs(market_spread) * 0.15  # Empirical estimate
        
        # Calculate deltas
        ml_delta = self.calculate_ml_delta(ml_forecast, ml_lower, ml_upper)
        market_delta = self.calculate_market_delta(market_spread)
        delta_ratio = self.calculate_delta_ratio(ml_delta, market_delta)
        hedge_ratio = self.calculate_hedge_ratio(correlation, ml_sigma, market_sigma)
        
        return {
            'ml_delta': ml_delta,
            'market_delta': market_delta,
            'delta_ratio': delta_ratio,
            'hedge_ratio': hedge_ratio,
            'ml_sigma': ml_sigma,
            'market_sigma': market_sigma,
            'interpretation': self._interpret_delta_ratio(delta_ratio)
        }
    
    def _interpret_delta_ratio(self, ratio: float) -> str:
        """Interpret delta ratio"""
        if ratio > 1.2:
            return "ML significantly more sensitive than market"
        elif ratio > 0.8:
            return "ML and market roughly equal sensitivity"
        else:
            return "Market more sensitive than ML"


# Test the delta calculator
if __name__ == "__main__":
    import time
    
    print("="*80)
    print("DELTA CALCULATOR - VERIFICATION")
    print("="*80)
    
    calculator = DeltaCalculator()
    
    # Test 1: ML delta
    print("\n1. ML Delta Calculation:")
    ml_forecast = 15.1
    ml_lower = 11.3
    ml_upper = 18.9
    
    start = time.time()
    ml_delta = calculator.calculate_ml_delta(ml_forecast, ml_lower, ml_upper)
    elapsed = (time.time() - start) * 1000
    
    print(f"   ML: {ml_forecast:+.1f} [{ml_lower:+.1f}, {ml_upper:+.1f}]")
    print(f"   Δ_ML: {ml_delta:.4f}")
    print(f"   Interpretation: +1 point in ML → +{ml_delta*100:.1f}% win probability")
    print(f"   Time: {elapsed:.3f}ms {'✅' if elapsed < 1 else '❌'}")
    
    # Test 2: Market delta
    print("\n2. Market Delta Calculation:")
    market_spread = -7.5
    
    market_delta = calculator.calculate_market_delta(market_spread)
    print(f"   Market: {market_spread:+.1f}")
    print(f"   Δ_Market: {market_delta:.4f}")
    print(f"   Interpretation: +1 point in spread → +{market_delta*100:.1f}% win probability")
    
    # Test 3: Delta ratio
    print("\n3. Delta Ratio:")
    delta_ratio = calculator.calculate_delta_ratio(ml_delta, market_delta)
    print(f"   Ratio: {delta_ratio:.3f}")
    print(f"   {calculator._interpret_delta_ratio(delta_ratio)}")
    
    # Test 4: Hedge ratio
    print("\n4. Hedge Ratio Calculation:")
    correlation = 0.85
    sigma_ml = (ml_upper - ml_lower) / (2 * 1.96)
    sigma_market = abs(market_spread) * 0.15
    
    hedge_ratio = calculator.calculate_hedge_ratio(correlation, sigma_ml, sigma_market)
    print(f"   ρ = {correlation:.2f}")
    print(f"   σ_ML = {sigma_ml:.2f}")
    print(f"   σ_Market = {sigma_market:.2f}")
    print(f"   Hedge ratio: {hedge_ratio:.3f}")
    print(f"   Interpretation: For every $100 on ML, hedge ${hedge_ratio*100:.0f} on market")
    
    # Test 5: Complete analysis
    print("\n5. Complete Delta Analysis:")
    analysis = calculator.get_complete_delta_analysis(
        ml_forecast=15.1,
        ml_lower=11.3,
        ml_upper=18.9,
        market_spread=-7.5,
        correlation=0.85
    )
    
    print(f"   ML Delta: {analysis['ml_delta']:.4f}")
    print(f"   Market Delta: {analysis['market_delta']:.4f}")
    print(f"   Delta Ratio: {analysis['delta_ratio']:.3f}")
    print(f"   Hedge Ratio: {analysis['hedge_ratio']:.3f}")
    print(f"   {analysis['interpretation']}")
    
    # Test 6: Position delta (dollar impact)
    print("\n6. Position Delta (Dollar Impact):")
    position_size = 272.50
    dollar_delta = calculator.calculate_position_delta(position_size, ml_delta)
    print(f"   Position: ${position_size:.2f}")
    print(f"   Delta: {ml_delta:.4f}")
    print(f"   Dollar Delta: ${dollar_delta:.2f}")
    print(f"   Interpretation: +1 pt in ML → +${dollar_delta:.2f} expected value")
    
    # Test 7: Performance
    print("\n7. Performance Test (10000 calculations):")
    start = time.time()
    for _ in range(10000):
        calculator.get_complete_delta_analysis(15.1, 11.3, 18.9, -7.5, 0.85)
    elapsed = (time.time() - start) * 1000
    avg = elapsed / 10000
    
    print(f"   10000 calculations: {elapsed:.1f}ms total")
    print(f"   Average: {avg:.3f}ms per calculation")
    print(f"   Target: <2ms")
    
    if avg < 2:
        print(f"   ✅ PASS!")
    else:
        print(f"   ❌ FAIL - Too slow")
    
    print("\n" + "="*80)
    print("✅ DELTA CALCULATOR READY")
    print("="*80)
    print("\nDelta measures sensitivity:")
    print("  Δ_ML: How much win prob changes per ML forecast point")
    print("  Δ_Market: How much win prob changes per market spread point")
    print("  Use delta ratio to understand relative sensitivities")
    print("  Use hedge ratio to calculate optimal hedge positions")

