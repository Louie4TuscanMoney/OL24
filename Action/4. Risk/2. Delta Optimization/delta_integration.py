"""
Delta Optimization - Complete Integration
Combines Correlation + Gap Analysis + Delta + Hedging

Usage: Takes Kelly-optimal bet → Returns correlation-adjusted position
Performance: <15ms total
"""

import time
from typing import Dict
from correlation_tracker import CorrelationTracker
from delta_calculator import DeltaCalculator
from hedge_optimizer import HedgeOptimizer


class DeltaOptimization:
    """
    Complete Delta Optimization System
    
    Flow:
    1. Track correlation between ML and market
    2. Calculate gap z-score (rubber band stretch)
    3. Analyze delta (sensitivity)
    4. Optimize position (amplify or hedge)
    
    Performance: <15ms total
    """
    
    def __init__(self):
        """Initialize all components"""
        self.correlation_tracker = CorrelationTracker(window_size=50)
        self.delta_calculator = DeltaCalculator()
        self.hedge_optimizer = HedgeOptimizer(max_amplification=2.0)
        
        print("\n" + "="*80)
        print("DELTA OPTIMIZATION SYSTEM INITIALIZED")
        print("="*80)
        print("Components:")
        print("  ✅ Correlation Tracker (rubber band monitoring)")
        print("  ✅ Delta Calculator (sensitivity analysis)")
        print("  ✅ Hedge Optimizer (position management)")
    
    def update_history(self, ml_forecast: float, market_spread: float):
        """
        Add historical observation
        
        Call this after each game to build correlation history
        
        Args:
            ml_forecast: ML halftime prediction
            market_spread: Market full-game spread
        """
        self.correlation_tracker.update(ml_forecast, market_spread)
    
    def optimize_bet(
        self,
        base_bet: float,
        ml_prediction: Dict,
        market_odds: Dict,
        ml_confidence: float
    ) -> Dict:
        """
        Optimize bet using correlation-based delta analysis
        
        INPUT from Kelly Criterion:
            base_bet = $272.50 (Kelly-optimal)
            ml_prediction = {
                'point_forecast': 15.1,
                'interval_lower': 11.3,
                'interval_upper': 18.9
            }
            market_odds = {
                'spread': -7.5,
                'odds': -110
            }
            ml_confidence = 0.759
        
        PROCESS:
            1. Get correlation (ρ)
            2. Calculate gap z-score
            3. Analyze delta
            4. Optimize position
        
        OUTPUT:
            {
                'strategy': 'AMPLIFICATION',
                'primary_bet': 354.00,
                'hedge_bet': 0.00,
                'net_exposure': 354.00,
                'amplification': 1.30,
                'correlation': 0.85,
                'gap_z_score': 5.14,
                'tension': 3.39,
                'reasoning': '...',
                'performance_ms': 12
            }
        
        Time: <15ms
        """
        start = time.time()
        
        # Step 1: Get correlation analysis
        correlation = self.correlation_tracker.get_correlation()
        gap_stats = self.correlation_tracker.get_gap_statistics()
        tension = self.correlation_tracker.get_tension_metric()
        
        # Step 2: Calculate delta
        delta_analysis = self.delta_calculator.get_complete_delta_analysis(
            ml_forecast=ml_prediction['point_forecast'],
            ml_lower=ml_prediction['interval_lower'],
            ml_upper=ml_prediction['interval_upper'],
            market_spread=market_odds['spread'],
            correlation=correlation
        )
        
        # Step 3: Optimize position
        gap_z_score = gap_stats.get('z_score', 0.0) if 'z_score' in gap_stats else 0.0
        
        position = self.hedge_optimizer.optimize_position(
            base_bet=base_bet,
            ml_prediction=ml_prediction,
            market_odds=market_odds,
            correlation=correlation,
            gap_z_score=gap_z_score,
            ml_confidence=ml_confidence
        )
        
        # Add analytics
        position['correlation'] = correlation
        position['gap_z_score'] = gap_z_score
        position['tension'] = tension
        position['delta_analysis'] = delta_analysis
        position['gap_stats'] = gap_stats
        
        # Performance tracking
        elapsed = (time.time() - start) * 1000
        position['performance_ms'] = round(elapsed, 2)
        
        return position
    
    def get_system_status(self) -> Dict:
        """
        Get current system status
        
        Returns correlation health, data quality, readiness
        """
        analysis = self.correlation_tracker.get_complete_analysis()
        
        return {
            'correlation': analysis['correlation'],
            'sample_size': analysis['sample_size'],
            'ready': analysis['sample_size'] >= 20,
            'momentum': analysis['momentum'],
            'gap_stats': analysis['gap_stats'],
            'tension': analysis['tension']
        }


# Complete integration test
if __name__ == "__main__":
    print("="*80)
    print("DELTA OPTIMIZATION - COMPLETE INTEGRATION TEST")
    print("="*80)
    
    # Initialize system
    delta_system = DeltaOptimization()
    
    # Step 1: Build historical data
    print("\n1. Building Historical Correlation Data...")
    import numpy as np
    np.random.seed(42)
    
    for i in range(50):
        # Simulate correlated ML and market
        base = np.random.normal(12, 2)
        ml_forecast = base + np.random.normal(0, 0.5)
        market_spread = -(base * 0.55) + np.random.normal(0, 0.3)
        
        delta_system.update_history(ml_forecast, market_spread)
    
    print(f"   ✅ Added 50 historical games")
    
    # Step 2: Check system status
    print("\n2. System Status:")
    status = delta_system.get_system_status()
    print(f"   Correlation: {status['correlation']:.3f}")
    print(f"   Sample size: {status['sample_size']}")
    print(f"   Ready: {status['ready']}")
    print(f"   Tension: {status['tension']:.2f}")
    
    # Step 3: Optimize bet (HUGE gap scenario!)
    print("\n3. Optimizing Bet (HUGE GAP - Rubber Band Stretched!):")
    print("   Input from Kelly Criterion: $272.50")
    print("   ML: LAL +15.1 [+11.3, +18.9]")
    print("   Market: LAL -7.5 @ -110")
    print("   Confidence: 0.759")
    
    # Add this unusual game to history first
    delta_system.update_history(ml_forecast=20.0, market_spread=-8.0)
    
    # Optimize
    result = delta_system.optimize_bet(
        base_bet=272.50,
        ml_prediction={
            'point_forecast': 20.0,
            'interval_lower': 17.0,
            'interval_upper': 23.0
        },
        market_odds={
            'spread': -8.0,
            'odds': -110
        },
        ml_confidence=0.85
    )
    
    print(f"\n   DELTA OPTIMIZATION RESULT:")
    print(f"   ─" * 40)
    print(f"   Strategy: {result['strategy']}")
    print(f"   Primary bet: ${result['primary_bet']:.2f}")
    print(f"   Hedge bet: ${result['hedge_bet']:.2f}")
    print(f"   Net exposure: ${result['net_exposure']:.2f}")
    
    if 'amplification' in result:
        print(f"   Amplification: {result['amplification']}x")
    
    print(f"\n   ANALYTICS:")
    print(f"   ─" * 40)
    print(f"   Correlation: {result['correlation']:.3f}")
    print(f"   Gap Z-score: {result['gap_z_score']:.2f}σ")
    print(f"   Tension: {result['tension']:.2f}")
    print(f"   Performance: {result['performance_ms']:.2f}ms")
    
    print(f"\n   REASONING:")
    print(f"   {result['reasoning']}")
    
    # Step 4: Compare to Kelly-only
    print("\n4. Comparison:")
    print(f"   Kelly-only bet: $272.50")
    print(f"   Delta-optimized: ${result['primary_bet']:.2f}")
    
    increase = ((result['primary_bet'] / 272.50) - 1) * 100
    print(f"   Change: {increase:+.1f}%")
    
    if result['strategy'] == 'AMPLIFICATION':
        print(f"   ✅ Amplified due to extreme gap + high correlation!")
    
    # Step 5: Performance summary
    print("\n5. Performance Summary:")
    if result['performance_ms'] < 15:
        print(f"   ✅ {result['performance_ms']:.2f}ms < 15ms target")
    else:
        print(f"   ❌ {result['performance_ms']:.2f}ms > 15ms target")
    
    print("\n" + "="*80)
    print("✅ DELTA OPTIMIZATION COMPLETE")
    print("="*80)
    print("\nThe Rubber Band System is ready!")
    print("  When ML and market diverge → rubber band stretches")
    print("  High correlation + large gap → amplify bet")
    print("  Low confidence + gap → partial hedge")
    print("  System optimizes every bet for maximum edge")

