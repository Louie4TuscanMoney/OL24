"""
Final Calibrator - Complete Safety System
The Responsible Adult Layer

Integrates: Absolute Limiter + Safety Modes + Confidence Scaling + Portfolio Validation
Performance: <10ms per calibration
"""

import time
from typing import List, Dict
from absolute_limiter import AbsoluteLimiter
from safety_mode_manager import SafetyModeManager, SafetyMode


class FinalCalibrator:
    """
    Final safety layer before trade execution
    
    Has VETO POWER over all previous layers
    
    Responsibilities:
    1. Enforce absolute maximum ($750 per bet)
    2. Apply safety mode limits (GREEN/YELLOW/RED)
    3. Scale by confidence within limits
    4. Enforce portfolio limits ($2,500 total)
    5. Validate reserve requirements ($2,500 held)
    
    The Responsible Adult: "I appreciate your enthusiasm, but here's what's safe."
    
    Performance: <10ms
    """
    
    def __init__(self, original_bankroll: float = 5000):
        """
        Initialize final calibrator
        
        Args:
            original_bankroll: Starting bankroll ($5,000)
        """
        self.original_bankroll = original_bankroll
        self.absolute_limiter = AbsoluteLimiter(original_bankroll)
        self.safety_mode_manager = SafetyModeManager(original_bankroll)
        
        print("\n" + "="*80)
        print("FINAL CALIBRATOR INITIALIZED")
        print("="*80)
        print("THE RESPONSIBLE ADULT LAYER")
        print(f"  Original bankroll: ${original_bankroll:,.0f}")
        print(f"  Absolute max: ${original_bankroll * 0.15:.0f} per bet")
        print(f"  Portfolio max: ${original_bankroll * 0.50:.0f} total")
        print(f"  Reserve required: ${original_bankroll * 0.50:.0f} always")
    
    def calibrate_single_bet(
        self,
        recommended_bet: float,
        ml_confidence: float,
        edge: float,
        calibration_status: str,
        current_bankroll: float,
        recent_win_rate: float,
        current_drawdown: float
    ) -> Dict:
        """
        Calibrate single bet with all safety checks
        
        INPUT from Decision Tree:
            recommended_bet: $1,750 (from TURBO mode)
            ml_confidence: 0.92
            edge: 0.226
            calibration_status: 'EXCELLENT'
            current_bankroll: $5,000
            recent_win_rate: 0.62
            current_drawdown: 0.05
        
        PROCESS:
            1. Apply absolute limit ($750 max)
            2. Check safety mode (GREEN/YELLOW/RED)
            3. Apply confidence scaling
            4. Final cap check
        
        OUTPUT:
            {
                'final_bet': 750.00,
                'original_recommended': 1750.00,
                'absolute_max_applied': True,
                'safety_mode': 'GREEN',
                'reduction_total_pct': 0.571,
                'reasoning': [...]
            }
        
        Time: <5ms
        """
        start = time.time()
        reasoning = []
        
        # Step 1: Apply absolute limit
        limit_result = self.absolute_limiter.apply_limit(recommended_bet)
        bet = limit_result['final_bet']
        
        if limit_result['was_capped']:
            reasoning.append(
                f"Absolute limit: ${recommended_bet:.0f} → ${bet:.0f} " +
                f"({limit_result['reduction_pct']:.0%} reduction)"
            )
        
        # Step 2: Determine safety mode
        mode_result = self.safety_mode_manager.determine_mode(
            current_bankroll=current_bankroll,
            calibration_status=calibration_status,
            recent_win_rate=recent_win_rate,
            current_drawdown=current_drawdown
        )
        
        # Apply mode limit
        mode_max = mode_result['max_bet']
        if bet > mode_max:
            old_bet = bet
            bet = mode_max
            reasoning.append(
                f"Safety mode {mode_result['mode_emoji']} {mode_result['mode_name']}: " +
                f"${old_bet:.0f} → ${bet:.0f}"
            )
        
        # Step 3: Apply confidence scaling (within limits)
        confidence_factor = self._calculate_confidence_factor(ml_confidence)
        edge_factor = self._calculate_edge_factor(edge)
        calib_factor = self._get_calibration_factor(calibration_status)
        health_factor = min(1.0, current_bankroll / self.original_bankroll)
        
        combined_scaling = confidence_factor * edge_factor * calib_factor * health_factor
        
        bet_scaled = bet * combined_scaling
        
        # Can't exceed mode max even after scaling
        if bet_scaled > bet:
            bet_scaled = bet
        
        if bet_scaled < bet:
            reasoning.append(
                f"Confidence scaling: {combined_scaling:.3f}× " +
                f"({confidence_factor:.2f} conf × {edge_factor:.2f} edge × " +
                f"{calib_factor:.2f} calib × {health_factor:.2f} health)"
            )
        
        bet_final = bet_scaled
        
        # Step 4: Round to nearest $10
        bet_final = round(bet_final / 10) * 10
        
        if reasoning:
            reasoning.append(f"Rounded to ${bet_final:.0f}")
        
        # Calculate total reduction
        total_reduction_pct = (recommended_bet - bet_final) / recommended_bet if recommended_bet > 0 else 0
        
        elapsed = (time.time() - start) * 1000
        
        return {
            'final_bet': bet_final,
            'original_recommended': round(recommended_bet, 2),
            'absolute_maximum': self.absolute_limiter.absolute_max,
            'mode_maximum': mode_max,
            'absolute_max_applied': limit_result['was_capped'],
            'safety_mode': mode_result['mode_name'],
            'mode_emoji': mode_result['mode_emoji'],
            'confidence_scaling_applied': bet_scaled < bet,
            'combined_scaling': round(combined_scaling, 3),
            'scaling_factors': {
                'confidence': confidence_factor,
                'edge': edge_factor,
                'calibration': calib_factor,
                'health': health_factor
            },
            'reduction_total_pct': round(total_reduction_pct, 3),
            'reduction_total_amount': round(recommended_bet - bet_final, 2),
            'reasoning': reasoning,
            'protection_level': 'MAXIMUM',
            'performance_ms': round(elapsed, 2)
        }
    
    def calibrate_portfolio(
        self,
        individual_bets: List[Dict],
        current_bankroll: float,
        mode_result: Dict
    ) -> Dict:
        """
        Calibrate entire portfolio
        
        Ensures:
        1. Total ≤ 50% of original ($2,500)
        2. Reserve ≥ 50% of original ($2,500)
        3. All bets within individual limits
        
        Time: <5ms
        """
        # Calculate proposed total
        total_proposed = sum(bet['final_bet'] for bet in individual_bets)
        
        # Check portfolio limit
        portfolio_limit = mode_result['max_total']
        
        if total_proposed > portfolio_limit:
            # Scale all proportionally
            scaling_factor = portfolio_limit / total_proposed
            
            for bet in individual_bets:
                old_bet = bet['final_bet']
                bet['final_bet'] = round(bet['final_bet'] * scaling_factor, 2)
                bet['portfolio_scaling_applied'] = True
                bet['portfolio_scaling_factor'] = scaling_factor
                bet['portfolio_reduction'] = old_bet - bet['final_bet']
            
            total_final = portfolio_limit
            portfolio_scaled = True
        else:
            total_final = total_proposed
            portfolio_scaled = False
        
        # Check reserve requirement
        reserve_required = self.original_bankroll * 0.50
        reserve_actual = current_bankroll - total_final
        
        if reserve_actual < reserve_required:
            # Insufficient reserve - reduce further
            max_total_allowed = current_bankroll - reserve_required
            
            if max_total_allowed < 0:
                max_total_allowed = 0  # Can't bet if reserve would be violated
            
            if total_final > 0:
                additional_scaling = max_total_allowed / total_final
                
                for bet in individual_bets:
                    old_bet = bet['final_bet']
                    bet['final_bet'] = round(bet['final_bet'] * additional_scaling, 2)
                    bet['reserve_scaling_applied'] = True
                    bet['reserve_scaling_factor'] = additional_scaling
                
                total_final = max_total_allowed
            
            reserve_scaled = True
        else:
            reserve_scaled = False
        
        return {
            'individual_bets': individual_bets,
            'total_allocation': round(total_final, 2),
            'portfolio_limit': portfolio_limit,
            'portfolio_scaled': portfolio_scaled,
            'reserve_required': reserve_required,
            'reserve_actual': round(current_bankroll - total_final, 2),
            'reserve_scaled': reserve_scaled,
            'reserve_check': 'PASS' if not reserve_scaled else 'SCALED_TO_FIT',
            'safety_level': 'MAXIMUM'
        }
    
    def _calculate_confidence_factor(self, confidence: float) -> float:
        """
        ML confidence scaling
        
        Formula (MATH_BREAKDOWN.txt 2.1):
            f = max(0.60, min(1.05, 0.45 + 0.60 × confidence))
        """
        factor = 0.45 + 0.60 * confidence
        return max(0.60, min(1.05, factor))
    
    def _calculate_edge_factor(self, edge: float) -> float:
        """
        Edge size scaling
        
        Formula (MATH_BREAKDOWN.txt 2.2):
            f = max(0.70, min(1.00, edge / 0.20))
        """
        factor = edge / 0.20
        return max(0.70, min(1.00, factor))
    
    def _get_calibration_factor(self, status: str) -> float:
        """
        Calibration status scaling
        
        Formula (MATH_BREAKDOWN.txt 2.3)
        """
        factors = {
            'EXCELLENT': 1.05,
            'GOOD': 1.00,
            'FAIR': 0.90,
            'POOR': 0.70,
            'VERY_POOR': 0.50
        }
        return factors.get(status, 1.00)


# Test the final calibrator
if __name__ == "__main__":
    print("="*80)
    print("FINAL CALIBRATOR - VERIFICATION")
    print("="*80)
    
    calibrator = FinalCalibrator(original_bankroll=5000)
    
    # Test 1: Monster bet from TURBO mode
    print("\n" + "="*80)
    print("TEST 1: TURBO MODE MONSTER BET")
    print("="*80)
    print("Scenario: All layers aligned for massive bet")
    print("  Decision Tree recommends: $1,750 (TURBO 125%)")
    print("  Edge: 22.6% (huge!)")
    print("  Confidence: 92% (very high)")
    print("  Calibration: EXCELLENT")
    
    result1 = calibrator.calibrate_single_bet(
        recommended_bet=1750.00,
        ml_confidence=0.92,
        edge=0.226,
        calibration_status='EXCELLENT',
        current_bankroll=5000,
        recent_win_rate=0.62,
        current_drawdown=0.05
    )
    
    print(f"\nFINAL CALIBRATION RESULT:")
    print(f"  Original recommended: ${result1['original_recommended']:,.0f}")
    print(f"  Safety mode: {result1['mode_emoji']} {result1['safety_mode']}")
    print(f"  Absolute max: ${result1['absolute_maximum']:.0f}")
    print(f"  Mode max: ${result1['mode_maximum']:.0f}")
    print(f"  Combined scaling: {result1['combined_scaling']:.3f}×")
    print(f"  ")
    print(f"  💰 FINAL BET: ${result1['final_bet']:,.0f}")
    print(f"  Total reduction: {result1['reduction_total_pct']:.0%} (${result1['reduction_total_amount']:.0f})")
    print(f"  Performance: {result1['performance_ms']:.2f}ms")
    
    print(f"\nReasoning:")
    for reason in result1['reasoning']:
        print(f"  • {reason}")
    
    print(f"\n✅ The responsible adult said: ${result1['final_bet']:.0f}. Not ${result1['original_recommended']:.0f}.")
    
    # Test 2: RED mode (defensive)
    print("\n" + "="*80)
    print("TEST 2: RED MODE (Defensive Conditions)")
    print("="*80)
    print("Scenario: Poor performance, big drawdown")
    print("  Decision Tree recommends: $800")
    print("  Bankroll: $2,800 (56% of original)")
    print("  Win rate: 48%")
    print("  Calibration: POOR")
    
    result2 = calibrator.calibrate_single_bet(
        recommended_bet=800.00,
        ml_confidence=0.65,
        edge=0.12,
        calibration_status='POOR',
        current_bankroll=2800,
        recent_win_rate=0.48,
        current_drawdown=0.32
    )
    
    print(f"\n  Safety mode: {result2['mode_emoji']} {result2['safety_mode']}")
    print(f"  Mode max: ${result2['mode_maximum']:.0f}")
    print(f"  💰 FINAL BET: ${result2['final_bet']:,.0f}")
    print(f"  Reduction: {result2['reduction_total_pct']:.0%}")
    print(f"\n  ⚠️  RED MODE - Heavily reduced for safety")
    
    # Test 3: Portfolio calibration (6 games)
    print("\n" + "="*80)
    print("TEST 3: PORTFOLIO CALIBRATION (6 GAMES)")
    print("="*80)
    
    # Simulate 6 individual calibrated bets
    individual_bets = [
        {'game_id': 'Game 1', 'final_bet': 750.00},
        {'game_id': 'Game 2', 'final_bet': 600.00},
        {'game_id': 'Game 3', 'final_bet': 500.00},
        {'game_id': 'Game 4', 'final_bet': 550.00},
        {'game_id': 'Game 5', 'final_bet': 480.00},
        {'game_id': 'Game 6', 'final_bet': 520.00},
    ]
    
    total_proposed = sum(b['final_bet'] for b in individual_bets)
    bet_list = [f"${b['final_bet']:.0f}" for b in individual_bets]
    print(f"Individual bets: {bet_list}")
    print(f"Total proposed: ${total_proposed:,.0f}")
    
    mode_result = calibrator.safety_mode_manager.determine_mode(
        current_bankroll=5000,
        calibration_status='EXCELLENT',
        recent_win_rate=0.62,
        current_drawdown=0.05
    )
    
    portfolio_result = calibrator.calibrate_portfolio(
        individual_bets=individual_bets,
        current_bankroll=5000,
        mode_result=mode_result
    )
    
    print(f"\nPortfolio limit: ${portfolio_result['portfolio_limit']:,.0f}")
    print(f"Exceeds limit: {portfolio_result['portfolio_scaled']}")
    
    if portfolio_result['portfolio_scaled']:
        print(f"Scaling applied: {individual_bets[0].get('portfolio_scaling_factor', 1.0):.3f}×")
        print(f"\nScaled bets:")
        for bet in portfolio_result['individual_bets']:
            print(f"  {bet['game_id']}: ${bet['final_bet']:.0f}")
        print(f"Total final: ${portfolio_result['total_allocation']:,.0f}")
    
    print(f"\nReserve check:")
    print(f"  Required: ${portfolio_result['reserve_required']:,.0f}")
    print(f"  Actual: ${portfolio_result['reserve_actual']:,.0f}")
    print(f"  Status: {portfolio_result['reserve_check']}")
    
    if portfolio_result['reserve_check'] == 'PASS':
        print(f"  ✅ Reserve requirement satisfied")
    
    print("\n" + "="*80)
    print("✅ FINAL CALIBRATOR READY")
    print("="*80)
    print("\nThe Responsible Adult is watching:")
    print("  ✅ No bet exceeds $750")
    print("  ✅ Total never exceeds $2,500")
    print("  ✅ Reserve always maintained at $2,500")
    print("  ✅ Safety modes enforce graduated limits")
    print("  ✅ Confidence scaling within limits")

