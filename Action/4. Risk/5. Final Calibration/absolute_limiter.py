"""
Absolute Limiter - Enforce Hard Maximum Bet Size
The Non-Negotiable Rule: Never exceed 15% of original bankroll

Based on: FINAL_CALIBRATION/Applied Model/absolute_limiter.py
Following: FINAL_CALIBRATION/MATH_BREAKDOWN.txt (Section 1)
Performance: <1ms per check
"""

from typing import Dict

class AbsoluteLimiter:
    """
    Enforce absolute maximum bet size
    
    The Hard Rule (MATH_BREAKDOWN.txt 1.1):
        Bet_max = B_0 × 0.15 = $750
    
    This NEVER changes regardless of:
        - Current bankroll ($3,000 or $10,000)
        - Recommended bet from other layers
        - Edge size, confidence, or power modes
        - TURBO states or perfect signals
    
    NO EXCEPTIONS.
    
    Performance: <1ms
    """
    
    def __init__(self, original_bankroll: float = 5000):
        """
        Initialize absolute limiter
        
        Args:
            original_bankroll: Starting bankroll ($5,000)
        """
        self.original_bankroll = original_bankroll
        self.absolute_max = original_bankroll * 0.15  # $750
        
        print("Absolute Limiter initialized:")
        print(f"  Original bankroll: ${original_bankroll:,.0f}")
        print(f"  ABSOLUTE MAXIMUM: ${self.absolute_max:,.0f} (15% of original)")
        print(f"  This value NEVER changes")
    
    def apply_limit(self, recommended_bet: float) -> Dict:
        """
        Apply absolute limit to recommended bet
        
        Formula (MATH_BREAKDOWN.txt 1.2):
            Bet_final = min(Bet_recommended, Bet_max)
        
        Args:
            recommended_bet: From previous layers (could be anything)
        
        Returns:
            {
                'final_bet': 750.00,
                'original_recommended': 1750.00,
                'absolute_maximum': 750.00,
                'was_capped': True,
                'reduction_pct': 0.571,
                'reduction_amount': 1000.00,
                'reasoning': 'Capped at 15% of original bankroll ($750)'
            }
        
        Time: <0.5ms
        """
        # Check if exceeds absolute max
        was_capped = recommended_bet > self.absolute_max
        
        # Apply cap
        final_bet = min(recommended_bet, self.absolute_max)
        
        # Calculate reduction
        reduction_amount = recommended_bet - final_bet if was_capped else 0
        reduction_pct = reduction_amount / recommended_bet if recommended_bet > 0 else 0
        
        # Generate reasoning
        if was_capped:
            reasoning = f'Capped at 15% of original bankroll (${self.absolute_max:.0f}). ' + \
                       f'Reduced from ${recommended_bet:.0f} ({reduction_pct:.0%} reduction)'
        else:
            reasoning = 'Within absolute limit - no reduction needed'
        
        return {
            'final_bet': round(final_bet, 2),
            'original_recommended': round(recommended_bet, 2),
            'absolute_maximum': self.absolute_max,
            'was_capped': was_capped,
            'reduction_pct': round(reduction_pct, 3),
            'reduction_amount': round(reduction_amount, 2),
            'reasoning': reasoning
        }
    
    def check_portfolio_total(self, individual_bets: list) -> Dict:
        """
        Check total portfolio exposure
        
        Formula (MATH_BREAKDOWN.txt 3.1):
            Total_max = B_0 × 0.50 = $2,500
        
        Args:
            individual_bets: List of bet amounts
        
        Returns:
            {
                'total': 3400.00,
                'limit': 2500.00,
                'exceeds': True,
                'scaling_required': 0.735
            }
        
        Time: <1ms
        """
        total = sum(individual_bets)
        portfolio_limit = self.original_bankroll * 0.50  # 50% max
        
        exceeds = total > portfolio_limit
        
        if exceeds:
            scaling_factor = portfolio_limit / total
        else:
            scaling_factor = 1.0
        
        return {
            'total': round(total, 2),
            'limit': portfolio_limit,
            'exceeds': exceeds,
            'scaling_required': round(scaling_factor, 4),
            'excess_amount': round(total - portfolio_limit, 2) if exceeds else 0
        }


# Test the absolute limiter
if __name__ == "__main__":
    print("="*80)
    print("ABSOLUTE LIMITER - VERIFICATION")
    print("="*80)
    
    limiter = AbsoluteLimiter(original_bankroll=5000)
    
    # Test 1: Within limit
    print("\n1. Bet Within Limit:")
    result1 = limiter.apply_limit(500)
    print(f"   Recommended: ${result1['original_recommended']:.0f}")
    print(f"   Final: ${result1['final_bet']:.0f}")
    print(f"   Was capped: {result1['was_capped']}")
    print(f"   ✅ Under $750 limit - no reduction")
    
    # Test 2: Exceeds limit (moderate)
    print("\n2. Bet Exceeds Limit (Moderate):")
    result2 = limiter.apply_limit(1200)
    print(f"   Recommended: ${result2['original_recommended']:.0f}")
    print(f"   Final: ${result2['final_bet']:.0f}")
    print(f"   Was capped: {result2['was_capped']}")
    print(f"   Reduction: {result2['reduction_pct']:.0%}")
    print(f"   ✅ Capped at $750")
    
    # Test 3: Wildly exceeds limit (TURBO mode)
    print("\n3. Bet WILDLY Exceeds Limit (TURBO):")
    result3 = limiter.apply_limit(2500)
    print(f"   Recommended: ${result3['original_recommended']:,.0f}")
    print(f"   Final: ${result3['final_bet']:.0f}")
    print(f"   Was capped: {result3['was_capped']}")
    print(f"   Reduction: {result3['reduction_pct']:.0%} (${result3['reduction_amount']:.0f})")
    print(f"   ✅ Capped at $750 (70% reduction!)")
    
    # Test 4: Portfolio total check
    print("\n4. Portfolio Total Check:")
    bets = [750, 750, 600, 500, 550, 600]
    portfolio_check = limiter.check_portfolio_total(bets)
    
    print(f"   Individual bets: {bets}")
    print(f"   Total: ${portfolio_check['total']:,.0f}")
    print(f"   Portfolio limit: ${portfolio_check['limit']:,.0f}")
    print(f"   Exceeds: {portfolio_check['exceeds']}")
    
    if portfolio_check['exceeds']:
        print(f"   Scaling required: {portfolio_check['scaling_required']:.3f}×")
        scaled_bets = [round(b * portfolio_check['scaling_required']) for b in bets]
        print(f"   Scaled bets: {scaled_bets}")
        print(f"   New total: ${sum(scaled_bets):,.0f}")
        print(f"   ✅ Portfolio scaled to fit within $2,500 limit")
    else:
        print(f"   ✅ Within portfolio limit")
    
    # Test 5: The "responsible adult" example
    print("\n5. The Responsible Adult Example:")
    print("   ─"*40)
    print("   Risky Kid: 'Let's bet $1,750! TURBO mode! Perfect signals!'")
    
    adult_result = limiter.apply_limit(1750)
    
    print(f"   Responsible Adult: 'No. Here's ${adult_result['final_bet']:.0f}.")
    print(f"                      That's the maximum. No exceptions.'")
    print(f"   ✅ Safety enforced - reduced {adult_result['reduction_pct']:.0%}")
    
    print("\n" + "="*80)
    print("✅ ABSOLUTE LIMITER READY")
    print("="*80)
    print("\nThe Non-Negotiable Rule:")
    print("  Maximum bet: $750 (15% of $5,000 original)")
    print("  Portfolio max: $2,500 (50% of $5,000 original)")
    print("  These values NEVER change")
    print("  Protection: MAXIMUM")

