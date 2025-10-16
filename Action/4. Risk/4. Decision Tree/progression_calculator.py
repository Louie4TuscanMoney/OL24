"""
Progression Calculator - Calculate Recovery Bet Sizes
Kelly-adjusted progressive betting (NOT pure martingale)

Based on: DECISION_TREE/Applied Model/progression_calculator.py
Following: DECISION_TREE/MATH_BREAKDOWN.txt (Section 3: Progressive Bet Sizing)
Performance: <5ms per calculation
"""

import numpy as np
from typing import Dict

class ProgressionCalculator:
    """
    Calculate bet sizes for each progression level
    
    Goal at each level:
        Recover all previous losses + achieve original target profit
    
    Formula (MATH_BREAKDOWN.txt 3.1):
        Bet_required = (Cumulative_loss + Target) / net_odds
    
    Then apply Kelly limits (MATH_BREAKDOWN.txt 3.2):
        Bet_final = min(Bet_required, Kelly_max, Hard_limits)
    
    Performance: <5ms per calculation
    """
    
    def __init__(self, max_level: int = 3):
        """
        Initialize progression calculator
        
        Args:
            max_level: Maximum progression depth (3 default)
        """
        self.max_level = max_level
        
        print("Progression Calculator initialized:")
        print(f"  Max level: {max_level}")
        print(f"  Method: Kelly-adjusted recovery")
    
    def calculate_level_bet(
        self,
        level: int,
        cumulative_loss: float,
        target_profit: float,
        current_bankroll: float,
        kelly_fraction: float,
        odds: float,
        win_probability: float = 0.60
    ) -> Dict:
        """
        Calculate bet for given progression level
        
        Args:
            level: 1, 2, or 3
            cumulative_loss: Total losses so far in sequence
            target_profit: Original target profit from Level 1
            current_bankroll: Current bankroll after losses
            kelly_fraction: Current Kelly fraction for opportunity
            odds: Decimal odds (e.g., 1.909 for -110)
            win_probability: Estimated win prob (default 60%)
        
        Returns:
            {
                'bet_size': 571.00,
                'level': 2,
                'required_win': 519.50,
                'bet_needed_uncapped': 571.00,
                'kelly_limit': 750.00,
                'hard_limit': 1000.00,
                'is_capped': False,
                'p_reach_here': 0.40,
                'p_lose_from_here': 0.16,
                'p_ruin_total': 0.064,
                'reasoning': '...'
            }
        
        Time: <5ms
        """
        # Level 1: Base betting (calculated by portfolio management)
        if level == 1:
            return {
                'bet_size': 0.0,  # Set by portfolio layer
                'level': 1,
                'is_base': True,
                'reasoning': 'Base level - no progression',
                'p_reach_here': 1.0,
                'p_lose_from_here': win_probability ** 1
            }
        
        # Calculate required win amount (MATH_BREAKDOWN.txt 3.1)
        required_win = cumulative_loss + target_profit
        
        # Calculate bet needed to achieve required win
        net_odds = odds - 1  # 1.909 → 0.909
        bet_needed_uncapped = required_win / net_odds if net_odds > 0 else required_win * 2
        
        # Apply Kelly limit
        kelly_max = current_bankroll * kelly_fraction
        
        # Apply hard limits
        hard_limit_single = current_bankroll * 0.20  # 20% max per bet
        hard_limit_progression = current_bankroll * 0.30  # 30% max in progression
        
        # Take minimum (MATH_BREAKDOWN.txt 3.2)
        bet_size = min(
            bet_needed_uncapped,
            kelly_max,
            hard_limit_single,
            hard_limit_progression
        )
        
        # Calculate probabilities (MATH_BREAKDOWN.txt 1.2, 1.3)
        p_loss = 1 - win_probability
        p_reach_here = p_loss ** (level - 1)
        p_lose_from_here = p_loss ** level
        p_ruin_total = p_loss ** self.max_level
        
        # Determine which limit applied
        limiting_factor = 'None'
        if bet_size == kelly_max:
            limiting_factor = 'Kelly'
        elif bet_size == hard_limit_single:
            limiting_factor = 'Hard 20%'
        elif bet_size == hard_limit_progression:
            limiting_factor = 'Progression 30%'
        
        return {
            'bet_size': round(bet_size, 2),
            'level': level,
            'required_win': round(required_win, 2),
            'bet_needed_uncapped': round(bet_needed_uncapped, 2),
            'kelly_limit': round(kelly_max, 2),
            'hard_limit': round(hard_limit_single, 2),
            'is_capped': bet_size < bet_needed_uncapped,
            'limiting_factor': limiting_factor,
            'p_reach_here': round(p_reach_here, 4),
            'p_lose_from_here': round(p_lose_from_here, 4),
            'p_ruin_total': round(p_ruin_total, 4),
            'cumulative_loss': cumulative_loss,
            'reasoning': self._generate_reasoning(level, bet_size, required_win, limiting_factor)
        }
    
    def calculate_complete_sequence(
        self,
        base_bet: float,
        target_profit: float,
        current_bankroll: float,
        kelly_fraction: float,
        odds: float
    ) -> Dict:
        """
        Calculate complete 3-level sequence
        
        Shows what bets would be at each level if reached
        
        Returns:
            {
                'level_1': {...},
                'level_2': {...},
                'level_3': {...},
                'max_cumulative_loss': 2043.50,
                'p_max_loss': 0.064,
                'expected_sequence_length': 1.56
            }
        
        Time: <15ms
        """
        sequence = {}
        cumulative_loss = 0.0
        
        for level in range(1, self.max_level + 1):
            if level == 1:
                sequence[f'level_{level}'] = {
                    'bet_size': base_bet,
                    'level': 1,
                    'is_base': True
                }
                cumulative_loss = base_bet
            else:
                level_calc = self.calculate_level_bet(
                    level=level,
                    cumulative_loss=cumulative_loss - base_bet,  # Don't include current level
                    target_profit=target_profit,
                    current_bankroll=current_bankroll - cumulative_loss,
                    kelly_fraction=kelly_fraction,
                    odds=odds
                )
                sequence[f'level_{level}'] = level_calc
                cumulative_loss += level_calc['bet_size']
        
        # Calculate sequence statistics
        p_loss = 0.40  # Assumed
        expected_length = sum(
            level * (p_loss ** (level - 1) * (1 if level < self.max_level else 1))
            for level in range(1, self.max_level + 1)
        )
        
        return {
            **sequence,
            'max_cumulative_loss': round(cumulative_loss, 2),
            'p_max_loss': round(p_loss ** self.max_level, 4),
            'expected_sequence_length': round(expected_length, 2)
        }
    
    def _generate_reasoning(
        self,
        level: int,
        bet_size: float,
        required_win: float,
        limiting_factor: str
    ) -> str:
        """Generate human-readable reasoning"""
        if level == 2:
            return f"Level 2: Recovering ${required_win:.0f}. Bet ${bet_size:.0f} (limited by {limiting_factor})"
        elif level == 3:
            return f"Level 3: Final recovery ${required_win:.0f}. Bet ${bet_size:.0f} (limited by {limiting_factor})"
        else:
            return f"Level {level}: Recovery bet"


# Test the progression calculator
if __name__ == "__main__":
    import time
    
    print("="*80)
    print("PROGRESSION CALCULATOR - VERIFICATION")
    print("="*80)
    
    calculator = ProgressionCalculator(max_level=3)
    
    # Test 1: Calculate Level 2 bet
    print("\n1. Level 2 Bet Calculation:")
    print("   Scenario: Lost $272.50 at Level 1")
    print("   Goal: Recover loss + make original $247 profit")
    
    level2 = calculator.calculate_level_bet(
        level=2,
        cumulative_loss=272.50,
        target_profit=247.00,
        current_bankroll=4727.50,
        kelly_fraction=0.15,
        odds=1.909
    )
    
    print(f"\n   Required win: ${level2['required_win']:.2f}")
    print(f"   Bet needed (uncapped): ${level2['bet_needed_uncapped']:.2f}")
    print(f"   Kelly limit: ${level2['kelly_limit']:.2f}")
    print(f"   Hard limit: ${level2['hard_limit']:.2f}")
    print(f"   → Final bet: ${level2['bet_size']:.2f}")
    print(f"   Is capped: {level2['is_capped']}")
    print(f"   Limiting factor: {level2['limiting_factor']}")
    print(f"\n   Probabilities:")
    print(f"   P(Reach Level 2): {level2['p_reach_here']:.1%}")
    print(f"   P(Lose from here): {level2['p_lose_from_here']:.1%}")
    print(f"   P(Lose all 3): {level2['p_ruin_total']:.1%}")
    
    # Test 2: Calculate Level 3 bet
    print("\n2. Level 3 Bet Calculation:")
    print("   Scenario: Lost $272.50 + $571 = $843.50")
    
    level3 = calculator.calculate_level_bet(
        level=3,
        cumulative_loss=843.50,
        target_profit=247.00,
        current_bankroll=4156.50,
        kelly_fraction=0.15,
        odds=1.909
    )
    
    print(f"\n   Required win: ${level3['required_win']:.2f}")
    print(f"   Bet needed (uncapped): ${level3['bet_needed_uncapped']:.2f}")
    print(f"   → Final bet: ${level3['bet_size']:.2f}")
    print(f"   Limiting factor: {level3['limiting_factor']}")
    print(f"\n   ⚠️  This is max depth!")
    print(f"   P(Reach Level 3): {level3['p_reach_here']:.1%}")
    print(f"   P(Lose all 3): {level3['p_ruin_total']:.1%}")
    
    # Test 3: Complete sequence
    print("\n3. Complete 3-Level Sequence:")
    sequence = calculator.calculate_complete_sequence(
        base_bet=272.50,
        target_profit=247.00,
        current_bankroll=5000.00,
        kelly_fraction=0.15,
        odds=1.909
    )
    
    print(f"\n   {'Level':<10} {'Bet Size':<12} {'P(Reach)':<12} {'Cumulative Loss':<18}")
    print(f"   {'-'*52}")
    for i in range(1, 4):
        level_data = sequence[f'level_{i}']
        if i == 1:
            print(f"   Level {i}    ${level_data['bet_size']:>8.2f}   100.0%       $0.00")
            cum = level_data['bet_size']
        else:
            print(f"   Level {i}    ${level_data['bet_size']:>8.2f}   {level_data['p_reach_here']*100:>5.1f}%      ${level_data['cumulative_loss']:>8.2f}")
    
    print(f"\n   Max cumulative loss: ${sequence['max_cumulative_loss']:.2f}")
    print(f"   P(Max loss): {sequence['p_max_loss']:.1%}")
    print(f"   Expected sequence length: {sequence['expected_sequence_length']:.2f} levels")
    
    # Test 4: Performance
    print("\n4. Performance Test (1000 calculations):")
    start = time.time()
    for _ in range(1000):
        calculator.calculate_level_bet(2, 272.50, 247.00, 4727.50, 0.15, 1.909)
    elapsed = (time.time() - start) * 1000
    avg = elapsed / 1000
    
    print(f"   1000 calculations: {elapsed:.1f}ms total")
    print(f"   Average: {avg:.2f}ms per calculation")
    print(f"   Target: <5ms")
    
    if avg < 5:
        print(f"   ✅ PASS!")
    else:
        print(f"   ❌ FAIL - Too slow")
    
    print("\n" + "="*80)
    print("✅ PROGRESSION CALCULATOR READY")
    print("="*80)
    print("\nKelly-Adjusted Progressive Betting:")
    print("  Level 1: Base Kelly bet")
    print("  Level 2: Recover L1 + target (Kelly-capped)")
    print("  Level 3: Recover L1+L2 + target (Kelly-capped)")
    print("  Max depth: 3 (then forced reset)")
    print("  P(Lose all 3): 6.4% (low risk of ruin)")

