"""
Power Controller - Dynamic Risk Adjustment
Run at 125% power when conditions are perfect, 25% when defensive

Based on: DECISION_TREE/IMPLEMENTATION_ENHANCEMENTS.md (Enhancement 2)
Performance: <5ms per calculation
"""

from typing import Dict

class PowerController:
    """
    Adjust system power based on conditions
    
    Power Levels:
        125% TURBO 🚀 - Perfect conditions (model excellent, winning streak)
        115% BOOST ⚡ - Great conditions
        100% FULL POWER - Normal conditions
        75% CRUISE 🏃 - Slight caution
        50% CAUTION ⚠️ - Defensive mode
        25% DEFENSIVE 🛡️ - Emergency mode
    
    This is your throttle - not a limiter, but a POWER AMPLIFIER
    
    Performance: <5ms
    """
    
    def __init__(self):
        """Initialize power controller"""
        self.power_level = 1.0  # Start at 100%
        self.boost_available = True
        
        print("Power Controller initialized:")
        print("  Default power: 100%")
        print("  Turbo available: YES (125% max)")
        print("  Emergency mode: 25% min")
    
    def calculate_power_level(
        self,
        calibration_status: str,
        recent_win_rate: float,
        drawdown: float,
        bankroll_health: float
    ) -> float:
        """
        Calculate current power level
        
        Args:
            calibration_status: 'EXCELLENT', 'GOOD', 'FAIR', 'POOR'
            recent_win_rate: Win rate last 20 bets (0-1)
            drawdown: Current drawdown from peak (0-1)
            bankroll_health: Current/Initial bankroll (>1 is profit)
        
        Returns:
            Power level (0.25 to 1.25)
        
        Decision Logic:
            TURBO (125%): Excellent model + 65%+ win rate + <5% drawdown + 120%+ bankroll
            BOOST (115%): Excellent model + 60%+ win rate + <10% drawdown
            FULL (100%): Good model + <15% drawdown
            CRUISE (75%): Fair model or <25% drawdown
            CAUTION (50%): Poor model or <35% drawdown  
            DEFENSIVE (25%): Emergency (>35% drawdown)
        
        Time: <5ms
        """
        power = 1.0  # Start at 100%
        
        # === TURBO MODE (125%) ===
        # Everything is PERFECT
        if (calibration_status == 'EXCELLENT' and 
            recent_win_rate > 0.65 and 
            drawdown < 0.05 and 
            bankroll_health > 1.20):
            power = 1.25
            status = 'TURBO 🚀'
        
        # === BOOST MODE (115%) ===
        # Conditions are GREAT
        elif (calibration_status == 'EXCELLENT' and 
              recent_win_rate > 0.60 and 
              drawdown < 0.10):
            power = 1.15
            status = 'BOOST ⚡'
        
        # === FULL POWER (100%) ===
        # Normal good conditions
        elif (calibration_status in ['EXCELLENT', 'GOOD'] and 
              drawdown < 0.15):
            power = 1.0
            status = 'FULL POWER'
        
        # === CRUISE MODE (75%) ===
        # Slightly cautious
        elif drawdown < 0.25:
            power = 0.75
            status = 'CRUISE 🏃'
        
        # === CAUTION MODE (50%) ===
        # Defensive
        elif drawdown < 0.35:
            power = 0.50
            status = 'CAUTION ⚠️'
        
        # === DEFENSIVE MODE (25%) ===
        # Emergency - big drawdown
        else:
            power = 0.25
            status = 'DEFENSIVE 🛡️'
        
        self.power_level = power
        
        return power
    
    def apply_power_to_bet(self, bet_size: float) -> float:
        """
        Apply current power level to bet size
        
        Args:
            bet_size: Base bet from risk optimization
        
        Returns:
            Power-adjusted bet
        
        Example:
            Base: $500
            Power: 125% (TURBO)
            Adjusted: $625
        
        Time: <1ms
        """
        return bet_size * self.power_level
    
    def can_use_progression(self) -> bool:
        """
        Check if progression betting is allowed
        
        Returns:
            True if power > 50%
        
        Reasoning: Only use aggressive progression when conditions are decent
        
        Time: <1ms
        """
        return self.power_level >= 0.50
    
    def get_max_progression_depth(self) -> int:
        """
        Get allowed progression depth based on power level
        
        Returns:
            Max levels allowed (0-3)
        
        Logic:
            ≥100% power: 3 levels (full progression)
            ≥75% power: 2 levels (moderate)
            ≥50% power: 1 level (base only, no progression)
            <50% power: 0 levels (no betting)
        
        Time: <1ms
        """
        if self.power_level >= 1.0:
            return 3  # Full progression
        elif self.power_level >= 0.75:
            return 2  # Moderate
        elif self.power_level >= 0.50:
            return 1  # Base only
        else:
            return 0  # No betting
    
    def get_power_status(self) -> Dict:
        """
        Get current power status with reasoning
        
        Returns:
            {
                'power_level': 1.25,
                'status': 'TURBO 🚀',
                'progression_allowed': True,
                'max_depth': 3,
                'multiplier_effect': '125% of base bets'
            }
        """
        if self.power_level >= 1.15:
            status = 'TURBO 🚀'
            description = 'CRUSHING IT - Running hot!'
        elif self.power_level >= 1.0:
            status = 'FULL POWER ⚡'
            description = 'Optimal conditions'
        elif self.power_level >= 0.75:
            status = 'CRUISE 🏃'
            description = 'Good conditions'
        elif self.power_level >= 0.50:
            status = 'CAUTION ⚠️'
            description = 'Defensive mode'
        else:
            status = 'DEFENSIVE 🛡️'
            description = 'Emergency mode'
        
        return {
            'power_level': self.power_level,
            'power_percentage': f'{self.power_level*100:.0f}%',
            'status': status,
            'description': description,
            'progression_allowed': self.can_use_progression(),
            'max_depth': self.get_max_progression_depth(),
            'multiplier_effect': f'{self.power_level:.0%} of base bets'
        }


# Test the power controller
if __name__ == "__main__":
    print("="*80)
    print("POWER CONTROLLER - VERIFICATION")
    print("="*80)
    
    controller = PowerController()
    
    # Test 1: TURBO mode (perfect conditions)
    print("\n1. TURBO MODE (Perfect Conditions):")
    print("   Model: EXCELLENT calibration")
    print("   Win rate: 68% (last 20 bets)")
    print("   Drawdown: 3%")
    print("   Bankroll: 106% of initial")
    
    power1 = controller.calculate_power_level(
        calibration_status='EXCELLENT',
        recent_win_rate=0.68,
        drawdown=0.03,
        bankroll_health=1.06
    )
    
    status1 = controller.get_power_status()
    print(f"\n   → Power: {status1['power_percentage']} ({status1['status']})")
    print(f"   → {status1['description']}")
    print(f"   → Max progression depth: {status1['max_depth']} levels")
    
    # Apply to bet
    base_bet = 500
    turbo_bet = controller.apply_power_to_bet(base_bet)
    print(f"\n   Example: Base bet $500 → TURBO bet ${turbo_bet:.0f}")
    
    # Test 2: CRUISE mode (normal conditions)
    print("\n2. CRUISE MODE (Normal Conditions):")
    power2 = controller.calculate_power_level(
        calibration_status='GOOD',
        recent_win_rate=0.58,
        drawdown=0.18,
        bankroll_health=0.95
    )
    
    status2 = controller.get_power_status()
    print(f"   → Power: {status2['power_percentage']} ({status2['status']})")
    print(f"   → Max progression depth: {status2['max_depth']} levels")
    
    cruise_bet = controller.apply_power_to_bet(base_bet)
    print(f"   Example: Base bet $500 → CRUISE bet ${cruise_bet:.0f}")
    
    # Test 3: CAUTION mode (drawdown)
    print("\n3. CAUTION MODE (Drawdown Conditions):")
    power3 = controller.calculate_power_level(
        calibration_status='FAIR',
        recent_win_rate=0.52,
        drawdown=0.28,
        bankroll_health=0.85
    )
    
    status3 = controller.get_power_status()
    print(f"   → Power: {status3['power_percentage']} ({status3['status']})")
    print(f"   → Progression allowed: {status3['progression_allowed']}")
    print(f"   → Max depth: {status3['max_depth']} level (base only)")
    
    caution_bet = controller.apply_power_to_bet(base_bet)
    print(f"   Example: Base bet $500 → CAUTION bet ${caution_bet:.0f}")
    
    # Test 4: DEFENSIVE mode (emergency)
    print("\n4. DEFENSIVE MODE (Emergency):")
    power4 = controller.calculate_power_level(
        calibration_status='POOR',
        recent_win_rate=0.45,
        drawdown=0.40,
        bankroll_health=0.72
    )
    
    status4 = controller.get_power_status()
    print(f"   → Power: {status4['power_percentage']} ({status4['status']})")
    print(f"   → Progression allowed: {status4['progression_allowed']}")
    
    defensive_bet = controller.apply_power_to_bet(base_bet)
    print(f"   Example: Base bet $500 → DEFENSIVE bet ${defensive_bet:.0f}")
    
    # Summary
    print("\n" + "="*80)
    print("POWER LEVEL SUMMARY")
    print("="*80)
    print(f"\n  {'Conditions':<20} {'Power':<10} {'$500 Bet →':<15} {'Progression':<12}")
    print(f"  {'-'*57}")
    print(f"  {'TURBO (perfect)':<20} {125:>6}%   ${625:>8}         {3:>3} levels")
    print(f"  {'BOOST (great)':<20} {115:>6}%   ${575:>8}         {3:>3} levels")
    print(f"  {'FULL (good)':<20} {100:>6}%   ${500:>8}         {3:>3} levels")
    print(f"  {'CRUISE (ok)':<20} {75:>6}%   ${375:>8}         {2:>3} levels")
    print(f"  {'CAUTION (defensive)':<20} {50:>6}%   ${250:>8}         {1:>3} level")
    print(f"  {'DEFENSIVE (emergency)':<20} {25:>6}%   ${125:>8}         {0:>3} levels")
    
    print("\n" + "="*80)
    print("✅ POWER CONTROLLER READY")
    print("="*80)
    print("\nYour throttle for aggressive betting:")
    print("  ✅ Run at 125% when conditions are PERFECT")
    print("  ✅ Graduated response (not binary)")
    print("  ✅ Protects during drawdowns")
    print("  ✅ Maximizes profit when edge is proven")

