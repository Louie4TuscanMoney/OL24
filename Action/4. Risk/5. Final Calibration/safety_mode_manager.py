"""
Safety Mode Manager - Determine Current Safety Level
GREEN (15% max) / YELLOW (12% max) / RED (8% max)

Based on: FINAL_CALIBRATION/Applied Model/safety_mode_manager.py
Following: FINAL_CALIBRATION/MATH_BREAKDOWN.txt (Section 4)
Performance: <2ms
"""

from enum import Enum
from typing import Dict

class SafetyMode(Enum):
    """Safety modes for risk management"""
    GREEN = "GREEN"      # Normal operations (15% max)
    YELLOW = "YELLOW"    # Caution (12% max)
    RED = "RED"          # Defensive (8% max)


class SafetyModeManager:
    """
    Determine current safety mode based on system health
    
    Modes (MATH_BREAKDOWN.txt 4.2):
        GREEN:  15% max ($750) - Normal operations
        YELLOW: 12% max ($600) - Caution mode
        RED:    8% max ($400)  - Defensive mode
    
    Performance: <2ms
    """
    
    def __init__(self, original_bankroll: float = 5000):
        """
        Initialize safety mode manager
        
        Args:
            original_bankroll: Starting bankroll
        """
        self.original_bankroll = original_bankroll
        
        # Mode-specific limits (% of original bankroll)
        self.mode_limits = {
            SafetyMode.GREEN: 0.15,   # $750
            SafetyMode.YELLOW: 0.12,  # $600
            SafetyMode.RED: 0.08      # $400
        }
        
        # Portfolio limits (% of original bankroll)
        self.portfolio_limits = {
            SafetyMode.GREEN: 0.50,   # $2,500
            SafetyMode.YELLOW: 0.40,  # $2,000
            SafetyMode.RED: 0.20      # $1,000
        }
        
        print("Safety Mode Manager initialized:")
        print(f"  GREEN mode: ${self.mode_limits[SafetyMode.GREEN] * original_bankroll:.0f} max")
        print(f"  YELLOW mode: ${self.mode_limits[SafetyMode.YELLOW] * original_bankroll:.0f} max")
        print(f"  RED mode: ${self.mode_limits[SafetyMode.RED] * original_bankroll:.0f} max")
    
    def determine_mode(
        self,
        current_bankroll: float,
        calibration_status: str,
        recent_win_rate: float,
        current_drawdown: float
    ) -> Dict:
        """
        Determine current safety mode
        
        Decision Logic (MATH_BREAKDOWN.txt 4.1):
            RED if:
                - Calibration POOR/VERY_POOR OR
                - Bankroll < 60% of original OR
                - Win rate < 50% OR
                - Drawdown > 25%
            
            YELLOW if:
                - Calibration FAIR OR
                - Bankroll 60-80% of original OR
                - Win rate 50-55% OR
                - Drawdown 15-25%
            
            GREEN otherwise:
                - Calibration GOOD/EXCELLENT
                - Bankroll > 80% of original
                - Win rate > 55%
                - Drawdown < 15%
        
        Args:
            current_bankroll: Current bankroll amount
            calibration_status: 'EXCELLENT', 'GOOD', 'FAIR', 'POOR'
            recent_win_rate: Win rate last 20 bets (0-1)
            current_drawdown: Current drawdown from peak (0-1)
        
        Returns:
            {
                'mode': SafetyMode.GREEN,
                'mode_emoji': '🟢',
                'max_bet': 750.00,
                'max_total': 2500.00,
                'reserve_required': 2500.00,
                'reasoning': 'All systems healthy - normal operations'
            }
        
        Time: <2ms
        """
        # Calculate bankroll ratio
        bankroll_ratio = current_bankroll / self.original_bankroll
        
        # === RED MODE (Most Restrictive) ===
        if (calibration_status in ['POOR', 'VERY_POOR'] or
            bankroll_ratio < 0.60 or
            recent_win_rate < 0.50 or
            current_drawdown > 0.25):
            
            mode = SafetyMode.RED
            reasoning = self._get_red_reasoning(
                calibration_status, bankroll_ratio, recent_win_rate, current_drawdown
            )
            emoji = '🔴'
        
        # === YELLOW MODE (Moderate Restriction) ===
        elif (calibration_status == 'FAIR' or
              0.60 <= bankroll_ratio < 0.80 or
              0.50 <= recent_win_rate < 0.55 or
              0.15 <= current_drawdown < 0.25):
            
            mode = SafetyMode.YELLOW
            reasoning = self._get_yellow_reasoning(
                calibration_status, bankroll_ratio, recent_win_rate, current_drawdown
            )
            emoji = '🟡'
        
        # === GREEN MODE (Normal Operations) ===
        else:
            mode = SafetyMode.GREEN
            reasoning = 'All systems healthy - normal operations'
            emoji = '🟢'
        
        # Calculate mode-specific limits
        max_bet = self.original_bankroll * self.mode_limits[mode]
        max_total = self.original_bankroll * self.portfolio_limits[mode]
        reserve_required = self.original_bankroll * 0.50  # Always 50%
        
        return {
            'mode': mode,
            'mode_name': mode.value,
            'mode_emoji': emoji,
            'max_bet': round(max_bet, 2),
            'max_total': round(max_total, 2),
            'reserve_required': round(reserve_required, 2),
            'reasoning': reasoning,
            'limits': {
                'single_bet_pct': self.mode_limits[mode],
                'portfolio_pct': self.portfolio_limits[mode],
                'reserve_pct': 0.50
            }
        }
    
    def _get_red_reasoning(self, calib: str, ratio: float, win_rate: float, dd: float) -> str:
        """Generate reasoning for RED mode"""
        reasons = []
        
        if calib in ['POOR', 'VERY_POOR']:
            reasons.append(f'Model calibration {calib}')
        if ratio < 0.60:
            reasons.append(f'Bankroll {ratio:.0%} of original')
        if win_rate < 0.50:
            reasons.append(f'Win rate {win_rate:.0%}')
        if dd > 0.25:
            reasons.append(f'Drawdown {dd:.0%}')
        
        return '🔴 RED MODE: ' + ' + '.join(reasons)
    
    def _get_yellow_reasoning(self, calib: str, ratio: float, win_rate: float, dd: float) -> str:
        """Generate reasoning for YELLOW mode"""
        reasons = []
        
        if calib == 'FAIR':
            reasons.append('Model calibration FAIR')
        if 0.60 <= ratio < 0.80:
            reasons.append(f'Bankroll {ratio:.0%} of original')
        if 0.50 <= win_rate < 0.55:
            reasons.append(f'Win rate {win_rate:.0%}')
        if 0.15 <= dd < 0.25:
            reasons.append(f'Drawdown {dd:.0%}')
        
        return '🟡 YELLOW MODE: ' + ' + '.join(reasons)


# Test the safety mode manager
if __name__ == "__main__":
    print("="*80)
    print("SAFETY MODE MANAGER - VERIFICATION")
    print("="*80)
    
    manager = SafetyModeManager(original_bankroll=5000)
    
    # Test 1: GREEN mode (healthy)
    print("\n1. GREEN MODE (Healthy System):")
    print("   Calibration: EXCELLENT")
    print("   Bankroll: $5,000 (100%)")
    print("   Win rate: 62%")
    print("   Drawdown: 5%")
    
    mode1 = manager.determine_mode(
        current_bankroll=5000,
        calibration_status='EXCELLENT',
        recent_win_rate=0.62,
        current_drawdown=0.05
    )
    
    print(f"\n   → Mode: {mode1['mode_emoji']} {mode1['mode_name']}")
    print(f"   → Max bet: ${mode1['max_bet']:.0f}")
    print(f"   → Max portfolio: ${mode1['max_total']:.0f}")
    print(f"   → Reasoning: {mode1['reasoning']}")
    
    # Test 2: YELLOW mode (caution)
    print("\n2. YELLOW MODE (Caution - Drawdown):")
    print("   Bankroll: $4,000 (80% of original)")
    print("   Win rate: 54%")
    print("   Drawdown: 18%")
    
    mode2 = manager.determine_mode(
        current_bankroll=4000,
        calibration_status='GOOD',
        recent_win_rate=0.54,
        current_drawdown=0.18
    )
    
    print(f"\n   → Mode: {mode2['mode_emoji']} {mode2['mode_name']}")
    print(f"   → Max bet: ${mode2['max_bet']:.0f} (reduced from $750)")
    print(f"   → Max portfolio: ${mode2['max_total']:.0f} (reduced from $2,500)")
    print(f"   → Reasoning: {mode2['reasoning']}")
    
    # Test 3: RED mode (defensive)
    print("\n3. RED MODE (Defensive - Bad Conditions):")
    print("   Bankroll: $2,800 (56% of original)")
    print("   Win rate: 48%")
    print("   Drawdown: 32%")
    
    mode3 = manager.determine_mode(
        current_bankroll=2800,
        calibration_status='POOR',
        recent_win_rate=0.48,
        current_drawdown=0.32
    )
    
    print(f"\n   → Mode: {mode3['mode_emoji']} {mode3['mode_name']}")
    print(f"   → Max bet: ${mode3['max_bet']:.0f} (HEAVILY reduced)")
    print(f"   → Max portfolio: ${mode3['max_total']:.0f} (20% only)")
    print(f"   → Reasoning: {mode3['reasoning']}")
    print(f"   → ⚠️  DEFENSIVE MODE - Consider pausing betting")
    
    # Summary
    print("\n" + "="*80)
    print("SAFETY MODE SUMMARY")
    print("="*80)
    
    print(f"\n  {'Mode':<8} {'Emoji':<6} {'Max Bet':<12} {'Max Portfolio':<15} {'When':<30}")
    print(f"  {'-'*71}")
    print(f"  {'GREEN':<8} {'🟢':<6} {'$750':<12} {'$2,500':<15} {'Healthy (>80% bankroll, >55% win)':<30}")
    print(f"  {'YELLOW':<8} {'🟡':<6} {'$600':<12} {'$2,000':<15} {'Caution (60-80%, 50-55% win)':<30}")
    print(f"  {'RED':<8} {'🔴':<6} {'$400':<12} {'$1,000':<15} {'Defensive (<60%, <50% win)':<30}")
    
    print("\n" + "="*80)
    print("✅ SAFETY MODE MANAGER READY")
    print("="*80)
    print("\nGraduated safety response:")
    print("  🟢 GREEN: Full operations")
    print("  🟡 YELLOW: Caution mode")
    print("  🔴 RED: Defensive or stop")

