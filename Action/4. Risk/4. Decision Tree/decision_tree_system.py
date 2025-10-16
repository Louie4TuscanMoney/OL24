"""
Decision Tree System - Complete Integration
Progressive betting with Kelly safeguards + Power control

Flow: Portfolio → Decision Tree → Final Bet (with state management)
Performance: <20ms per calculation
"""

import time
from typing import Dict, Optional
from state_manager import StateManager, ProgressionLevel
from progression_calculator import ProgressionCalculator
from power_controller import PowerController


class DecisionTreeSystem:
    """
    Complete Decision Tree Risk Management
    
    Manages:
    - Progression state tracking
    - Recovery bet calculations
    - Power-based adjustments
    - Safety limit enforcement
    
    Formula (MATH_BREAKDOWN.txt):
        P(Lose N consecutive) = P(Loss)^N
        
        Level 1: Base betting
        Level 2: Recover + target (if lose 1)
        Level 3: Recover all + target (if lose 2)
        Max depth: 3 (then reset)
    
    Performance: <20ms per calculation
    """
    
    def __init__(self, initial_bankroll: float = 5000):
        """
        Initialize decision tree system
        
        Args:
            initial_bankroll: Starting bankroll ($5,000 default)
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        
        # Components
        self.state_manager = StateManager(max_active_progressions=5)
        self.progression_calculator = ProgressionCalculator(max_level=3)
        self.power_controller = PowerController()
        
        # Performance tracking
        self.total_bets = 0
        self.total_wins = 0
        self.peak_bankroll = initial_bankroll
        
        print("\n" + "="*80)
        print("DECISION TREE SYSTEM INITIALIZED")
        print("="*80)
        print("Progressive betting with Kelly safeguards")
        print(f"  Initial bankroll: ${initial_bankroll:,.0f}")
        print(f"  Max progression depth: 3 levels")
        print(f"  Power control: 25%-125%")
    
    def calculate_final_bet(
        self,
        portfolio_bet: float,
        game_context_id: str,
        kelly_fraction: float,
        target_profit: float,
        odds: float,
        ml_confidence: float = 0.75
    ) -> Dict:
        """
        Calculate final bet with decision tree logic
        
        INPUT from Portfolio Management:
            portfolio_bet: $375 (Markowitz-optimized)
            game_context_id: 'LAL@BOS_2025-10-15'
            kelly_fraction: 0.15
            target_profit: $341 (what we'd win at Level 1)
            odds: 1.909 (-110 American)
            ml_confidence: 0.85
        
        PROCESS:
            1. Check current progression state
            2. Calculate bet for current level
            3. Apply power controller
            4. Return final bet
        
        OUTPUT:
            {
                'final_bet': 375.00,
                'level': 1,
                'power_level': 1.15,
                'progression_active': False,
                'reasoning': '...',
                'performance_ms': 12
            }
        
        Time: <20ms
        """
        start = time.time()
        
        # Get current state
        state = self.state_manager.get_state(game_context_id)
        
        # Calculate power level
        calibration_status = self._get_calibration_status()
        recent_win_rate = self._get_recent_win_rate()
        drawdown = self._get_current_drawdown()
        bankroll_health = self.current_bankroll / self.initial_bankroll
        
        power_level = self.power_controller.calculate_power_level(
            calibration_status=calibration_status,
            recent_win_rate=recent_win_rate,
            drawdown=drawdown,
            bankroll_health=bankroll_health
        )
        
        # === LEVEL 1 (Base) ===
        if state.level == ProgressionLevel.LEVEL_1:
            # Use portfolio bet
            base_bet = portfolio_bet
            
            # Apply power adjustment
            power_adjusted_bet = self.power_controller.apply_power_to_bet(base_bet)
            
            # Don't exceed portfolio bet by more than 35% (even with power)
            max_amplified = portfolio_bet * 1.35
            final_bet = min(power_adjusted_bet, max_amplified)
            
            result = {
                'final_bet': round(final_bet, 2),
                'base_bet': round(portfolio_bet, 2),
                'power_adjusted': round(power_adjusted_bet, 2),
                'level': 1,
                'power_level': power_level,
                'power_status': self.power_controller.get_power_status()['status'],
                'progression_active': False,
                'reasoning': f"Level 1 base betting at {power_level:.0%} power"
            }
        
        # === LEVEL 2-3 (Progression) ===
        else:
            # Check if progression allowed
            if not self.power_controller.can_use_progression():
                # Power too low - skip progression, use small base bet
                result = {
                    'final_bet': round(portfolio_bet * 0.50, 2),
                    'level': state.level.value,
                    'power_level': power_level,
                    'power_status': self.power_controller.get_power_status()['status'],
                    'progression_active': False,
                    'progression_skipped': True,
                    'reasoning': f"Power too low ({power_level:.0%}) - progression disabled"
                }
            else:
                # Calculate progression bet
                progression_bet_calc = self.progression_calculator.calculate_level_bet(
                    level=state.level.value,
                    cumulative_loss=state.cumulative_loss,
                    target_profit=state.target_profit,
                    current_bankroll=self.current_bankroll,
                    kelly_fraction=kelly_fraction,
                    odds=odds
                )
                
                # Apply power adjustment to progression bet
                power_adjusted = self.power_controller.apply_power_to_bet(
                    progression_bet_calc['bet_size']
                )
                
                # Don't exceed 2x original progression bet (even with turbo)
                max_amplified = progression_bet_calc['bet_size'] * 2.0
                final_bet = min(power_adjusted, max_amplified)
                
                result = {
                    'final_bet': round(final_bet, 2),
                    'base_progression': round(progression_bet_calc['bet_size'], 2),
                    'power_adjusted': round(power_adjusted, 2),
                    'level': state.level.value,
                    'power_level': power_level,
                    'power_status': self.power_controller.get_power_status()['status'],
                    'progression_active': True,
                    'cumulative_loss': progression_bet_calc['cumulative_loss'],
                    'required_win': progression_bet_calc['required_win'],
                    'p_lose_from_here': progression_bet_calc['p_lose_from_here'],
                    'is_capped': progression_bet_calc['is_capped'],
                    'reasoning': progression_bet_calc['reasoning']
                }
        
        # Performance tracking
        elapsed = (time.time() - start) * 1000
        result['performance_ms'] = round(elapsed, 2)
        result['game_context_id'] = game_context_id
        
        return result
    
    def record_outcome(
        self,
        game_context_id: str,
        won: bool,
        bet_size: float,
        profit_or_loss: float
    ):
        """
        Record bet outcome and update state
        
        Args:
            game_context_id: Game identifier
            won: True if won, False if lost
            bet_size: Amount bet
            profit_or_loss: Amount won (positive) or lost (negative)
        """
        state = self.state_manager.get_state(game_context_id)
        
        if won:
            # Win resets to Level 1
            result = state.record_win(profit_or_loss)
            self.current_bankroll += profit_or_loss
            self.total_wins += 1
            
            print(f"✅ WIN: +${profit_or_loss:.2f}")
            print(f"   Sequence complete - reset to Level 1")
            
        else:
            # Loss progresses to next level
            result = state.record_loss(bet_size)
            self.current_bankroll -= bet_size
            
            if result.get('hit_max_depth'):
                print(f"❌ LOSS: -${bet_size:.2f}")
                print(f"   ⚠️  MAX DEPTH HIT - Forced reset to Level 1")
                print(f"   Total sequence loss: ${result['cumulative_loss']:.2f}")
            else:
                print(f"❌ LOSS: -${bet_size:.2f}")
                print(f"   Progress to Level {result['to_level']}")
                print(f"   Cumulative loss: ${result['cumulative_loss']:.2f}")
        
        self.total_bets += 1
        
        # Update peak bankroll
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll
    
    def _get_calibration_status(self) -> str:
        """Get current model calibration status"""
        # Simplified - would connect to calibration monitor
        win_rate = self._get_recent_win_rate()
        
        if win_rate > 0.65:
            return 'EXCELLENT'
        elif win_rate > 0.58:
            return 'GOOD'
        elif win_rate > 0.52:
            return 'FAIR'
        else:
            return 'POOR'
    
    def _get_recent_win_rate(self) -> float:
        """Get recent win rate (last 20 bets)"""
        if self.total_bets == 0:
            return 0.60  # Default assumption
        
        return self.total_wins / self.total_bets
    
    def _get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if self.peak_bankroll == 0:
            return 0.0
        
        drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
        return max(0.0, drawdown)
    
    def get_system_status(self) -> Dict:
        """
        Get complete system status
        
        Returns all key metrics
        """
        power_status = self.power_controller.get_power_status()
        active_progressions = self.state_manager.get_active_progression_count()
        
        return {
            'bankroll': {
                'current': self.current_bankroll,
                'initial': self.initial_bankroll,
                'peak': self.peak_bankroll,
                'return_pct': (self.current_bankroll / self.initial_bankroll - 1) * 100,
                'drawdown_pct': self._get_current_drawdown() * 100
            },
            'performance': {
                'total_bets': self.total_bets,
                'total_wins': self.total_wins,
                'win_rate': self._get_recent_win_rate(),
                'calibration': self._get_calibration_status()
            },
            'power': power_status,
            'progressions': {
                'active_count': active_progressions,
                'max_allowed': self.state_manager.max_active_progressions,
                'total_exposure': self.state_manager.get_total_progression_exposure()
            }
        }


# Test the complete decision tree system
if __name__ == "__main__":
    print("="*80)
    print("DECISION TREE SYSTEM - COMPLETE SEQUENCE TEST")
    print("="*80)
    
    system = DecisionTreeSystem(initial_bankroll=5000)
    
    print("\n" + "="*80)
    print("SCENARIO: 3-LEVEL PROGRESSION SEQUENCE")
    print("="*80)
    print("Simulating: Lose → Lose → Win (recovery!)")
    
    # === GAME 1 (Level 1) ===
    print("\n" + "─"*80)
    print("GAME 1 - Level 1 (Base Betting)")
    print("─"*80)
    
    result1 = system.calculate_final_bet(
        portfolio_bet=375.00,
        game_context_id='sequence_test',
        kelly_fraction=0.15,
        target_profit=341.00,
        odds=1.909,
        ml_confidence=0.85
    )
    
    print(f"\nPortfolio suggested: ${result1['base_bet']:.2f}")
    print(f"Power level: {result1['power_level']:.0%} ({result1['power_status']})")
    print(f"Final bet: ${result1['final_bet']:.2f}")
    print(f"Level: {result1['level']}")
    print(f"Reasoning: {result1['reasoning']}")
    print(f"Performance: {result1['performance_ms']:.2f}ms")
    
    # Simulate LOSS
    print(f"\n>>> OUTCOME: LOSE -${result1['final_bet']:.2f}")
    system.record_outcome('sequence_test', won=False, bet_size=result1['final_bet'], profit_or_loss=-result1['final_bet'])
    print(f">>> Bankroll: ${system.current_bankroll:,.2f}")
    
    # === GAME 2 (Level 2) ===
    print("\n" + "─"*80)
    print("GAME 2 - Level 2 (First Recovery Attempt)")
    print("─"*80)
    
    # Set target profit for state
    state = system.state_manager.get_state('sequence_test')
    state.target_profit = 341.00
    
    result2 = system.calculate_final_bet(
        portfolio_bet=0,  # Not used at Level 2
        game_context_id='sequence_test',
        kelly_fraction=0.15,
        target_profit=341.00,
        odds=1.909
    )
    
    print(f"\nProgression active: {result2['progression_active']}")
    print(f"Level: {result2['level']}")
    print(f"Cumulative loss: ${result2['cumulative_loss']:.2f}")
    print(f"Required win: ${result2['required_win']:.2f}")
    print(f"Final bet: ${result2['final_bet']:.2f}")
    print(f"P(Lose from here): {result2['p_lose_from_here']:.1%}")
    print(f"Reasoning: {result2['reasoning']}")
    
    # Simulate LOSS again
    print(f"\n>>> OUTCOME: LOSE -${result2['final_bet']:.2f}")
    system.record_outcome('sequence_test', won=False, bet_size=result2['final_bet'], profit_or_loss=-result2['final_bet'])
    print(f">>> Bankroll: ${system.current_bankroll:,.2f}")
    
    # === GAME 3 (Level 3) ===
    print("\n" + "─"*80)
    print("GAME 3 - Level 3 (Final Recovery - MAX DEPTH!)")
    print("─"*80)
    
    result3 = system.calculate_final_bet(
        portfolio_bet=0,
        game_context_id='sequence_test',
        kelly_fraction=0.15,
        target_profit=341.00,
        odds=1.909
    )
    
    print(f"\n⚠️  AT MAX DEPTH!")
    print(f"Level: {result3['level']}")
    print(f"Cumulative loss: ${result3['cumulative_loss']:.2f}")
    print(f"Required win: ${result3['required_win']:.2f}")
    print(f"Final bet: ${result3['final_bet']:.2f}")
    print(f"P(Lose all 3): {result3['p_lose_from_here']:.1%} (LOW!)")
    print(f"If lose: Sequence ends, reset to Level 1")
    
    # Simulate WIN (recovery!)
    win_amount = result3['final_bet'] * (1.909 - 1)
    print(f"\n>>> OUTCOME: WIN +${win_amount:.2f}")
    system.record_outcome('sequence_test', won=True, bet_size=result3['final_bet'], profit_or_loss=win_amount)
    print(f">>> Bankroll: ${system.current_bankroll:,.2f}")
    
    # === SEQUENCE SUMMARY ===
    print("\n" + "="*80)
    print("SEQUENCE COMPLETE")
    print("="*80)
    
    print(f"\nResults:")
    print(f"  Started: ${5000:,.2f}")
    print(f"  After 2 losses: ${system.current_bankroll - win_amount:,.2f}")
    print(f"  After recovery: ${system.current_bankroll:,.2f}")
    print(f"  Net P/L: ${system.current_bankroll - 5000:+,.2f}")
    
    net_pl = system.current_bankroll - 5000
    if net_pl > 0:
        print(f"\n  ✅ Recovered AND made profit!")
    elif net_pl > -200:
        print(f"\n  ✅ Mostly recovered losses")
    else:
        print(f"\n  ⚠️  Still down (but progression limited losses)")
    
    # System status
    print("\n" + "="*80)
    print("SYSTEM STATUS")
    print("="*80)
    
    status = system.get_system_status()
    print(f"\nBankroll:")
    print(f"  Current: ${status['bankroll']['current']:,.2f}")
    print(f"  Return: {status['bankroll']['return_pct']:+.1f}%")
    print(f"  Drawdown: {status['bankroll']['drawdown_pct']:.1f}%")
    
    print(f"\nPerformance:")
    print(f"  Win rate: {status['performance']['win_rate']:.1%}")
    print(f"  Calibration: {status['performance']['calibration']}")
    
    print(f"\nPower:")
    print(f"  Level: {status['power']['power_percentage']}")
    print(f"  Status: {status['power']['status']}")
    print(f"  Max depth: {status['power']['max_depth']} levels")
    
    print(f"\nProgressions:")
    print(f"  Active: {status['progressions']['active_count']}")
    print(f"  Max allowed: {status['progressions']['max_allowed']}")
    
    print("\n" + "="*80)
    print("✅ DECISION TREE SYSTEM READY")
    print("="*80)
    print("\nFinite Mathematics Progressive Betting:")
    print("  P(Lose 1) = 40%")
    print("  P(Lose 2 consecutive) = 16%")
    print("  P(Lose 3 consecutive) = 6.4% (VERY LOW!)")
    print("  Each level can recover all previous losses + target")
    print("  Kelly limits enforce safe betting at each level")
    print("  Power controller adjusts based on conditions")

