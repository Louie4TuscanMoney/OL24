"""
ONTORISK PHASE 4: RISK MANAGEMENT SYSTEM

Purpose: Enforce limits, prevent ruin, manage drawdowns
Author: Ontologic XYZ
Date: October 20, 2025

This implements the critical risk controls that prevent bankroll destruction.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    daily_loss_limit_pct: float = 0.10  # Max 10% loss per day
    weekly_loss_limit_pct: float = 0.20  # Max 20% loss per week
    max_drawdown_pct: float = 0.30  # Circuit breaker at 30% drawdown
    max_concurrent_bets: int = 5  # Max 5 bets at once
    max_exposure_pct: float = 0.25  # Max 25% in open bets
    max_single_bet_pct: float = 0.10  # Max 10% per bet
    max_bets_per_day: int = 10  # Max 10 bets per day


@dataclass
class RiskState:
    """Current risk state"""
    current_bankroll: float
    peak_bankroll: float
    daily_start_bankroll: float
    weekly_start_bankroll: float
    daily_loss: float
    weekly_loss: float
    current_drawdown: float
    open_positions: int
    total_exposure: float
    bets_today: int
    last_reset: datetime


class RiskManager:
    """
    Comprehensive risk management system
    """
    
    def __init__(
        self,
        starting_bankroll: float = 10000,
        limits: Optional[RiskLimits] = None
    ):
        """
        Args:
            starting_bankroll: Initial capital
            limits: RiskLimits object (uses defaults if None)
        """
        self.starting_bankroll = starting_bankroll
        self.limits = limits or RiskLimits()
        
        # Initialize state
        self.state = RiskState(
            current_bankroll=starting_bankroll,
            peak_bankroll=starting_bankroll,
            daily_start_bankroll=starting_bankroll,
            weekly_start_bankroll=starting_bankroll,
            daily_loss=0.0,
            weekly_loss=0.0,
            current_drawdown=0.0,
            open_positions=0,
            total_exposure=0.0,
            bets_today=0,
            last_reset=datetime.now()
        )
        
        self.alerts: List[str] = []
        
    def update_bankroll(self, new_bankroll: float):
        """Update current bankroll"""
        self.state.current_bankroll = new_bankroll
        
        # Update peak
        if new_bankroll > self.state.peak_bankroll:
            self.state.peak_bankroll = new_bankroll
        
        # Calculate drawdown
        self.state.current_drawdown = (
            self.state.peak_bankroll - new_bankroll
        ) / self.state.peak_bankroll
        
        # Update daily/weekly loss
        self.state.daily_loss = (
            self.state.daily_start_bankroll - new_bankroll
        )
        self.state.weekly_loss = (
            self.state.weekly_start_bankroll - new_bankroll
        )
    
    def reset_daily(self):
        """Reset daily counters (call at start of each day)"""
        self.state.daily_start_bankroll = self.state.current_bankroll
        self.state.daily_loss = 0.0
        self.state.bets_today = 0
        self.state.last_reset = datetime.now()
    
    def reset_weekly(self):
        """Reset weekly counters (call at start of each week)"""
        self.state.weekly_start_bankroll = self.state.current_bankroll
        self.state.weekly_loss = 0.0
        self.reset_daily()
    
    def check_limits(self) -> Dict[str, bool]:
        """
        Check all risk limits
        
        Returns:
            Dict with status of each check
        """
        checks = {}
        self.alerts = []
        
        # Daily loss check
        daily_loss_pct = self.state.daily_loss / self.state.daily_start_bankroll
        checks['daily_loss_ok'] = daily_loss_pct < self.limits.daily_loss_limit_pct
        
        if not checks['daily_loss_ok']:
            self.alerts.append(
                f"🚨 DAILY LOSS LIMIT HIT: {daily_loss_pct:.1%} > {self.limits.daily_loss_limit_pct:.1%}"
            )
        
        # Weekly loss check
        weekly_loss_pct = self.state.weekly_loss / self.state.weekly_start_bankroll
        checks['weekly_loss_ok'] = weekly_loss_pct < self.limits.weekly_loss_limit_pct
        
        if not checks['weekly_loss_ok']:
            self.alerts.append(
                f"🚨 WEEKLY LOSS LIMIT HIT: {weekly_loss_pct:.1%} > {self.limits.weekly_loss_limit_pct:.1%}"
            )
        
        # Drawdown check
        checks['drawdown_ok'] = self.state.current_drawdown < self.limits.max_drawdown_pct
        
        if not checks['drawdown_ok']:
            self.alerts.append(
                f"🚨 CIRCUIT BREAKER: Drawdown {self.state.current_drawdown:.1%} > {self.limits.max_drawdown_pct:.1%}"
            )
        
        # Position limits
        checks['positions_ok'] = self.state.open_positions < self.limits.max_concurrent_bets
        
        if not checks['positions_ok']:
            self.alerts.append(
                f"⚠️ MAX POSITIONS: {self.state.open_positions} >= {self.limits.max_concurrent_bets}"
            )
        
        # Exposure check
        exposure_pct = self.state.total_exposure / self.state.current_bankroll
        checks['exposure_ok'] = exposure_pct < self.limits.max_exposure_pct
        
        if not checks['exposure_ok']:
            self.alerts.append(
                f"⚠️ MAX EXPOSURE: {exposure_pct:.1%} >= {self.limits.max_exposure_pct:.1%}"
            )
        
        # Daily bet count
        checks['bet_count_ok'] = self.state.bets_today < self.limits.max_bets_per_day
        
        if not checks['bet_count_ok']:
            self.alerts.append(
                f"⚠️ MAX BETS TODAY: {self.state.bets_today} >= {self.limits.max_bets_per_day}"
            )
        
        # Overall status
        checks['can_bet'] = all([
            checks['daily_loss_ok'],
            checks['weekly_loss_ok'],
            checks['drawdown_ok'],
            checks['positions_ok'],
            checks['exposure_ok'],
            checks['bet_count_ok']
        ])
        
        return checks
    
    def validate_bet_size(self, proposed_stake: float) -> Tuple[bool, float, str]:
        """
        Validate and adjust bet size
        
        Args:
            proposed_stake: Proposed bet amount
            
        Returns:
            (is_valid, adjusted_stake, reason)
        """
        # Check limits first
        checks = self.check_limits()
        
        if not checks['can_bet']:
            return False, 0.0, "Risk limits exceeded"
        
        # Max single bet
        max_single = self.state.current_bankroll * self.limits.max_single_bet_pct
        
        if proposed_stake > max_single:
            adjusted = max_single
            reason = f"Reduced to {self.limits.max_single_bet_pct:.0%} of bankroll"
            return True, adjusted, reason
        
        # Check exposure
        new_exposure = self.state.total_exposure + proposed_stake
        max_exposure = self.state.current_bankroll * self.limits.max_exposure_pct
        
        if new_exposure > max_exposure:
            available = max_exposure - self.state.total_exposure
            if available < 50:  # Min bet threshold
                return False, 0.0, "Max exposure reached"
            adjusted = available
            reason = "Reduced to fit exposure limit"
            return True, adjusted, reason
        
        # All good
        return True, proposed_stake, "OK"
    
    def open_position(self, stake: float):
        """Open a new position"""
        self.state.open_positions += 1
        self.state.total_exposure += stake
        self.state.bets_today += 1
    
    def close_position(self, stake: float, profit: float):
        """Close a position"""
        self.state.open_positions -= 1
        self.state.total_exposure -= stake
        self.update_bankroll(self.state.current_bankroll + profit)
    
    def get_status(self) -> Dict:
        """Get current risk status"""
        checks = self.check_limits()
        
        return {
            'current_bankroll': self.state.current_bankroll,
            'peak_bankroll': self.state.peak_bankroll,
            'current_drawdown': self.state.current_drawdown,
            'daily_loss': self.state.daily_loss,
            'weekly_loss': self.state.weekly_loss,
            'open_positions': self.state.open_positions,
            'total_exposure': self.state.total_exposure,
            'bets_today': self.state.bets_today,
            'can_bet': checks['can_bet'],
            'alerts': self.alerts,
            'limits': {
                'daily_loss_limit': self.limits.daily_loss_limit_pct,
                'weekly_loss_limit': self.limits.weekly_loss_limit_pct,
                'max_drawdown': self.limits.max_drawdown_pct,
                'max_positions': self.limits.max_concurrent_bets,
                'max_exposure': self.limits.max_exposure_pct
            }
        }
    
    def print_status(self):
        """Print formatted status"""
        status = self.get_status()
        
        print("\n" + "="*80)
        print("🛡️ RISK MANAGEMENT STATUS")
        print("="*80 + "\n")
        
        print(f"💰 Bankroll: ${status['current_bankroll']:,.0f}")
        print(f"📈 Peak: ${status['peak_bankroll']:,.0f}")
        print(f"📉 Drawdown: {status['current_drawdown']:.1%}")
        print()
        
        print(f"📅 Daily Loss: ${status['daily_loss']:,.0f} " +
              f"(Limit: {status['limits']['daily_loss_limit']:.0%})")
        print(f"📅 Weekly Loss: ${status['weekly_loss']:,.0f} " +
              f"(Limit: {status['limits']['weekly_loss_limit']:.0%})")
        print()
        
        print(f"🎯 Open Positions: {status['open_positions']} " +
              f"(Max: {status['limits']['max_positions']})")
        print(f"💵 Total Exposure: ${status['total_exposure']:,.0f} " +
              f"(Max: {status['limits']['max_exposure']:.0%})")
        print(f"📊 Bets Today: {status['bets_today']}")
        print()
        
        if status['can_bet']:
            print("✅ CAN BET - All limits OK")
        else:
            print("🚨 CANNOT BET - Limits exceeded:")
            for alert in status['alerts']:
                print(f"   {alert}")
        
        print("\n" + "="*80)


def example_usage():
    """
    Example: Risk management in action
    """
    print("\n" + "="*80)
    print("🔥 ONTORISK PHASE 4: RISK MANAGEMENT")
    print("="*80 + "\n")
    
    # Initialize risk manager
    rm = RiskManager(starting_bankroll=10000)
    
    # Show initial status
    rm.print_status()
    
    # Simulate some bets
    print("\n" + "="*80)
    print("📊 SIMULATING BETTING SCENARIO")
    print("="*80 + "\n")
    
    # Bet 1: Normal bet
    print("Bet 1: Proposed stake $500")
    is_valid, stake, reason = rm.validate_bet_size(500)
    print(f"  Valid: {is_valid}, Stake: ${stake:.0f}, Reason: {reason}")
    
    if is_valid:
        rm.open_position(stake)
        print(f"  ✅ Position opened")
    
    # Bet 2: Another bet
    print("\nBet 2: Proposed stake $600")
    is_valid, stake, reason = rm.validate_bet_size(600)
    print(f"  Valid: {is_valid}, Stake: ${stake:.0f}, Reason: {reason}")
    
    if is_valid:
        rm.open_position(stake)
        print(f"  ✅ Position opened")
    
    # Show status with open positions
    rm.print_status()
    
    # Simulate losing bet
    print("\n" + "="*80)
    print("💸 SIMULATING LOSSES")
    print("="*80 + "\n")
    
    print("Bet 1 settles: LOSS of $500")
    rm.close_position(stake=500, profit=-500)
    
    print("Bet 2 settles: LOSS of $600")
    rm.close_position(stake=600, profit=-600)
    
    rm.print_status()
    
    # Try to bet more (should be limited due to daily loss)
    print("\n" + "="*80)
    print("🚨 TESTING DAILY LOSS LIMIT")
    print("="*80 + "\n")
    
    print("Bet 3: Proposed stake $500")
    is_valid, stake, reason = rm.validate_bet_size(500)
    print(f"  Valid: {is_valid}, Stake: ${stake:.0f}, Reason: {reason}")
    
    checks = rm.check_limits()
    if not checks['daily_loss_ok']:
        print("\n  🚨 DAILY LOSS LIMIT HIT - CANNOT BET")
        print(f"     Lost: ${rm.state.daily_loss:,.0f}")
        print(f"     Limit: ${rm.state.daily_start_bankroll * rm.limits.daily_loss_limit_pct:,.0f}")
    
    # Simulate circuit breaker
    print("\n" + "="*80)
    print("🚨 TESTING CIRCUIT BREAKER (30% Drawdown)")
    print("="*80 + "\n")
    
    # Simulate major loss
    print("Simulating catastrophic loss...")
    rm.update_bankroll(7000)  # Down 30% from peak
    
    rm.print_status()
    
    checks = rm.check_limits()
    if not checks['drawdown_ok']:
        print("\n🚨🚨🚨 CIRCUIT BREAKER TRIGGERED 🚨🚨🚨")
        print("     STOP ALL BETTING")
        print("     REVIEW SYSTEM")
        print("     DO NOT RESUME WITHOUT ANALYSIS")
    
    print("\n" + "="*80)
    print("✅ RISK MANAGEMENT SYSTEM COMPLETE")
    print("="*80)


class AdaptiveKellyManager:
    """
    Adjust Kelly fraction based on performance and risk state
    """
    
    def __init__(
        self,
        base_kelly_fraction: float = 0.25,
        reduce_on_losing_streak: int = 5,
        reduce_on_drawdown: float = 0.15
    ):
        """
        Args:
            base_kelly_fraction: Normal Kelly fraction
            reduce_on_losing_streak: Reduce after N losses
            reduce_on_drawdown: Reduce when drawdown exceeds this
        """
        self.base_kelly = base_kelly_fraction
        self.reduce_on_streak = reduce_on_losing_streak
        self.reduce_on_dd = reduce_on_drawdown
        
        self.current_streak = 0
        self.last_results: List[str] = []
    
    def record_result(self, outcome: str):
        """Record bet outcome"""
        self.last_results.append(outcome)
        
        if outcome == "LOSS":
            self.current_streak += 1
        else:
            self.current_streak = 0
    
    def get_adjusted_kelly(self, current_drawdown: float) -> float:
        """
        Get Kelly fraction adjusted for current risk state
        
        Args:
            current_drawdown: Current drawdown %
            
        Returns:
            Adjusted Kelly fraction
        """
        kelly = self.base_kelly
        
        # Reduce on losing streak
        if self.current_streak >= self.reduce_on_streak:
            kelly *= 0.5
            print(f"⚠️ Kelly reduced 50% due to {self.current_streak}-bet losing streak")
        
        # Reduce on drawdown
        if current_drawdown >= self.reduce_on_dd:
            kelly *= 0.5
            print(f"⚠️ Kelly reduced 50% due to {current_drawdown:.1%} drawdown")
        
        # Further reduce on severe drawdown
        if current_drawdown >= self.reduce_on_dd * 1.5:
            kelly *= 0.5
            print(f"🚨 Kelly reduced 75% total due to severe drawdown")
        
        return kelly


def example_adaptive_kelly():
    """Example: Adaptive Kelly in action"""
    print("\n" + "="*80)
    print("🧠 ADAPTIVE KELLY MANAGER")
    print("="*80 + "\n")
    
    akm = AdaptiveKellyManager(base_kelly_fraction=0.25)
    
    print(f"Base Kelly: {akm.base_kelly:.0%}\n")
    
    # Normal state
    kelly = akm.get_adjusted_kelly(current_drawdown=0.05)
    print(f"Normal state (5% DD): Kelly = {kelly:.0%}\n")
    
    # Simulate losing streak
    print("Simulating 5-bet losing streak...")
    for i in range(5):
        akm.record_result("LOSS")
    
    kelly = akm.get_adjusted_kelly(current_drawdown=0.10)
    print(f"After 5 losses (10% DD): Kelly = {kelly:.0%}\n")
    
    # Win breaks streak
    print("Win breaks streak...")
    akm.record_result("WIN")
    kelly = akm.get_adjusted_kelly(current_drawdown=0.08)
    print(f"After win (8% DD): Kelly = {kelly:.0%}\n")
    
    # Severe drawdown
    print("Severe drawdown (25%)...")
    kelly = akm.get_adjusted_kelly(current_drawdown=0.25)
    print(f"Severe DD (25%): Kelly = {kelly:.0%}\n")
    
    print("="*80)
    print("✅ ADAPTIVE KELLY PROTECTS BANKROLL")
    print("="*80)


if __name__ == "__main__":
    example_usage()
    print("\n")
    example_adaptive_kelly()

