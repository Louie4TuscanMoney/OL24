"""
🎯 ONTORISK DEMONSTRATION

Purpose: Show OntoRisk working end-to-end with simulated betting
Author: Ontologic XYZ
Date: October 20, 2025

This demonstrates the COMPLETE system working.
"""

import numpy as np
import sys
sys.path.append('4. Risk')

from ontorisk_phase1_probability_calibration import ProbabilityCalibrator
from ontorisk_phase4_risk_management import RiskManager, AdaptiveKellyManager


def demo_complete_workflow():
    """
    Demonstrate complete OntoRisk workflow
    """
    print("\n" + "="*80)
    print("🔥 ONTORISK COMPLETE WORKFLOW DEMONSTRATION")
    print("="*80 + "\n")
    
    # Initialize components
    calibrator = ProbabilityCalibrator(mae=9.029)
    risk_manager = RiskManager(starting_bankroll=10000)
    kelly_manager = AdaptiveKellyManager()
    
    print("✅ System initialized")
    print(f"   Bankroll: $10,000")
    print(f"   Kelly Fraction: 25%")
    print(f"   MAE: 9.029")
    print()
    
    # Simulate 10 betting opportunities
    print("="*80)
    print("📊 SIMULATING 10 BETTING OPPORTUNITIES")
    print("="*80 + "\n")
    
    # Game scenarios (realistic NBA situations)
    scenarios = [
        {'pred': +2.5, 'spread': -3.5, 'actual': +4.0, 'home': 'LAL', 'away': 'BOS'},
        {'pred': -8.0, 'spread': -5.0, 'actual': -7.0, 'home': 'GSW', 'away': 'PHX'},
        {'pred': +6.0, 'spread': +1.0, 'actual': +8.0, 'home': 'MIA', 'away': 'DEN'},
        {'pred': -12.0, 'spread': -8.0, 'actual': -6.0, 'home': 'MIL', 'away': 'CLE'},
        {'pred': +0.5, 'spread': -2.0, 'actual': -1.0, 'home': 'DAL', 'away': 'PHI'},
        {'pred': +10.0, 'spread': +4.0, 'actual': +12.0, 'home': 'OKC', 'away': 'SAS'},
        {'pred': -5.5, 'spread': -10.0, 'actual': -4.0, 'home': 'BKN', 'away': 'ATL'},
        {'pred': +3.0, 'spread': -4.0, 'actual': +2.0, 'home': 'LAC', 'away': 'SAC'},
        {'pred': -15.0, 'spread': -9.0, 'actual': -18.0, 'home': 'BOS', 'away': 'DET'},
        {'pred': +8.0, 'spread': +2.0, 'actual': +5.0, 'home': 'NYK', 'away': 'WAS'},
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{'='*80}")
        print(f"GAME {i}: {scenario['away']} @ {scenario['home']}")
        print(f"{'='*80}\n")
        
        # Calculate probability
        prob = calibrator.calculate_probability(
            prediction=scenario['pred'],
            spread_line=scenario['spread'],
            home_team=scenario['home'],
            away_team=scenario['away']
        )
        
        edge = abs(scenario['pred'] - scenario['spread'])
        
        print(f"Our Prediction: {scenario['home']} {scenario['pred']:+.1f}")
        print(f"Market Spread: {scenario['home']} {scenario['spread']:+.1f}")
        print(f"Edge: {edge:.1f} points")
        print()
        
        # Check if we should bet
        if edge < 5.0:
            print("❌ SKIP: Edge < 5 points")
            print()
            continue
        
        if prob.p_win < 0.55:
            print("❌ SKIP: P(Win) < 55%")
            print()
            continue
        
        # Check risk limits
        checks = risk_manager.check_limits()
        if not checks['can_bet']:
            print("🚨 SKIP: Risk limits exceeded")
            print()
            continue
        
        # Calculate stake
        kelly_adj = kelly_manager.get_adjusted_kelly(risk_manager.state.current_drawdown)
        full_kelly = risk_manager.state.current_bankroll * kelly_adj * prob.kelly_edge
        is_valid, stake, reason = risk_manager.validate_bet_size(full_kelly)
        
        if not is_valid:
            print(f"❌ SKIP: {reason}")
            print()
            continue
        
        print(f"✅ BET RECOMMENDED: {prob.bet_line}")
        print(f"   P(Win): {prob.p_win:.1%}")
        print(f"   Kelly Edge: {prob.kelly_edge:.1%}")
        print(f"   Stake: ${stake:,.0f}")
        print()
        
        # Place bet
        risk_manager.open_position(stake)
        
        # Determine outcome
        actual = scenario['actual']
        if prob.bet_side == "OVER":
            win = actual >= scenario['spread']
        else:
            win = actual <= scenario['spread']
        
        if win:
            profit = stake * (100 / 110)
            outcome = "WIN"
            emoji = "✅"
        else:
            profit = -stake
            outcome = "LOSS"
            emoji = "❌"
        
        print(f"Actual Result: {scenario['home']} {actual:+.1f}")
        print(f"{emoji} {outcome}: {profit:+,.2f}")
        
        # Settle bet
        risk_manager.close_position(stake, profit)
        kelly_manager.record_result(outcome)
        
        print(f"Bankroll: ${risk_manager.state.current_bankroll:,.0f}")
        print()
    
    # Final summary
    print("="*80)
    print("📊 FINAL SUMMARY")
    print("="*80 + "\n")
    
    risk_manager.print_status()
    
    print("\n" + "="*80)
    print("✅ ONTORISK DEMONSTRATION COMPLETE")
    print("="*80)
    print("\n🎯 This demonstrates:")
    print("   ✅ Probability calibration (MAE → P(win))")
    print("   ✅ Kelly position sizing (optimal stakes)")
    print("   ✅ Risk management (limits enforced)")
    print("   ✅ Adaptive Kelly (reduces on losses)")
    print("   ✅ Complete workflow (end-to-end)")
    print("\n📋 Next: Week 2 - Real spreads → TRUE backtestperformance")
    print("="*80)


if __name__ == "__main__":
    demo_complete_workflow()

