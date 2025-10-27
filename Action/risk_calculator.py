#!/usr/bin/env python3
"""
💰 RISK CALCULATOR - Kelly Criterion Bet Sizing

Calculates optimal bet size based on:
- Predicted edge
- Model confidence
- Current bankroll
- Risk limits

CONSERVATIVE SETTINGS for Week 1:
- Use 50% Kelly (half-Kelly for safety)
- Max bet: $750 (15% of $5,000 bankroll)
- Min edge: 2 points (be selective)
"""

import numpy as np

class RiskCalculator:
    """Calculate bet sizes using Kelly criterion"""
    
    def __init__(self, bankroll=5000, max_bet_pct=0.15, kelly_fraction=0.5):
        """
        Initialize risk calculator
        
        Args:
            bankroll: Current bankroll ($)
            max_bet_pct: Max % of bankroll per bet (0.15 = 15%)
            kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly)
        """
        self.bankroll = bankroll
        self.max_bet_pct = max_bet_pct
        self.kelly_fraction = kelly_fraction
        self.max_bet = bankroll * max_bet_pct
        
        print(f"💰 Risk Calculator Initialized")
        print(f"   Bankroll: ${bankroll:,}")
        print(f"   Max bet: ${self.max_bet:.0f} ({max_bet_pct*100:.0f}%)")
        print(f"   Kelly fraction: {kelly_fraction*100:.0f}% (conservative)")
    
    def calculate_bet(self, predicted_edge, confidence, odds=-110):
        """
        Calculate bet size
        
        Args:
            predicted_edge: Points edge (e.g., 3.5)
            confidence: 'HIGH', 'MEDIUM', 'LOW'
            odds: Betting odds (e.g., -110)
        
        Returns:
            {
                'bet_size': Dollar amount,
                'kelly_pct': Kelly percentage,
                'recommendation': 'BET' or 'SKIP',
                'reason': Why we bet or skip
            }
        """
        # Confidence multipliers
        confidence_mult = {
            'HIGH': 1.0,
            'MEDIUM': 0.6,
            'LOW': 0.3
        }
        
        mult = confidence_mult.get(confidence, 0.5)
        
        # Edge threshold - must be at least 2 points
        edge = abs(predicted_edge)
        
        if edge < 2.0:
            return {
                'bet_size': 0,
                'kelly_pct': 0,
                'recommendation': 'SKIP',
                'reason': f'Edge too small ({edge:.1f} < 2.0 pts)'
            }
        
        # Convert edge to Kelly fraction
        # Rule of thumb: 1 point edge ≈ 5% Kelly
        kelly_raw = (edge / 20.0) * mult
        
        # Apply Kelly fraction (e.g., 50% of Kelly)
        kelly_pct = kelly_raw * self.kelly_fraction
        
        # Cap at max bet percentage
        kelly_pct = min(kelly_pct, self.max_bet_pct)
        
        # Calculate dollar amount
        bet_size = self.bankroll * kelly_pct
        bet_size = min(bet_size, self.max_bet)
        
        # Round to nearest $10
        bet_size = round(bet_size / 10) * 10
        
        # Decision
        if bet_size < 50:
            return {
                'bet_size': 0,
                'kelly_pct': kelly_pct,
                'recommendation': 'SKIP',
                'reason': f'Bet too small (${bet_size:.0f} < $50 min)'
            }
        
        return {
            'bet_size': bet_size,
            'kelly_pct': kelly_pct,
            'recommendation': 'BET',
            'reason': f'{confidence} confidence, {edge:.1f} pt edge'
        }
    
    def update_bankroll(self, new_bankroll):
        """Update bankroll after wins/losses"""
        old_bankroll = self.bankroll
        self.bankroll = new_bankroll
        self.max_bet = new_bankroll * self.max_bet_pct
        
        change = new_bankroll - old_bankroll
        print(f"\n💰 Bankroll updated: ${old_bankroll:,.0f} → ${new_bankroll:,.0f} ({change:+,.0f})")
        print(f"   New max bet: ${self.max_bet:.0f}")


# Demo usage
if __name__ == "__main__":
    print("="*80)
    print("💰 RISK CALCULATOR - Demo")
    print("="*80)
    
    calc = RiskCalculator(bankroll=5000, max_bet_pct=0.15, kelly_fraction=0.5)
    
    print("\n" + "="*80)
    print("EXAMPLE SCENARIOS")
    print("="*80)
    
    scenarios = [
        {"edge": 4.5, "conf": "HIGH", "desc": "Strong edge, high confidence"},
        {"edge": 2.5, "conf": "MEDIUM", "desc": "Small edge, medium confidence"},
        {"edge": 1.5, "conf": "HIGH", "desc": "Tiny edge (should skip)"},
        {"edge": 6.0, "conf": "LOW", "desc": "Big edge but low confidence"},
    ]
    
    for i, s in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {s['desc']}")
        print(f"   Edge: {s['edge']:.1f} pts, Confidence: {s['conf']}")
        
        result = calc.calculate_bet(s['edge'], s['conf'])
        
        print(f"   → {result['recommendation']}: ${result['bet_size']:.0f}")
        print(f"   Reason: {result['reason']}")
    
    print("\n" + "="*80)
    print("✅ Risk calculator ready for Monday!")
    print("="*80)



