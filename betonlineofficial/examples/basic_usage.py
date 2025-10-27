#!/usr/bin/env python3
"""
🕷️ BASIC BETONLINE SCRAPER USAGE

Simple example showing how to use the BetOnline scraper.

Status: ⚠️ Currently returns synthetic odds (scraper needs fixing)
Fix: Follow documentation/SOLUTIONS_GUIDE.md to get real odds
"""

import sys
sys.path.append('..')

from scrapers.betonline_live_lines import BetOnlineScraper
from utilities.implied_probability_calculator import ImpliedProbabilityCalculator


def main():
    """Basic scraper usage example"""
    
    print("🕷️ BETONLINE SCRAPER - BASIC USAGE")
    print("=" * 80)
    print()
    
    # Initialize scraper
    print("📊 Initializing scraper...")
    scraper = BetOnlineScraper()
    calc = ImpliedProbabilityCalculator()
    
    # Get live lines
    print("🔍 Fetching live lines...")
    lines = scraper.get_live_lines()
    
    print(f"\n✅ Found {len(lines)} live games:")
    print("=" * 80)
    
    # Display each game
    for i, line in enumerate(lines, 1):
        print(f"\n🏀 Game {i}: {line['away_team']} @ {line['home_team']}")
        print(f"   Game ID: {line['game_id']}")
        print(f"   Source: {line['source']}")
        print()
        
        # Spread
        spread = line.get('spread')
        if spread is not None:
            print(f"   📊 SPREAD: {spread:+.1f}")
            if 'spread_display' in line:
                print(f"      {line['spread_display']}")
        
        # Total
        total = line.get('total')
        if total is not None:
            print(f"   📊 TOTAL: {total:.1f}")
        
        # Moneyline
        home_ml = line.get('moneyline_home')
        away_ml = line.get('moneyline_away')
        
        if home_ml and away_ml:
            print(f"   📊 MONEYLINE:")
            print(f"      Home: {home_ml:+d}")
            print(f"      Away: {away_ml:+d}")
            
            # Calculate implied probabilities
            home_prob = calc.american_to_probability(home_ml)
            away_prob = calc.american_to_probability(away_ml)
            
            print(f"   💡 IMPLIED PROBABILITY:")
            print(f"      Home: {home_prob:.1%}")
            print(f"      Away: {away_prob:.1%}")
            
            # Calculate no-vig probabilities
            no_vig = calc.no_vig_probability(home_ml, away_ml)
            vig = calc.vig_percentage(home_ml, away_ml)
            
            print(f"   💡 NO-VIG PROBABILITY:")
            print(f"      Home: {no_vig['home']:.1%}")
            print(f"      Away: {no_vig['away']:.1%}")
            print(f"   💰 VIG: {vig:.2f}%")
        
        print()
    
    print("=" * 80)
    print()
    
    # Warning if using synthetic odds
    if lines and 'synthetic' in lines[0]['source'].lower():
        print("⚠️  WARNING: Using synthetic odds (scraper needs fixing)")
        print("📖 Follow documentation/SOLUTIONS_GUIDE.md to get real odds")
        print()


if __name__ == '__main__':
    main()

