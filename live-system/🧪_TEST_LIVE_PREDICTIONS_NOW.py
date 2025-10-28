#!/usr/bin/env python3
"""
TEST LIVE PREDICTIONS - RIGHT NOW
Test the full prediction pipeline on live games
"""

import sys
sys.path.append('.')

from live_trading_engine import LiveTradingEngine
from datetime import datetime

print("=" * 80)
print("🧪 TESTING LIVE PREDICTIONS ON REAL GAMES")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")

# Initialize engine
print("Initializing Live Trading Engine...")
engine = LiveTradingEngine()

print("\n" + "=" * 80)
print("🔍 SCANNING FOR PREDICTIONS...")
print("=" * 80)

# Scan for opportunities (now returns ALL predictions)
predictions = engine.scan_live_opportunities()

print("\n" + "=" * 80)
print(f"📊 RESULTS: {len(predictions)} PREDICTIONS")
print("=" * 80)

if predictions:
    for i, pred in enumerate(predictions, 1):
        print(f"\n🎯 PREDICTION #{i}")
        print("-" * 80)
        print(f"Game: {pred.get('away_team')} @ {pred.get('home_team')}")
        print(f"Game ID: {pred.get('game_id')}")
        print(f"Period: Q{pred.get('period')} | Clock: {pred.get('clock')}")
        print(f"Current Score: {pred.get('away_score')}-{pred.get('home_score')}")
        print()
        print(f"🤖 ML PREDICTION:")
        print(f"   Final Margin: {pred.get('ml_prediction', 0):.1f}")
        print(f"   Confidence: [{pred.get('ml_lower', 0):.1f}, {pred.get('ml_upper', 0):.1f}]")
        print()
        print(f"💰 MARKET:")
        print(f"   Spread: {pred.get('market_spread', 0):.1f}")
        print(f"   Source: {pred.get('line_source', 'UNKNOWN')}")
        print()
        print(f"📊 EDGE:")
        print(f"   Edge: {pred.get('edge', 0):.1f} points")
        print(f"   Is Opportunity: {pred.get('is_opportunity', False)}")
        print()
        print(f"✅ 33 FEATURES EXTRACTED: {pred.get('features_extracted', 'UNKNOWN')}")
        print("-" * 80)
else:
    print("\n❌ NO PREDICTIONS MADE")
    print("\nDEBUG INFO:")
    print("  - Check if games are at Q2 6:00 or later")
    print("  - Check if model is loaded")
    print("  - Check daemon logs for errors")

print("\n" + "=" * 80)
print("🧪 TEST COMPLETE")
print("=" * 80)

