#!/usr/bin/env python3
"""
🕷️ CRAWLEE SCRAPER EXAMPLE

Example showing how to use the Crawlee-based BetOnline scraper.

Status: ❌ Currently returns 0 games (needs selector fixes)
Fix: Follow documentation/SOLUTIONS_GUIDE.md Section 1
"""

import sys
import asyncio
sys.path.append('..')

from scrapers.crawlee_betonline_scraper import get_crawlee_betonline_odds


async def main():
    """Crawlee scraper example"""
    
    print("🕷️ BETONLINE CRAWLEE SCRAPER")
    print("=" * 80)
    print()
    
    print("📊 Launching Crawlee scraper...")
    print("   - Using Playwright browser automation")
    print("   - With stealth plugin")
    print("   - Headless mode")
    print()
    
    # Get odds using Crawlee
    try:
        odds = await get_crawlee_betonline_odds()
        
        if odds and len(odds) > 0:
            print(f"✅ SUCCESS: Found {len(odds)} games!")
            print("=" * 80)
            
            for i, game in enumerate(odds, 1):
                print(f"\n🏀 Game {i}:")
                print(f"   {game.get('away_team', 'N/A')} @ {game.get('home_team', 'N/A')}")
                print(f"   Spread: {game.get('spread', 'N/A')}")
                print(f"   Total: {game.get('total', 'N/A')}")
                print(f"   Timestamp: {game.get('timestamp', 'N/A')}")
        
        else:
            print("❌ FAILED: Scraper returned 0 games")
            print()
            print("🔍 ROOT CAUSE:")
            print("   The scraper is using PLACEHOLDER CSS selectors")
            print("   These selectors don't match BetOnline's actual HTML")
            print()
            print("🔧 HOW TO FIX:")
            print("   1. Open https://www.betonline.ag/sportsbook/basketball/nba")
            print("   2. Press F12 (Developer Tools)")
            print("   3. Inspect game elements")
            print("   4. Find real CSS selectors")
            print("   5. Update scrapers/crawlee_betonline_scraper.py")
            print()
            print("📖 Detailed Guide:")
            print("   documentation/SOLUTIONS_GUIDE.md - Section 1")
            print()
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print()
        print("🔍 COMMON ISSUES:")
        print("   1. Playwright not installed: pip install playwright")
        print("   2. Chromium not installed: python -m playwright install chromium")
        print("   3. Anti-bot blocking: See documentation/SOLUTIONS_GUIDE.md Section 2")
        print()
    
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())

