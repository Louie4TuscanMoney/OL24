#!/usr/bin/env python3
"""
🕷️ MULTI-METHOD SCRAPER EXAMPLE

Example showing the fallback chain: Crawlee → API → HTML → Synthetic

This demonstrates how the scraper tries multiple methods to get odds.

Status: ⚠️ Currently falls back to synthetic (all real methods need fixing)
"""

import sys
sys.path.append('..')

from scrapers.betonline_live_lines import BetOnlineScraper


def main():
    """Multi-method scraper with fallback chain"""
    
    print("🕷️ BETONLINE MULTI-METHOD SCRAPER")
    print("=" * 80)
    print()
    
    print("📊 Fallback Chain:")
    print("   1. 🕷️  Crawlee (Playwright + Stealth)")
    print("   2. 🌐 BetOnline API endpoints")
    print("   3. 📄 HTML scraping")
    print("   4. 🔮 Synthetic odds (fallback)")
    print()
    print("=" * 80)
    print()
    
    # Initialize scraper
    scraper = BetOnlineScraper()
    
    # Get lines (will try all methods)
    print("🔍 Attempting to scrape BetOnline...")
    print()
    
    lines = scraper.get_live_lines()
    
    if lines:
        print(f"✅ Found {len(lines)} games")
        print()
        
        # Check which method worked
        source = lines[0].get('source', 'Unknown')
        
        if 'Crawlee' in source:
            print("✅ Method: Crawlee scraper (BEST!)")
            print("   - Real-time odds")
            print("   - Browser automation")
            print("   - Stealth enabled")
        
        elif 'API' in source:
            print("✅ Method: BetOnline API")
            print("   - Direct API access")
            print("   - Fast and reliable")
            print("   - Structured data")
        
        elif 'HTML' in source:
            print("✅ Method: HTML scraping")
            print("   - Direct page scraping")
            print("   - Moderate speed")
            print("   - Requires parsing")
        
        elif 'synthetic' in source.lower():
            print("⚠️  Method: Synthetic odds (FALLBACK)")
            print("   - Generated odds (NOT REAL!)")
            print("   - All real methods failed")
            print("   - Needs fixing")
            print()
            print("🔧 TO GET REAL ODDS:")
            print("   Follow documentation/SOLUTIONS_GUIDE.md")
        
        print()
        print("=" * 80)
        print()
        
        # Display games
        for i, line in enumerate(lines[:3], 1):  # Show first 3 games
            print(f"🏀 Game {i}: {line['away_team']} @ {line['home_team']}")
            print(f"   Spread: {line.get('spread', 'N/A')}")
            print(f"   Total: {line.get('total', 'N/A')}")
            print(f"   ML: {line.get('moneyline_home', 'N/A')} / {line.get('moneyline_away', 'N/A')}")
            print()
        
        if len(lines) > 3:
            print(f"   ... and {len(lines) - 3} more games")
            print()
    
    else:
        print("❌ No games found")
        print("   All scraping methods failed")
        print()
    
    print("=" * 80)
    print()
    
    # Show what needs fixing
    if lines and 'synthetic' in lines[0]['source'].lower():
        print("🔍 WHAT'S NOT WORKING:")
        print()
        print("   1. ❌ Crawlee: Using placeholder selectors")
        print("      Fix: Inspect HTML, find real selectors")
        print("      Time: 1-2 hours")
        print()
        print("   2. ❌ API: Endpoints not public or need auth")
        print("      Fix: Reverse engineer API in Network tab")
        print("      Time: 2-4 hours (or may be impossible)")
        print()
        print("   3. ❌ HTML: 403 Forbidden (anti-bot)")
        print("      Fix: Better stealth or residential proxy")
        print("      Time: 1-2 hours")
        print()
        print("📖 Detailed Solutions:")
        print("   documentation/SOLUTIONS_GUIDE.md")
        print()


if __name__ == '__main__':
    main()

