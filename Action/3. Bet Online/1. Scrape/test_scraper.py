"""
Test BetOnline Scraper
Verify scraping works and meets 5-second target
"""

import asyncio
from betonline_scraper import BetOnlineScraper

async def test_single_scrape():
    """Test a single scrape"""
    print("="*80)
    print("BETONLINE SCRAPER - SINGLE SCRAPE TEST")
    print("="*80)
    
    scraper = BetOnlineScraper()
    
    # Initialize (one-time cost)
    await scraper.initialize()
    
    # Single scrape
    print("\nPerforming test scrape...")
    result = await scraper.scrape_odds()
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    if result['success']:
        print(f"✅ Success")
        print(f"   Scrape time: {result['scrape_time_ms']:.0f}ms")
        print(f"   Games found: {result['games_found']}")
        print(f"   Target: <1000ms")
        
        if result['scrape_time_ms'] < 1000:
            print(f"   ✅ PASS - Within target!")
        else:
            print(f"   ❌ FAIL - Too slow")
        
        # Show sample data
        if result['odds'].get('games'):
            print(f"\n   Sample game:")
            game = result['odds']['games'][0]
            print(f"   {game.get('away_team', 'N/A')} @ {game.get('home_team', 'N/A')}")
            print(f"   Spread: {game.get('spread_away', 'N/A')} / {game.get('spread_home', 'N/A')}")
    else:
        print(f"❌ Failed: {result['error']}")
    
    # Cleanup
    await scraper.cleanup()
    
    return result['success']

if __name__ == "__main__":
    success = asyncio.run(test_single_scrape())
    
    if success:
        print("\n" + "="*80)
        print("✅ SCRAPER TEST PASSED")
        print("="*80)
        print("\nNext: Run continuous polling")
        print("  python betonline_scraper.py")
    else:
        print("\n" + "="*80)
        print("❌ SCRAPER TEST FAILED")
        print("="*80)

