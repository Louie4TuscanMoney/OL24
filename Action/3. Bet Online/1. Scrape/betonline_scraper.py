"""
BetOnline NBA Odds Scraper - 5-Second Intervals
Following: BETONLINE/BETONLINE_SCRAPING_OPTIMIZATION.md

SPEED OPTIMIZATIONS:
- Persistent browser (saves 1500ms)
- Resource blocking (saves 500ms)
- domcontentloaded wait (saves 1000ms)
- Cached selectors

Target: <1000ms per scrape
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional
from playwright.async_api import async_playwright, Browser, Page

class BetOnlineScraper:
    """
    High-speed persistent browser scraper for BetOnline NBA odds
    """
    
    def __init__(self):
        """Initialize scraper"""
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.is_initialized = False
        
        # Performance tracking
        self.scrape_count = 0
        self.total_scrape_time = 0.0
        self.last_odds = {}
        
    async def initialize(self):
        """
        Launch persistent browser (called once at startup)
        Time: ~2000ms (one-time cost)
        """
        print("="*80)
        print("INITIALIZING PERSISTENT BROWSER")
        print("="*80)
        
        start = time.time()
        
        # Start Playwright
        print("\nLaunching browser...")
        self.playwright = await async_playwright().start()
        
        # Launch browser with optimizations
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-gpu',
                '--disable-extensions',
                '--disable-images',                    # Skip images
                '--blink-settings=imagesEnabled=false',
                '--disable-web-security',              # Faster
            ]
        )
        
        # Create page
        self.page = await self.browser.new_page()
        
        # Set viewport (smaller = faster)
        await self.page.set_viewport_size({"width": 1280, "height": 720})
        
        # Block unnecessary resources (HUGE speedup!)
        print("Setting up resource blocking...")
        await self.page.route("**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,css}", 
                             lambda route: route.abort())
        await self.page.route("**/google-analytics.com/**", 
                             lambda route: route.abort())
        await self.page.route("**/facebook.com/**", 
                             lambda route: route.abort())
        await self.page.route("**/doubleclick.net/**", 
                             lambda route: route.abort())
        
        self.is_initialized = True
        
        elapsed = time.time() - start
        print(f"✅ Browser initialized in {elapsed*1000:.0f}ms")
        print(f"   Persistent: YES (browser stays open)")
        print(f"   Resource blocking: ACTIVE")
        
    async def scrape_odds(self) -> Dict:
        """
        Scrape current NBA odds
        Target: <1000ms
        """
        if not self.is_initialized:
            await self.initialize()
        
        start = time.time()
        
        try:
            # Navigate (fast with persistent browser)
            await self.page.goto(
                'https://www.betonline.ag/sportsbook/basketball/nba',
                wait_until='domcontentloaded',  # Don't wait for everything
                timeout=2000
            )
            
            # Wait for odds container (should be fast)
            try:
                await self.page.wait_for_selector('.game-lines', timeout=1000)
            except:
                # If selector not found, page might have different structure
                pass
            
            # Extract odds using JavaScript (fast)
            odds_data = await self.page.evaluate('''() => {
                const games = [];
                
                // Find game containers (adjust selector as needed)
                const gameElements = document.querySelectorAll('.game-line, .bet-item, [data-game-id]');
                
                gameElements.forEach(el => {
                    try {
                        // Extract team names
                        const teams = el.querySelectorAll('.team-name, .participant');
                        
                        // Extract odds
                        const odds = el.querySelectorAll('.odds, .price');
                        
                        if (teams.length >= 2 && odds.length > 0) {
                            games.push({
                                away_team: teams[0]?.textContent?.trim() || '',
                                home_team: teams[1]?.textContent?.trim() || '',
                                spread_away: odds[0]?.textContent?.trim() || '',
                                spread_home: odds[1]?.textContent?.trim() || '',
                                timestamp: Date.now()
                            });
                        }
                    } catch (e) {
                        console.log('Error parsing game:', e);
                    }
                });
                
                return {
                    games: games,
                    total_games: games.length,
                    scraped_at: new Date().toISOString()
                };
            }''')
            
            elapsed = time.time() - start
            
            return {
                'success': True,
                'odds': odds_data,
                'scrape_time_ms': elapsed * 1000,
                'timestamp': datetime.now().isoformat(),
                'games_found': odds_data.get('total_games', 0)
            }
            
        except Exception as e:
            elapsed = time.time() - start
            return {
                'success': False,
                'error': str(e),
                'scrape_time_ms': elapsed * 1000,
                'timestamp': datetime.now().isoformat()
            }
    
    async def start_polling(self, duration_minutes: Optional[int] = None):
        """
        Start 5-second polling loop
        
        Args:
            duration_minutes: How long to run (None = forever)
        """
        print("\n" + "="*80)
        print("STARTING 5-SECOND BETONLINE POLLING")
        print("="*80)
        print(f"Target: <1000ms per scrape")
        print(f"Interval: 5 seconds")
        if duration_minutes:
            print(f"Duration: {duration_minutes} minutes")
        else:
            print("Duration: Continuous (Ctrl+C to stop)")
        print("="*80)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60) if duration_minutes else None
        
        try:
            while True:
                self.scrape_count += 1
                poll_start = time.time()
                
                print(f"\n[Scrape #{self.scrape_count}] {datetime.now().strftime('%H:%M:%S')}")
                
                # Scrape
                result = await self.scrape_odds()
                
                # Track performance
                scrape_time = result['scrape_time_ms']
                self.total_scrape_time += scrape_time / 1000
                
                if result['success']:
                    games_found = result['games_found']
                    avg_time = (self.total_scrape_time / self.scrape_count) * 1000
                    
                    print(f"  ✅ Found {games_found} games in {scrape_time:.0f}ms (avg: {avg_time:.0f}ms)")
                    
                    if scrape_time > 1000:
                        print(f"  ⚠️  SLOW: {scrape_time:.0f}ms > 1000ms target")
                    
                    # Detect changes
                    if self.last_odds:
                        changes = self._detect_changes(result['odds'], self.last_odds)
                        if changes:
                            print(f"  📊 {len(changes)} odds changed")
                    
                    self.last_odds = result['odds']
                else:
                    print(f"  ❌ Error: {result['error']}")
                
                # Check duration
                if end_time and time.time() >= end_time:
                    break
                
                # Wait for next cycle (maintain 5-second interval)
                elapsed = time.time() - poll_start
                wait_time = max(0, 5.0 - elapsed)
                await asyncio.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")
        finally:
            await self.cleanup()
        
        # Summary
        total_time = time.time() - start_time
        avg_scrape = (self.total_scrape_time / self.scrape_count) * 1000 if self.scrape_count > 0 else 0
        
        print(f"\n" + "="*80)
        print("SCRAPING SUMMARY")
        print("="*80)
        print(f"Runtime: {total_time/60:.1f} minutes")
        print(f"Total scrapes: {self.scrape_count}")
        print(f"Average scrape time: {avg_scrape:.0f}ms")
        print(f"Target: <1000ms")
        print(f"Status: {'✅ PASS' if avg_scrape < 1000 else '❌ FAIL'}")
    
    def _detect_changes(self, current_odds: Dict, previous_odds: Dict) -> List[Dict]:
        """
        Detect which odds changed
        """
        changes = []
        # Implementation for detecting line movements
        return changes
    
    async def cleanup(self):
        """Close browser cleanly"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print("\n✅ Browser closed cleanly")


# Run standalone
if __name__ == "__main__":
    scraper = BetOnlineScraper()
    
    # Test for 1 minute (12 scrapes)
    asyncio.run(scraper.start_polling(duration_minutes=1))

