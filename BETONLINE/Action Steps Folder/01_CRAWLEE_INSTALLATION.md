# Step 1: Crawlee Installation & Setup

**Objective:** Install and configure Crawlee for 5-second BetOnline scraping  
**Duration:** 15 minutes  
**Output:** Working Crawlee scraper with optimal configuration

---

## 1.1 Choose Your Stack (Python or JavaScript)

### Option A: Python (Recommended if ML backend is Python)

```bash
# Install Crawlee for Python
pip install 'crawlee[all]'

# Install Playwright browser
playwright install chromium

# Optional: Install additional tools
pip install beautifulsoup4 lxml orjson redis
```

---

### Option B: JavaScript/TypeScript (If using Node.js)

```bash
# Install Crawlee for JavaScript
npm install crawlee playwright

# Install browser
npx playwright install chromium

# Optional: TypeScript
npm install -D typescript @types/node
```

**For this guide, we'll use Python** (matches your ML backend stack)

---

## 1.2 Verify Installation (2 minutes)

**Create `test_crawlee.py`:**

```python
"""
Test Crawlee installation
"""

from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext

async def test_crawlee():
    print("🧪 Testing Crawlee installation...")
    
    crawler = PlaywrightCrawler(
        max_requests_per_crawl=1,
        headless=True
    )
    
    @crawler.router.default_handler
    async def handler(context: PlaywrightCrawlingContext) -> None:
        title = await context.page.title()
        context.log.info(f"✅ Crawlee working! Page title: {title}")
        
        await context.push_data({
            'url': context.request.url,
            'title': title
        })
    
    # Test with simple page
    await crawler.run(['https://example.com'])
    
    # Get data
    data = await crawler.get_data()
    print(f"✅ Crawlee test passed! Extracted {len(data.items)} item(s)")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_crawlee())
```

**Run:**
```bash
python test_crawlee.py
```

**Expected output:**
```
🧪 Testing Crawlee installation...
[INFO] Crawlee: Processing https://example.com ...
[INFO] Crawlee: ✅ Crawlee working! Page title: Example Domain
✅ Crawlee test passed! Extracted 1 item(s)
```

---

## 1.3 Configure for Production (5 minutes)

**Create `config/crawlee_config.py`:**

```python
"""
Crawlee Configuration - Optimized for BetOnline 5-second scraping
"""

# ============================================
# BROWSER CONFIGURATION
# ============================================

# Keep browser alive (CRITICAL for speed!)
PERSISTENT_BROWSER = True

# Browser type
BROWSER_TYPE = 'chromium'  # Fastest, most compatible

# Headless mode (production)
HEADLESS = True  # False for debugging

# Browser launch options (speed optimized)
BROWSER_LAUNCH_OPTIONS = {
    'headless': HEADLESS,
    'args': [
        '--disable-dev-shm-usage',
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-gpu',
        '--disable-extensions',
        '--disable-images',  # Faster loading
        '--blink-settings=imagesEnabled=false',  # No images
    ]
}

# ============================================
# SCRAPING CONFIGURATION
# ============================================

# Scrape interval
SCRAPE_INTERVAL_SECONDS = 5  # 5-second intervals

# Request timeout
REQUEST_TIMEOUT_SECONDS = 3  # Fail fast

# Max retries
MAX_RETRIES = 2

# ============================================
# PERFORMANCE OPTIMIZATION
# ============================================

# Navigation timeout (critical for speed!)
NAVIGATION_TIMEOUT_MS = 2000  # 2 seconds max

# Wait for selector timeout
SELECTOR_TIMEOUT_MS = 1000  # 1 second max

# Reuse browser context
BROWSER_CONTEXT_REUSE = True

# Connection pooling
HTTP_POOL_CONNECTIONS = 5
HTTP_POOL_MAXSIZE = 10

# ============================================
# ANTI-DETECTION (For internal use)
# ============================================

# User agent rotation
ROTATE_USER_AGENTS = True

# Fingerprint generation
USE_FINGERPRINTS = True

# Proxy (optional, if needed)
USE_PROXIES = False
PROXY_LIST = []

# ============================================
# DATA STORAGE
# ============================================

# Store scraped data (optional)
STORE_RAW_DATA = False  # Set to True for debugging

# Storage directory
STORAGE_DIR = './storage/betonline'

# ============================================
# ERROR HANDLING
# ============================================

# Continue on error
CONTINUE_ON_ERROR = True

# Alert threshold
ALERT_AFTER_N_FAILURES = 3

# Fallback behavior
USE_CACHED_ON_FAILURE = True
```

---

## 1.4 Create Base Scraper Class (5 minutes)

**Create `services/betonline_scraper.py`:**

```python
"""
BetOnline Scraper - Base Implementation
Optimized for 5-second intervals with persistent browser
"""

import asyncio
import time
from typing import Optional, Dict, List
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
from crawlee.fingerprint_suite import FingerprintSuite
from playwright.async_api import Browser, Page
from config.crawlee_config import (
    BROWSER_LAUNCH_OPTIONS,
    NAVIGATION_TIMEOUT_MS,
    SELECTOR_TIMEOUT_MS,
    SCRAPE_INTERVAL_SECONDS
)

class BetOnlineScraper:
    """
    High-speed BetOnline.ag scraper
    Optimized for 5-second polling with persistent browser
    """
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.crawler: Optional[PlaywrightCrawler] = None
        
        # Performance tracking
        self.scrape_count = 0
        self.total_scrape_time = 0.0
        self.last_scrape_time = 0.0
        
        # Cache last known good selectors
        self.selector_cache = {}
        
        # Last scraped data (for delta detection)
        self.last_odds_data = {}
    
    async def initialize(self):
        """
        Initialize scraper with persistent browser
        Called once at startup
        """
        print("🚀 Initializing BetOnline scraper...")
        
        # Create Crawlee crawler
        self.crawler = PlaywrightCrawler(
            max_requests_per_crawl=1,
            headless=BROWSER_LAUNCH_OPTIONS['headless'],
            browser_type='chromium',
            
            # Stealth configuration
            fingerprint_suite=FingerprintSuite(),
            
            # Performance settings
            max_request_retries=2,
            request_handler_timeout_secs=5,
        )
        
        print("✅ Crawlee initialized")
        print(f"   Browser: chromium (headless={BROWSER_LAUNCH_OPTIONS['headless']})")
        print(f"   Scrape interval: {SCRAPE_INTERVAL_SECONDS}s")
    
    async def start_polling(self):
        """
        Main polling loop - scrapes every 5 seconds
        """
        print("\n⚡ Starting 5-second polling loop...")
        print("   Target: <1000ms per scrape")
        print()
        
        while True:
            try:
                start_time = time.time()
                
                # Scrape BetOnline
                odds_data = await self.scrape_odds()
                
                # Performance tracking
                elapsed = time.time() - start_time
                self.total_scrape_time += elapsed
                self.scrape_count += 1
                self.last_scrape_time = elapsed
                
                avg_time = self.total_scrape_time / self.scrape_count
                
                print(f"✅ Scrape #{self.scrape_count}: "
                      f"{len(odds_data)} games, "
                      f"{elapsed*1000:.0f}ms "
                      f"(avg: {avg_time*1000:.0f}ms)")
                
                if elapsed > 1.0:
                    print(f"⚠️  Slow scrape! {elapsed*1000:.0f}ms > 1000ms target")
                
                # Detect changes
                changes = self.detect_changes(odds_data)
                if changes:
                    print(f"   📊 {len(changes)} odds changed")
                    await self.emit_changes(changes)
                
                # Wait for next cycle
                wait_time = max(0, SCRAPE_INTERVAL_SECONDS - elapsed)
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                print(f"❌ Scraping error: {e}")
                await asyncio.sleep(5)
    
    async def scrape_odds(self) -> List[Dict]:
        """
        Scrape current NBA odds from BetOnline
        Target: <1000ms
        """
        odds_data = []
        
        @self.crawler.router.default_handler
        async def handler(context: PlaywrightCrawlingContext) -> None:
            # Extract odds (implementation in next step)
            data = await self._extract_odds_from_page(context.page)
            odds_data.extend(data)
        
        # Run scraper
        await self.crawler.run(['https://www.betonline.ag/sportsbook/basketball/nba'])
        
        return odds_data
    
    async def _extract_odds_from_page(self, page: Page) -> List[Dict]:
        """
        Extract odds from BetOnline page
        Placeholder - implemented in Step 02
        """
        # Wait for odds to load
        await page.wait_for_selector('.game-container', timeout=SELECTOR_TIMEOUT_MS)
        
        # Extract data
        odds = await page.evaluate('''() => {
            // Placeholder - actual selectors in Step 02
            return [{
                game_id: 'test',
                spread: -7.5,
                total: 215.5
            }];
        }''')
        
        return odds
    
    def detect_changes(self, current_odds: List[Dict]) -> List[Dict]:
        """
        Detect what changed since last scrape
        Returns only games with changed odds
        """
        changes = []
        
        for game in current_odds:
            game_id = game.get('game_id')
            
            if game_id not in self.last_odds_data:
                # New game
                changes.append(game)
            elif game != self.last_odds_data[game_id]:
                # Odds changed
                changes.append(game)
        
        # Update cache
        self.last_odds_data = {g['game_id']: g for g in current_odds}
        
        return changes
    
    async def emit_changes(self, changes: List[Dict]):
        """
        Emit changed odds to WebSocket
        Override in production
        """
        # Placeholder - WebSocket integration in Step 05
        pass
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.browser:
            await self.browser.close()
```

**Key Features:**
- ✅ Persistent browser (reuse across scrapes)
- ✅ Change detection (only emit updates)
- ✅ Performance tracking
- ✅ Error handling

---

## 1.5 Test BetOnline Connection (3 minutes)

**Create `test_betonline.py`:**

```python
"""
Test BetOnline scraping
"""

import asyncio
from services.betonline_scraper import BetOnlineScraper

async def test_betonline():
    scraper = BetOnlineScraper()
    await scraper.initialize()
    
    # Single scrape test
    print("🧪 Testing BetOnline scrape...")
    odds = await scraper.scrape_odds()
    
    print(f"✅ Extracted odds for {len(odds)} games")
    
    if odds:
        print(f"\nSample odds:")
        print(odds[0])
    
    await scraper.cleanup()

if __name__ == "__main__":
    asyncio.run(test_betonline())
```

**Run:**
```bash
python test_betonline.py
```

---

## ✅ Validation Checklist

- [ ] Crawlee installed (`pip install crawlee[all]`)
- [ ] Playwright browser installed (`playwright install chromium`)
- [ ] Test script runs successfully
- [ ] Can access BetOnline.ag
- [ ] Browser launches in <2 seconds
- [ ] Configuration file created
- [ ] Base scraper class created

---

## 🚀 Performance Expectations

After setup, you should achieve:

| Metric | Target | Typical |
|--------|--------|---------|
| First browser launch | <3000ms | ~2000ms |
| Subsequent scrapes | <1000ms | ~500ms |
| Parse time | <100ms | ~50ms |

**Next Step:** Proceed to Step 02 to build the actual BetOnline scraper with selectors

---

*Action Step 1 of 5 - Crawlee Installation*

