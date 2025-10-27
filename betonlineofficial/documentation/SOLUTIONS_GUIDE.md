# 🔧 SOLUTIONS GUIDE - How to Fix Everything

Step-by-step guide to fix the BetOnline scraper and get it production-ready.

---

## 🎯 OVERVIEW

This guide will walk you through fixing the three main issues:

1. **CSS Selectors** (80% of problem) - 1-2 hours
2. **Anti-Bot Protection** (15% of problem) - 1-2 hours  
3. **Dynamic Content** (5% of problem) - 30-60 min

**Total Time: 3-5 hours of focused work**

---

## 🚀 SOLUTION #1: FIX CSS SELECTORS

**Time: 1-2 hours**  
**Difficulty: Easy-Medium**  
**Impact: Will make scraper work!**

### Step 1.1: Inspect BetOnline HTML (30 min)

**Open BetOnline:**
```bash
# Open in your browser
open https://www.betonline.ag/sportsbook/basketball/nba
```

**Open Developer Tools:**
- Press `F12` (Windows/Linux)
- Press `Cmd+Option+I` (Mac)
- Or right-click → "Inspect"

**Find Game Elements:**

1. **Hover over a game** in the live games section
2. **Right-click → Inspect Element**
3. **Find the container div** that holds the entire game
4. **Note the class names**

**Example of what you might find:**
```html
<div class="event-row" data-event-id="12345">
    <div class="event-teams">
        <span class="team-name-1">Los Angeles Lakers</span>
        <span class="team-name-2">Golden State Warriors</span>
    </div>
    <div class="event-markets">
        <span class="spread-home">-3.5</span>
        <span class="spread-odds">-110</span>
        <span class="total-value">225.5</span>
    </div>
</div>
```

**Document the structure:**
```
Game Container: .event-row
Home Team: .team-name-1
Away Team: .team-name-2
Spread: .spread-home
Total: .total-value
Moneyline: (find this too)
```

### Step 1.2: Test Selectors in Browser (15 min)

**Open Browser Console:**
- Press `F12` → Console tab

**Test your selectors:**
```javascript
// Test game container
document.querySelectorAll('.event-row')
// Should return array of game elements

// Test team names
document.querySelectorAll('.team-name-1')
// Should return array of team names

// Test spread
document.querySelectorAll('.spread-home')
// Should return array of spreads
```

**Verify data:**
```javascript
// Get first game
const game = document.querySelector('.event-row');

// Get teams
const home = game.querySelector('.team-name-1').textContent;
const away = game.querySelector('.team-name-2').textContent;

console.log(`${away} @ ${home}`);
// Should print something like: "Lakers @ Warriors"
```

**If selectors don't work:**
- Try different selectors
- Check for typos
- Look for dynamic IDs
- Try parent-child relationships

### Step 1.3: Update Scraper Code (30 min)

**Open:** `scrapers/crawlee_betonline_scraper.py`

**Find this section (around line 65):**
```python
self.selectors = {
    'game_list': {
        'container': '.game-container, .event-container, [data-game-id]',  # ❌ OLD
        'game_card': '.game-line, .event-line',  # ❌ OLD
        'home_team': '.home-team, .team-home',  # ❌ OLD
        'away_team': '.away-team, .team-away',  # ❌ OLD
        # ... more old selectors
    }
}
```

**Replace with REAL selectors:**
```python
self.selectors = {
    'game_list': {
        'container': '.event-row',  # ✅ REAL selector from inspection
        'game_card': '.event-row',  # ✅ Same as container
        'home_team': '.team-name-1',  # ✅ REAL selector
        'away_team': '.team-name-2',  # ✅ REAL selector
        'game_time': '.event-time',  # ✅ Find this in HTML
        'game_url': 'a.event-link',  # ✅ Find this too
        'live_indicator': '.live-badge',  # ✅ Find this too
    },
    'odds': {
        'spread_home': '.spread-home',  # ✅ REAL
        'spread_away': '.spread-away',  # ✅ REAL
        'total_over': '.total-over',  # ✅ REAL
        'total_under': '.total-under',  # ✅ REAL
        'moneyline_home': '.ml-home',  # ✅ REAL
        'moneyline_away': '.ml-away',  # ✅ REAL
    }
}
```

**Update the scraping logic:**

**Find** `_scrape_game_list` method (around line 135):
```python
async def _scrape_game_list(self, page) -> List[Dict]:
    try:
        # OLD: Wait for placeholder selector
        await page.wait_for_selector(self.selectors['game_list']['container'], timeout=10000)
```

**Update to:**
```python
async def _scrape_game_list(self, page) -> List[Dict]:
    try:
        # ✅ NEW: Wait for REAL selector
        await page.wait_for_selector('.event-row', timeout=10000)  # Your real selector
        
        # Extract game data
        games = await page.evaluate("""
            () => {
                const games = [];
                const gameElements = document.querySelectorAll('.event-row');  // ✅ REAL
                
                gameElements.forEach(element => {
                    try {
                        const homeTeam = element.querySelector('.team-name-1')?.textContent?.trim();  // ✅ REAL
                        const awayTeam = element.querySelector('.team-name-2')?.textContent?.trim();  // ✅ REAL
                        const spread = element.querySelector('.spread-home')?.textContent?.trim();  // ✅ REAL
                        const total = element.querySelector('.total-value')?.textContent?.trim();  // ✅ REAL
                        
                        if (homeTeam && awayTeam) {
                            games.push({
                                home_team: homeTeam,
                                away_team: awayTeam,
                                spread: parseFloat(spread) || null,
                                total: parseFloat(total) || null,
                                is_live: true,
                                timestamp: new Date().toISOString()
                            });
                        }
                    } catch (e) {
                        console.error('Error parsing game:', e);
                    }
                });
                
                return games;
            }
        """)
        
        return games
```

### Step 1.4: Test the Scraper (15 min)

**Run the scraper:**
```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research/betonlineofficial"
python3 -c "
from scrapers.crawlee_betonline_scraper import get_crawlee_betonline_odds

print('🕷️ Testing scraper with REAL selectors...')
odds = get_crawlee_betonline_odds()

if odds and len(odds) > 0:
    print(f'✅ SUCCESS: Found {len(odds)} games!')
    for game in odds[:3]:
        print(f'  {game}')
else:
    print('❌ FAILED: Still returning 0 games')
    print('Check selectors in browser again!')
"
```

**If it works:**
🎉 Congrats! You fixed 80% of the problem!

**If it doesn't work:**
1. Double-check selectors in browser
2. Add debug logging:
   ```python
   print(f"Looking for selector: {selector}")
   print(f"Found elements: {await page.query_selector_all(selector)}")
   ```
3. Take screenshot:
   ```python
   await page.screenshot(path='debug.png')
   ```
4. Check console errors:
   ```python
   page.on('console', lambda msg: print(f"Console: {msg.text}"))
   ```

---

## 🛡️ SOLUTION #2: BYPASS ANTI-BOT PROTECTION

**Time: 1-2 hours**  
**Difficulty: Medium-Hard**  
**Impact: Prevents 403 errors**

### Step 2.1: Better Stealth Configuration (30 min)

**Update browser context:**

Find where browser is launched (around line 97):
```python
browser = await p.chromium.launch(
    headless=self.config['headless'],
    args=['--no-sandbox', '--disable-setuid-sandbox']
)
```

**Enhance with better stealth:**
```python
browser = await p.chromium.launch(
    headless=self.config['headless'],
    args=[
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-blink-features=AutomationControlled',  # ✅ NEW
        '--disable-dev-shm-usage',  # ✅ NEW
        '--disable-web-security',  # ✅ NEW (use carefully)
    ]
)

context = await browser.new_context(
    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    viewport={'width': 1920, 'height': 1080'},  # ✅ NEW
    locale='en-US',  # ✅ NEW
    timezone_id='America/New_York',  # ✅ NEW
    device_scale_factor=1,  # ✅ NEW
    is_mobile=False,  # ✅ NEW
    has_touch=False,  # ✅ NEW
)
```

**Add more stealth scripts:**
```python
await page.add_init_script("""
    // Hide webdriver
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
    });
    
    // Override plugins
    Object.defineProperty(navigator, 'plugins', {
        get: () => [1, 2, 3, 4, 5],
    });
    
    // Override languages
    Object.defineProperty(navigator, 'languages', {
        get: () => ['en-US', 'en'],
    });
    
    // Override chrome
    window.chrome = {
        runtime: {},
    };
    
    // Override permissions
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
            Promise.resolve({ state: Notification.permission }) :
            originalQuery(parameters)
    );
""")
```

### Step 2.2: Add Realistic Behavior (30 min)

**Add delays:**
```python
# After page load
await page.wait_for_timeout(random.randint(1000, 3000))

# Before clicking
await page.wait_for_timeout(random.randint(500, 1500))
```

**Add mouse movements:**
```python
# Move mouse randomly
async def random_mouse_move(page):
    x = random.randint(100, 1800)
    y = random.randint(100, 1000)
    await page.mouse.move(x, y, steps=random.randint(5, 15))

# Use it
await random_mouse_move(page)
await page.wait_for_timeout(random.randint(500, 1000))
await random_mouse_move(page)
```

**Add scrolling:**
```python
# Scroll down naturally
await page.evaluate("""
    async () => {
        await new Promise((resolve) => {
            let totalHeight = 0;
            const distance = 100;
            const timer = setInterval(() => {
                const scrollHeight = document.body.scrollHeight;
                window.scrollBy(0, distance);
                totalHeight += distance;
                
                if(totalHeight >= scrollHeight / 2){
                    clearInterval(timer);
                    resolve();
                }
            }, 100);
        });
    }
""")
```

### Step 2.3: Use Residential Proxy (Optional, 30 min)

**If stealth doesn't work, use proxies:**

**Get a residential proxy:**
- Bright Data: https://brightdata.com (starts at $500/mo)
- Smartproxy: https://smartproxy.com (starts at $75/mo)
- Oxylabs: https://oxylabs.io (starts at $300/mo)

**Update browser context:**
```python
context = await browser.new_context(
    proxy={
        'server': 'http://proxy.example.com:8000',
        'username': 'your_username',
        'password': 'your_password'
    },
    # ... other options
)
```

### Step 2.4: Alternative - Use ScraperAPI (Easiest)

**Instead of fighting anti-bot yourself:**

```python
import requests

# Use ScraperAPI (handles everything)
scraper_api_key = "YOUR_API_KEY"  # Get from https://www.scraperapi.com
url = "https://www.betonline.ag/sportsbook/basketball/nba"

response = requests.get(
    'http://api.scraperapi.com',
    params={
        'api_key': scraper_api_key,
        'url': url,
        'render': 'true',  # Execute JavaScript
    }
)

html = response.text
# Parse with BeautifulSoup
```

**Cost:** $50-200/month depending on usage  
**Benefit:** They handle all anti-bot issues!

---

## ⏱️ SOLUTION #3: HANDLE DYNAMIC CONTENT

**Time: 30-60 min**  
**Difficulty: Easy-Medium**  
**Impact: Ensures odds loaded**

### Step 3.1: Wait for Specific Content (15 min)

**Instead of:**
```python
await page.wait_for_timeout(3000)  # ❌ Arbitrary wait
```

**Do this:**
```python
# ✅ Wait for specific element
await page.wait_for_selector('.spread-home', state='visible', timeout=10000)

# ✅ OR wait for text content
await page.wait_for_function("""
    () => {
        const spread = document.querySelector('.spread-home');
        return spread && spread.textContent.trim() !== '' && spread.textContent !== '--';
    }
""", timeout=10000)
```

### Step 3.2: Intercept Network Requests (15 min)

**Find API calls that load odds:**

```python
# Log all requests
page.on('request', lambda request: print(f"Request: {request.url}"))

# Log all responses
page.on('response', lambda response: print(f"Response: {response.url} - {response.status}"))

# Intercept specific requests
api_data = []

async def handle_response(response):
    if 'odds' in response.url or 'api' in response.url:
        try:
            data = await response.json()
            api_data.append(data)
            print(f"API data: {data}")
        except:
            pass

page.on('response', handle_response)
```

**If you find an API:**
You can skip HTML scraping and use the API directly!

### Step 3.3: Handle WebSocket Connections (30 min)

**Many betting sites use WebSocket for real-time odds:**

```python
# Monitor WebSocket connections
def handle_websocket(ws):
    print(f"WebSocket: {ws.url}")
    
    ws.on('framereceived', lambda payload: print(f"WS received: {payload}"))
    ws.on('framesent', lambda payload: print(f"WS sent: {payload}"))

page.on('websocket', handle_websocket)
```

**If WebSocket is used:**
1. Find the WebSocket URL
2. Find the message format
3. Connect directly to WebSocket
4. Parse real-time messages

**Example WebSocket scraper:**
```python
import websockets
import json

async def scrape_via_websocket():
    uri = "wss://betonline.ag/odds/websocket"  # Find real URL
    
    async with websockets.connect(uri) as websocket:
        # Send subscription message
        await websocket.send(json.dumps({
            "action": "subscribe",
            "channel": "nba_odds"
        }))
        
        # Receive messages
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Odds update: {data}")
```

---

## 🎯 SOLUTION #4: PRODUCTION HARDENING

**Time: 1 hour**  
**Difficulty: Easy**  
**Impact: Reliability**

### Step 4.1: Add Error Handling (20 min)

```python
async def scrape_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            games = await scrape_live_games()
            if games:
                return games
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(5 * (attempt + 1))  # Exponential backoff
            else:
                raise
    
    return []
```

### Step 4.2: Add Rate Limiting (15 min)

```python
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
    
    async def wait_if_needed(self):
        now = time.time()
        
        # Remove old requests
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        
        # Wait if at limit
        if len(self.requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.requests.append(now)

# Use it
limiter = RateLimiter(max_requests=10, time_window=60)  # 10 req/min

async def scrape():
    await limiter.wait_if_needed()
    # ... scrape
```

### Step 4.3: Add Monitoring (15 min)

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use it
logger.info(f"Starting scrape of {url}")
logger.warning(f"Attempt {attempt} failed")
logger.error(f"Critical error: {e}")
```

### Step 4.4: Add Health Checks (10 min)

```python
async def health_check():
    """Test if scraper is working"""
    try:
        games = await scrape_live_games()
        return {
            'status': 'healthy' if games else 'degraded',
            'games_found': len(games),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
```

---

## ✅ TESTING CHECKLIST

### Unit Tests:
- [ ] Selectors return correct elements
- [ ] Data extraction parses correctly
- [ ] Error handling catches exceptions
- [ ] Rate limiter enforces limits

### Integration Tests:
- [ ] Scraper retrieves 1+ games
- [ ] Odds data is valid (numbers, not null)
- [ ] Multiple requests don't get blocked
- [ ] Fallback chain works

### Production Tests:
- [ ] Runs for 1 hour without errors
- [ ] Handles page structure changes
- [ ] Recovers from temporary failures
- [ ] Logs useful debugging info

---

## 🚀 DEPLOYMENT CHECKLIST

- [ ] Real CSS selectors (not placeholders)
- [ ] Anti-bot protection bypassed
- [ ] Dynamic content handled
- [ ] Error handling added
- [ ] Rate limiting configured
- [ ] Monitoring enabled
- [ ] Health checks working
- [ ] Tested for 1+ hours
- [ ] Documentation updated
- [ ] Team trained on system

---

## 📞 TROUBLESHOOTING

### Issue: Still getting 0 games

**Check:**
1. Are selectors correct? Test in browser console
2. Is content loaded? Add `page.screenshot()`
3. Are you waiting long enough? Increase timeout
4. Is page structure different? Re-inspect HTML

### Issue: Still getting 403 errors

**Try:**
1. Better stealth configuration
2. Residential proxy
3. ScraperAPI or similar service
4. Real Chrome (not Chromium)

### Issue: Data is incomplete

**Check:**
1. Are all selectors correct?
2. Is data in different elements?
3. Is data loaded asynchronously?
4. Add more wait conditions

---

**🔧 Follow this guide step-by-step and you'll have a working scraper!**

**Estimated total time: 3-5 hours of focused work**

