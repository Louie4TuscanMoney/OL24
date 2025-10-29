# 🕵️ REVERSE ENGINEER ESPN REAL-TIME API

**Goal:** Find ESPN's internal WebSocket or API that powers their website's instant updates

**Why:** ESPN.com shows scores with <1 second lag. We want that same feed.

---

## 🔍 STEP 1: INSPECT ESPN NETWORK TRAFFIC

### **Method A: Chrome DevTools (Easiest)**

1. **Open ESPN Scoreboard:**
   ```
   https://www.espn.com/nba/scoreboard
   ```

2. **Open DevTools:**
   - Press `F12` or `Cmd+Option+I` (Mac)
   - Go to **"Network"** tab
   - Click **"WS"** (WebSocket) filter at the top

3. **Look for WebSocket connections:**
   ```
   You should see something like:
   - wss://push.api.espn.com/...
   - wss://realtime.espn.com/...
   - wss://live.espn.com/...
   ```

4. **Click on the WebSocket connection:**
   - See the **"Messages"** tab
   - Watch messages come in every 1-2 seconds
   - Copy the **WebSocket URL**

5. **Also check XHR/Fetch:**
   - Filter by "Fetch/XHR"
   - Look for API calls like:
     - `/live/scores`
     - `/realtime/nba`
     - `/streaming/games`

---

## 🎯 STEP 2: INSPECT THE WEBSOCKET PROTOCOL

### **Once you find the WebSocket URL:**

**In Chrome DevTools → Network → WS:**

1. **Headers Tab:**
   - Copy the URL
   - Note any query parameters (`?league=nba&sport=basketball`)
   - Check for auth tokens in headers

2. **Messages Tab:**
   - See what data ESPN sends/receives
   - Look for JSON message format
   - Note the message structure

**Example message might look like:**
```json
{
  "type": "score_update",
  "game_id": "401585370",
  "home_score": 60,
  "away_score": 58,
  "period": 2,
  "clock": "5:30"
}
```

---

## 🛠️ STEP 3: TEST THE WEBSOCKET

**I'll create a Python script to connect to ESPN's WebSocket:**

### **espn_websocket_test.py:**

```python
import asyncio
import websockets
import json

async def connect_espn_websocket():
    # Replace with the WebSocket URL you found
    url = "wss://push.api.espn.com/streaming/nba/scoreboard"
    
    # Add headers if needed (copy from DevTools)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Origin": "https://www.espn.com"
    }
    
    try:
        async with websockets.connect(url, extra_headers=headers) as ws:
            print(f"✅ Connected to: {url}")
            
            # Listen for messages
            while True:
                message = await ws.recv()
                print(f"\n📦 Received:")
                
                # Try to parse as JSON
                try:
                    data = json.loads(message)
                    print(json.dumps(data, indent=2))
                except:
                    print(message)
    
    except Exception as e:
        print(f"❌ Error: {e}")

# Run it
asyncio.run(connect_espn_websocket())
```

---

## 🔬 STEP 4: ANALYZE THE API STRUCTURE

### **What to look for:**

1. **Authentication:**
   - Does it require a token?
   - Is there a cookie/session needed?
   - Can you connect without auth?

2. **Message Format:**
   - How are games identified? (game_id, event_id, etc.)
   - What fields are included?
   - How often do updates arrive?

3. **Subscription:**
   - Do you need to send a subscribe message?
   - Example:
     ```json
     {"action": "subscribe", "channel": "nba.scoreboard"}
     ```

---

## 🚀 STEP 5: INTEGRATE INTO OUR SYSTEM

**Once you find the working WebSocket:**

```python
# In nba_live_scores.py:

import asyncio
import websockets

class ESPNWebSocketClient:
    def __init__(self):
        self.url = "wss://[ESPN_WEBSOCKET_URL]"
        self.games = {}
    
    async def connect(self):
        async with websockets.connect(self.url) as ws:
            # Subscribe to NBA scoreboard
            await ws.send(json.dumps({
                "action": "subscribe",
                "channel": "nba.live.scores"
            }))
            
            # Listen for updates
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                
                # Update games dict
                self.games[data['game_id']] = {
                    'home_score': data['home_score'],
                    'away_score': data['away_score'],
                    'period': data['period'],
                    'clock': data['clock']
                }
                
                print(f"⚡ LIVE UPDATE: {data}")
    
    def get_games(self):
        return list(self.games.values())
```

---

## 🎮 STEP 6: ACTUALLY DO IT NOW

### **Open ESPN right now and inspect:**

1. Go to: https://www.espn.com/nba/scoreboard
2. Open DevTools (F12)
3. Go to Network → WS
4. Refresh the page
5. **Copy the WebSocket URL you see**

**Send me:**
- The WebSocket URL
- Any headers it uses
- Example messages you see

**I'll write the integration code immediately!**

---

## ⚠️ LEGAL CONSIDERATIONS

**Is this allowed?**

- ✅ **Technically:** You're just using publicly exposed APIs
- ⚠️ **Legally:** Might violate ESPN Terms of Service
- ⚠️ **Risk:** ESPN could block your IP or change the protocol

**However:**
- Many developers do this
- As long as you're not DDOSing them (1-2 req/sec is fine)
- Don't redistribute the data commercially
- Personal/internal use is usually OK

---

## 🎯 ALTERNATIVE: Use Browser Automation

**If WebSocket is too complex:**

```python
from playwright.async_api import async_playwright

async def get_espn_scores():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Intercept WebSocket or API calls
        page.on("websocket", lambda ws: print(f"WebSocket: {ws.url}"))
        page.on("request", lambda req: 
            print(req.url) if 'score' in req.url else None
        )
        
        await page.goto('https://www.espn.com/nba/scoreboard')
        await page.wait_for_timeout(5000)
        
        # Extract scores from page
        scores = await page.eval_on_selector_all('.ScoreCell', '''
            elements => elements.map(e => e.textContent)
        ''')
```

---

**Go inspect ESPN.com NOW and send me what you find in the Network tab!** 🕵️
