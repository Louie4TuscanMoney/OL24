# 🤖 AUTONOMOUS ONTORISK INTEGRATION PLAN

**Goal:** OntoRisk runs autonomously after getting signal from live PBP + Mamba + BetOnline odds

---

## **🎯 THE AUTONOMOUS FLOW:**

```
LIVE GAME (Q2 6:00)
        ↓
1. ESPN/NBA API → Game State
        ↓
2. Play-by-Play → 18-min Pattern
        ↓
3. Mamba Extractor → 67 Features
        ↓
4. Mamba Model → Prediction
        ↓
5. BetOnline Scraper → REAL Odds ⚠️ CRITICAL!
        ↓
6. OntoRisk → Probability Calibration
        ↓
7. OntoRisk → Kelly Criterion
        ↓
8. OntoRisk → Risk Validation
        ↓
9. Decision: BET or SKIP
        ↓
10. Auto-Logger → Store Everything
```

---

## **🚨 CRITICAL BOTTLENECK: BETONLINE SCRAPER**

### **Current Status:**
❌ **NOT WORKING** - BetOnline blocks automated scraping
❌ Using **synthetic fallback odds**
❌ OntoRisk **CANNOT work** without real odds

### **Why It's Critical:**
```python
# OntoRisk needs REAL odds for:
1. Implied probability calculation
2. Vig percentage
3. No-vig probability
4. Market efficiency analysis
5. Kelly Criterion optimal sizing
6. Edge validation
```

**Without real odds, OntoRisk is meaningless!** ⚠️

---

## **🔧 BETONLINE SCRAPER SOLUTIONS:**

### **Option 1: Manual Odds Entry (IMMEDIATE)**

Create a quick UI for you to enter odds manually:

```python
# API endpoint
POST /api/betonline/manual-odds
{
  "game_id": "0022500123",
  "spread": -6.0,
  "total": 215.5,
  "home_ml": -250,
  "away_ml": +210
}
```

**Pros:**
- ✅ Works immediately
- ✅ 100% accurate
- ✅ OntoRisk can run

**Cons:**
- ❌ Manual entry required
- ❌ Not fully autonomous

### **Option 2: Odds API Service (RECOMMENDED)**

Use a paid odds API:

**The Odds API** (https://the-odds-api.com/)
- ✅ Real-time NBA odds
- ✅ BetOnline included
- ✅ $0-$10/month
- ✅ REST API
- ✅ Live odds updates

```python
import requests

def get_betonline_odds_via_api(game_id):
    API_KEY = "your_odds_api_key"
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
    
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'spreads,totals,h2h',
        'bookmakers': 'betonline'
    }
    
    response = requests.get(url, params=params)
    return response.json()
```

**Pros:**
- ✅ Fully autonomous
- ✅ Real BetOnline odds
- ✅ No scraping needed
- ✅ OntoRisk works perfectly

**Cons:**
- ❌ Costs $10/month
- ❌ API key required

### **Option 3: Residential Proxy + Crawlee (ADVANCED)**

Use rotating residential proxies to avoid blocking:

```javascript
// Crawlee with residential proxy
const { PlaywrightCrawler } = require('crawlee');

const crawler = new PlaywrightCrawler({
    proxyConfiguration: {
        proxyUrls: ['http://user:pass@residential-proxy.com:8080']
    },
    // ... rest of scraper
});
```

**Pros:**
- ✅ Free (if you have proxies)
- ✅ Real scraping
- ✅ Autonomous

**Cons:**
- ❌ Complex setup
- ❌ Proxies cost money
- ❌ Still might get blocked

---

## **🎯 RECOMMENDED SOLUTION:**

### **SHORT-TERM (TODAY):**
Use **Manual Odds Entry** via API:

1. You watch BetOnline during live games
2. Enter odds via API when Mamba triggers
3. OntoRisk calculates optimal bet
4. System logs everything

### **LONG-TERM (THIS WEEK):**
Subscribe to **The Odds API**:

1. Sign up: https://the-odds-api.com/
2. Get API key
3. Add to Railway environment variables
4. Update `betonline_live_lines.py` to use API
5. Fully autonomous system

---

## **🔧 IMPLEMENTATION:**

### **Step 1: Manual Odds Entry (Now)**

Add to `trading_dashboard_api.py`:

```python
@app.post("/api/betonline/manual-entry")
async def manual_betonline_entry(request: dict):
    """
    Manually enter BetOnline odds for a game
    
    Body:
        {
            "game_id": "0022500123",
            "home_team": "LAL",
            "away_team": "GSW",
            "spread": -6.0,
            "total": 215.5,
            "home_ml": -250,
            "away_ml": +210,
            "timestamp": "2025-10-27T20:00:00"
        }
    """
    global _manual_betonline_odds
    if '_manual_betonline_odds' not in globals():
        _manual_betonline_odds = {}
    
    game_id = request['game_id']
    
    # Calculate implied probabilities
    home_ml = request['home_ml']
    away_ml = request['away_ml']
    
    if home_ml < 0:
        home_implied = abs(home_ml) / (abs(home_ml) + 100)
    else:
        home_implied = 100 / (home_ml + 100)
    
    if away_ml < 0:
        away_implied = abs(away_ml) / (abs(away_ml) + 100)
    else:
        away_implied = 100 / (away_ml + 100)
    
    total_implied = home_implied + away_implied
    vig_pct = (total_implied - 1) * 100
    
    home_no_vig = home_implied / total_implied
    away_no_vig = away_implied / total_implied
    
    _manual_betonline_odds[game_id] = {
        **request,
        'home_implied_prob': home_implied,
        'away_implied_prob': away_implied,
        'home_no_vig_prob': home_no_vig,
        'away_no_vig_prob': away_no_vig,
        'vig_percentage': vig_pct,
        'source': 'BetOnline (MANUAL ENTRY)',
        'entered_at': datetime.now().isoformat()
    }
    
    return {
        "status": "✅ Odds entered!",
        "game_id": game_id,
        "odds": _manual_betonline_odds[game_id]
    }
```

### **Step 2: Update BetOnline Scraper (This Week)**

```python
# betonline_live_lines.py

import os

def get_live_lines(self):
    """Get live lines - try multiple sources"""
    
    # Priority 1: Manual entry (highest priority)
    if hasattr(self, '_manual_odds') and self._manual_odds:
        print("📊 Using manually entered BetOnline odds")
        return list(self._manual_odds.values())
    
    # Priority 2: The Odds API (if key available)
    odds_api_key = os.getenv('ODDS_API_KEY')
    if odds_api_key:
        try:
            odds = self._fetch_from_odds_api(odds_api_key)
            if odds:
                print("📊 Using The Odds API (BetOnline)")
                return odds
        except Exception as e:
            print(f"⚠️ Odds API error: {e}")
    
    # Priority 3: Crawlee scraper (if working)
    try:
        odds = self._fetch_with_crawlee()
        if odds:
            print("📊 Using Crawlee scraper")
            return odds
    except Exception as e:
        print(f"⚠️ Crawlee error: {e}")
    
    # Priority 4: Synthetic fallback (LAST RESORT)
    print("⚠️ Using synthetic odds (NOT REAL)")
    return self._generate_synthetic_lines()
```

---

## **🎯 ONTORISK INTEGRATION:**

Once you have **REAL odds** (manual or API), OntoRisk runs autonomously:

```python
# In live_trading_engine.py (already integrated!)

def make_live_prediction(self, game, line):
    # 1. Mamba prediction
    prediction = self.model.predict(features)
    
    # 2. Get REAL BetOnline odds
    spread_line = line['spread']  # ⚠️ MUST BE REAL!
    
    # 3. OntoRisk probability calibration
    if self.ontorisk_enabled:
        prob = self.calibrator.calculate_probability(
            prediction=prediction,
            spread_line=spread_line,
            home_team=game['home_team'],
            away_team=game['away_team']
        )
        
        # 4. OntoRisk risk validation
        checks = self.risk_manager.check_limits()
        can_bet = checks['can_bet'] and edge >= 5.0
        
        # 5. Kelly Criterion sizing
        if can_bet:
            kelly_fraction = prob.kelly_edge * 0.25
            kelly_stake = self.risk_manager.state.current_bankroll * kelly_fraction
            is_valid, stake, reason = self.risk_manager.validate_bet_size(kelly_stake)
        
        # 6. Decision: BET or SKIP
        return {
            'bet_recommended': is_valid,
            'recommended_stake': stake,
            'reason': reason,
            'p_win': prob.p_win,
            'kelly_edge': prob.kelly_edge
        }
```

---

## **✅ AUTONOMOUS CHECKLIST:**

- [x] Live game detection (ESPN/NBA API)
- [x] Play-by-play extraction (18-min pattern)
- [x] Mamba feature extraction (67 features)
- [x] Mamba prediction (model ready)
- [ ] **BetOnline REAL odds** ⚠️ CRITICAL BLOCKER!
- [x] OntoRisk probability calibration
- [x] OntoRisk Kelly Criterion
- [x] OntoRisk risk validation
- [x] Auto-logger (stores everything)

---

## **🚨 THE BLOCKER:**

**Without REAL BetOnline odds, OntoRisk is useless!**

You have 3 options:
1. **Manual entry** (works today, not autonomous)
2. **The Odds API** ($10/month, fully autonomous)
3. **Residential proxies** (complex, might still fail)

---

## **💡 MY RECOMMENDATION:**

### **TODAY:**
- Use manual odds entry
- Test OntoRisk calculations
- Verify Kelly sizing works
- Monitor live game behavior

### **THIS WEEK:**
- Subscribe to The Odds API ($10/month)
- Add API key to Railway
- Update scraper to use API
- **FULLY AUTONOMOUS SYSTEM** ✅

---

## **🎯 FINAL ANSWER:**

**YES, OntoRisk CAN run autonomously!**

**BUT**: You need REAL BetOnline odds first.

**Fastest solution**: The Odds API ($10/month)

**Alternative**: Manual entry (not autonomous, but works)

**Once you have real odds**: System is 100% autonomous! 🚀


