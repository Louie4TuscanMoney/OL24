# BetOnline Scraping - 5-Second Updates

**Status:** Building  
**Purpose:** Scrape live NBA odds from betonline.ag  
**Target:** 5-second refresh rate  
**Speed:** SPEED SPEED SPEED

---

## Objective

Scrape `https://www.betonline.ag/sportsbook/basketball/nba` every 5 seconds to get:
- Live game odds (spread, moneyline, total)
- Halftime lines
- Line movements

**Why:** Compare ML predictions with market odds to find edges

---

## Architecture

```
BetOnline.ag
    ↓ Every 5 seconds
Crawlee Scraper (persistent browser)
    ↓ Extract odds
Odds Database (in-memory + log)
    ↓ Compare
ML Predictions (from NBA_API)
    ↓ Calculate edge
Edge Detection System
    ↓ If edge > threshold
Risk Management (bet sizing)
```

---

## Technical Approach

**Library:** Crawlee (Python)  
**Browser:** Persistent (stays open)  
**Optimizations:**
- Resource blocking (images, ads)
- Cached selectors
- Parallel extraction

**From BETONLINE specs:**
- 5000ms total budget
- Includes calibration against live scores
- Forward ML outputs to SolidJS

---

## Next Steps

1. Install Crawlee
2. Build scraper with persistent browser
3. Extract odds data structure
4. Store in database
5. Compare with ML predictions
6. Detect edges

---

*Ready to build BetOnline scraper*

