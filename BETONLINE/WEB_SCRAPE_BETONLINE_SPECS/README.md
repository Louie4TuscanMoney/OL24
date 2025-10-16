# Web Scraping - BetOnline Live Scores

**Real-Time NBA Score Collection with Crawlee for ML Forecasting**

---

## Overview

This folder contains complete documentation and implementation strategies for scraping live NBA game scores from BetOnline.ag using Crawlee (Node.js). The scraped data feeds directly into your ML forecasting pipeline (Informer, Conformal Prediction, and Dejavu) for real-time halftime predictions.

**Status:** Production-ready blueprint  
**Technology:** Crawlee + Playwright + Stealth Plugin  
**Target Site:** https://www.betonline.ag/sportsbook/basketball/  
**Use Case:** Collect live scores every 30-60 seconds from game start to halftime

---

## What's Inside

| Document | Purpose | Status |
|----------|---------|--------|
| **DATA_COLLECTION_BETONLINE.md** | Complete implementation guide with code examples | ✅ Complete |
| **DEFINITION.md** | System overview and architecture | ✅ Complete |
| **BETONLINE_IMPLEMENTATION_SPEC.md** | Technical specifications and deployment guide | ✅ Complete |
| **README.md** (this file) | Quick start and navigation | ✅ Complete |

---

## Quick Start

### Prerequisites

```bash
# Install Node.js 18+ (if not already installed)
node --version  # Should be 18.x or higher

# Install dependencies
npm install crawlee playwright puppeteer-extra-plugin-stealth
```

### 5-Minute Test Scraper

```javascript
// test-scraper.js - Test if BetOnline is scrapable
import { PlaywrightCrawler } from 'crawlee';

const crawler = new PlaywrightCrawler({
  headless: false,  // See what's happening
  requestHandler: async ({ page, request, log }) => {
    log.info(`Scraping ${request.url}`);
    
    // Wait for page to load
    await page.waitForLoadState('domcontentloaded');
    
    // Take screenshot
    await page.screenshot({ path: 'betonline-screenshot.png' });
    
    // Extract page title
    const title = await page.title();
    log.info(`Page title: ${title}`);
    
    // Try to find game elements (adjust selectors as needed)
    const gameElements = await page.$$('.game-line, .event-line, [data-game-id]');
    log.info(`Found ${gameElements.length} game elements`);
  },
});

// Test with main basketball page
await crawler.run(['https://www.betonline.ag/sportsbook/basketball/nba']);
```

**Run it:**
```bash
node test-scraper.js
```

**Expected Result:**
- Browser window opens showing BetOnline
- Screenshot saved to `betonline-screenshot.png`
- Console logs showing game elements found

---

## Documentation Roadmap

### 1. Start Here: DEFINITION.md

**Read this first to understand:**
- What we're building and why
- System architecture overview
- Data flow from BetOnline → ML models
- Key challenges and solutions
- Success criteria

**Time:** 10 minutes

### 2. Implementation Guide: DATA_COLLECTION_BETONLINE.md

**Complete walkthrough including:**
- BetOnline URL structure analysis
- Crawlee setup with anti-detection
- Proxy rotation and rate limiting
- Live game monitoring strategy
- Complete working code examples
- Data transformation for ML models
- Deployment architecture

**Time:** 30-60 minutes to read, 15-20 hours to implement

**Key Sections:**
- **Section 1:** URL Structure Analysis (identify what to scrape)
- **Section 2:** Crawlee Architecture (system design)
- **Section 3:** Anti-Detection Strategies (avoid blocks)
- **Section 4:** Scraping Strategy & Timing (when to scrape)
- **Section 5:** Implementation Code (complete scraper)
- **Section 6:** Integration with ML Models (data export)
- **Section 7:** Deployment & Monitoring (production ready)

### 3. Technical Specs: BETONLINE_IMPLEMENTATION_SPEC.md

**Detailed technical specifications:**
- System architecture diagrams
- Configuration schemas (TypeScript interfaces)
- Database schemas (MongoDB, PostgreSQL)
- Scraping schedules (cron syntax)
- Error handling logic
- Monitoring specifications
- Docker deployment configs
- Performance targets and cost estimates

**Time:** 20-30 minutes to read, reference during implementation

---

## Implementation Workflow

### Phase 1: Setup & Exploration (2-4 hours)

**Goal:** Get a basic scraper working on one page

1. **Install Tools**
   ```bash
   npm init -y
   npm install crawlee playwright puppeteer-extra-plugin-stealth
   npm install winston mongodb node-cron express
   ```

2. **Inspect BetOnline**
   - Open https://www.betonline.ag/sportsbook/basketball/nba
   - Right-click → Inspect Element
   - Identify CSS selectors for:
     - Game cards/containers
     - Team names
     - Scores (if live)
     - Game times
     - Links to individual games
   
3. **Test Basic Scraper**
   - Use the 5-minute test scraper above
   - Verify you can access the page
   - Check for CAPTCHA or blocks
   - Confirm selectors work

4. **Document Selectors**
   ```javascript
   // selectors.js
   export const SELECTORS = {
     gameCard: '.your-actual-selector',  // From inspection
     homeTeam: '.home-team',
     awayTeam: '.away-team',
     // etc.
   };
   ```

**Checkpoint:** Can you load BetOnline and find game elements?

### Phase 2: Anti-Detection (2-3 hours)

**Goal:** Avoid getting blocked

1. **Add Stealth Plugin**
   ```javascript
   import { chromium } from 'playwright-extra';
   import StealthPlugin from 'puppeteer-extra-plugin-stealth';
   chromium.use(StealthPlugin());
   ```

2. **Configure Proxies**
   - Sign up for proxy service (Bright Data, Oxylabs, SmartProxy)
   - Get residential proxy URLs
   - Add to Crawlee config:
   ```javascript
   proxyConfiguration: {
     proxyUrls: ['http://user:pass@proxy1.com:8080'],
   }
   ```

3. **Add Rate Limiting**
   ```javascript
   maxConcurrency: 2,  // Only 2 requests at once
   requestDelayMs: 3000,  // 3 seconds between requests
   ```

4. **Test for 30 Minutes**
   - Run scraper continuously
   - Monitor for 429 errors (rate limiting)
   - Monitor for 403 errors (blocked)
   - Adjust delays as needed

**Checkpoint:** Can you scrape for 30 minutes without blocks?

### Phase 3: Live Game Monitoring (3-4 hours)

**Goal:** Scrape scores from live games

1. **Find Live Games**
   - Scrape main page to get list of live games
   - Extract game URLs
   - Store in database or queue

2. **Scrape Individual Games**
   - Visit each game URL
   - Extract current score
   - Extract quarter and time
   - Calculate score differential

3. **Implement Scheduler**
   ```javascript
   import cron from 'node-cron';
   
   // Every minute during games
   cron.schedule('* * * * *', async () => {
     await monitorLiveGames();
   });
   ```

4. **Store Data**
   ```javascript
   // Save to MongoDB
   await db.collection('live_scores').insertOne({
     gameId: 'LAL-BOS-20251015',
     quarter: '2Q',
     timeRemaining: '6:32',
     scoreHome: 52,
     scoreAway: 48,
     differential: 4,
     timestamp: new Date(),
   });
   ```

**Checkpoint:** Can you track a live game and collect scores?

### Phase 4: Data Pipeline (2-3 hours)

**Goal:** Transform scraped data for ML models

1. **Time-Series Conversion**
   ```javascript
   // Convert event-based to minute-by-minute
   const timeSeries = convertToMinuteSeries(scrapedData);
   // Output: [{ minute: 0, differential: 0 }, { minute: 1, differential: 2 }, ...]
   ```

2. **Fill Missing Data**
   ```javascript
   // Interpolate missing minutes
   const complete = fillMissingMinutes(timeSeries);
   ```

3. **Export for ML Models**
   ```javascript
   // Informer format
   const informerData = {
     xEnc: timeSeries.slice(0, 18).map(p => p.differential),
     y: timeSeries.slice(18, 24).map(p => p.differential),
   };
   
   // Dejavu format
   const dejavuData = {
     pattern: timeSeries.slice(0, 18),
     outcome: timeSeries.slice(18, 24),
   };
   ```

4. **Validate Data Quality**
   ```javascript
   // Check for issues
   assert(timeSeries.length === 48, 'Wrong length');
   assert(!timeSeries.some(p => isNaN(p.differential)), 'NaN values');
   ```

**Checkpoint:** Can you export data that matches ML model requirements?

### Phase 5: Monitoring (2-3 hours)

**Goal:** Know when things break

1. **Add Logging**
   ```javascript
   import winston from 'winston';
   
   const logger = winston.createLogger({
     transports: [
       new winston.transports.File({ filename: 'error.log' }),
       new winston.transports.Console(),
     ],
   });
   ```

2. **Health Check Endpoint**
   ```javascript
   app.get('/health', (req, res) => {
     res.json({
       status: 'healthy',
       lastScrape: lastScrapeTime,
       successRate: successfulRequests / totalRequests,
     });
   });
   ```

3. **Set Up Alerts**
   ```javascript
   // Telegram bot for alerts
   if (errorRate > 0.2) {
     await bot.sendMessage(chatId, '⚠️ High error rate!');
   }
   ```

4. **Create Dashboard**
   - Use Grafana or simple HTML page
   - Show: success rate, latency, games monitored
   - Update every minute

**Checkpoint:** Will you know if scraper stops working?

### Phase 6: Production Deployment (2-3 hours)

**Goal:** Run reliably 24/7

1. **Dockerize**
   ```dockerfile
   FROM node:18-alpine
   COPY . /app
   WORKDIR /app
   RUN npm install
   CMD ["node", "src/index.js"]
   ```

2. **Deploy to Server**
   ```bash
   # Option A: Docker Compose
   docker-compose up -d
   
   # Option B: PM2
   pm2 start ecosystem.config.js
   ```

3. **Configure Auto-Restart**
   ```javascript
   // PM2 config
   autorestart: true,
   max_restarts: 10,
   ```

4. **Monitor for 24 Hours**
   - Check logs for errors
   - Verify data is being collected
   - Confirm no rate limiting
   - Adjust as needed

**Checkpoint:** Has scraper run for 24 hours without intervention?

### Phase 7: Optimization (2-4 hours)

**Goal:** Fine-tune performance

1. **Optimize Scraping Schedule**
   - More frequent during critical window (6:00-0:00 2Q)
   - Less frequent after halftime

2. **Reduce Costs**
   - Use cheaper proxies if possible
   - Optimize database queries
   - Cache static data

3. **Improve Data Quality**
   - Better interpolation for missing data
   - Anomaly detection
   - Cross-validation with other sources

4. **Scale if Needed**
   - Add more workers for multiple games
   - Distribute across servers
   - Load balancing

**Checkpoint:** Is system optimized for cost and performance?

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    BetOnline Scraping System                    │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Game Discovery (Every 5-10 minutes)                            │
│  • Scrape main basketball pages                                 │
│  • Find upcoming/live games                                     │
│  • Extract game URLs and times                                  │
│  • Add to monitoring queue                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Live Game Monitoring (Every 30-60 seconds per game)            │
│  • Visit individual game pages                                  │
│  • Extract current scores and game state                        │
│  • Calculate differential                                       │
│  • Store with timestamp                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Data Storage (MongoDB/PostgreSQL)                              │
│  • Raw scraped data (all events)                                │
│  • Processed time series (minute-by-minute)                     │
│  • Error logs and metrics                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Data Transformation                                             │
│  • Convert to minute-by-minute format                           │
│  • Interpolate missing values                                   │
│  • Validate data quality                                        │
│  • Export for ML models                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ML Model Integration (Python)                                  │
│  • Informer: 18-min input → 6-min forecast                      │
│  • Conformal: Uncertainty quantification                        │
│  • Dejavu: Pattern matching                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### ✅ Anti-Detection
- **Stealth Plugin:** Removes automation markers
- **Proxy Rotation:** Residential IPs for each request
- **User-Agent Rotation:** Mimic different browsers/devices
- **Rate Limiting:** Respectful delays between requests
- **Session Management:** Maintain cookies and state

### ✅ Reliability
- **Automatic Retries:** Exponential backoff on failures
- **Error Recovery:** Intelligent handling of different error types
- **Health Monitoring:** Real-time metrics and alerts
- **Auto-Restart:** PM2/Docker keeps scraper running
- **Data Validation:** Comprehensive quality checks

### ✅ Scalability
- **Parallel Crawling:** Monitor multiple games simultaneously
- **Queue Management:** Efficient request scheduling
- **Database Indexing:** Fast data retrieval
- **Caching:** Reduce redundant requests
- **Horizontal Scaling:** Add more workers as needed

### ✅ Data Quality
- **Validation:** Check every data point for consistency
- **Interpolation:** Fill missing minutes intelligently
- **Deduplication:** Remove duplicate scrapes
- **Timestamping:** Precise temporal tracking
- **Versioning:** Track data schema changes

### ✅ Integration
- **ML Model Ready:** Direct export to Informer/Conformal/Dejavu
- **Real-Time Streaming:** Push data as it's collected
- **API Endpoints:** REST API for model queries
- **Format Flexibility:** JSON, CSV, Parquet export
- **Backward Compatible:** Version your data schemas

---

## Data Output Examples

### Raw Scraped Data

```json
{
  "gameId": "LAL-BOS-20251015",
  "homeTeam": "Los Angeles Lakers",
  "awayTeam": "Boston Celtics",
  "scoreHome": 52,
  "scoreAway": 48,
  "differential": 4,
  "quarter": "2Q",
  "timeRemaining": "6:32",
  "timestamp": "2025-10-15T19:45:32Z",
  "sourceUrl": "https://www.betonline.ag/sportsbook/basketball/game/...",
  "odds": {
    "homeOdds": -110,
    "awayOdds": +105,
    "overUnder": 220.5
  }
}
```

### Processed Time Series

```json
{
  "gameId": "LAL-BOS-20251015",
  "timeSeries": [
    { "minute": 0, "quarter": 1, "differential": 0 },
    { "minute": 1, "quarter": 1, "differential": 2 },
    { "minute": 2, "quarter": 1, "differential": 1 },
    ...
    { "minute": 17, "quarter": 2, "differential": 4 },  // 6:00 2Q - INPUT
    { "minute": 18, "quarter": 2, "differential": 3 },
    ...
    { "minute": 23, "quarter": 2, "differential": 5 }   // 0:00 2Q - TARGET
  ]
}
```

### ML Model Format (Informer)

```json
{
  "gameId": "LAL-BOS-20251015",
  "xEnc": [0, 2, 1, 3, 2, 4, 5, 3, 4, 6, 5, 7, 6, 5, 4, 5, 3, 4],  // 18 values
  "xMarkEnc": [
    [10, 15, 1, 19],  // [month, day, weekday, hour] for minute 0
    [10, 15, 1, 19],  // for minute 1
    // ... 18 rows total
  ],
  "y": [3, 4, 5, 5, 6, 5],  // 6 values (minutes 18-23)
  "metadata": {
    "gameDate": "2025-10-15",
    "teams": "LAL vs BOS",
    "actualHalftimeDiff": 5
  }
}
```

---

## Common Issues & Solutions

### Issue 1: Can't Access BetOnline
**Symptoms:** 403 Forbidden or connection timeout  
**Solutions:**
- Use residential proxy (not datacenter)
- Enable stealth plugin
- Check if your IP is blocked (try different network)
- Contact BetOnline support with your approval reference

### Issue 2: Selectors Don't Work
**Symptoms:** No data extracted, empty arrays  
**Solutions:**
- Inspect BetOnline page manually to find correct selectors
- Check if page uses dynamic loading (wait for elements)
- Take screenshot to debug: `await page.screenshot({ path: 'debug.png' })`
- Use browser dev tools to test selectors

### Issue 3: Rate Limited (429 Errors)
**Symptoms:** "Too Many Requests" errors  
**Solutions:**
- Increase delay between requests (5-10 seconds)
- Reduce concurrent workers (maxConcurrency: 1)
- Rotate proxies more frequently
- Contact BetOnline for guidance on acceptable rate

### Issue 4: Missing Data
**Symptoms:** Gaps in time series  
**Solutions:**
- Implement interpolation for missing minutes
- Increase scraping frequency during critical windows
- Store raw data for later backfill
- Cross-validate with other sources

### Issue 5: Scraper Crashes
**Symptoms:** Process stops unexpectedly  
**Solutions:**
- Use PM2 for auto-restart: `pm2 start app.js`
- Add try-catch blocks around critical sections
- Implement graceful error handling
- Monitor memory usage (may need to restart periodically)

### Issue 6: Data Quality Issues
**Symptoms:** NaN values, extreme differentials  
**Solutions:**
- Validate every data point before storing
- Set reasonable bounds (differential typically -40 to +40)
- Log suspicious data for review
- Cross-check with official NBA scores

---

## Testing Checklist

### Unit Tests
- [ ] Can extract game list from main page
- [ ] Can extract scores from individual game page
- [ ] Selectors work for different game states
- [ ] Data validation catches bad data
- [ ] Time-series conversion is correct

### Integration Tests
- [ ] End-to-end scrape of one game
- [ ] Data successfully stored in database
- [ ] Export functions produce correct format
- [ ] Error handling works for common errors
- [ ] Monitoring alerts are triggered

### Production Tests
- [ ] Scraper runs for 24 hours without intervention
- [ ] Success rate >95%
- [ ] No rate limiting after tuning
- [ ] Data quality score >90%
- [ ] Health checks pass consistently

### Performance Tests
- [ ] Latency <2 seconds per request
- [ ] Can monitor 5+ games simultaneously
- [ ] Memory usage stable over time
- [ ] Database queries are fast (<100ms)
- [ ] System handles failures gracefully

---

## Cost Analysis

### Development Costs
| Item | Time | Notes |
|------|------|-------|
| Setup & exploration | 2-4 hours | Install tools, inspect site |
| Core scraper | 3-4 hours | Basic functionality |
| Anti-detection | 2-3 hours | Stealth, proxies, rate limiting |
| Live monitoring | 3-4 hours | Scheduling, queue management |
| Data pipeline | 2-3 hours | Transform, validate, export |
| Monitoring | 2-3 hours | Logs, alerts, dashboard |
| Deployment | 2-3 hours | Docker, PM2, production setup |
| Testing & refinement | 2-4 hours | Fix issues, optimize |
| **Total** | **18-28 hours** | |

### Monthly Operating Costs
| Item | Cost | Notes |
|------|------|-------|
| Residential Proxies | $50-150 | Bright Data, ~1-3 GB/month |
| Server (AWS EC2 t3.small) | $15-20 | 2 vCPU, 2 GB RAM |
| Database (MongoDB Atlas) | $0-25 | Shared tier or self-hosted |
| Monitoring (Grafana Cloud) | $0-50 | Free tier available |
| **Total** | **$65-245/month** | Can optimize to <$100 |

**Ways to Reduce Costs:**
- Self-host MongoDB and Grafana ($0)
- Use cheaper proxy service (~$30-50/month)
- Optimize scraping schedule (reduce bandwidth)
- Use existing infrastructure if available

---

## Best Practices Summary

### 1. Start Slow
- Begin with 10-second delays
- Monitor for blocks
- Gradually increase frequency
- Contact BetOnline for guidance

### 2. Be Respectful
- Don't overload their servers
- Identify yourself in User-Agent
- Follow their rate limits
- Have approval documented

### 3. Handle Errors Gracefully
- Expect failures
- Retry with backoff
- Log everything
- Alert on critical issues

### 4. Validate Everything
- Check data quality
- Set reasonable bounds
- Cross-validate when possible
- Store raw + processed data

### 5. Monitor Continuously
- Health checks every minute
- Alerts on failures
- Dashboard for visibility
- Log rotation to save space

### 6. Document Changes
- Version your selectors
- Track config changes
- Document BetOnline updates
- Keep approval reference

---

## Next Steps

### Immediate (First Session)
1. Read DEFINITION.md (10 min)
2. Skim DATA_COLLECTION_BETONLINE.md (20 min)
3. Run 5-minute test scraper (10 min)
4. Inspect BetOnline and document selectors (20 min)

**Time: 1 hour**

### Short-Term (First Week)
1. Implement basic scraper with anti-detection
2. Test live game monitoring
3. Set up database and storage
4. Create data transformation pipeline
5. Deploy with monitoring

**Time: 15-20 hours over 1 week**

### Long-Term (First Month)
1. Run in production for NBA season
2. Monitor and optimize performance
3. Integrate with ML forecasting models
4. Collect data for model training/validation
5. Refine based on real-world usage

**Time: 5-10 hours of refinement over 1 month**

---

## Related Documentation

### In This Folder
- **DATA_COLLECTION_BETONLINE.md** - Complete implementation guide
- **DEFINITION.md** - System overview and rationale
- **BETONLINE_IMPLEMENTATION_SPEC.md** - Technical specifications

### In ML Research Project
- **Action Steps Folder/01_DATA_COLLECTION_SETUP.md** - Basketball-Reference scraping
- **Action Steps Folder/08_LIVE_SCORE_INTEGRATION.md** - ML model integration
- **Informer-Beyond.../DATA_ENGINEERING_INFORMER.md** - Informer data requirements
- **Dejavu-A.../DATA_ENGINEERING_DEJAVU.md** - Dejavu pattern database
- **Conformal.../DATA_ENGINEERING_CONFORMAL.md** - Conformal calibration

---

## Support & Resources

### Crawlee Resources
- **Docs:** https://crawlee.dev/
- **Examples:** https://crawlee.dev/docs/examples
- **GitHub:** https://github.com/apify/crawlee
- **Discord:** https://discord.com/invite/jyEM2PRvMU

### Proxy Providers
- **Bright Data:** https://brightdata.com (recommended)
- **Oxylabs:** https://oxylabs.io
- **SmartProxy:** https://smartproxy.com
- **Apify Proxy:** https://apify.com/proxy

### Monitoring Tools
- **Grafana:** https://grafana.com
- **Prometheus:** https://prometheus.io
- **Winston (logging):** https://github.com/winstonjs/winston
- **PM2 (process mgmt):** https://pm2.keymetrics.io

### Community
- **Stack Overflow:** Tag [crawlee] or [web-scraping]
- **Reddit:** r/webscraping
- **Apify Discord:** Active community for Crawlee support

---

## Summary

This folder provides everything you need to build a production-ready BetOnline scraper:

✅ **Complete Documentation** - From concept to deployment  
✅ **Working Code Examples** - Copy-paste ready  
✅ **Anti-Detection Strategies** - Avoid blocks and bans  
✅ **Production Deployment** - Docker, PM2, monitoring  
✅ **ML Integration** - Direct export to your models  
✅ **Cost Optimization** - Keep monthly costs <$100  
✅ **Best Practices** - Respectful, reliable, scalable

**Start with the 5-minute test scraper above, then dive into DATA_COLLECTION_BETONLINE.md for complete implementation!**

---

## Questions?

If you encounter issues:
1. Check "Common Issues & Solutions" section above
2. Review error logs and screenshots
3. Test selectors manually on BetOnline
4. Consult Crawlee documentation
5. Adjust rate limits and retry logic

**Remember:** Web scraping requires patience and iteration. Start slow, test thoroughly, and refine based on real-world behavior.

**Good luck scraping! 🚀🏀**

---

*Version 1.0.0 - October 15, 2025*

