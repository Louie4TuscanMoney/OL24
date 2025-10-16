# BetOnline.ag Live Score Data Collection

**Strategic Web Scraping with Crawlee for Real-Time NBA Betting Odds & Scores**

**Date:** October 15, 2025  
**Objective:** Collect real-time NBA game scores and odds from BetOnline.ag for live forecasting  
**Technology:** Crawlee (Node.js) with anti-detection and rate limiting

---

## Executive Summary

This document provides a complete strategy for scraping live NBA scores and betting odds from BetOnline.ag using Crawlee, a modern Node.js scraping framework designed for resilient, scalable data collection with built-in anti-bot evasion.

**Key Challenges:**
1. BetOnline has anti-scraping measures (rate limiting, bot detection)
2. Dynamic content loaded via JavaScript
3. Need to identify game URLs and optimal scraping times
4. Must maintain approval status and avoid 429 errors

**Solution Approach:**
1. Crawlee with Playwright for browser automation + stealth
2. Proxy rotation and user-agent management
3. Intelligent scheduling based on game times
4. Respectful rate limiting with backoff strategies

**Data Output:** Real-time score differentials synced with your ML forecasting models (Informer, Conformal, Dejavu)

---

## Table of Contents

1. [BetOnline.ag URL Structure Analysis](#betonlineag-url-structure-analysis)
2. [Crawlee Architecture for BetOnline](#crawlee-architecture-for-betonline)
3. [Anti-Detection Strategies](#anti-detection-strategies)
4. [Scraping Strategy & Timing](#scraping-strategy--timing)
5. [Implementation Code](#implementation-code)
6. [Integration with ML Models](#integration-with-ml-models)
7. [Deployment & Monitoring](#deployment--monitoring)

---

## BetOnline.ag URL Structure Analysis

### Primary Target URL

```
https://www.betonline.ag/sportsbook/basketball/preseason
```

**What's on This Page:**
- List of upcoming/live NBA games
- Current odds for each game
- Live scores (if games are in progress)
- Links to individual game pages

### URL Patterns to Discover

**Step 1: Analyze the main page to extract:**

```javascript
// Expected structure on the preseason page
const gameElements = {
  gameCard: '.game-card',           // Container for each game
  gameUrl: 'a.game-link',           // Link to individual game
  teamNames: '.team-name',          // Team names
  gameTime: '.game-time',           // Scheduled start time
  liveIndicator: '.live-badge',     // "LIVE" indicator if game active
  currentScore: '.current-score'    // Current score (if live)
}
```

**Step 2: Individual game URLs typically follow patterns like:**
```
https://www.betonline.ag/sportsbook/basketball/game/[GAME-ID]
https://www.betonline.ag/sportsbook/basketball/nba/[TEAM1-vs-TEAM2]
https://www.betonline.ag/sportsbook/live/basketball/[GAME-ID]
```

**Step 3: Live betting/scores page:**
```
https://www.betonline.ag/sportsbook/live
```

### Data to Extract

| Field | Description | Example | Format |
|-------|-------------|---------|--------|
| `game_id` | Unique identifier | "lakers-vs-celtics-20251015" | String |
| `home_team` | Home team name | "Los Angeles Lakers" | String |
| `away_team` | Away team name | "Boston Celtics" | String |
| `game_time` | Scheduled start | "2025-10-15T19:30:00Z" | ISO timestamp |
| `current_quarter` | Current quarter | "2Q" | String |
| `time_remaining` | Time on clock | "6:32" | String |
| `score_home` | Home team score | 52 | Integer |
| `score_away` | Away team score | 48 | Integer |
| `differential` | Score differential | 4 | Integer |
| `odds_home` | Home team odds | -110 | Integer |
| `odds_away` | Away team odds | +105 | Integer |
| `total_over_under` | O/U line | 220.5 | Float |

---

## Crawlee Architecture for BetOnline

### Why Crawlee?

**Advantages over BeautifulSoup/Selenium:**
- ✅ Built-in session management and cookies
- ✅ Automatic retries with exponential backoff
- ✅ Request queue management
- ✅ Integrated proxy rotation
- ✅ Stealth plugins for bot detection evasion
- ✅ Parallel crawling with concurrency control
- ✅ TypeScript/JavaScript ecosystem (modern, maintained)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Crawlee Main Process                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────┐      ┌──────────────────┐               │
│  │ Request Queue  │─────▶│  Crawler Pool    │               │
│  │  (Game URLs)   │      │  (5-10 workers)  │               │
│  └────────────────┘      └──────────────────┘               │
│                                  │                            │
│                                  ▼                            │
│                    ┌──────────────────────────┐              │
│                    │  Playwright Browser      │              │
│                    │  + Stealth Plugin        │              │
│                    └──────────────────────────┘              │
│                                  │                            │
│                                  ▼                            │
│         ┌────────────────────────────────────────┐           │
│         │  Proxy Rotation Layer                  │           │
│         │  (Residential/Datacenter IPs)          │           │
│         └────────────────────────────────────────┘           │
│                                  │                            │
│                                  ▼                            │
│                    ┌──────────────────────────┐              │
│                    │    BetOnline.ag          │              │
│                    │    (Target Website)      │              │
│                    └──────────────────────────┘              │
│                                  │                            │
│                                  ▼                            │
│         ┌────────────────────────────────────────┐           │
│         │  Data Pipeline                         │           │
│         │  • Parse HTML/JSON                     │           │
│         │  • Validate data                       │           │
│         │  • Transform to time series            │           │
│         │  • Store in database                   │           │
│         └────────────────────────────────────────┘           │
│                                  │                            │
│                                  ▼                            │
│         ┌────────────────────────────────────────┐           │
│         │  PostgreSQL/MongoDB                    │           │
│         │  (Time-series data storage)            │           │
│         └────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## Anti-Detection Strategies

### 1. Proxy Rotation

**Why Needed:** Prevent IP-based rate limiting and blocks

**Implementation:**

```javascript
// crawlee-config.js
import { PlaywrightCrawler } from 'crawlee';

const crawler = new PlaywrightCrawler({
  proxyConfiguration: {
    // Option A: Use Apify Proxy (recommended for production)
    useApifyProxy: true,
    apifyProxyGroups: ['RESIDENTIAL'],  // More expensive but less detectable
    
    // Option B: Custom proxy list
    // proxyUrls: [
    //   'http://username:password@proxy1.com:8080',
    //   'http://username:password@proxy2.com:8080',
    //   'http://username:password@proxy3.com:8080',
    // ],
  },
  
  // Rotate proxies per request
  sessionPoolOptions: {
    maxPoolSize: 20,
    sessionOptions: {
      maxUsageCount: 10,  // Retire session after 10 uses
      maxErrorScore: 3,   // Retire if too many errors
    },
  },
});
```

**Proxy Providers (Recommended):**
- **Bright Data** (formerly Luminati): Enterprise-grade residential proxies
- **Oxylabs**: Sports betting friendly
- **SmartProxy**: Cost-effective residential proxies
- **Apify Proxy**: Built-in to Crawlee ecosystem

**Cost Estimate:**
- Residential proxies: $5-15 per GB
- For ~1000 game scrapes/day: ~$50-150/month

### 2. User-Agent Rotation

**Why Needed:** Mimic different devices/browsers

```javascript
// user-agent-pool.js
const USER_AGENTS = [
  // Chrome on Windows
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
  // Chrome on Mac
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
  // Firefox on Windows
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
  // Safari on Mac
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
  // Chrome on Android
  'Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.144 Mobile Safari/537.36',
];

// Rotate randomly
const getRandomUserAgent = () => {
  return USER_AGENTS[Math.floor(Math.random() * USER_AGENTS.length)];
};

// In crawler
requestHandler: async ({ page, request }) => {
  await page.setUserAgent(getRandomUserAgent());
  // ... rest of handler
}
```

### 3. Stealth Plugin (Critical for BetOnline)

**Why Needed:** Bypass JavaScript-based bot detection

```javascript
// stealth-crawler.js
import { PlaywrightCrawler } from 'crawlee';
import { chromium } from 'playwright-extra';
import StealthPlugin from 'puppeteer-extra-plugin-stealth';

// Apply stealth plugin
const stealth = StealthPlugin();

const crawler = new PlaywrightCrawler({
  launchContext: {
    launcher: chromium,
    useChrome: true,  // Use real Chrome instead of Chromium
    launchOptions: {
      headless: true,  // Can set to false for debugging
      args: [
        '--disable-blink-features=AutomationControlled',
        '--disable-dev-shm-usage',
        '--disable-web-security',
        '--no-sandbox',
      ],
    },
  },
  
  preNavigationHooks: [
    async ({ page }) => {
      // Override webdriver property
      await page.evaluateOnNewDocument(() => {
        Object.defineProperty(navigator, 'webdriver', {
          get: () => false,
        });
      });
      
      // Randomize screen resolution
      const viewports = [
        { width: 1920, height: 1080 },
        { width: 1366, height: 768 },
        { width: 1440, height: 900 },
      ];
      const viewport = viewports[Math.floor(Math.random() * viewports.length)];
      await page.setViewportSize(viewport);
      
      // Add mouse movement (looks more human)
      await page.mouse.move(
        Math.random() * viewport.width,
        Math.random() * viewport.height
      );
    },
  ],
});
```

**What Stealth Plugin Does:**
- Removes `navigator.webdriver` flag
- Masks automation markers in Chrome DevTools Protocol
- Randomizes canvas/WebGL fingerprints
- Mimics human-like mouse movements and timings

### 4. Rate Limiting & Backoff

**Why Needed:** Respect BetOnline's servers and avoid 429 errors

```javascript
// rate-limiter.js
const crawler = new PlaywrightCrawler({
  // Limit concurrent requests
  maxConcurrency: 3,  // Only 3 pages at once
  
  // Delay between requests
  minConcurrency: 1,
  maxRequestsPerCrawl: 100,  // Limit per session
  
  // Time between requests (in milliseconds)
  requestHandlerTimeoutSecs: 60,
  
  // Custom rate limiting
  autoscaledPoolOptions: {
    desiredConcurrency: 2,
    minConcurrency: 1,
    maxConcurrency: 5,
  },
});

// Add random delays
const randomDelay = (min = 2000, max = 5000) => {
  return new Promise(resolve => 
    setTimeout(resolve, Math.random() * (max - min) + min)
  );
};

// In request handler
requestHandler: async ({ page, request }) => {
  // ... scrape data ...
  
  // Random delay before next request (2-5 seconds)
  await randomDelay(2000, 5000);
}
```

**Recommended Rate Limits (With BetOnline Approval):**
- **Development/Testing:** 1 request every 5-10 seconds
- **Production (approved):** 1 request every 2-3 seconds
- **Live game monitoring:** 1 request every 30-60 seconds per game

**Contact BetOnline for:**
- Recommended rate limits
- IP whitelisting (if possible)
- API access (if available, far better than scraping!)

### 5. Session Management

**Why Needed:** Maintain cookies and state across requests

```javascript
// session-management.js
const crawler = new PlaywrightCrawler({
  useSessionPool: true,
  persistCookiesPerSession: true,
  
  sessionPoolOptions: {
    maxPoolSize: 10,
    sessionOptions: {
      maxUsageCount: 20,
      maxErrorScore: 3,
    },
  },
  
  // Handle session errors
  failedRequestHandler: async ({ request, error }) => {
    console.log(`Request ${request.url} failed: ${error.message}`);
    
    // If 429 (rate limited), increase delay
    if (error.message.includes('429')) {
      console.log('Rate limited! Waiting 60 seconds...');
      await new Promise(resolve => setTimeout(resolve, 60000));
    }
  },
});
```

### 6. Error Handling & Retries

```javascript
// error-handling.js
const crawler = new PlaywrightCrawler({
  maxRequestRetries: 5,  // Retry failed requests up to 5 times
  
  // Exponential backoff for retries
  maxRequestsPerMinute: 20,
  
  failedRequestHandler: async ({ request, error, log }) => {
    log.error(`Request failed for ${request.url}`, { error: error.message });
    
    // Log to monitoring system
    await logToDatabase({
      url: request.url,
      error: error.message,
      timestamp: new Date(),
    });
    
    // Handle specific errors
    if (error.message.includes('429')) {
      // Rate limited - increase delay
      log.warning('Rate limited - backing off for 2 minutes');
      await new Promise(resolve => setTimeout(resolve, 120000));
    } else if (error.message.includes('403')) {
      // Forbidden - might be blocked
      log.error('403 Forbidden - check if IP is blocked');
      // Switch proxy or notify admin
    } else if (error.message.includes('timeout')) {
      // Timeout - BetOnline might be slow
      log.warning('Timeout - will retry with longer timeout');
    }
  },
});
```

---

## Scraping Strategy & Timing

### Phase 1: Game Discovery (Pre-Game)

**Objective:** Identify upcoming games and their URLs

**When:** 1-2 hours before game time

**Process:**
```javascript
// 1. Scrape main basketball page
const gamesPage = 'https://www.betonline.ag/sportsbook/basketball/preseason';

// 2. Extract game list
const games = await page.$$eval('.game-card', (cards) => {
  return cards.map(card => ({
    gameId: card.getAttribute('data-game-id'),
    homeTeam: card.querySelector('.home-team')?.textContent,
    awayTeam: card.querySelector('.away-team')?.textContent,
    gameTime: card.querySelector('.game-time')?.getAttribute('datetime'),
    gameUrl: card.querySelector('a')?.href,
  }));
});

// 3. Store in database for scheduling
await saveGamesToDatabase(games);
```

**Output:** List of games with URLs and start times

### Phase 2: Live Score Monitoring (During Game)

**Objective:** Collect real-time scores at strategic intervals

**When:** From game start until halftime (0:00 2Q)

**Key Timestamps for ML Models:**
- Every minute from 0:00 1Q to 6:00 2Q (18 data points)
- Critical: 6:00 2Q (input for forecast)
- Critical: 0:00 2Q (halftime - target prediction)

**Scraping Frequency:**
```javascript
// scraping-schedule.js
const SCRAPING_SCHEDULE = {
  // First quarter: Every minute
  Q1: { interval: 60, unit: 'seconds' },
  
  // Second quarter until 6:00: Every minute
  Q2_before_prediction: { interval: 60, unit: 'seconds' },
  
  // From 6:00 2Q to halftime: Every 30 seconds (critical window)
  Q2_prediction_window: { interval: 30, unit: 'seconds' },
  
  // After halftime: Can reduce frequency or stop
  halftime: { interval: 300, unit: 'seconds' },  // Every 5 min
};

// Calculate next scrape time
const getNextScrapeTime = (currentQuarter, currentTime) => {
  if (currentQuarter === 1) {
    return SCRAPING_SCHEDULE.Q1.interval;
  } else if (currentQuarter === 2 && currentTime > '6:00') {
    return SCRAPING_SCHEDULE.Q2_before_prediction.interval;
  } else if (currentQuarter === 2 && currentTime <= '6:00') {
    return SCRAPING_SCHEDULE.Q2_prediction_window.interval;
  } else {
    return SCRAPING_SCHEDULE.halftime.interval;
  }
};
```

### Phase 3: Data Validation & Storage

**Objective:** Validate scraped data and store for ML models

```javascript
// data-validation.js
const validateScrapedData = (data) => {
  const issues = [];
  
  // Check required fields
  if (!data.gameId) issues.push('Missing game ID');
  if (!data.currentQuarter) issues.push('Missing quarter');
  if (!data.timeRemaining) issues.push('Missing time');
  if (data.scoreHome === undefined) issues.push('Missing home score');
  if (data.scoreAway === undefined) issues.push('Missing away score');
  
  // Check data ranges
  if (data.currentQuarter < 1 || data.currentQuarter > 4) {
    issues.push(`Invalid quarter: ${data.currentQuarter}`);
  }
  
  if (data.scoreHome < 0 || data.scoreAway < 0) {
    issues.push('Negative scores detected');
  }
  
  // Check differential is calculated correctly
  const expectedDiff = data.scoreHome - data.scoreAway;
  if (data.differential !== expectedDiff) {
    issues.push(`Differential mismatch: ${data.differential} vs ${expectedDiff}`);
  }
  
  return {
    valid: issues.length === 0,
    issues: issues,
  };
};

// Store to database
const storeGameData = async (data) => {
  const validation = validateScrapedData(data);
  
  if (!validation.valid) {
    console.error('Data validation failed:', validation.issues);
    return false;
  }
  
  // Store in time-series database
  await db.collection('live_scores').insertOne({
    ...data,
    timestamp: new Date(),
    scrapedAt: new Date(),
    dataSource: 'betonline.ag',
  });
  
  return true;
};
```

---

## Implementation Code

### Complete Crawlee Scraper for BetOnline

```javascript
// betonline-scraper.js
import { PlaywrightCrawler, Dataset } from 'crawlee';
import { chromium } from 'playwright-extra';
import StealthPlugin from 'puppeteer-extra-plugin-stealth';

// Apply stealth
chromium.use(StealthPlugin());

class BetOnlineScraper {
  constructor(config = {}) {
    this.config = {
      maxConcurrency: config.maxConcurrency || 2,
      requestDelay: config.requestDelay || 3000,  // 3 seconds
      useProxy: config.useProxy || true,
      headless: config.headless !== false,
      ...config,
    };
    
    this.crawler = null;
    this.initCrawler();
  }
  
  initCrawler() {
    this.crawler = new PlaywrightCrawler({
      launchContext: {
        launcher: chromium,
        useChrome: true,
        launchOptions: {
          headless: this.config.headless,
          args: [
            '--disable-blink-features=AutomationControlled',
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-web-security',
          ],
        },
      },
      
      // Proxy configuration
      proxyConfiguration: this.config.useProxy ? {
        proxyUrls: this.config.proxyUrls || [],
      } : undefined,
      
      // Rate limiting
      maxConcurrency: this.config.maxConcurrency,
      minConcurrency: 1,
      maxRequestRetries: 5,
      requestHandlerTimeoutSecs: 60,
      
      // Session management
      useSessionPool: true,
      persistCookiesPerSession: true,
      
      // Main request handler
      requestHandler: async ({ page, request, log }) => {
        log.info(`Scraping ${request.url}`);
        
        // Set random user agent
        await page.setUserAgent(this.getRandomUserAgent());
        
        // Wait for page to load
        await page.waitForLoadState('domcontentloaded');
        
        // Add human-like delay
        await this.randomDelay(1000, 3000);
        
        // Check if it's game list or individual game
        if (request.url.includes('/preseason') || request.url.includes('/nba')) {
          await this.scrapeGameList(page, log);
        } else if (request.url.includes('/game/') || request.url.includes('/live/')) {
          await this.scrapeLiveGame(page, log);
        }
        
        // Random delay before next request
        await this.randomDelay(this.config.requestDelay, this.config.requestDelay + 2000);
      },
      
      // Error handler
      failedRequestHandler: async ({ request, error, log }) => {
        log.error(`Request ${request.url} failed: ${error.message}`);
        
        if (error.message.includes('429')) {
          log.warning('Rate limited! Backing off for 2 minutes...');
          await new Promise(resolve => setTimeout(resolve, 120000));
        }
      },
    });
  }
  
  async scrapeGameList(page, log) {
    log.info('Scraping game list page');
    
    try {
      // Wait for game cards to load (adjust selector based on actual site)
      await page.waitForSelector('.game-line, .event-line, [data-game-id]', { timeout: 10000 });
      
      // Extract games
      const games = await page.$$eval('[data-game-id], .game-line, .event-line', (elements) => {
        return elements.map(el => {
          // Adjust selectors based on BetOnline's actual HTML structure
          const homeTeam = el.querySelector('.home-team, .team-home')?.textContent?.trim();
          const awayTeam = el.querySelector('.away-team, .team-away')?.textContent?.trim();
          const gameTime = el.querySelector('.game-time, .event-time')?.textContent?.trim();
          const gameUrl = el.querySelector('a')?.href;
          const liveIndicator = el.querySelector('.live, .in-play');
          
          return {
            homeTeam,
            awayTeam,
            gameTime,
            gameUrl,
            isLive: !!liveIndicator,
            scrapedAt: new Date().toISOString(),
          };
        }).filter(game => game.homeTeam && game.awayTeam);
      });
      
      log.info(`Found ${games.length} games`);
      
      // Save to dataset
      await Dataset.pushData(games);
      
      // Add live games to queue for detailed scraping
      for (const game of games) {
        if (game.isLive && game.gameUrl) {
          await this.crawler.addRequests([game.gameUrl]);
        }
      }
      
    } catch (error) {
      log.error(`Error scraping game list: ${error.message}`);
    }
  }
  
  async scrapeLiveGame(page, log) {
    log.info('Scraping live game page');
    
    try {
      // Wait for score elements (adjust selectors)
      await page.waitForSelector('.score, .current-score, [data-score]', { timeout: 10000 });
      
      // Extract live game data
      const gameData = await page.evaluate(() => {
        // Adjust these selectors based on BetOnline's actual HTML structure
        const homeScore = document.querySelector('.home-score, .score-home')?.textContent?.trim();
        const awayScore = document.querySelector('.away-score, .score-away')?.textContent?.trim();
        const quarter = document.querySelector('.quarter, .period')?.textContent?.trim();
        const timeRemaining = document.querySelector('.time-remaining, .clock')?.textContent?.trim();
        const homeTeam = document.querySelector('.home-team-name')?.textContent?.trim();
        const awayTeam = document.querySelector('.away-team-name')?.textContent?.trim();
        
        // Parse scores
        const scoreHome = parseInt(homeScore) || 0;
        const scoreAway = parseInt(awayScore) || 0;
        const differential = scoreHome - scoreAway;
        
        return {
          homeTeam,
          awayTeam,
          scoreHome,
          scoreAway,
          differential,
          quarter,
          timeRemaining,
          timestamp: new Date().toISOString(),
          url: window.location.href,
        };
      });
      
      log.info(`Live game: ${gameData.homeTeam} ${gameData.scoreHome} - ${gameData.scoreAway} ${gameData.awayTeam} (${gameData.quarter} ${gameData.timeRemaining})`);
      
      // Save to dataset
      await Dataset.pushData(gameData);
      
      return gameData;
      
    } catch (error) {
      log.error(`Error scraping live game: ${error.message}`);
      return null;
    }
  }
  
  getRandomUserAgent() {
    const userAgents = [
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    ];
    return userAgents[Math.floor(Math.random() * userAgents.length)];
  }
  
  async randomDelay(min = 2000, max = 5000) {
    const delay = Math.random() * (max - min) + min;
    await new Promise(resolve => setTimeout(resolve, delay));
  }
  
  async run(urls) {
    await this.crawler.run(urls);
  }
}

export default BetOnlineScraper;
```

### Usage Example

```javascript
// run-scraper.js
import BetOnlineScraper from './betonline-scraper.js';

const main = async () => {
  const scraper = new BetOnlineScraper({
    maxConcurrency: 2,
    requestDelay: 3000,  // 3 seconds between requests
    useProxy: true,
    proxyUrls: [
      'http://user:pass@proxy1.com:8080',
      'http://user:pass@proxy2.com:8080',
    ],
    headless: true,
  });
  
  // Start with main basketball page
  const startUrls = [
    'https://www.betonline.ag/sportsbook/basketball/preseason',
    'https://www.betonline.ag/sportsbook/basketball/nba',
  ];
  
  console.log('Starting BetOnline scraper...');
  await scraper.run(startUrls);
  console.log('Scraping complete!');
};

main().catch(console.error);
```

### Scheduled Scraping for Live Games

```javascript
// scheduler.js
import BetOnlineScraper from './betonline-scraper.js';
import cron from 'node-cron';

class LiveGameScheduler {
  constructor() {
    this.scraper = new BetOnlineScraper({
      maxConcurrency: 1,  // One at a time for live monitoring
      requestDelay: 30000,  // 30 seconds between updates
    });
    
    this.activeGames = new Map();  // gameId -> gameUrl
  }
  
  async discoverGames() {
    console.log('Discovering live games...');
    // Scrape main page to find live games
    await this.scraper.run([
      'https://www.betonline.ag/sportsbook/basketball/nba',
      'https://www.betonline.ag/sportsbook/live',
    ]);
    
    // TODO: Parse results and update activeGames map
  }
  
  async monitorLiveGames() {
    if (this.activeGames.size === 0) {
      console.log('No active games to monitor');
      return;
    }
    
    console.log(`Monitoring ${this.activeGames.size} live games`);
    
    for (const [gameId, gameUrl] of this.activeGames) {
      await this.scraper.run([gameUrl]);
      // Add delay between games
      await new Promise(resolve => setTimeout(resolve, 10000));  // 10 sec
    }
  }
  
  start() {
    console.log('Starting live game scheduler...');
    
    // Discover games every 5 minutes
    cron.schedule('*/5 * * * *', async () => {
      await this.discoverGames();
    });
    
    // Monitor live games every minute
    cron.schedule('* * * * *', async () => {
      await this.monitorLiveGames();
    });
    
    // Initial discovery
    this.discoverGames();
  }
}

// Run scheduler
const scheduler = new LiveGameScheduler();
scheduler.start();
```

---

## Integration with ML Models

### Data Format for Informer/Conformal/Dejavu

**Transform scraped data to match ML model requirements:**

```javascript
// transform-to-timeseries.js
import { MongoClient } from 'mongodb';

class DataTransformer {
  constructor(mongoUrl) {
    this.client = new MongoClient(mongoUrl);
    this.db = null;
  }
  
  async connect() {
    await this.client.connect();
    this.db = this.client.db('nba_forecasting');
  }
  
  async getGameTimeSeries(gameId) {
    // Get all scraped data points for this game
    const dataPoints = await this.db
      .collection('live_scores')
      .find({ gameId })
      .sort({ timestamp: 1 })
      .toArray();
    
    // Convert to minute-by-minute format
    const timeSeries = this.convertToMinuteSeries(dataPoints);
    
    return timeSeries;
  }
  
  convertToMinuteSeries(dataPoints) {
    // Map to minute indices (0-47 for full game, 0-23 for halftime)
    const minuteSeries = [];
    
    for (const point of dataPoints) {
      const minute = this.parseGameMinute(point.quarter, point.timeRemaining);
      
      minuteSeries.push({
        minute,
        quarter: point.quarter,
        timeRemaining: point.timeRemaining,
        differential: point.differential,
        scoreHome: point.scoreHome,
        scoreAway: point.scoreAway,
        timestamp: point.timestamp,
      });
    }
    
    // Fill in missing minutes with interpolation
    return this.fillMissingMinutes(minuteSeries);
  }
  
  parseGameMinute(quarter, timeRemaining) {
    // Convert quarter and time to minute index
    // Quarter 1: minutes 0-11
    // Quarter 2: minutes 12-23
    // etc.
    
    const quarterInt = parseInt(quarter.replace(/[^0-9]/g, ''));
    const [minutes, seconds] = timeRemaining.split(':').map(Number);
    
    // Minutes elapsed in quarter
    const minutesInQuarter = 12 - minutes - (seconds > 30 ? 0 : 1);
    
    // Total minute index
    const totalMinute = (quarterInt - 1) * 12 + minutesInQuarter;
    
    return totalMinute;
  }
  
  fillMissingMinutes(minuteSeries) {
    // Create complete 48-minute series with interpolation
    const complete = Array(48).fill(null).map((_, i) => {
      const existing = minuteSeries.find(p => p.minute === i);
      if (existing) return existing;
      
      // Interpolate
      const before = minuteSeries.filter(p => p.minute < i).slice(-1)[0];
      const after = minuteSeries.find(p => p.minute > i);
      
      if (before && after) {
        // Linear interpolation
        const ratio = (i - before.minute) / (after.minute - before.minute);
        return {
          minute: i,
          differential: before.differential + (after.differential - before.differential) * ratio,
          interpolated: true,
        };
      }
      
      return { minute: i, differential: null, interpolated: true };
    });
    
    return complete;
  }
  
  async exportForInformer(gameId) {
    const timeSeries = await this.getGameTimeSeries(gameId);
    
    // Extract input sequence (minutes 0-17, i.e., up to 6:00 2Q)
    const xEnc = timeSeries.slice(0, 18).map(p => p.differential);
    
    // Extract target (minutes 18-23, i.e., 6:00 2Q to halftime)
    const y = timeSeries.slice(18, 24).map(p => p.differential);
    
    // Get game metadata for temporal features
    const gameData = await this.db.collection('games').findOne({ gameId });
    const gameDate = new Date(gameData.gameTime);
    
    const temporalFeatures = Array(18).fill({
      month: gameDate.getMonth() + 1,
      day: gameDate.getDate(),
      weekday: gameDate.getDay(),
      hour: gameDate.getHours(),
    });
    
    return {
      x_enc: xEnc,
      x_mark_enc: temporalFeatures,
      y: y,
      gameId: gameId,
    };
  }
  
  async exportForDejavu(gameId) {
    const timeSeries = await this.getGameTimeSeries(gameId);
    
    // Dejavu pattern: 18-minute pattern + 6-minute outcome
    const pattern = timeSeries.slice(0, 18).map(p => p.differential);
    const outcome = timeSeries.slice(18, 24).map(p => p.differential);
    
    const gameData = await this.db.collection('games').findOne({ gameId });
    
    return {
      pattern: pattern,
      outcome: outcome,
      halftime_differential: outcome[5],  // At minute 23 (0:00 2Q)
      game_id: gameId,
      date: gameData.gameTime,
      teams: `${gameData.homeTeam} vs ${gameData.awayTeam}`,
      metadata: {
        season: gameData.season,
        game_type: gameData.gameType,
      },
    };
  }
}

export default DataTransformer;
```

---

## Deployment & Monitoring

### Production Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AWS/Cloud Infrastructure              │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │  EC2/Container (Node.js)                           │ │
│  │  ┌──────────────────────────────────────────────┐ │ │
│  │  │  Crawlee Scraper (Always Running)            │ │ │
│  │  │  • Game discovery scheduler                  │ │ │
│  │  │  • Live game monitor                         │ │ │
│  │  │  • Error handling & retry logic              │ │ │
│  │  └──────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────┘ │
│                          │                               │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │  MongoDB/PostgreSQL (Time-Series DB)              │ │
│  │  • Raw scraped data                               │ │
│  │  • Processed time series                         │ │
│  │  • Error logs                                     │ │
│  └────────────────────────────────────────────────────┘ │
│                          │                               │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │  ML Model Pipeline (Python)                       │ │
│  │  • Informer forecasting                           │ │
│  │  • Conformal prediction                           │ │
│  │  • Dejavu pattern matching                        │ │
│  └────────────────────────────────────────────────────┘ │
│                          │                               │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Monitoring & Alerting                            │ │
│  │  • Grafana dashboards                             │ │
│  │  • Alert on scraping failures                     │ │
│  │  • Data quality monitoring                        │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Monitoring Setup

```javascript
// monitoring.js
import { Telegraf } from 'telegraf';  // Telegram bot for alerts
import winston from 'winston';

class ScraperMonitoring {
  constructor(telegramToken, chatId) {
    // Set up logging
    this.logger = winston.createLogger({
      level: 'info',
      format: winston.format.json(),
      transports: [
        new winston.transports.File({ filename: 'error.log', level: 'error' }),
        new winston.transports.File({ filename: 'combined.log' }),
        new winston.transports.Console(),
      ],
    });
    
    // Set up Telegram alerts
    this.bot = new Telegraf(telegramToken);
    this.chatId = chatId;
    
    // Metrics
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      rateLimitHits: 0,
      lastScrapeTime: null,
    };
  }
  
  logSuccess(url, data) {
    this.metrics.totalRequests++;
    this.metrics.successfulRequests++;
    this.metrics.lastScrapeTime = new Date();
    
    this.logger.info('Scrape successful', { url, dataPoints: data.length });
  }
  
  logError(url, error) {
    this.metrics.totalRequests++;
    this.metrics.failedRequests++;
    
    this.logger.error('Scrape failed', { url, error: error.message });
    
    // Alert on repeated failures
    const failureRate = this.metrics.failedRequests / this.metrics.totalRequests;
    if (failureRate > 0.2) {  // More than 20% failure rate
      this.sendAlert(`⚠️ High failure rate: ${(failureRate * 100).toFixed(1)}%`);
    }
  }
  
  logRateLimit(url) {
    this.metrics.rateLimitHits++;
    this.logger.warn('Rate limit hit', { url });
    
    // Alert on rate limiting
    this.sendAlert(`⚠️ Rate limited on ${url}`);
  }
  
  async sendAlert(message) {
    try {
      await this.bot.telegram.sendMessage(this.chatId, message);
    } catch (error) {
      this.logger.error('Failed to send alert', { error: error.message });
    }
  }
  
  getMetrics() {
    return {
      ...this.metrics,
      successRate: this.metrics.successfulRequests / this.metrics.totalRequests,
      minutesSinceLastScrape: this.metrics.lastScrapeTime 
        ? (Date.now() - this.metrics.lastScrapeTime) / 60000
        : null,
    };
  }
}

export default ScraperMonitoring;
```

### Health Check Endpoint

```javascript
// server.js
import express from 'express';
import ScraperMonitoring from './monitoring.js';

const app = express();
const monitoring = new ScraperMonitoring(
  process.env.TELEGRAM_TOKEN,
  process.env.TELEGRAM_CHAT_ID
);

// Health check endpoint
app.get('/health', (req, res) => {
  const metrics = monitoring.getMetrics();
  
  const isHealthy = 
    metrics.successRate > 0.8 &&  // 80%+ success rate
    metrics.minutesSinceLastScrape < 10;  // Scraped within last 10 min
  
  res.status(isHealthy ? 200 : 503).json({
    status: isHealthy ? 'healthy' : 'unhealthy',
    metrics: metrics,
  });
});

// Metrics endpoint
app.get('/metrics', (req, res) => {
  res.json(monitoring.getMetrics());
});

app.listen(3000, () => {
  console.log('Monitoring server running on port 3000');
});
```

---

## Complete Setup Checklist

### Development Environment
- [ ] Install Node.js 18+ and npm
- [ ] Install Crawlee: `npm install crawlee playwright`
- [ ] Install stealth plugin: `npm install puppeteer-extra-plugin-stealth`
- [ ] Set up development proxy (optional)
- [ ] Configure BetOnline approval/API key (if provided)

### Scraper Configuration
- [ ] Identify correct CSS selectors for BetOnline (inspect page)
- [ ] Test scraper on single game page (headless=false for debugging)
- [ ] Implement error handling and retries
- [ ] Configure rate limiting (start conservative: 5-10 sec)
- [ ] Set up proxy rotation
- [ ] Add user-agent rotation
- [ ] Enable stealth plugin

### Anti-Detection
- [ ] Test with BetOnline approval documentation ready
- [ ] Monitor for 429 errors and adjust rate limits
- [ ] Use residential proxies for production
- [ ] Rotate sessions every 10-20 requests
- [ ] Add random delays (2-5 seconds)
- [ ] Human-like mouse movements

### Data Pipeline
- [ ] Set up MongoDB or PostgreSQL for storage
- [ ] Create schema for live scores
- [ ] Implement data validation
- [ ] Build time-series transformation
- [ ] Export functions for Informer/Conformal/Dejavu

### Scheduling
- [ ] Game discovery: Every 5-10 minutes
- [ ] Live monitoring: Every 30-60 seconds during games
- [ ] Focus on Q1 start to halftime (0:00 2Q)
- [ ] Reduce/stop after halftime

### Monitoring
- [ ] Set up logging (Winston or similar)
- [ ] Create health check endpoint
- [ ] Configure alerts (Telegram/email)
- [ ] Dashboard for metrics (Grafana/custom)
- [ ] Track success rate, failures, rate limits

### Production Deployment
- [ ] Deploy to AWS EC2 or Container (Docker)
- [ ] Set up database (managed MongoDB/PostgreSQL)
- [ ] Configure environment variables
- [ ] Set up process manager (PM2)
- [ ] Enable auto-restart on failure
- [ ] Schedule backups
- [ ] Document API endpoints

---

## Best Practices Summary

### 1. Respect BetOnline
- ✅ Start slow (5-10 sec/request), speed up gradually
- ✅ Contact them for rate limit guidance
- ✅ Use their API if available (always better than scraping!)
- ✅ Document your approval
- ✅ Monitor for 429s and back off immediately

### 2. Anti-Detection
- ✅ Use residential proxies (not datacenter)
- ✅ Rotate user-agents
- ✅ Enable stealth plugin
- ✅ Add random delays
- ✅ Maintain sessions/cookies

### 3. Data Quality
- ✅ Validate every scraped data point
- ✅ Handle missing data gracefully
- ✅ Store raw + processed data
- ✅ Log all errors
- ✅ Monitor data freshness

### 4. Reliability
- ✅ Automatic retries with exponential backoff
- ✅ Health checks and alerts
- ✅ Failover proxies
- ✅ Process monitoring (PM2/systemd)
- ✅ Graceful error handling

### 5. Integration
- ✅ Export data in ML model format
- ✅ Real-time streaming to forecasting pipeline
- ✅ Version your data schema
- ✅ Document data transformations

---

## Resources

### Crawlee Documentation
- **Official Docs:** https://crawlee.dev/
- **GitHub:** https://github.com/apify/crawlee
- **Examples:** https://crawlee.dev/docs/examples

### Proxy Providers
- **Bright Data:** https://brightdata.com
- **Oxylabs:** https://oxylabs.io
- **SmartProxy:** https://smartproxy.com
- **Apify Proxy:** https://apify.com/proxy

### Stealth & Anti-Detection
- **puppeteer-extra:** https://github.com/berstend/puppeteer-extra
- **Stealth Plugin:** https://github.com/berstend/puppeteer-extra/tree/master/packages/puppeteer-extra-plugin-stealth

### Monitoring
- **Winston (logging):** https://github.com/winstonjs/winston
- **Telegraf (Telegram bot):** https://telegraf.js.org
- **Grafana:** https://grafana.com

---

## Summary

**Strategic BetOnline Scraping with Crawlee:**
- ✅ Use Crawlee + Playwright for resilient scraping
- ✅ Apply stealth plugin to evade JavaScript detection
- ✅ Rotate proxies and user-agents
- ✅ Respect rate limits (start 5-10 sec/request)
- ✅ Schedule intelligently (focus on Q1-halftime)
- ✅ Transform data for ML models (Informer, Conformal, Dejavu)
- ✅ Monitor health and alert on failures

**Result:** Reliable real-time NBA score collection for live forecasting! 🏀

---

*Version 1.0.0 - October 15, 2025*

