# BetOnline Implementation Specification

**Technical Specifications for Production Web Scraping System**

**Date:** October 15, 2025  
**Version:** 1.0.0  
**Status:** Production-Ready Blueprint

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Production System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Scraping Layer (Node.js + Crawlee)                      │   │
│  │  ┌────────────────┐  ┌────────────────┐                 │   │
│  │  │ Game Discovery │  │ Live Monitoring│                 │   │
│  │  │ Scheduler      │  │ Workers (x5)   │                 │   │
│  │  └────────────────┘  └────────────────┘                 │   │
│  │           │                    │                          │   │
│  │           ▼                    ▼                          │   │
│  │  ┌─────────────────────────────────────┐                │   │
│  │  │  Crawlee + Playwright + Stealth     │                │   │
│  │  └─────────────────────────────────────┘                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Data Storage Layer                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │ MongoDB     │  │ Redis       │  │ PostgreSQL  │     │   │
│  │  │ (Raw Data)  │  │ (Cache)     │  │ (Time Series│     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Processing Layer (Node.js/Python)                       │   │
│  │  ┌────────────────┐  ┌────────────────┐                 │   │
│  │  │ Data Transform │  │ Validation     │                 │   │
│  │  │ Pipeline       │  │ & QC           │                 │   │
│  │  └────────────────┘  └────────────────┘                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ML Integration Layer (Python)                           │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐                     │   │
│  │  │Informer│  │Conformal│ │Dejavu  │                     │   │
│  │  └────────┘  └────────┘  └────────┘                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Monitoring & Alerting                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │ Grafana     │  │ Prometheus  │  │ Telegram    │     │   │
│  │  │ Dashboard   │  │ Metrics     │  │ Alerts      │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Specifications

### 1. Scraper Service Specification

**Technology Stack:**
- **Runtime:** Node.js 18+ LTS
- **Framework:** Crawlee 3.5+
- **Browser:** Playwright 1.40+
- **Stealth:** puppeteer-extra-plugin-stealth 2.11+
- **Language:** TypeScript 5.x (or JavaScript ES2022)

**Configuration Schema:**

```typescript
interface ScraperConfig {
  // Rate limiting
  maxConcurrency: number;           // Default: 2
  requestDelayMs: number;           // Default: 3000 (3 sec)
  maxRequestsPerMinute: number;     // Default: 20
  
  // Proxy settings
  useProxy: boolean;                // Default: true (production)
  proxyUrls: string[];              // List of proxy URLs
  proxyRotationStrategy: 'round-robin' | 'random' | 'sticky';
  
  // Browser settings
  headless: boolean;                // Default: true
  browserTimeout: number;           // Default: 60000 (60 sec)
  useChrome: boolean;               // Default: true (vs Chromium)
  
  // Retry logic
  maxRetries: number;               // Default: 5
  retryDelayMs: number;             // Default: 5000 (5 sec)
  exponentialBackoff: boolean;      // Default: true
  
  // Session management
  useSessionPool: boolean;          // Default: true
  maxSessionPoolSize: number;       // Default: 10
  maxSessionUsageCount: number;     // Default: 20
  
  // Data storage
  storageType: 'mongodb' | 'postgresql';
  databaseUrl: string;
  
  // Monitoring
  enableMetrics: boolean;           // Default: true
  metricsPort: number;              // Default: 3000
  telegramBotToken?: string;
  telegramChatId?: string;
}
```

**Example Production Config:**

```javascript
// config/production.js
export default {
  maxConcurrency: 3,
  requestDelayMs: 3000,
  maxRequestsPerMinute: 15,
  
  useProxy: true,
  proxyUrls: [
    'http://user:pass@proxy1.brightdata.com:24000',
    'http://user:pass@proxy2.brightdata.com:24000',
    'http://user:pass@proxy3.brightdata.com:24000',
  ],
  proxyRotationStrategy: 'round-robin',
  
  headless: true,
  browserTimeout: 60000,
  useChrome: true,
  
  maxRetries: 5,
  retryDelayMs: 5000,
  exponentialBackoff: true,
  
  useSessionPool: true,
  maxSessionPoolSize: 10,
  maxSessionUsageCount: 15,
  
  storageType: 'mongodb',
  databaseUrl: process.env.MONGODB_URL,
  
  enableMetrics: true,
  metricsPort: 3000,
  telegramBotToken: process.env.TELEGRAM_BOT_TOKEN,
  telegramChatId: process.env.TELEGRAM_CHAT_ID,
};
```

---

### 2. URL Discovery Patterns

**BetOnline URL Structure:**

```javascript
const BETONLINE_URLS = {
  // Main pages (for game discovery)
  mainPages: [
    'https://www.betonline.ag/sportsbook/basketball/nba',
    'https://www.betonline.ag/sportsbook/basketball/preseason',
    'https://www.betonline.ag/sportsbook/live',  // Live games only
  ],
  
  // Game page patterns (to construct or extract)
  gamePagePatterns: [
    'https://www.betonline.ag/sportsbook/basketball/game/{gameId}',
    'https://www.betonline.ag/sportsbook/basketball/nba/{team1-vs-team2}',
    'https://www.betonline.ag/sportsbook/live/basketball/{gameId}',
  ],
};
```

**CSS Selectors (To be refined via inspection):**

```javascript
// NOTE: These selectors are EXAMPLES - inspect BetOnline.ag to get actual selectors
const SELECTORS = {
  gameList: {
    container: '.game-container, .event-container, [data-game-id]',
    gameCard: '.game-line, .event-line',
    homeTeam: '.home-team, .team-home',
    awayTeam: '.away-team, .team-away',
    gameTime: '.game-time, .event-time, [data-start-time]',
    gameUrl: 'a.game-link, a[href*="/game/"]',
    liveIndicator: '.live, .in-play, .live-indicator',
  },
  
  liveGame: {
    homeScore: '.home-score, .score-home, [data-home-score]',
    awayScore: '.away-score, .score-away, [data-away-score]',
    quarter: '.quarter, .period, [data-quarter]',
    gameClock: '.game-clock, .time-remaining, [data-time]',
    homeTeamName: '.home-team-name',
    awayTeamName: '.away-team-name',
  },
  
  odds: {
    homeOdds: '.home-odds, [data-home-odds]',
    awayOdds: '.away-odds, [data-away-odds]',
    overUnder: '.over-under, .total',
  },
};
```

**Selector Discovery Script:**

```javascript
// tools/inspect-selectors.js
import { chromium } from 'playwright';

async function discoverSelectors(url) {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  
  await page.goto(url);
  
  console.log('Page loaded. Inspect and copy selectors.');
  console.log('Press Ctrl+C when done.');
  
  // Wait indefinitely for manual inspection
  await page.waitForTimeout(3600000);  // 1 hour
  
  await browser.close();
}

// Run: node tools/inspect-selectors.js
discoverSelectors('https://www.betonline.ag/sportsbook/basketball/nba');
```

---

### 3. Data Schema

**Raw Scraped Data (MongoDB):**

```typescript
interface RawScrapedData {
  _id: ObjectId;
  gameId: string;               // e.g., "LAL-BOS-20251015"
  homeTeam: string;             // "Los Angeles Lakers"
  awayTeam: string;             // "Boston Celtics"
  scoreHome: number;            // 52
  scoreAway: number;            // 48
  differential: number;         // 4 (home - away)
  quarter: string;              // "2Q"
  timeRemaining: string;        // "6:32"
  gameMinute: number;           // 17 (computed)
  timestamp: Date;              // When scraped
  sourceUrl: string;            // BetOnline URL
  odds: {
    homeOdds: number;           // -110
    awayOdds: number;           // +105
    overUnder: number;          // 220.5
  };
  metadata: {
    scraperId: string;          // Which scraper instance
    userAgent: string;
    proxyUsed: string;
    latencyMs: number;
  };
}
```

**Processed Time Series (PostgreSQL):**

```sql
CREATE TABLE game_time_series (
  id SERIAL PRIMARY KEY,
  game_id VARCHAR(100) NOT NULL,
  game_date DATE NOT NULL,
  home_team VARCHAR(50) NOT NULL,
  away_team VARCHAR(50) NOT NULL,
  minute INTEGER NOT NULL,           -- 0-47
  quarter INTEGER NOT NULL,          -- 1-4
  time_remaining_quarter INTEGER,    -- 12-0
  differential FLOAT NOT NULL,
  score_home INTEGER,
  score_away INTEGER,
  interpolated BOOLEAN DEFAULT FALSE,
  timestamp TIMESTAMP NOT NULL,
  
  -- Indexes for fast queries
  INDEX idx_game_id (game_id),
  INDEX idx_game_date (game_date),
  INDEX idx_minute (minute),
  UNIQUE (game_id, minute)
);
```

**ML Model Export Format:**

```typescript
interface InformerFormat {
  gameId: string;
  xEnc: number[];                // [18] - differentials from minute 0-17
  xMarkEnc: number[][];          // [18, 4] - temporal features
  y: number[];                   // [6] - differentials from minute 18-23
  metadata: {
    gameDate: Date;
    teams: string;
    actualHalftimeDiff: number;
  };
}

interface DejavuFormat {
  pattern: number[];             // [18] - same as xEnc
  outcome: number[];             // [6] - same as y
  halftimeDifferential: number;  // outcome[5]
  gameId: string;
  date: Date;
  teams: string;
  metadata: {
    season: string;
    gameType: string;
  };
}
```

---

### 4. Scraping Schedule Specification

**Game Discovery Schedule:**

```javascript
// Using node-cron syntax
const DISCOVERY_SCHEDULE = {
  // Every 10 minutes during NBA season
  regularSeason: '*/10 * * * *',
  
  // Every 5 minutes on game days
  gameDays: '*/5 * * * *',
  
  // Every 2 minutes during prime time (7-11 PM ET)
  primeTime: '*/2 19-23 * * *',
};
```

**Live Game Monitoring Schedule:**

```javascript
const MONITORING_SCHEDULE = {
  // Quarters and frequency
  periods: [
    {
      quarter: 1,
      startMinute: 0,
      endMinute: 11,
      intervalSeconds: 60,        // Every 60 seconds
    },
    {
      quarter: 2,
      startMinute: 12,
      endMinute: 17,
      intervalSeconds: 60,        // Every 60 seconds
    },
    {
      quarter: 2,
      startMinute: 18,            // 6:00 2Q
      endMinute: 23,              // Halftime
      intervalSeconds: 30,        // Every 30 seconds (CRITICAL)
    },
    {
      quarter: 2,
      startMinute: 24,
      endMinute: 47,
      intervalSeconds: 300,       // Every 5 minutes (post-halftime, optional)
    },
  ],
  
  // Stop conditions
  stopAt: 'halftime',  // or 'end-of-game'
  
  // Retry on errors
  retryOnError: true,
  maxConsecutiveErrors: 3,
};
```

**Implementation:**

```javascript
// scheduler.js
import cron from 'node-cron';

class GameScheduler {
  constructor(scraper) {
    this.scraper = scraper;
    this.activeGames = new Map();
  }
  
  start() {
    // Game discovery
    cron.schedule(DISCOVERY_SCHEDULE.regularSeason, async () => {
      await this.discoverGames();
    });
    
    // Live monitoring (runs every second, checks if any game needs update)
    cron.schedule('* * * * * *', async () => {  // Every second
      await this.checkAndMonitorGames();
    });
  }
  
  async discoverGames() {
    // Scrape main pages for game list
    const games = await this.scraper.scrapeGameList();
    
    for (const game of games) {
      if (game.isLive && !this.activeGames.has(game.gameId)) {
        this.activeGames.set(game.gameId, {
          ...game,
          lastScraped: null,
          errorCount: 0,
        });
      }
    }
  }
  
  async checkAndMonitorGames() {
    const now = Date.now();
    
    for (const [gameId, gameInfo] of this.activeGames) {
      // Determine scraping interval based on game state
      const interval = this.getScrapingInterval(gameInfo);
      
      // Check if it's time to scrape
      if (!gameInfo.lastScraped || (now - gameInfo.lastScraped) >= interval) {
        try {
          await this.scraper.scrapeLiveGame(gameInfo.url);
          gameInfo.lastScraped = now;
          gameInfo.errorCount = 0;
        } catch (error) {
          gameInfo.errorCount++;
          
          if (gameInfo.errorCount >= MONITORING_SCHEDULE.maxConsecutiveErrors) {
            console.error(`Stopping monitoring for ${gameId} due to errors`);
            this.activeGames.delete(gameId);
          }
        }
      }
      
      // Check if game is complete
      if (this.isGameComplete(gameInfo)) {
        this.activeGames.delete(gameId);
      }
    }
  }
  
  getScrapingInterval(gameInfo) {
    // Determine interval based on quarter and minute
    const currentMinute = this.parseGameMinute(gameInfo.quarter, gameInfo.timeRemaining);
    
    for (const period of MONITORING_SCHEDULE.periods) {
      if (currentMinute >= period.startMinute && currentMinute <= period.endMinute) {
        return period.intervalSeconds * 1000;  // Convert to ms
      }
    }
    
    return 60000;  // Default 1 minute
  }
  
  parseGameMinute(quarter, timeRemaining) {
    // Parse "6:32" in "2Q" to minute index
    const quarterInt = parseInt(quarter.replace(/[^0-9]/g, ''));
    const [minutes, seconds] = timeRemaining.split(':').map(Number);
    const minutesInQuarter = 12 - minutes - (seconds > 30 ? 0 : 1);
    return (quarterInt - 1) * 12 + minutesInQuarter;
  }
  
  isGameComplete(gameInfo) {
    if (MONITORING_SCHEDULE.stopAt === 'halftime') {
      return this.parseGameMinute(gameInfo.quarter, gameInfo.timeRemaining) >= 24;
    } else {
      return gameInfo.quarter === '4Q' && gameInfo.timeRemaining === '0:00';
    }
  }
}

export default GameScheduler;
```

---

### 5. Error Handling Specification

**Error Types and Responses:**

```typescript
enum ErrorType {
  RATE_LIMIT = 'RATE_LIMIT',           // 429 error
  BLOCKED = 'BLOCKED',                 // 403 error
  TIMEOUT = 'TIMEOUT',                 // Request timeout
  NETWORK = 'NETWORK',                 // Connection error
  PARSING = 'PARSING',                 // Can't parse HTML
  VALIDATION = 'VALIDATION',           // Data validation failed
  UNKNOWN = 'UNKNOWN',                 // Other errors
}

interface ErrorHandler {
  errorType: ErrorType;
  retryable: boolean;
  backoffMs: number;
  action: () => Promise<void>;
}
```

**Error Handling Logic:**

```javascript
const ERROR_HANDLERS = {
  [ErrorType.RATE_LIMIT]: {
    retryable: true,
    backoffMs: 120000,  // 2 minutes
    action: async (context) => {
      console.warn('Rate limited - backing off');
      await context.monitor.logRateLimit(context.url);
      await context.sleep(120000);
    },
  },
  
  [ErrorType.BLOCKED]: {
    retryable: true,
    backoffMs: 300000,  // 5 minutes
    action: async (context) => {
      console.error('Blocked - switching proxy');
      await context.monitor.sendAlert('⚠️ IP blocked, switching proxy');
      await context.proxyManager.rotateProxy();
    },
  },
  
  [ErrorType.TIMEOUT]: {
    retryable: true,
    backoffMs: 10000,  // 10 seconds
    action: async (context) => {
      console.warn('Timeout - will retry');
    },
  },
  
  [ErrorType.NETWORK]: {
    retryable: true,
    backoffMs: 30000,  // 30 seconds
    action: async (context) => {
      console.warn('Network error - will retry');
    },
  },
  
  [ErrorType.PARSING]: {
    retryable: false,
    backoffMs: 0,
    action: async (context) => {
      console.error('Parsing error - selectors may have changed');
      await context.monitor.sendAlert('❌ Parsing error - check selectors!');
      await context.logErrorPage(context.page);
    },
  },
  
  [ErrorType.VALIDATION]: {
    retryable: true,
    backoffMs: 5000,  // 5 seconds
    action: async (context) => {
      console.warn('Validation failed - data quality issue');
    },
  },
};
```

---

### 6. Monitoring Specification

**Metrics to Track:**

```typescript
interface ScraperMetrics {
  // Request metrics
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  rateLimitHits: number;
  avgLatencyMs: number;
  
  // Data metrics
  gamesDiscovered: number;
  gamesMonitored: number;
  dataPointsCollected: number;
  dataQualityScore: number;  // 0-100
  
  // System metrics
  uptime: number;            // Milliseconds
  lastScrapeTime: Date;
  activeWorkers: number;
  
  // Error metrics
  errorsByType: Record<ErrorType, number>;
  consecutiveErrors: number;
}
```

**Health Check Endpoints:**

```javascript
// server.js
import express from 'express';

const app = express();

// Health check
app.get('/health', (req, res) => {
  const metrics = monitoringService.getMetrics();
  
  const isHealthy = 
    metrics.successRate > 0.8 &&
    metrics.minutesSinceLastScrape < 10 &&
    metrics.consecutiveErrors < 3;
  
  res.status(isHealthy ? 200 : 503).json({
    status: isHealthy ? 'healthy' : 'unhealthy',
    timestamp: new Date(),
    metrics: metrics,
  });
});

// Detailed metrics (Prometheus format)
app.get('/metrics', (req, res) => {
  const metrics = monitoringService.getMetrics();
  
  // Prometheus exposition format
  const prometheusMetrics = `
# HELP scraper_requests_total Total number of scraping requests
# TYPE scraper_requests_total counter
scraper_requests_total{status="success"} ${metrics.successfulRequests}
scraper_requests_total{status="failed"} ${metrics.failedRequests}

# HELP scraper_latency_ms Average latency in milliseconds
# TYPE scraper_latency_ms gauge
scraper_latency_ms ${metrics.avgLatencyMs}

# HELP scraper_data_quality Data quality score (0-100)
# TYPE scraper_data_quality gauge
scraper_data_quality ${metrics.dataQualityScore}
  `.trim();
  
  res.type('text/plain').send(prometheusMetrics);
});

app.listen(3000);
```

---

### 7. Deployment Specification

**Docker Configuration:**

```dockerfile
# Dockerfile
FROM node:18-alpine

# Install Playwright dependencies
RUN apk add --no-cache \
    chromium \
    nss \
    freetype \
    harfbuzz \
    ca-certificates \
    ttf-freefont

# Set Playwright to use system chromium
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium-browser

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application files
COPY . .

# Expose metrics port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD node health-check.js || exit 1

# Run scraper
CMD ["node", "src/index.js"]
```

**Docker Compose:**

```yaml
# docker-compose.yml
version: '3.8'

services:
  scraper:
    build: .
    container_name: betonline-scraper
    restart: unless-stopped
    environment:
      - NODE_ENV=production
      - MONGODB_URL=mongodb://mongo:27017/nba_forecasting
      - REDIS_URL=redis://redis:6379
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
      - PROXY_URL_1=${PROXY_URL_1}
      - PROXY_URL_2=${PROXY_URL_2}
      - PROXY_URL_3=${PROXY_URL_3}
    ports:
      - "3000:3000"
    depends_on:
      - mongo
      - redis
    volumes:
      - ./logs:/app/logs
    networks:
      - nba-network
  
  mongo:
    image: mongo:7
    container_name: mongo-db
    restart: unless-stopped
    volumes:
      - mongo-data:/data/db
    networks:
      - nba-network
  
  redis:
    image: redis:7-alpine
    container_name: redis-cache
    restart: unless-stopped
    networks:
      - nba-network
  
  grafana:
    image: grafana/grafana:latest
    container_name: grafana-monitoring
    restart: unless-stopped
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - nba-network

volumes:
  mongo-data:
  grafana-data:

networks:
  nba-network:
    driver: bridge
```

**Environment Variables:**

```bash
# .env
NODE_ENV=production

# Database
MONGODB_URL=mongodb://localhost:27017/nba_forecasting
REDIS_URL=redis://localhost:6379

# Proxies (from Bright Data or similar)
PROXY_URL_1=http://user:pass@proxy1.brightdata.com:24000
PROXY_URL_2=http://user:pass@proxy2.brightdata.com:24000
PROXY_URL_3=http://user:pass@proxy3.brightdata.com:24000

# Monitoring
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
GRAFANA_PASSWORD=secure_password

# Scraper config
MAX_CONCURRENCY=3
REQUEST_DELAY_MS=3000
MAX_RETRIES=5
```

**Process Management (PM2):**

```javascript
// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'betonline-scraper',
    script: 'src/index.js',
    instances: 1,
    exec_mode: 'fork',
    watch: false,
    max_memory_restart: '500M',
    env: {
      NODE_ENV: 'production',
    },
    error_file: './logs/pm2-error.log',
    out_file: './logs/pm2-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    
    // Auto-restart on failure
    autorestart: true,
    max_restarts: 10,
    min_uptime: '10s',
    
    // Graceful shutdown
    kill_timeout: 5000,
  }],
};
```

---

## Implementation Checklist

### Phase 1: Setup (2-4 hours)
- [ ] Initialize Node.js project with TypeScript
- [ ] Install Crawlee, Playwright, stealth plugin
- [ ] Set up MongoDB/PostgreSQL
- [ ] Configure environment variables
- [ ] Set up logging (Winston)

### Phase 2: Core Scraper (3-4 hours)
- [ ] Inspect BetOnline pages to identify correct selectors
- [ ] Implement game discovery scraper
- [ ] Implement live game scraper
- [ ] Add stealth plugin and anti-detection measures
- [ ] Configure proxy rotation

### Phase 3: Scheduling (2-3 hours)
- [ ] Implement game discovery scheduler
- [ ] Implement live monitoring scheduler
- [ ] Add adaptive scraping intervals
- [ ] Test with real games

### Phase 4: Data Pipeline (2-3 hours)
- [ ] Implement data validation
- [ ] Build time-series transformation
- [ ] Create export functions for ML models
- [ ] Test data quality

### Phase 5: Monitoring (2-3 hours)
- [ ] Set up metrics collection
- [ ] Create health check endpoints
- [ ] Configure Telegram alerts
- [ ] Build Grafana dashboard

### Phase 6: Deployment (2-3 hours)
- [ ] Create Dockerfile
- [ ] Set up docker-compose
- [ ] Configure PM2
- [ ] Deploy to production server
- [ ] Run initial tests
- [ ] Monitor for 24 hours

### Phase 7: Refinement (2-4 hours)
- [ ] Adjust rate limits based on performance
- [ ] Fine-tune error handling
- [ ] Optimize scraping schedule
- [ ] Add additional data quality checks

**Total Estimated Time: 15-24 hours**

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Success Rate | >95% | Successful scrapes / total attempts |
| Data Freshness | <60 seconds | Time since last scrape for active games |
| Latency | <2 seconds | Average request duration |
| Uptime | >99% | System availability over 30 days |
| Data Completeness | >90% | Percentage of minutes with data |
| Error Recovery | <5 minutes | Time to recover from failures |

---

## Cost Estimate

**Monthly Operational Costs:**

| Item | Cost | Notes |
|------|------|-------|
| Residential Proxies | $50-150 | Bright Data, ~1-3 GB/month |
| Server (AWS EC2 t3.small) | $15-20 | Or use existing infrastructure |
| MongoDB Atlas (Shared) | $0-25 | Or self-hosted |
| Monitoring (Grafana Cloud) | $0-50 | Or self-hosted |
| **Total** | **$65-245/month** | Can be optimized lower |

**One-Time Costs:**
- Development time: 15-24 hours × your hourly rate
- Setup and testing: Included in development time

---

## Summary

This specification provides a production-ready blueprint for scraping BetOnline.ag with Crawlee. Key features:

- ✅ **Robust architecture** with retry logic and error handling
- ✅ **Anti-detection** via stealth plugin, proxies, and rate limiting
- ✅ **Intelligent scheduling** based on game state
- ✅ **Data quality** validation and transformation
- ✅ **ML integration** for Informer, Conformal, and Dejavu
- ✅ **Monitoring** with alerts and dashboards
- ✅ **Production deployment** with Docker and PM2

**Ready for implementation!** 🚀

---

*Version 1.0.0 - October 15, 2025*

