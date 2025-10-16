# NBA_API Setup & Configuration

**Objective:** Install and configure NBA_API for optimal performance  
**Duration:** 5-10 minutes  
**Output:** Working NBA_API client with test verification

---

## Installation

### Requirements

```bash
# Python 3.7 or higher required
python --version  # Should be 3.7+

# Install NBA_API
pip install nba-api

# Verify installation
python -c "import nba_api; print(nba_api.__version__)"
```

**Expected output:** `1.4.1` or higher

---

### Optional Dependencies (Recommended for SPEED)

```bash
# Faster JSON parsing (20-30% speedup)
pip install orjson

# Async HTTP requests (for concurrent polling)
pip install aiohttp

# Redis for caching (optional, reduces API calls)
pip install redis

# All together
pip install nba-api orjson aiohttp redis
```

---

## Quick Verification Test

### Test 1: Basic Import

```python
# test_nba_api.py
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.static import teams, players

print("✅ NBA_API imported successfully")

# Get teams
nba_teams = teams.get_teams()
print(f"✅ Fetched {len(nba_teams)} NBA teams")

# Get players
nba_players = players.get_players()
print(f"✅ Fetched {len(nba_players)} NBA players")

print("\n🎉 NBA_API is ready!")
```

**Run:**
```bash
python test_nba_api.py
```

**Expected output:**
```
✅ NBA_API imported successfully
✅ Fetched 30 NBA teams
✅ Fetched 4500+ NBA players

🎉 NBA_API is ready!
```

---

### Test 2: Live Data (Most Important!)

```python
# test_live_data.py
from nba_api.live.nba.endpoints import scoreboard
import time

print("Fetching live NBA games...")
start = time.time()

try:
    board = scoreboard.ScoreBoard()
    games = board.games.get_dict()
    
    elapsed = (time.time() - start) * 1000  # Convert to ms
    
    print(f"✅ Response time: {elapsed:.0f}ms")
    print(f"✅ Games found: {len(games)}")
    
    if games:
        game = games[0]
        print(f"\nSample game:")
        print(f"  {game['awayTeam']['teamName']} @ {game['homeTeam']['teamName']}")
        print(f"  Score: {game['awayTeam']['score']} - {game['homeTeam']['score']}")
        print(f"  Status: {game['gameStatus']}")
    else:
        print("\nℹ️  No live games right now (check during NBA season)")
    
    print("\n🎉 Live data working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Troubleshooting tips:")
    print("  1. Check internet connection")
    print("  2. Verify NBA.com is accessible")
    print("  3. Try again in a few seconds")
```

**Run:**
```bash
python test_live_data.py
```

**Expected output (during live games):**
```
Fetching live NBA games...
✅ Response time: 245ms
✅ Games found: 8

Sample game:
  Lakers @ Celtics
  Score: 52 - 48
  Status: 2  (2 = live)

🎉 Live data working!
```

---

## Configuration File

### Create `config/nba_api_config.py`

```python
"""
NBA_API Configuration
Optimized for SPEED and reliability
"""

# ============================================
# POLLING CONFIGURATION
# ============================================

# How often to poll NBA.com for live updates
POLL_INTERVAL_SECONDS = 10  # NBA.com updates ~every 10s

# How often to emit to WebSocket (can be faster than polling)
EMIT_INTERVAL_SECONDS = 5   # Emit every 5s for smooth UX

# Timeout for NBA_API requests
REQUEST_TIMEOUT_SECONDS = 3  # Fail fast if NBA.com is slow

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

# ============================================
# CACHING CONFIGURATION  
# ============================================

# Cache static data (teams, players)
CACHE_STATIC_DATA = True
STATIC_DATA_TTL_HOURS = 24  # Refresh daily

# Cache game data (avoid redundant processing)
CACHE_GAME_DATA = True
GAME_DATA_TTL_SECONDS = 5   # Short TTL for live data

# ============================================
# DATA PROCESSING
# ============================================

# Track games automatically or manual selection
AUTO_TRACK_ALL_GAMES = True  # Track all live games

# Specific teams to track (empty = all teams)
TRACKED_TEAMS = []  # e.g., ['LAL', 'BOS', 'GSW']

# Minimum quarter to start tracking (1 = from start)
MIN_QUARTER_TO_TRACK = 1

# Maximum games to track simultaneously
MAX_CONCURRENT_GAMES = 15  # Prevent overload

# ============================================
# ML MODEL TRIGGER
# ============================================

# Trigger prediction at this minute mark
PREDICTION_TRIGGER_MINUTE = 18  # 6:00 Q2 = 18th minute

# Pattern length for ML model
PATTERN_LENGTH = 18  # First 18 minutes

# ============================================
# PERFORMANCE OPTIMIZATION
# ============================================

# Use faster JSON parser if available
USE_ORJSON = True  # Requires: pip install orjson

# Use async HTTP for concurrent requests
USE_ASYNC_HTTP = True  # Requires: pip install aiohttp

# Connection pooling
HTTP_POOL_SIZE = 10
HTTP_POOL_MAX_SIZE = 20

# ============================================
# REDIS CACHING (OPTIONAL)
# ============================================

# Enable Redis for distributed caching
USE_REDIS = False  # Set to True if Redis available

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None

# ============================================
# LOGGING
# ============================================

# Log level
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR

# Log NBA_API requests
LOG_API_REQUESTS = True

# Log processing times
LOG_PERFORMANCE = True

# ============================================
# ERROR HANDLING
# ============================================

# Continue on API errors
CONTINUE_ON_ERROR = True

# Alert on consecutive failures
ALERT_THRESHOLD_FAILURES = 3

# Graceful degradation (use cached data if API fails)
USE_CACHED_ON_FAILURE = True
```

---

## Environment Variables (Production)

### Create `.env` file

```bash
# NBA_API Configuration
NBA_API_POLL_INTERVAL=10
NBA_API_EMIT_INTERVAL=5
NBA_API_REQUEST_TIMEOUT=3

# Redis (optional)
REDIS_ENABLED=false
REDIS_HOST=localhost
REDIS_PORT=6379

# Logging
NBA_API_LOG_LEVEL=INFO

# Performance
NBA_API_USE_ORJSON=true
NBA_API_USE_ASYNC=true
```

### Load in Python

```python
# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

POLL_INTERVAL = int(os.getenv('NBA_API_POLL_INTERVAL', 10))
EMIT_INTERVAL = int(os.getenv('NBA_API_EMIT_INTERVAL', 5))
REQUEST_TIMEOUT = int(os.getenv('NBA_API_REQUEST_TIMEOUT', 3))

# ... rest of config
```

---

## Performance Tuning

### 1. Enable orjson (Faster JSON)

```python
# utils/json_parser.py
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    import json
    USE_ORJSON = False

def parse_json(data):
    """Fast JSON parsing"""
    if USE_ORJSON:
        return orjson.loads(data)
    return json.loads(data)

def dumps_json(data):
    """Fast JSON serialization"""
    if USE_ORJSON:
        return orjson.dumps(data).decode('utf-8')
    return json.dumps(data)
```

**Speedup:** 20-30% faster JSON parsing

---

### 2. Connection Pooling

```python
# utils/http_client.py
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Create session with connection pooling
session = requests.Session()

# Configure retries
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504]
)

adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=retries
)

session.mount('http://', adapter)
session.mount('https://', adapter)

# Use this session for all NBA_API calls
# Results in 30-50% faster subsequent requests
```

---

### 3. Async Polling (Multiple Games)

```python
# services/async_poller.py
import asyncio
import aiohttp
from nba_api.live.nba.endpoints import scoreboard

async def poll_live_games_async():
    """Poll NBA_API asynchronously"""
    async with aiohttp.ClientSession() as session:
        while True:
            start = asyncio.get_event_loop().time()
            
            # Fetch scoreboard
            board = scoreboard.ScoreBoard()
            games = board.games.get_dict()
            
            # Process all games concurrently
            tasks = [process_game(game) for game in games]
            await asyncio.gather(*tasks)
            
            elapsed = asyncio.get_event_loop().time() - start
            print(f"Processed {len(games)} games in {elapsed*1000:.0f}ms")
            
            await asyncio.sleep(10)

async def process_game(game):
    """Process single game (async)"""
    # Extract data, build pattern, etc.
    pass
```

**Speedup:** Process 10 games in parallel vs sequential

---

## Directory Structure

```
project/
├── config/
│   ├── nba_api_config.py       # Configuration
│   └── settings.py             # Environment variables
│
├── services/
│   ├── nba_data_service.py     # Main polling service
│   ├── score_buffer.py         # Pattern building
│   └── cache_service.py        # Redis caching
│
├── utils/
│   ├── json_parser.py          # Fast JSON
│   ├── http_client.py          # Connection pooling
│   └── logger.py               # Logging
│
├── tests/
│   ├── test_nba_api.py         # Basic tests
│   └── test_live_data.py       # Live data tests
│
├── .env                        # Environment variables
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## Troubleshooting

### Issue: Slow API Responses

**Symptoms:** NBA_API calls taking >1 second

**Solutions:**
1. Check internet connection speed
2. Enable connection pooling (see above)
3. Use orjson for faster parsing
4. Check NBA.com status (may be slow during peak traffic)

**Test:**
```python
import time
from nba_api.live.nba.endpoints import scoreboard

start = time.time()
board = scoreboard.ScoreBoard()
elapsed = time.time() - start

print(f"Response time: {elapsed*1000:.0f}ms")
# Should be 200-500ms
# If >1000ms, there's a problem
```

---

### Issue: Rate Limiting

**Symptoms:** `429 Too Many Requests` errors

**Cause:** Polling too frequently (NBA.com has rate limits)

**Solutions:**
1. Increase POLL_INTERVAL to 10+ seconds
2. Cache static data (teams, players)
3. Use Redis to share cache across instances
4. Implement exponential backoff on errors

**Example:**
```python
import time

last_poll = 0
MIN_INTERVAL = 10  # Minimum 10 seconds between polls

def poll_with_rate_limit():
    global last_poll
    
    # Check if enough time has passed
    elapsed = time.time() - last_poll
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    
    # Make request
    board = scoreboard.ScoreBoard()
    last_poll = time.time()
    
    return board
```

---

### Issue: Connection Errors

**Symptoms:** `ConnectionError`, `Timeout`

**Solutions:**
1. Increase REQUEST_TIMEOUT
2. Implement retry logic
3. Check firewall/proxy settings
4. Verify NBA.com is accessible

**Test connection:**
```bash
# Test NBA.com connectivity
curl -I https://stats.nba.com

# Should return 200 OK
```

---

## Performance Benchmarks

### Expected Performance

| Operation | Target | Notes |
|-----------|--------|-------|
| **First API call** | <1s | Cold start |
| **Subsequent calls** | <500ms | Warm connection |
| **With orjson** | <400ms | 20% faster |
| **With pooling** | <300ms | 30% faster |
| **JSON parsing** | <10ms | With orjson |
| **Pattern building** | <5ms | Simple math |

### Measure Your Performance

```python
# benchmark.py
import time
from nba_api.live.nba.endpoints import scoreboard

# Warm up
_ = scoreboard.ScoreBoard()
time.sleep(1)

# Benchmark
times = []
for i in range(10):
    start = time.time()
    board = scoreboard.ScoreBoard()
    elapsed = (time.time() - start) * 1000
    times.append(elapsed)
    time.sleep(2)

avg = sum(times) / len(times)
print(f"Average response time: {avg:.0f}ms")
print(f"Min: {min(times):.0f}ms, Max: {max(times):.0f}ms")

# Target: <500ms average
```

---

## ✅ Setup Verification Checklist

- [ ] Python 3.7+ installed
- [ ] `pip install nba-api` successful
- [ ] `pip install orjson aiohttp` (optional but recommended)
- [ ] Basic import test passes
- [ ] Live data test passes
- [ ] Response time <500ms
- [ ] Configuration file created
- [ ] Environment variables set
- [ ] Connection pooling enabled
- [ ] Fast JSON parser enabled
- [ ] Directory structure created

---

## Next Steps

**After setup complete:**
1. ✅ Read LIVE_DATA_INTEGRATION.md (build real-time poller)
2. ✅ Read ML_MODEL_INTEGRATION.md (connect to ML backend)
3. ✅ Test with live games (verify end-to-end)

---

**Setup time:** 5-10 minutes  
**Result:** NBA_API ready for production with optimal performance

---

*Last Updated: October 15, 2025*  
*Part of ML Research / NBA_API documentation*

