# 🚀 Deploy Optimized Database to Railway

## Overview

This guide deploys the **frontend-optimized database schema** to your Railway PostgreSQL instance and populates it with all data needed for your Vercel frontend.

---

## 📋 Prerequisites

- ✅ Railway project with PostgreSQL database
- ✅ `DATABASE_URL` environment variable set in Railway
- ✅ Python 3.10+ on Railway

---

## 🎯 Step 1: Deploy the Schema

### Option A: Via Railway Dashboard (Easiest)

1. **Open Railway Dashboard**:
   - Go to your Railway project
   - Click on PostgreSQL database
   - Click "Query" tab

2. **Copy & Paste Schema**:
   ```bash
   # Copy the entire contents of:
   live-system/database_schema_OPTIMIZED_FOR_FRONTEND.sql
   ```

3. **Run the Schema**:
   - Paste into Railway query editor
   - Click "Run"
   - Wait for completion (~30 seconds)

### Option B: Via psql Command Line

```bash
# Set your DATABASE_URL
export DATABASE_URL="postgresql://postgres:password@your-railway-host/railway"

# Deploy schema
psql $DATABASE_URL < live-system/database_schema_OPTIMIZED_FOR_FRONTEND.sql
```

### ✅ Verify Schema Deployed

Run this query in Railway dashboard:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;
```

You should see:
- ✅ teams (with 30 rows)
- ✅ players
- ✅ player_season_stats
- ✅ nba_schedule
- ✅ player_injuries
- ✅ team_depth_charts
- ✅ ml_predictions
- ✅ player_last10
- ✅ standings
- ✅ player_transactions

---

## 🌊 Step 2: Populate the Database

### Run Population Script

```bash
cd live-system
python3 populate_database_for_frontend.py
```

### What This Does:

1. **Teams**: Verifies 30 teams with logos & brand colors ✅
2. **Players**: Fetches ~500 active players from `nba_api` ✅
3. **Stats**: Scrapes stats from Basketball Reference ✅
4. **Schedule**: Fetches upcoming games from `nba_api` ✅
5. **Injuries**: Fetches current injuries from ESPN ✅
6. **Depth Charts**: Computes from minutes per game ✅

### Expected Output:

```
================================================================================
🚀 COMPREHENSIVE DATABASE POPULATION FOR FRONTEND
================================================================================
   Started: 2025-10-29 12:00:00
   Target: Railway PostgreSQL
   Purpose: Populate ALL data needed for Vercel frontend
================================================================================

✅ Connected to Railway PostgreSQL

================================================================================
1️⃣  POPULATING TEAMS (with logos & colors)
================================================================================
   ✅ 30 teams with logos & colors

================================================================================
2️⃣  POPULATING PLAYERS (from nba_api)
================================================================================
   📊 Found 500 active players
   ✅ Inserted 450 new players, updated 50 existing

================================================================================
3️⃣  SCRAPING PLAYER STATS (Basketball Reference)
================================================================================
   ✅ Scraped stats for 450 players

================================================================================
4️⃣  POPULATING SCHEDULE (from nba_api)
================================================================================
   📅 Found 12 games
   ✅ Inserted/updated 12 games

================================================================================
5️⃣  POPULATING INJURIES (from ESPN)
================================================================================
   ✅ Inserted/updated 25 injuries

================================================================================
6️⃣  COMPUTING DEPTH CHARTS (from MPG)
================================================================================
   ✅ Computed depth charts for all teams (360 entries)

================================================================================
🔍 VERIFICATION: Data Ready for Frontend
================================================================================
   ✅ Teams with logos & colors: 30
   ✅ Active players: 450
   ✅ Player season stats: 450
   ✅ Scheduled games: 12
   ✅ Active injuries: 25
   ✅ Depth chart entries: 360

   Critical API Endpoint Checks:
      ✅ Teams with visuals: 30/30
      ✅ Active injuries: 25
      ✅ Upcoming games: 12
      ✅ Teams with depth charts: 30/30

================================================================================
✅ POPULATION COMPLETE!
================================================================================
```

---

## 🧪 Step 3: Test API Endpoints

### Test 1: Teams with Logos

```bash
curl https://ol24-production.up.railway.app/api/stats/teams | jq
```

Expected:
```json
{
  "teams": [
    {
      "abbreviation": "LAL",
      "full_name": "Los Angeles Lakers",
      "logo_url": "https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg",
      "primary_color": "#552583",
      "secondary_color": "#FDB927",
      "ppg": 115.2,
      "wins": 5,
      "losses": 3
    }
  ]
}
```

### Test 2: Injuries

```bash
curl https://ol24-production.up.railway.app/api/injuries | jq
```

Expected:
```json
{
  "injuries": [
    {
      "name": "LeBron James",
      "team_abbr": "LAL",
      "status": "Questionable",
      "injury_type": "Ankle",
      "description": "Left ankle soreness"
    }
  ],
  "count": 25
}
```

### Test 3: Schedule

```bash
curl "https://ol24-production.up.railway.app/api/schedule?days_ahead=7" | jq
```

Expected:
```json
{
  "games": [
    {
      "game_id": "0022500123",
      "date": "2025-10-29",
      "home_team": {"abbr": "LAL", "name": "Los Angeles Lakers"},
      "away_team": {"abbr": "GSW", "name": "Golden State Warriors"},
      "status": "Scheduled"
    }
  ],
  "count": 12
}
```

### Test 4: Depth Chart

```bash
curl https://ol24-production.up.railway.app/api/team/LAL/depth-chart | jq
```

Expected:
```json
{
  "team": {"abbreviation": "LAL", "name": "Los Angeles Lakers"},
  "starters": [
    {
      "name": "LeBron James",
      "position": "SF",
      "mpg": 35.2,
      "ppg": 25.7,
      "is_starter": true
    }
  ],
  "bench": [...],
  "total_players": 15
}
```

---

## 🔄 Step 4: Schedule Daily Updates

The database needs daily updates for:
- New games (schedule)
- Updated stats (Basketball Reference)
- New injuries (ESPN)

### Option A: Railway Cron Job (Recommended)

Add to `railway.json`:

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "cd live-system && python3 trading_dashboard_api.py",
    "healthcheckPath": "/",
    "healthcheckTimeout": 100
  },
  "cron": [
    {
      "schedule": "30 3 * * *",
      "command": "cd live-system && python3 populate_database_for_frontend.py"
    }
  ]
}
```

This runs the population script every day at 3:30 AM UTC.

### Option B: Integrated in API Startup

The `trading_dashboard_api.py` already has a daily scheduler:

```python
# In trading_dashboard_api.py startup
from daily_nba_scheduler import start_scheduler
start_scheduler()  # Runs at 3:30 AM UTC daily
```

---

## 🎨 Frontend Integration

Your Vercel frontend is already configured to use these endpoints!

### StatsPage (src/components/StatsPage.tsx)

```typescript
// Fetches teams with logos & colors
const teamsRes = await fetch(`${API_BASE}/api/stats/teams`);
const teamsData = await teamsRes.json();
setTeams(teamsData.teams);

// Fetches injuries
const injuriesRes = await fetch(`${API_BASE}/api/injuries`);
const injuriesData = await injuriesRes.json();
setInjuries(injuriesData.injuries);
```

### SchedulePage (src/components/SchedulePage.tsx)

```typescript
// Fetches schedule
const response = await fetch(`${API_BASE}/api/schedule?days_ahead=${daysAhead()}`);
const data = await response.json();
setGames(data.games);
```

### TeamPage (src/components/TeamPage.tsx)

```typescript
// Fetches team depth chart
const depthRes = await fetch(`${API_BASE}/api/team/${teamAbbr}/depth-chart`);
const depthData = await depthRes.json();
setStarters(depthData.starters);
setBench(depthData.bench);

// Fetches team schedule
const schedRes = await fetch(`${API_BASE}/api/team/${teamAbbr}/schedule?days_ahead=14`);
const schedData = await schedRes.json();
setSchedule(schedData.games);
```

---

## 🔧 Troubleshooting

### Issue: "No teams found"

**Solution**: Schema didn't deploy correctly. Re-run schema SQL in Railway dashboard.

### Issue: "No players found"

**Solution**: Run population script:
```bash
python3 live-system/populate_database_for_frontend.py
```

### Issue: "Basketball Reference scraping failed"

**Solution**: Cloudflare block. Try:
1. Add delays between requests
2. Use proxy (optional)
3. Or manually run on local machine and copy data

### Issue: "API returns empty arrays"

**Solution**: Check Railway logs:
```bash
railway logs
```

Look for database connection errors or missing tables.

---

## 📊 Database Size

After full population:

| Table | Rows | Size |
|-------|------|------|
| teams | 30 | 10 KB |
| players | 500 | 50 KB |
| player_season_stats | 500 | 100 KB |
| player_box_scores | 40,000+ | 5 MB |
| nba_schedule | 1,230 | 50 KB |
| player_injuries | 50 | 10 KB |
| ml_predictions | 1,000+ | 500 KB |

**Total**: ~10 MB (very lightweight!)

---

## ✅ Success Checklist

- [ ] Schema deployed to Railway PostgreSQL
- [ ] 30 teams with logos & colors populated
- [ ] 500+ players populated
- [ ] Player stats scraped from Basketball Reference
- [ ] Schedule populated with upcoming games
- [ ] Injuries populated from ESPN
- [ ] Depth charts computed
- [ ] All API endpoints tested and working
- [ ] Daily cron job scheduled (3:30 AM UTC)
- [ ] Frontend connected and displaying data

---

## 🎯 Next Steps

1. **Monitor Daily Updates**:
   - Check Railway logs at 3:30 AM UTC
   - Verify stats update daily

2. **Add Missing Data**:
   - Player headshots (from nba.com CDN)
   - Team logos (already in schema!)
   - Historical game data (optional)

3. **Performance Optimization**:
   - Add indexes for slow queries
   - Use materialized views for complex aggregations
   - Enable connection pooling

---

## 🚀 You're Ready!

Your Railway PostgreSQL database is now **fully optimized** for your frontend. All API endpoints are working, data is fresh, and daily updates are automated.

Frontend should now display:
- ✅ Team cards with logos and brand colors
- ✅ Player stats with advanced metrics
- ✅ Upcoming games schedule
- ✅ Current injuries
- ✅ Team depth charts
- ✅ ML predictions (when games are live)

**Your system is production-ready! 🎉**

