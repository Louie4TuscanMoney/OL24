# 📦 DEPLOY NBA ANALYTICS TO RAILWAY

## 🚀 **STEP-BY-STEP DEPLOYMENT**

### **1. Add PostgreSQL to Railway**

1. Go to Railway dashboard: https://railway.app/dashboard
2. Click your project (OL24)
3. Click "+ New" → "Database" → "PostgreSQL"
4. PostgreSQL provisions automatically
5. Railway auto-creates `DATABASE_URL` environment variable

**Your Railway will now have:**
- ✅ Main Service (trading_dashboard_api.py)
- ✅ PostgreSQL Database (new!)

---

### **2. Deploy Schema to PostgreSQL**

**Option A: Railway Web UI**
1. Click PostgreSQL service
2. Click "Data" tab
3. Click "Query"
4. Copy/paste contents of `database_schema.sql`
5. Click "Execute"

**Option B: Local psql**
```bash
# Get DATABASE_URL from Railway (click PostgreSQL → Connect)
psql $DATABASE_URL < database_schema.sql
```

You should see:
```
CREATE TABLE
CREATE TABLE
CREATE TABLE
... (7 tables created)
INSERT 0 30  (teams inserted)
```

---

### **3. Push Code to Railway**

```bash
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"

# Commit new files
git add live-system/nba_stats_collector.py
git add live-system/database_schema.sql
git add live-system/Procfile
git add live-system/requirements.txt

git commit -m "🏀 Add NBA Analytics Rolling Model

- PostgreSQL schema for teams, players, stats
- Stats collector (runs every 24 hours)
- Procfile: web + worker processes
- Rolling model: continuously updated, no historical yet"

git push origin main
```

Railway automatically:
- Detects Procfile
- Starts TWO processes:
  - `web`: Your live betting API
  - `worker`: Stats collector

---

### **4. Verify Deployment**

**Check Railway Logs:**

**Web process (trading_dashboard_api.py):**
```
✅ Model downloaded: 307.5 MB
✅ NBA API initialized
✅ Trading engine ready
INFO: Uvicorn running on 0.0.0.0:8080
```

**Worker process (nba_stats_collector.py):**
```
🚂 NBA STATS COLLECTOR - RAILWAY
Started: 2025-10-28 12:00:00
Mode: Rolling (continuous daily updates)
================================================================================
🔄 DAILY UPDATE STARTING
================================================================================
📊 Updating teams...
✅ Updated 30 teams
📊 Updating players...
✅ Updated 100 players
📊 Updating standings...
✅ Updated standings for 30 teams
📸 Snapshot: 30 teams, 100 players [success]
================================================================================
✅ DAILY UPDATE COMPLETED
================================================================================
⏰ Next update in 24 hours
💤 Sleeping...
```

---

### **5. Test API Endpoints**

Visit your Railway URL with new endpoints:

**Get All Teams:**
```
https://ol24-production.up.railway.app/api/stats/teams
```

**Get Players (Lakers):**
```
https://ol24-production.up.railway.app/api/stats/players?team=LAL
```

**Get Standings:**
```
https://ol24-production.up.railway.app/api/stats/standings
```

**Get Player Stats (LeBron James = 2544):**
```
https://ol24-production.up.railway.app/api/stats/player/2544
```

---

### **6. Monitor Database**

**Railway PostgreSQL UI:**
1. Click PostgreSQL service
2. Click "Data" tab
3. View tables: teams, players, standings, etc.

**Or use psql:**
```bash
psql $DATABASE_URL

# Check tables
\dt

# Count records
SELECT COUNT(*) FROM teams;      -- Should be 30
SELECT COUNT(*) FROM players;    -- Should be 100+
SELECT COUNT(*) FROM standings;  -- Should be 30 (per day)

# View latest snapshot
SELECT * FROM daily_snapshots ORDER BY snapshot_date DESC LIMIT 1;
```

---

## 🔄 **HOW IT WORKS**

### **Daily Update Cycle:**

```
Day 1 (Oct 28):
12:00 AM - Collector runs
          ├─ Updates all 30 teams
          ├─ Updates 450+ active players  
          ├─ Updates standings
          └─ Creates snapshot
12:00 AM - 11:59 PM - Sleep
11:59 PM - End of day

Day 2 (Oct 29):
12:00 AM - Collector runs again
          ├─ Updates teams (any trades?)
          ├─ Updates players (roster changes?)
          ├─ Updates standings (new W/L records)
          └─ Creates new snapshot
[Repeat forever...]
```

### **Rolling Model:**
- Database always has CURRENT season data
- Updates overwrite existing records (UPSERT)
- No historical accumulation (yet)
- Lightweight and fast

### **Future (Historical Model):**
- Keep snapshots for every day
- Store every season separately
- Time-travel queries (e.g., "Lakers roster on Jan 15, 2024")

---

## 📊 **WHAT'S COLLECTED:**

### **Current (Phase 1):**
- ✅ 30 teams (static info)
- ✅ 450+ active players
- ✅ Daily standings
- ✅ Collection status tracking

### **Soon (Phase 2):**
- ⏳ Player season stats (PPG, RPG, APG, etc.)
- ⏳ Player recent games (last 10)
- ⏳ Team season stats (offense/defense)
- ⏳ Advanced metrics (TS%, eFG%, BPM, etc.)

### **Later (Phase 3):**
- ⏳ Historical data (every day/season)
- ⏳ KenPom-style ratings
- ⏳ RAPTOR impact metrics
- ⏳ Similarity scores
- ⏳ ML feature engineering

---

## 🐛 **TROUBLESHOOTING**

### **Worker not running:**
```
# Check Railway logs for worker process
# Make sure Procfile exists in repo
# Verify DATABASE_URL is set
```

### **Database connection fails:**
```
❌ DATABASE_URL not set!
```
**Fix:** Make sure PostgreSQL is added to Railway project

### **Rate limiting:**
```
⚠️ nba_api failed: 429 Too Many Requests
```
**Fix:** Collector has built-in 0.6s delays, should be fine

### **Empty database:**
```
SELECT COUNT(*) FROM teams;
-- Returns 0
```
**Fix:** Run `database_schema.sql` to insert 30 teams

---

## ✅ **SUCCESS CHECKLIST**

- [ ] PostgreSQL added to Railway
- [ ] Schema deployed (`database_schema.sql`)
- [ ] Code pushed to Railway
- [ ] Web process running (API)
- [ ] Worker process running (collector)
- [ ] 30 teams in database
- [ ] 100+ players in database
- [ ] API endpoints returning data
- [ ] Daily updates scheduled

---

## 🎯 **NEXT STEPS**

1. **Deploy everything** (follow steps above)
2. **Verify it works** (check logs, test API)
3. **Add more data** (season stats, recent games)
4. **Build frontend** (stats pages, player cards)
5. **Calculate advanced metrics** (KenPom, RAPTOR)
6. **Integrate with ML** (use stats for predictions)

**Ready to deploy! 🚀**

