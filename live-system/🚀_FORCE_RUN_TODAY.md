# 🚀 FORCE RUN PIPELINE TODAY - DON'T WAIT FOR 3:30 AM!

## ⚡ **TEST THE SYSTEM IMMEDIATELY:**

---

## 📋 **DEPLOYMENT STEPS:**

### **STEP 1: Add PostgreSQL to Railway (30 seconds)**

1. Go to: https://railway.app/dashboard
2. Click your project
3. Click "+ New" → "Database" → "PostgreSQL"
4. Wait for provision

**Railway auto-sets:** `DATABASE_URL` environment variable

---

### **STEP 2: Deploy Schema (2 minutes)**

**Option A: Railway UI**
1. Click PostgreSQL service
2. Click "Data" tab
3. Click "Query"
4. Copy/paste ALL of: `database_schema_v3_SELF_COMPUTED.sql`
5. Click "Execute"

**Expected output:**
```
CREATE TABLE (15 times)
CREATE INDEX (40+ times)
INSERT 0 30 (teams)
INSERT 0 1 (season)
SELECT 'Schema deployed successfully!'
```

**Option B: Terminal (if you have psql)**
```bash
# Get DATABASE_URL from Railway → PostgreSQL → Connect
export DATABASE_URL="postgresql://postgres:..."

psql $DATABASE_URL < live-system/database_schema_v3_SELF_COMPUTED.sql
```

---

### **STEP 3: Populate Images (1 minute)**

Run on Railway or locally:

```bash
export DATABASE_URL="postgresql://..."  # From Railway
python live-system/populate_player_images.py
```

**Output:**
```
🖼️  POPULATING IMAGES + COLORS
📊 Updating team visuals...
   ✅ 30 team logos + colors
📊 Updating player images...
   ✅ 450 player headshots
✅ IMAGES POPULATED!
```

---

### **STEP 4: FORCE RUN PIPELINE NOW! (4 minutes)**

```bash
export DATABASE_URL="postgresql://..."  # From Railway
python live-system/force_run_pipeline_today.py
```

**Expected output:**
```
🔥 FORCE RUNNING NIGHTLY PIPELINE NOW
================================================================================
🌙 NIGHTLY PIPELINE: 2025-10-28 18:45:32
================================================================================

📥 [1/5] Pulling games + box scores...
      10 games...
      20 games...
   ✅ 82 games processed

🗑️  [2/5] Pruning old games + refreshing last10...
   ✅ Last 10 games refreshed

🧮 [3/5] Computing season aggregates...
      100 players...
      200 players...
      450 players total
   ✅ Season stats updated

📊 [4/5] Computing team metrics...
   ✅ Team stats updated

⏭️  [5/5] Skipping RAPM (only runs Sunday)

🏆 Updating standings...
   ✅ 30 teams ranked

================================================================================
✅ PIPELINE COMPLETED in 180s
================================================================================
```

---

### **STEP 5: Verify Data (30 seconds)**

```bash
# Connect to database
psql $DATABASE_URL

# Check what was populated
SELECT COUNT(*) FROM player_box_scores;      -- Should be ~1000+ (82 games × 13 players/team)
SELECT COUNT(*) FROM player_season_stats;    -- Should be 450+
SELECT COUNT(*) FROM teams;                  -- Should be 30

-- View top scorers
SELECT name, ppg, pts_100, lebron_total
FROM player_season_stats pss
JOIN players p ON pss.player_id = p.player_id
ORDER BY ppg DESC
LIMIT 10;

-- Check images
SELECT name, headshot_url, team_id
FROM players
WHERE headshot_url IS NOT NULL
LIMIT 5;

-- Check team colors
SELECT abbreviation, full_name, primary_color, logo_url
FROM teams
LIMIT 5;
```

---

## 🎯 **TEST API ENDPOINTS:**

### **1. Get Player Profile (LeBron James = 2544):**
```bash
curl https://ol24-production.up.railway.app/api/stats/player/2544 | python3 -m json.tool
```

**Expected response:**
```json
{
  "player_id": "2544",
  "name": "LeBron James",
  "headshot_url": "https://cdn.nba.com/headshots/nba/latest/1040x760/2544.png",
  "jersey": "23",
  "position": "F",
  "height": "6-9",
  "team": {
    "abbreviation": "LAL",
    "full_name": "Los Angeles Lakers",
    "logo_url": "https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg",
    "primary_color": "#552583",
    "secondary_color": "#FDB927"
  },
  "season_stats": {
    "ppg": 25.3,
    "rpg": 7.2,
    "apg": 7.8
  },
  "advanced_stats": {
    "ts_pct": 0.623,
    "pts_100": 38.4,
    "pts_36": 28.1,
    "lebron": 7.2,
    "rapm": 4.5
  },
  "last10_games": [
    {"date": "2025-10-27", "opponent": "DET", "pts": 28, "reb": 8, "ast": 10},
    ...
  ],
  "probability_metrics": {
    "sample_size": 82,
    "scoring_variance": 12.5,
    "consistency_score": 0.847
  }
}
```

### **2. Get All Teams:**
```bash
curl https://ol24-production.up.railway.app/api/stats/teams
```

### **3. Get Standings:**
```bash
curl https://ol24-production.up.railway.app/api/stats/standings
```

---

## 🖼️  **PLAYER IMAGE URLS (NBA OFFICIAL CDN):**

**Pattern:**
```
https://cdn.nba.com/headshots/nba/latest/1040x760/{PLAYER_ID}.png
```

**Examples:**
- LeBron (2544): https://cdn.nba.com/headshots/nba/latest/1040x760/2544.png
- Giannis (203507): https://cdn.nba.com/headshots/nba/latest/1040x760/203507.png
- Curry (201939): https://cdn.nba.com/headshots/nba/latest/1040x760/201939.png

**Team Logos:**
```
https://cdn.nba.com/logos/nba/{TEAM_ID}/primary/L/logo.svg
```

---

## 🎨 **TEAM COLORS (For Frontend Styling):**

All 30 teams have official brand colors stored:
- Lakers: `#552583` (purple) / `#FDB927` (gold)
- Celtics: `#007A33` (green) / `#BA9653` (gold)
- Warriors: `#1D428A` (blue) / `#FFC72C` (yellow)

Use these for player cards, team backgrounds, charts, etc.

---

## ✅ **SUCCESS CHECKLIST:**

- [ ] PostgreSQL added to Railway
- [ ] Schema deployed (18 tables created)
- [ ] Images populated (30 teams + 450 players)
- [ ] Pipeline force-run completed
- [ ] 82+ games in player_box_scores
- [ ] 450+ players in player_season_stats
- [ ] API endpoint returns player profile with image
- [ ] Team logos + colors working

---

## 🚀 **RUN IT NOW:**

```bash
# 1. Add PostgreSQL to Railway (web UI)

# 2. Deploy schema
psql $DATABASE_URL < live-system/database_schema_v3_SELF_COMPUTED.sql

# 3. Populate images
python live-system/populate_player_images.py

# 4. Force run pipeline
python live-system/force_run_pipeline_today.py

# 5. Test API
curl https://ol24-production.up.railway.app/api/stats/player/2544
```

**Don't wait for 3:30 AM - populate NOW!** ⚡

