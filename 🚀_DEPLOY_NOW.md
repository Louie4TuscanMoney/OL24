# 🚀 DEPLOY DATABASE TO RAILWAY NOW

## Quick Deploy Guide (15 minutes)

Follow these steps in order to get your optimized database live on Railway.

---

## STEP 1: Access Railway PostgreSQL (2 minutes)

### Option A: Railway Dashboard (Easiest)

1. **Go to Railway Dashboard**:
   ```
   https://railway.app/
   ```

2. **Select Your Project**: 
   - Click on your project (likely named "ol24-production")
   - You should see your PostgreSQL database

3. **Open PostgreSQL Query Tab**:
   - Click on the PostgreSQL service
   - Click "Query" tab at the top
   - You'll see a query editor

### Option B: Get Connection String

1. Click PostgreSQL service
2. Click "Variables" tab
3. Copy the `DATABASE_URL` value
4. Should look like: `postgresql://postgres:xxx@xxx.railway.app:5432/railway`

---

## STEP 2: Deploy the Schema (3 minutes)

### Via Railway Dashboard Query Tab

1. **Open the schema file** on your local machine:
   ```
   live-system/database_schema_OPTIMIZED_FOR_FRONTEND.sql
   ```

2. **Copy the ENTIRE contents** (Cmd+A, Cmd+C)

3. **Paste into Railway Query Editor**

4. **Click "Run" or "Execute"**

5. **Wait for completion** (~30 seconds)
   - You should see: "Query executed successfully"

### Verify Schema Deployed

Run this query in Railway dashboard:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;
```

You should see at least 10 tables including:
- `teams`
- `players`
- `player_season_stats`
- `nba_schedule`
- `player_injuries`

---

## STEP 3: Check Backend Connection (2 minutes)

Your backend (`trading_dashboard_api.py`) needs the `DATABASE_URL` environment variable.

### Verify Railway Has DATABASE_URL

1. **In Railway Dashboard**:
   - Click on your web service (the FastAPI backend)
   - Click "Variables" tab
   - Look for `DATABASE_URL`

2. **If Missing, Add It**:
   - Click "New Variable"
   - Name: `DATABASE_URL`
   - Value: Reference the PostgreSQL service
   - Railway should auto-suggest: `${{Postgres.DATABASE_URL}}`

3. **Redeploy if needed**:
   - If you added/changed the variable, click "Deploy" to restart

---

## STEP 4: Populate the Database (5 minutes)

Now we need to fill the database with data.

### Option A: Railway SSH (Recommended)

1. **Open Railway CLI or Dashboard Shell**:
   - In Railway dashboard, click web service
   - Click the "..." menu → "Deploy Logs"
   - Or install Railway CLI: `npm i -g @railway/cli`

2. **SSH into Railway**:
   ```bash
   railway run bash
   ```

3. **Run population script**:
   ```bash
   cd live-system
   python3 populate_database_for_frontend.py
   ```

4. **Wait for completion** (~5 minutes):
   - It will populate teams, players, stats, schedule, injuries
   - You'll see progress messages

### Option B: Run Locally (Alternative)

If Railway shell doesn't work, run locally:

```bash
# On your local machine
export DATABASE_URL="postgresql://postgres:xxx@xxx.railway.app:5432/railway"
cd "live-system"
python3 populate_database_for_frontend.py
```

---

## STEP 5: Test API Endpoints (3 minutes)

Now let's verify the backend is serving data correctly.

### Test 1: Health Check

```bash
curl https://ol24-production.up.railway.app/
```

Should return:
```json
{
  "status": "online",
  "system": "Ontologic XYZ Trading Dashboard",
  "version": "1.0.0"
}
```

### Test 2: Teams with Logos

```bash
curl https://ol24-production.up.railway.app/api/stats/teams
```

Should return teams with `logo_url`, `primary_color`, `secondary_color`:
```json
{
  "teams": [
    {
      "abbreviation": "LAL",
      "full_name": "Los Angeles Lakers",
      "logo_url": "https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg",
      "primary_color": "#552583",
      "secondary_color": "#FDB927"
    }
  ],
  "count": 30
}
```

### Test 3: Injuries

```bash
curl https://ol24-production.up.railway.app/api/injuries
```

Should return current injuries:
```json
{
  "injuries": [...],
  "count": 25
}
```

### Test 4: Schedule

```bash
curl https://ol24-production.up.railway.app/api/schedule
```

Should return upcoming games:
```json
{
  "games": [...],
  "count": 12
}
```

---

## STEP 6: Verify Frontend Connection (2 minutes)

Your Vercel frontend should now display data from the database.

### Check Frontend

1. **Open your Vercel site**:
   ```
   https://ontologicxyz.com
   ```

2. **Navigate to "Stats" page**

3. **You should see**:
   - ✅ Team cards with logos and colors
   - ✅ Team statistics (PPG, wins, losses)
   - ✅ Current injuries list

4. **Navigate to "Schedule" page**:
   - ✅ Upcoming games
   - ✅ Team logos visible
   - ✅ Game dates and times

5. **Click on a team**:
   - ✅ Depth chart with starters
   - ✅ Player stats
   - ✅ Team schedule

### If Frontend Shows Empty Data

1. **Check browser console** (F12 → Console tab)
2. **Look for errors** like:
   - CORS errors → Backend CORS is already configured
   - Network errors → Backend might be down
   - 404 errors → Endpoint might be wrong

3. **Verify API base URL** in frontend:
   ```typescript
   // Should be in frontend/src/components/*.tsx
   const API_BASE = 'https://ol24-production.up.railway.app';
   ```

---

## TROUBLESHOOTING

### Issue: "Table does not exist"

**Cause**: Schema didn't deploy correctly.

**Fix**: Re-run the schema SQL in Railway query tab.

### Issue: "No teams found"

**Cause**: Database is empty.

**Fix**: Run the population script:
```bash
python3 live-system/populate_database_for_frontend.py
```

### Issue: "Connection refused"

**Cause**: `DATABASE_URL` not set or wrong.

**Fix**: 
1. Check Railway variables tab
2. Make sure `DATABASE_URL` references `${{Postgres.DATABASE_URL}}`
3. Redeploy

### Issue: "Basketball Reference scraping failed"

**Cause**: Cloudflare blocking requests.

**Fix**: Run from local machine instead:
```bash
export DATABASE_URL="your-railway-url"
python3 live-system/populate_database_for_frontend.py
```

### Issue: "Frontend shows 'Loading...' forever"

**Cause**: CORS or network issue.

**Fix**: 
1. Check browser console for errors
2. Verify API URL is correct
3. Test API directly with curl

---

## VERIFICATION CHECKLIST

After deployment, verify:

- [ ] Railway PostgreSQL has all tables (run: `\dt` in query tab)
- [ ] Teams table has 30 rows with logos
- [ ] Players table has 400+ rows
- [ ] Player_season_stats has data
- [ ] Schedule has upcoming games
- [ ] Injuries table has current injuries
- [ ] Backend health check works (curl /)
- [ ] Backend /api/stats/teams returns data
- [ ] Backend /api/injuries returns data
- [ ] Backend /api/schedule returns data
- [ ] Frontend displays team cards with logos
- [ ] Frontend displays injuries
- [ ] Frontend displays schedule

---

## QUICK VERIFICATION COMMANDS

Run these to verify everything:

```bash
# 1. Backend health
curl https://ol24-production.up.railway.app/

# 2. Teams endpoint
curl https://ol24-production.up.railway.app/api/stats/teams | jq '.count'
# Should output: 30

# 3. Injuries endpoint
curl https://ol24-production.up.railway.app/api/injuries | jq '.count'
# Should output: 20-30

# 4. Schedule endpoint
curl https://ol24-production.up.railway.app/api/schedule | jq '.count'
# Should output: 10-20
```

---

## SUCCESS! 🎉

If all tests pass, your database is live and working!

### What You Have Now:

✅ Railway PostgreSQL with optimized schema
✅ 30 teams with logos and brand colors
✅ 400+ players with stats
✅ Complete advanced stats (PER, BPM, VORP, TS%)
✅ Upcoming games schedule
✅ Current player injuries
✅ Team depth charts
✅ All API endpoints working
✅ Frontend displaying data correctly

### Next Steps:

1. **Schedule daily updates** (add to `railway.json`)
2. **Monitor logs** at 3:30 AM UTC for auto-updates
3. **Add more data** (player headshots, historical games)

---

## NEED HELP?

If you run into issues:

1. **Check Railway logs**:
   - Dashboard → Web Service → "Deploy Logs"
   - Look for errors

2. **Check database directly**:
   - Dashboard → PostgreSQL → "Query"
   - Run: `SELECT COUNT(*) FROM teams;`

3. **Test API locally**:
   ```bash
   curl -v https://ol24-production.up.railway.app/api/stats/teams
   ```

---

**Ready to deploy? Start with Step 1! 🚀**

