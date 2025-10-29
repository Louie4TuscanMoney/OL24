# 📊 STATS ACCURACY AUDIT

**Last Updated:** October 29, 2025

---

## ⚠️ **CRITICAL ISSUES**

### **1. Player Stats: NOT POPULATED (0 records)**
```
Player stats (2025-26): 0  ❌
```

**Why?** Players not on current rosters are being skipped, but this might be skipping ALL players due to:
- Season hasn't started yet (or just started)
- `nba_api` might not have 2025-26 stats yet
- We need to verify the API is returning data

---

### **2. Advanced Stats: NOT CALCULATED**

**Currently MISSING:**
- ❌ RAPM (Regularized Adjusted Plus-Minus)
- ❌ LEBRON composite metric
- ❌ BPM (Box Plus-Minus)
- ❌ VORP (Value Over Replacement Player)
- ❌ Usage Rate
- ❌ PER (Player Efficiency Rating)
- ❌ True defensive metrics

**What we DO have:**
- ✅ TS% (True Shooting %) - Auto-calculated
- ✅ eFG% (Effective Field Goal %) - Auto-calculated
- ✅ PPG, RPG, APG, MPG - Auto-calculated from totals
- ⚠️ Per-100 - **ESTIMATED** (uses minutes * 1.2, NOT actual possessions)

---

### **3. Season Start Date: UNKNOWN**

We inserted games but don't know:
- When the season actually started
- If 2025-26 data is even available yet
- If we're pulling correct season data

**Action needed:**
```sql
SELECT MIN(game_date) FROM nba_schedule WHERE season_id = '2025-26';
```

---

## 📋 **WHAT WE'RE ACTUALLY REPORTING**

### **✅ ACCURATE (Generated Columns)**
These are auto-calculated by PostgreSQL from totals:

```sql
-- Player Stats
ppg = pts_total / games_played
rpg = reb_total / games_played
apg = ast_total / games_played
mpg = minutes_total / games_played
ts_pct = pts_total / (2 * (fga_total + 0.44 * fta_total))
efg_pct = (fgm_total + 0.5 * fg3m_total) / fga_total

-- Team Stats
win_pct = wins / games_played
ppg = pts_total / games_played
net_rating = offensive_rating - defensive_rating
luck = win_pct - (pythagorean_wins / games_played)
```

---

### **⚠️ ESTIMATED (Rough Approximations)**

**Per-100 Stats:**
```python
# Uses ESTIMATE: minutes * 1.2 possessions per minute
# NOT accurate - should use actual team possessions
poss_estimate = min_played * gp * 1.2
pts_100 = (pts * gp * 100) / poss_estimate
```

**Team Pace:**
```python
# Estimated from FGA, FTA, TO, OREB
# Should use play-by-play data for accuracy
team_poss = fga + 0.44 * fta - oreb + tov
pace = team_poss / gp
```

**Offensive/Defensive Rating:**
```python
# Simplified calculation
# Real ORtg/DRtg requires possession tracking
ortg = (pts / gp / pace * 100)
drtg = (opp_pts / gp / pace * 100)
```

---

### **❌ NOT IMPLEMENTED (Placeholders Only)**

These columns exist in the schema but are **NULL**:

```sql
-- Player Advanced Stats
rapm DECIMAL(7,3)        -- NULL
lebron DECIMAL(7,3)      -- NULL
bpm DECIMAL(7,3)         -- NULL

-- Missing entirely:
- Usage Rate
- PER
- Defensive Win Shares
- Offensive Win Shares
- True defensive metrics
```

---

## 🔗 **BACKLINKS STATUS**

### **Database Foreign Keys (✅ Exist)**
```sql
player_season_stats.player_id → players.player_id
player_season_stats.team_id → teams.team_id
team_season_stats.team_id → teams.team_id
games.home_team_id → teams.team_id
games.away_team_id → teams.team_id
```

### **Frontend URL Routing (❓ Partial)**
- `/team/{abbr}` - Team page exists
- `/team/{abbr}/depth-chart` - Depth chart API exists
- `/player/{id}` - **NOT IMPLEMENTED**
- `/game/{id}` - **PARTIAL** (GameDetailPage exists)

### **Missing Player Pages**
- No individual player profile pages yet
- No player comparison tool
- No player season history

---

## 🎯 **ACTION ITEMS TO FIX**

### **1. Verify 2025-26 Season Data**
```bash
# Test if nba_api has 2025-26 data
python3 -c "
from nba_api.stats.endpoints import leaguedashplayerstats
stats = leaguedashplayerstats.LeagueDashPlayerStats(
    season='2025-26',
    season_type_all_star='Regular Season'
)
df = stats.get_data_frames()[0]
print(f'Players with stats: {len(df)}')
print(f'Sample: {df.head(3)}')
"
```

### **2. Fix Per-100 Calculation**
Use actual team possessions instead of estimate:
```python
# Get team possessions from boxscoreadvancedv2
from nba_api.stats.endpoints import boxscoreadvancedv2
advanced = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id)
team_poss = advanced.get_data_frames()[0]['POSS'][0]
```

### **3. Implement Missing Advanced Stats**

**Priority 1 (Calculable from box scores):**
- Usage Rate: `(FGA + 0.44*FTA + TO) * (Team MIN / 5) / (MIN * (Team FGA + 0.44*Team FTA + Team TO))`
- PER: Complex formula (see Basketball Reference)

**Priority 2 (Require play-by-play):**
- RAPM: Ridge regression on lineup data
- BPM: Statistical model
- LEBRON: Hybrid metric

### **4. Add Player Profile Pages**
```
/player/{player_id}
  - Career stats
  - Season stats
  - Game log
  - Advanced metrics
  - Shot chart
  - Similar players
```

---

## 📊 **WHAT TO TELL USERS**

### **Honest Stats Description:**

**Available Now:**
- ✅ Traditional box score stats (PPG, RPG, APG, MPG)
- ✅ Shooting percentages (FG%, 3P%, FT%)
- ✅ Advanced shooting (TS%, eFG%)
- ✅ Team record, win%, net rating
- ⚠️ Estimated per-100 stats (not possession-accurate)

**Coming Soon:**
- ⏳ Accurate per-100 (with real possessions)
- ⏳ Usage Rate, PER
- ⏳ RAPM, LEBRON, BPM (require more data)
- ⏳ Player profile pages
- ⏳ Historical season comparisons

**Not Available:**
- ❌ Full KenPom-style metrics (requires season-long data)
- ❌ Defensive metrics (require play-by-play tracking)
- ❌ Lineup data
- ❌ Clutch stats

---

## ✅ **IMMEDIATE FIXES**

1. **Run populate script with verbose output:**
```bash
python3 backend/services/populate_comprehensive_nba_data.py
```

2. **Verify data exists:**
```sql
SELECT COUNT(*) FROM player_season_stats WHERE season_id = '2025-26';
SELECT COUNT(*) FROM team_season_stats WHERE season_id = '2025-26';
```

3. **If counts are 0, check API:**
```python
# Verify nba_api has 2025-26 data
from nba_api.stats.endpoints import leaguedashplayerstats
stats = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26')
print(len(stats.get_data_frames()[0]))
```

4. **Update frontend messaging:**
```typescript
// Display accurate stat labels
"Traditional Stats" instead of "Advanced Stats"
"Estimated Per-100" instead of "Per-100"
Add disclaimer: "Advanced metrics available after 10+ games"
```

---

## 🎓 **FOR TRUE ACCURACY**

To match **KenPom/Basketball Reference/RAPTOR** quality:

1. **Need:**
   - Play-by-play data (every possession)
   - Lineup tracking (who's on court)
   - Shot location data
   - Defensive matchups
   - 10+ games of data

2. **Timeline:**
   - Basic stats: ✅ Now
   - Better per-100: 📅 Week 1 (after games played)
   - Advanced metrics: 📅 Week 3-4 (after 10+ games)
   - Full KenPom style: 📅 Mid-season (after 20+ games)

---

**BOTTOM LINE:** We have **basic/traditional stats** implemented correctly, but **advanced stats are limited/estimated**. Be transparent about this on the frontend!

