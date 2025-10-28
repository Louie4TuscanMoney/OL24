# 🚂 RAILWAY NBA ANALYTICS - ROLLING MODEL

**Goal:** Live, continuously updating stats on Railway with PostgreSQL

**Future:** Historical data for every day/season (later)

---

## 🗄️ **RAILWAY SETUP**

### **Add PostgreSQL to Railway:**

1. Go to Railway dashboard
2. Click "New" → "Database" → "PostgreSQL"
3. Database auto-provisions with connection string
4. Railway sets environment variable: `DATABASE_URL`

---

## 📊 **ROLLING MODEL SCHEMA (Phase 1)**

**"Rolling" = Always current, updates daily**

### **Core Tables:**

```sql
-- ============================================================================
-- ROLLING MODEL: Current Season Only
-- ============================================================================

-- Teams (30 teams)
CREATE TABLE teams (
    team_id VARCHAR(10) PRIMARY KEY,
    abbreviation VARCHAR(3) UNIQUE NOT NULL,
    full_name VARCHAR(50),
    conference VARCHAR(10),
    division VARCHAR(20),
    city VARCHAR(50),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Players (450+ active players)
CREATE TABLE players (
    player_id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    position VARCHAR(5),
    height VARCHAR(10),
    weight INTEGER,
    jersey_number VARCHAR(3),
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Current Season Stats (ONE row per player, constantly updated)
CREATE TABLE player_season_stats (
    player_id VARCHAR(10) PRIMARY KEY REFERENCES players(player_id),
    season VARCHAR(10) DEFAULT '2024-25',
    
    -- Games
    games_played INTEGER,
    games_started INTEGER,
    minutes_per_game FLOAT,
    
    -- Basic Per-Game Stats
    ppg FLOAT,
    rpg FLOAT,
    apg FLOAT,
    spg FLOAT,
    bpg FLOAT,
    tov_pg FLOAT,
    
    -- Shooting
    fgp FLOAT,
    fg3p FLOAT,
    ftp FLOAT,
    
    -- Advanced (calculated nightly)
    ts_pct FLOAT,
    efg_pct FLOAT,
    usg_pct FLOAT,
    per FLOAT,
    
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Last N Games (rolling window)
CREATE TABLE player_recent_games (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(10) REFERENCES players(player_id),
    game_id VARCHAR(15),
    game_date DATE,
    opponent VARCHAR(3),
    
    -- Stats
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    minutes FLOAT,
    fgm INTEGER,
    fga INTEGER,
    fg3m INTEGER,
    fg3a INTEGER,
    plus_minus INTEGER,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(player_id, game_id)
);

-- Team Season Stats (ONE row per team, constantly updated)
CREATE TABLE team_season_stats (
    team_id VARCHAR(10) PRIMARY KEY REFERENCES teams(team_id),
    season VARCHAR(10) DEFAULT '2024-25',
    
    -- Record
    wins INTEGER,
    losses INTEGER,
    win_pct FLOAT,
    
    -- Per Game
    ppg FLOAT,
    opp_ppg FLOAT,
    
    -- Advanced (calculated nightly)
    ortg FLOAT,
    drtg FLOAT,
    netrtg FLOAT,
    pace FLOAT,
    
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Standings (updates daily)
CREATE TABLE standings (
    id SERIAL PRIMARY KEY,
    team_id VARCHAR(10) REFERENCES teams(team_id),
    conference VARCHAR(10),
    rank INTEGER,
    wins INTEGER,
    losses INTEGER,
    gb FLOAT,
    home_record VARCHAR(10),
    away_record VARCHAR(10),
    last_10 VARCHAR(10),
    streak VARCHAR(5),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(team_id, updated_at::date)
);

-- Daily Snapshots (for historical tracking)
CREATE TABLE daily_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE UNIQUE NOT NULL,
    teams_count INTEGER,
    players_count INTEGER,
    games_today INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🔄 **DATA COLLECTION SERVICE**

### **File: `nba_stats_collector.py`**

```python
"""
NBA Stats Collector - ROLLING MODEL
Runs on Railway, updates PostgreSQL daily
"""

import os
import time
from datetime import datetime, timedelta
import psycopg2
from nba_api.stats.endpoints import (
    leaguestandings,
    playerdashboardbygeneralsplits,
    teamdashboardbygeneralsplits,
    playergamelog
)
from nba_api.stats.static import teams, players

# Railway PostgreSQL connection
DATABASE_URL = os.environ.get('DATABASE_URL')

class NBAStatsCollector:
    """
    Collects and updates NBA stats in PostgreSQL
    Rolling model: Always current, updates daily
    """
    
    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_URL)
        self.cursor = self.conn.cursor()
        print("✅ Connected to Railway PostgreSQL")
    
    def run_daily_update(self):
        """
        Main update loop - runs every 24 hours
        """
        print(f"\n🔄 Starting daily update: {datetime.now()}")
        
        try:
            # 1. Update teams (rarely changes)
            self.update_teams()
            
            # 2. Update players (roster changes)
            self.update_players()
            
            # 3. Update season stats (daily)
            self.update_player_season_stats()
            self.update_team_season_stats()
            
            # 4. Update standings (daily)
            self.update_standings()
            
            # 5. Collect recent games (last 10)
            self.update_recent_games()
            
            # 6. Create daily snapshot
            self.create_snapshot()
            
            self.conn.commit()
            print(f"✅ Daily update completed: {datetime.now()}")
            
        except Exception as e:
            print(f"❌ Update failed: {e}")
            self.conn.rollback()
    
    def update_teams(self):
        """Update all 30 teams"""
        print("📊 Updating teams...")
        all_teams = teams.get_teams()
        
        for team in all_teams:
            self.cursor.execute("""
                INSERT INTO teams (team_id, abbreviation, full_name, conference, division, city)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (team_id) 
                DO UPDATE SET 
                    abbreviation = EXCLUDED.abbreviation,
                    full_name = EXCLUDED.full_name,
                    updated_at = NOW()
            """, (
                team['id'],
                team['abbreviation'],
                team['full_name'],
                'East' if team['id'] in EASTERN_TEAMS else 'West',
                team.get('division', ''),
                team.get('city', '')
            ))
        
        print(f"✅ Updated {len(all_teams)} teams")
    
    def update_player_season_stats(self):
        """Update season stats for all active players"""
        print("📊 Updating player season stats...")
        
        # Get all active players
        self.cursor.execute("SELECT player_id FROM players WHERE is_active = TRUE")
        player_ids = [row[0] for row in self.cursor.fetchall()]
        
        updated = 0
        for player_id in player_ids:
            try:
                # Rate limiting
                time.sleep(0.6)
                
                # Get season stats
                stats = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
                    player_id=player_id,
                    season='2024-25'
                ).get_data_frames()[0]
                
                if len(stats) > 0:
                    row = stats.iloc[0]
                    
                    self.cursor.execute("""
                        INSERT INTO player_season_stats (
                            player_id, games_played, minutes_per_game,
                            ppg, rpg, apg, spg, bpg, tov_pg,
                            fgp, fg3p, ftp
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (player_id)
                        DO UPDATE SET
                            games_played = EXCLUDED.games_played,
                            ppg = EXCLUDED.ppg,
                            rpg = EXCLUDED.rpg,
                            apg = EXCLUDED.apg,
                            updated_at = NOW()
                    """, (
                        player_id,
                        row['GP'],
                        row['MIN'],
                        row['PTS'],
                        row['REB'],
                        row['AST'],
                        row['STL'],
                        row['BLK'],
                        row['TOV'],
                        row['FG_PCT'],
                        row['FG3_PCT'],
                        row['FT_PCT']
                    ))
                    
                    updated += 1
                    
                    if updated % 50 == 0:
                        print(f"   Updated {updated} players...")
                        self.conn.commit()
                
            except Exception as e:
                print(f"⚠️ Failed to update player {player_id}: {e}")
                continue
        
        print(f"✅ Updated {updated} players")
    
    def update_recent_games(self):
        """Update last 10 games for each player"""
        print("📊 Updating recent games...")
        
        self.cursor.execute("SELECT player_id FROM players WHERE is_active = TRUE LIMIT 100")
        player_ids = [row[0] for row in self.cursor.fetchall()]
        
        for player_id in player_ids:
            try:
                time.sleep(0.6)
                
                log = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season='2024-25'
                ).get_data_frames()[0].head(10)
                
                for _, game in log.iterrows():
                    self.cursor.execute("""
                        INSERT INTO player_recent_games (
                            player_id, game_id, game_date, opponent,
                            points, rebounds, assists, minutes,
                            fgm, fga, fg3m, fg3a, plus_minus
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (player_id, game_id) DO NOTHING
                    """, (
                        player_id,
                        game['Game_ID'],
                        game['GAME_DATE'],
                        game['MATCHUP'].split()[-1],
                        game['PTS'],
                        game['REB'],
                        game['AST'],
                        game['MIN'],
                        game['FGM'],
                        game['FGA'],
                        game['FG3M'],
                        game['FG3A'],
                        game['PLUS_MINUS']
                    ))
            except:
                continue
        
        print(f"✅ Updated recent games")
    
    def create_snapshot(self):
        """Create daily snapshot"""
        today = datetime.now().date()
        
        self.cursor.execute("SELECT COUNT(*) FROM teams")
        teams_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM players WHERE is_active = TRUE")
        players_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            INSERT INTO daily_snapshots (snapshot_date, teams_count, players_count)
            VALUES (%s, %s, %s)
            ON CONFLICT (snapshot_date) DO NOTHING
        """, (today, teams_count, players_count))
        
        print(f"✅ Created snapshot: {teams_count} teams, {players_count} players")


def run_continuous_collector():
    """
    Run collector continuously on Railway
    Updates every 24 hours
    """
    collector = NBAStatsCollector()
    
    while True:
        try:
            # Run daily update
            collector.run_daily_update()
            
            # Wait 24 hours
            print(f"⏰ Next update in 24 hours...")
            time.sleep(24 * 60 * 60)
            
        except Exception as e:
            print(f"❌ Collector error: {e}")
            time.sleep(60 * 60)  # Wait 1 hour on error


if __name__ == "__main__":
    run_continuous_collector()
```

---

## 🚀 **RAILWAY DEPLOYMENT**

### **1. Add to `requirements.txt`:**

```txt
# Existing packages...

# Database
psycopg2-binary>=2.9.0,<3.0.0

# NBA API (already have this)
nba_api>=1.5.0,<2.0.0
```

### **2. Add to `Procfile` (if not exists):**

```
web: uvicorn trading_dashboard_api:app --host 0.0.0.0 --port $PORT
worker: python nba_stats_collector.py
```

### **3. Railway will run BOTH:**
- `web`: Your live betting API
- `worker`: Stats collector (runs continuously)

---

## 📊 **API ENDPOINTS (Add to `trading_dashboard_api.py`)**

```python
# Database connection
import psycopg2
DATABASE_URL = os.environ.get('DATABASE_URL')

def get_db():
    return psycopg2.connect(DATABASE_URL)

# ============================================================================
# STATS API ENDPOINTS
# ============================================================================

@app.get("/api/stats/players")
async def get_players(team: str = None, limit: int = 50):
    """Get all players (optionally filter by team)"""
    conn = get_db()
    cursor = conn.cursor()
    
    if team:
        cursor.execute("""
            SELECT p.*, pss.ppg, pss.rpg, pss.apg
            FROM players p
            LEFT JOIN player_season_stats pss ON p.player_id = pss.player_id
            WHERE p.team_id = (SELECT team_id FROM teams WHERE abbreviation = %s)
            AND p.is_active = TRUE
            LIMIT %s
        """, (team, limit))
    else:
        cursor.execute("""
            SELECT p.*, pss.ppg, pss.rpg, pss.apg
            FROM players p
            LEFT JOIN player_season_stats pss ON p.player_id = pss.player_id
            WHERE p.is_active = TRUE
            LIMIT %s
        """, (limit,))
    
    columns = [desc[0] for desc in cursor.description]
    players = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    conn.close()
    return {"players": players, "count": len(players)}

@app.get("/api/stats/player/{player_id}")
async def get_player_stats(player_id: str):
    """Get detailed player stats"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Player info + season stats
    cursor.execute("""
        SELECT p.*, pss.*
        FROM players p
        LEFT JOIN player_season_stats pss ON p.player_id = pss.player_id
        WHERE p.player_id = %s
    """, (player_id,))
    
    columns = [desc[0] for desc in cursor.description]
    player = dict(zip(columns, cursor.fetchone()))
    
    # Recent games
    cursor.execute("""
        SELECT * FROM player_recent_games
        WHERE player_id = %s
        ORDER BY game_date DESC
        LIMIT 10
    """, (player_id,))
    
    columns = [desc[0] for desc in cursor.description]
    recent_games = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    conn.close()
    return {
        "player": player,
        "recent_games": recent_games
    }

@app.get("/api/stats/standings")
async def get_standings():
    """Get current standings"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT t.abbreviation, t.full_name, s.*
        FROM standings s
        JOIN teams t ON s.team_id = t.team_id
        WHERE s.updated_at::date = (SELECT MAX(updated_at::date) FROM standings)
        ORDER BY s.conference, s.rank
    """)
    
    columns = [desc[0] for desc in cursor.description]
    standings = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    conn.close()
    return {"standings": standings}
```

---

## ⏰ **CRON SCHEDULE (Railway)**

Railway doesn't have native cron, but we use **continuous loop**:

```python
# In nba_stats_collector.py
while True:
    run_daily_update()
    sleep(24 * 60 * 60)  # 24 hours
```

**OR use Railway Cron (paid plan):**
```
0 0 * * * python nba_stats_collector.py  # Midnight every day
```

---

## 📋 **DEPLOYMENT CHECKLIST**

- [ ] Add PostgreSQL to Railway
- [ ] Update `requirements.txt`
- [ ] Create `nba_stats_collector.py`
- [ ] Add stats API endpoints
- [ ] Deploy to Railway
- [ ] Verify database connection
- [ ] Run initial data collection
- [ ] Confirm 24-hour updates

---

## 🎯 **NEXT STEPS**

1. **I'll create the files**
2. **You deploy to Railway**
3. **System starts collecting automatically**
4. **Later: Add historical data (every day/season)**

**Ready to create the files?** 🚀

