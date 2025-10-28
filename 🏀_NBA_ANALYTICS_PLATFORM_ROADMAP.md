# 🏀 NBA ANALYTICS PLATFORM - COMPLETE ROADMAP

**Vision:** Comprehensive NBA statistics platform with advanced metrics for ML predictions and similarity analysis

---

## 📊 **PHASE 1: DATA COLLECTION (nba_api)**

### **What We'll Track:**

#### **1. Teams Structure**
```python
teams = {
    "LAL": {
        "team_id": "1610612747",
        "full_name": "Los Angeles Lakers",
        "abbreviation": "LAL",
        "conference": "West",
        "division": "Pacific",
        "city": "Los Angeles",
        "arena": "Crypto.com Arena",
        "year_founded": 1947,
        
        # Link to other data
        "roster": [...player_ids],
        "current_season_stats": {...},
        "historical_seasons": [...]
    }
}
```

#### **2. Players Structure**
```python
players = {
    "2544": {  # LeBron James
        "player_id": "2544",
        "name": "LeBron James",
        "team_id": "1610612747",  # Links to LAL
        "team_abbr": "LAL",
        "position": "F",
        "height": "6-9",
        "weight": 250,
        "birthdate": "1984-12-30",
        "experience": 21,
        
        # Basic Stats (per game)
        "ppg": 25.3,
        "rpg": 7.2,
        "apg": 7.8,
        "spg": 1.3,
        "bpg": 0.6,
        "fgp": 0.523,
        "fg3p": 0.411,
        "ftp": 0.750,
        
        # Game log (last N games)
        "game_log": [...],
        
        # Career stats
        "career": {...}
    }
}
```

#### **3. Standings**
```python
standings = {
    "East": [
        {"rank": 1, "team": "BOS", "wins": 45, "losses": 12, "gb": 0.0, "streak": "W5"},
        {"rank": 2, "team": "CLE", "wins": 42, "losses": 15, "gb": 3.0, "streak": "W2"},
        ...
    ],
    "West": [...]
}
```

#### **4. Playoff Picture**
```python
playoffs = {
    "East": {
        "guaranteed": ["BOS", "CLE", "NYK"],
        "play_in_range": ["MIA", "PHI", "ORL", "IND"],
        "eliminated": ["WAS", "DET", "CHA"],
        "clinch_scenarios": {...}
    },
    "West": {...}
}
```

---

## 🗄️ **DATABASE SCHEMA**

### **PostgreSQL Tables:**

```sql
-- Teams
CREATE TABLE teams (
    team_id VARCHAR(10) PRIMARY KEY,
    abbreviation VARCHAR(3) UNIQUE NOT NULL,
    full_name VARCHAR(50),
    conference VARCHAR(10),
    division VARCHAR(20),
    city VARCHAR(50),
    arena VARCHAR(100),
    year_founded INTEGER
);

-- Players
CREATE TABLE players (
    player_id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    position VARCHAR(5),
    height VARCHAR(10),
    weight INTEGER,
    birthdate DATE,
    experience INTEGER
);

-- Player Game Stats
CREATE TABLE player_game_stats (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(10) REFERENCES players(player_id),
    game_id VARCHAR(15),
    game_date DATE,
    team_id VARCHAR(10) REFERENCES teams(team_id),
    opponent_id VARCHAR(10) REFERENCES teams(team_id),
    
    -- Basic Stats
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    fouls INTEGER,
    minutes_played FLOAT,
    
    -- Shooting
    fgm INTEGER,
    fga INTEGER,
    fg3m INTEGER,
    fg3a INTEGER,
    ftm INTEGER,
    fta INTEGER,
    
    -- Plus/Minus
    plus_minus INTEGER,
    
    UNIQUE(player_id, game_id)
);

-- Team Season Stats
CREATE TABLE team_season_stats (
    id SERIAL PRIMARY KEY,
    team_id VARCHAR(10) REFERENCES teams(team_id),
    season VARCHAR(10),
    
    -- Record
    wins INTEGER,
    losses INTEGER,
    win_pct FLOAT,
    
    -- Offense
    ppg FLOAT,
    fgp FLOAT,
    fg3p FLOAT,
    ftp FLOAT,
    apg FLOAT,
    
    -- Defense
    opp_ppg FLOAT,
    opp_fgp FLOAT,
    
    -- Advanced (to be calculated)
    ortg FLOAT,  -- Offensive Rating
    drtg FLOAT,  -- Defensive Rating
    netrtg FLOAT,  -- Net Rating
    pace FLOAT,
    
    UNIQUE(team_id, season)
);

-- Advanced Player Stats (calculated)
CREATE TABLE player_advanced_stats (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(10) REFERENCES players(player_id),
    season VARCHAR(10),
    
    -- Efficiency
    ts_pct FLOAT,  -- True Shooting %
    efg_pct FLOAT,  -- Effective FG%
    usg_pct FLOAT,  -- Usage %
    per FLOAT,  -- Player Efficiency Rating
    
    -- Impact
    bpm FLOAT,  -- Box Plus/Minus
    vorp FLOAT,  -- Value Over Replacement Player
    win_shares FLOAT,
    
    -- Advanced (KenPom-style)
    ortg FLOAT,  -- Offensive Rating
    drtg FLOAT,  -- Defensive Rating
    
    UNIQUE(player_id, season)
);

-- Similarity Scores
CREATE TABLE similarity_scores (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(10),  -- 'player' or 'team'
    entity_id_1 VARCHAR(10),
    entity_id_2 VARCHAR(10),
    similarity_score FLOAT,
    time_window VARCHAR(20),  -- 'last_10_games', 'season', etc.
    calculated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(entity_type, entity_id_1, entity_id_2, time_window)
);
```

---

## 🔌 **NBA_API ENDPOINTS WE'LL USE:**

### **1. Teams & Rosters**
```python
from nba_api.stats.endpoints import (
    teamdetails,
    commonteamroster,
    teaminfocommon
)

# Get all teams
teams = teams.get_teams()

# Get roster for each team
roster = commonteamroster.CommonTeamRoster(team_id="1610612747")
```

### **2. Players**
```python
from nba_api.stats.endpoints import (
    commonplayerinfo,
    playergamelog,
    playercareerstats
)

# Player info
info = commonplayerinfo.CommonPlayerInfo(player_id="2544")

# Game log (last N games)
log = playergamelog.PlayerGameLog(player_id="2544", season="2024-25")

# Career stats
career = playercareerstats.PlayerCareerStats(player_id="2544")
```

### **3. Standings**
```python
from nba_api.stats.endpoints import leaguestandings

standings = leaguestandings.LeagueStandings()
```

### **4. Team Stats**
```python
from nba_api.stats.endpoints import (
    teamdashboardbygeneralsplits,
    teamgamelog
)

# Team stats this season
stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
    team_id="1610612747",
    season="2024-25"
)
```

---

## 📈 **PHASE 2: ADVANCED STATS CALCULATIONS**

### **KenPom-Style Metrics:**

```python
def calculate_kenpom_metrics(team_stats):
    """
    Adjusted Efficiency Metrics (KenPom-style)
    """
    # Offensive Efficiency (points per 100 possessions)
    possessions = (
        team_stats['fga'] 
        + 0.44 * team_stats['fta'] 
        - team_stats['orb'] 
        + team_stats['tov']
    )
    ortg = (team_stats['points'] / possessions) * 100
    
    # Defensive Efficiency (opp points per 100 possessions)
    drtg = (opponent_stats['points'] / opponent_possessions) * 100
    
    # Tempo-Free Stats
    tempo = possessions / (team_stats['minutes'] / 5)
    
    return {
        'ortg': ortg,
        'drtg': drtg,
        'netrtg': ortg - drtg,
        'tempo': tempo
    }
```

### **RAPTOR (FiveThirtyEight-style):**

```python
def calculate_raptor_components(player_stats):
    """
    Robust Algorithm using Player Tracking and On/Off Ratings
    
    Combines:
    - Box score stats
    - On/Off court impact
    - Plus/Minus adjusted for teammates/opponents
    """
    # Offensive RAPTOR
    offensive_raptor = (
        0.3 * calculate_scoring_impact(player_stats)
        + 0.2 * calculate_playmaking_impact(player_stats)
        + 0.1 * calculate_spacing_impact(player_stats)
    )
    
    # Defensive RAPTOR
    defensive_raptor = (
        0.2 * calculate_rim_protection(player_stats)
        + 0.2 * calculate_perimeter_defense(player_stats)
        + 0.1 * calculate_help_defense(player_stats)
    )
    
    return {
        'raptor_offense': offensive_raptor,
        'raptor_defense': defensive_raptor,
        'raptor_total': offensive_raptor + defensive_raptor,
        'war': calculate_war(offensive_raptor + defensive_raptor)
    }
```

### **RAPM (Regularized Adjusted Plus-Minus):**

```python
def calculate_rapm(player_stints, num_iterations=10000):
    """
    Ridge regression to estimate player impact
    
    Accounts for:
    - Teammates on court
    - Opponents on court
    - Home/Away
    - Score differential during stint
    """
    # Ridge regression (L2 regularization)
    from sklearn.linear_model import Ridge
    
    # Build stint matrix
    X = build_stint_matrix(player_stints)  # Players on court
    y = stint_plus_minus  # Point differential
    
    # Regularized regression
    model = Ridge(alpha=1000)
    model.fit(X, y)
    
    # Player coefficients = RAPM
    rapm_values = dict(zip(player_ids, model.coef_))
    
    return rapm_values
```

### **True Shooting % & Advanced Efficiency:**

```python
def calculate_advanced_stats(player_stats):
    """
    Advanced efficiency metrics
    """
    # True Shooting %
    ts_pct = (
        player_stats['points'] / 
        (2 * (player_stats['fga'] + 0.44 * player_stats['fta']))
    )
    
    # Effective FG%
    efg_pct = (
        (player_stats['fgm'] + 0.5 * player_stats['fg3m']) / 
        player_stats['fga']
    )
    
    # Usage Rate
    usg_pct = (
        100 * ((player_stats['fga'] + 0.44 * player_stats['fta'] + player_stats['tov']) * 
        (team_stats['minutes'] / 5)) / 
        (player_stats['minutes'] * 
        (team_stats['fga'] + 0.44 * team_stats['fta'] + team_stats['tov']))
    )
    
    # Box Plus/Minus (simplified)
    bpm = calculate_bpm(player_stats, league_averages)
    
    return {
        'ts_pct': ts_pct,
        'efg_pct': efg_pct,
        'usg_pct': usg_pct,
        'bpm': bpm
    }
```

---

## 🔗 **PHASE 3: SIMILARITY SCORES**

### **Player Similarity:**

```python
def calculate_player_similarity(player1_stats, player2_stats, window="last_10_games"):
    """
    Calculate cosine similarity between player stat profiles
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Normalize stats
    features = [
        'ppg', 'rpg', 'apg', 'spg', 'bpg',
        'fgp', 'fg3p', 'ftp',
        'ts_pct', 'usg_pct', 'bpm'
    ]
    
    vec1 = np.array([player1_stats[f] for f in features])
    vec2 = np.array([player2_stats[f] for f in features])
    
    # Cosine similarity
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    
    return {
        'similarity_score': similarity,
        'comparison': compare_features(vec1, vec2, features),
        'window': window
    }
```

### **Team Similarity:**

```python
def calculate_team_similarity(team1_stats, team2_stats):
    """
    Compare team playing styles
    """
    style_features = [
        'pace',  # Tempo
        'ortg',  # Offensive efficiency
        'drtg',  # Defensive efficiency
        '3pa_per_game',  # 3-point volume
        'assist_ratio',  # Ball movement
        'rebound_rate',  # Rebounding
        'turnover_rate'  # Ball security
    ]
    
    similarity = calculate_feature_similarity(
        team1_stats, 
        team2_stats, 
        style_features
    )
    
    return {
        'style_similarity': similarity,
        'style_comparison': "Similar" if similarity > 0.8 else "Different"
    }
```

---

## 🏗️ **BACKEND API STRUCTURE**

### **FastAPI Endpoints:**

```python
# Teams
@app.get("/api/teams")
async def get_all_teams()

@app.get("/api/teams/{team_id}")
async def get_team_details(team_id: str)

@app.get("/api/teams/{team_id}/roster")
async def get_team_roster(team_id: str)

@app.get("/api/teams/{team_id}/stats")
async def get_team_stats(team_id: str, season: str = "2024-25")

# Players
@app.get("/api/players")
async def get_all_players(team: str = None)

@app.get("/api/players/{player_id}")
async def get_player_info(player_id: str)

@app.get("/api/players/{player_id}/stats")
async def get_player_stats(player_id: str, season: str = "2024-25")

@app.get("/api/players/{player_id}/gamelog")
async def get_player_gamelog(player_id: str, last_n: int = 10)

@app.get("/api/players/{player_id}/advanced")
async def get_player_advanced_stats(player_id: str)

# Standings
@app.get("/api/standings")
async def get_standings(conference: str = None)

# Playoffs
@app.get("/api/playoffs/picture")
async def get_playoff_picture()

# Advanced Stats
@app.get("/api/advanced/team/{team_id}")
async def get_team_advanced_stats(team_id: str)

@app.get("/api/advanced/player/{player_id}")
async def get_player_advanced_stats(player_id: str)

# Similarity
@app.get("/api/similarity/players/{player1_id}/{player2_id}")
async def compare_players(player1_id: str, player2_id: str, window: str = "season")

@app.get("/api/similarity/teams/{team1_id}/{team2_id}")
async def compare_teams(team1_id: str, team2_id: str)

# For ML
@app.get("/api/ml/features/game/{game_id}")
async def get_game_features(game_id: str)

@app.get("/api/ml/features/player/{player_id}")
async def get_player_ml_features(player_id: str, games: int = 10)
```

---

## 📅 **IMPLEMENTATION TIMELINE**

### **Week 1-2: Data Collection**
- [ ] Set up PostgreSQL database
- [ ] Create data collection scripts (nba_api)
- [ ] Populate teams, players, rosters
- [ ] Collect historical stats (last 3 seasons)

### **Week 3-4: Advanced Stats**
- [ ] Implement KenPom-style calculations
- [ ] Calculate efficiency metrics
- [ ] Build RAPM system
- [ ] Create nightly update job

### **Week 5-6: Similarity Engine**
- [ ] Player similarity algorithm
- [ ] Team similarity algorithm
- [ ] Historical comparison tools

### **Week 7-8: Frontend Integration**
- [ ] Build comprehensive stats pages
- [ ] Create comparison visualizations
- [ ] Add player/team search
- [ ] Integrate with ML predictions

---

## 🚀 **START NOW?**

**I can begin with Phase 1:**

1. Create database schema
2. Build data collection service
3. Set up nightly update cron job
4. Create initial API endpoints

**Ready to start? Tell me which phase to begin!**

