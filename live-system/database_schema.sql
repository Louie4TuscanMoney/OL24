-- ============================================================================
-- NBA ANALYTICS PLATFORM - ROLLING MODEL SCHEMA
-- Deploy this to Railway PostgreSQL
-- ============================================================================

-- Teams (30 NBA teams)
CREATE TABLE IF NOT EXISTS teams (
    team_id VARCHAR(10) PRIMARY KEY,
    abbreviation VARCHAR(3) UNIQUE NOT NULL,
    full_name VARCHAR(50),
    conference VARCHAR(10),
    division VARCHAR(20),
    city VARCHAR(50),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Players (450+ active NBA players)
CREATE TABLE IF NOT EXISTS players (
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

-- Player Season Stats (ONE row per player, updated daily)
CREATE TABLE IF NOT EXISTS player_season_stats (
    player_id VARCHAR(10) PRIMARY KEY REFERENCES players(player_id),
    season VARCHAR(10) DEFAULT '2024-25',
    
    -- Games
    games_played INTEGER DEFAULT 0,
    games_started INTEGER DEFAULT 0,
    minutes_per_game FLOAT DEFAULT 0,
    
    -- Basic Per-Game Stats
    ppg FLOAT DEFAULT 0,
    rpg FLOAT DEFAULT 0,
    apg FLOAT DEFAULT 0,
    spg FLOAT DEFAULT 0,
    bpg FLOAT DEFAULT 0,
    tov_pg FLOAT DEFAULT 0,
    
    -- Shooting
    fgp FLOAT DEFAULT 0,
    fg3p FLOAT DEFAULT 0,
    ftp FLOAT DEFAULT 0,
    
    -- Advanced (calculated nightly)
    ts_pct FLOAT,
    efg_pct FLOAT,
    usg_pct FLOAT,
    per FLOAT,
    
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Player Recent Games (rolling window of last 10 games)
CREATE TABLE IF NOT EXISTS player_recent_games (
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

-- Team Season Stats (ONE row per team, updated daily)
CREATE TABLE IF NOT EXISTS team_season_stats (
    team_id VARCHAR(10) PRIMARY KEY REFERENCES teams(team_id),
    season VARCHAR(10) DEFAULT '2024-25',
    
    -- Record
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    win_pct FLOAT DEFAULT 0,
    
    -- Per Game
    ppg FLOAT DEFAULT 0,
    opp_ppg FLOAT DEFAULT 0,
    
    -- Advanced (calculated nightly)
    ortg FLOAT,
    drtg FLOAT,
    netrtg FLOAT,
    pace FLOAT,
    
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Standings (updated daily)
CREATE TABLE IF NOT EXISTS standings (
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

-- Daily Snapshots (tracks collection status)
CREATE TABLE IF NOT EXISTS daily_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE UNIQUE NOT NULL,
    teams_count INTEGER,
    players_count INTEGER,
    games_today INTEGER,
    collection_status VARCHAR(20) DEFAULT 'success',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_players_active ON players(is_active);
CREATE INDEX IF NOT EXISTS idx_recent_games_player ON player_recent_games(player_id);
CREATE INDEX IF NOT EXISTS idx_recent_games_date ON player_recent_games(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_standings_date ON standings(updated_at DESC);

-- ============================================================================
-- INITIAL DATA (Run once to populate teams)
-- ============================================================================

-- Insert all 30 NBA teams
INSERT INTO teams (team_id, abbreviation, full_name, conference, division, city) VALUES
('1610612737', 'ATL', 'Atlanta Hawks', 'East', 'Southeast', 'Atlanta'),
('1610612738', 'BOS', 'Boston Celtics', 'East', 'Atlantic', 'Boston'),
('1610612751', 'BKN', 'Brooklyn Nets', 'East', 'Atlantic', 'Brooklyn'),
('1610612766', 'CHA', 'Charlotte Hornets', 'East', 'Southeast', 'Charlotte'),
('1610612741', 'CHI', 'Chicago Bulls', 'East', 'Central', 'Chicago'),
('1610612739', 'CLE', 'Cleveland Cavaliers', 'East', 'Central', 'Cleveland'),
('1610612742', 'DAL', 'Dallas Mavericks', 'West', 'Southwest', 'Dallas'),
('1610612743', 'DEN', 'Denver Nuggets', 'West', 'Northwest', 'Denver'),
('1610612765', 'DET', 'Detroit Pistons', 'East', 'Central', 'Detroit'),
('1610612744', 'GSW', 'Golden State Warriors', 'West', 'Pacific', 'San Francisco'),
('1610612745', 'HOU', 'Houston Rockets', 'West', 'Southwest', 'Houston'),
('1610612754', 'IND', 'Indiana Pacers', 'East', 'Central', 'Indianapolis'),
('1610612746', 'LAC', 'LA Clippers', 'West', 'Pacific', 'Los Angeles'),
('1610612747', 'LAL', 'Los Angeles Lakers', 'West', 'Pacific', 'Los Angeles'),
('1610612763', 'MEM', 'Memphis Grizzlies', 'West', 'Southwest', 'Memphis'),
('1610612748', 'MIA', 'Miami Heat', 'East', 'Southeast', 'Miami'),
('1610612749', 'MIL', 'Milwaukee Bucks', 'East', 'Central', 'Milwaukee'),
('1610612750', 'MIN', 'Minnesota Timberwolves', 'West', 'Northwest', 'Minneapolis'),
('1610612740', 'NOP', 'New Orleans Pelicans', 'West', 'Southwest', 'New Orleans'),
('1610612752', 'NYK', 'New York Knicks', 'East', 'Atlantic', 'New York'),
('1610612760', 'OKC', 'Oklahoma City Thunder', 'West', 'Northwest', 'Oklahoma City'),
('1610612753', 'ORL', 'Orlando Magic', 'East', 'Southeast', 'Orlando'),
('1610612755', 'PHI', 'Philadelphia 76ers', 'East', 'Atlantic', 'Philadelphia'),
('1610612756', 'PHX', 'Phoenix Suns', 'West', 'Pacific', 'Phoenix'),
('1610612757', 'POR', 'Portland Trail Blazers', 'West', 'Northwest', 'Portland'),
('1610612758', 'SAC', 'Sacramento Kings', 'West', 'Pacific', 'Sacramento'),
('1610612759', 'SAS', 'San Antonio Spurs', 'West', 'Southwest', 'San Antonio'),
('1610612761', 'TOR', 'Toronto Raptors', 'East', 'Atlantic', 'Toronto'),
('1610612762', 'UTA', 'Utah Jazz', 'West', 'Northwest', 'Salt Lake City'),
('1610612764', 'WAS', 'Washington Wizards', 'East', 'Southeast', 'Washington')
ON CONFLICT (team_id) DO NOTHING;

COMMENT ON TABLE teams IS 'All 30 NBA teams';
COMMENT ON TABLE players IS 'Active NBA players (450+)';
COMMENT ON TABLE player_season_stats IS 'Current season stats per player (rolling updates)';
COMMENT ON TABLE player_recent_games IS 'Last 10 games per player';
COMMENT ON TABLE team_season_stats IS 'Current season stats per team';
COMMENT ON TABLE standings IS 'Daily standings snapshots';
COMMENT ON TABLE daily_snapshots IS 'Daily collection status tracking';

