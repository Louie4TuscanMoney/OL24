-- ============================================================================
-- ONTOLOGIC XYZ - OPTIMIZED DATABASE SCHEMA FOR FRONTEND
-- Designed specifically for Railway + Vercel deployment
-- Includes ALL tables needed for frontend API endpoints
-- Version: 5.0 (Frontend Optimized)
-- Last Updated: 2025-10-29
-- ============================================================================

-- ============================================================================
-- SECTION 1: CORE ENTITIES (Teams, Players)
-- ============================================================================

-- 1.1 Teams Master Table (with frontend visual assets)
CREATE TABLE IF NOT EXISTS teams (
    team_id VARCHAR(10) PRIMARY KEY,
    abbreviation VARCHAR(3) UNIQUE NOT NULL,
    full_name VARCHAR(50) NOT NULL,
    conference VARCHAR(10) NOT NULL CHECK (conference IN ('East', 'West')),
    division VARCHAR(20) NOT NULL,
    
    -- Visual Identity (CRITICAL for frontend!)
    logo_url VARCHAR(200) DEFAULT 'https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg',
    primary_color VARCHAR(7) DEFAULT '#000000',
    secondary_color VARCHAR(7) DEFAULT '#FFFFFF',
    
    -- Location & Venue
    city VARCHAR(50),
    state VARCHAR(30),
    arena VARCHAR(100),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Populate teams with visual assets
INSERT INTO teams (team_id, abbreviation, full_name, conference, division, city, logo_url, primary_color, secondary_color) VALUES
('1610612737', 'ATL', 'Atlanta Hawks', 'East', 'Southeast', 'Atlanta', 
 'https://cdn.nba.com/logos/nba/1610612737/primary/L/logo.svg', '#E03A3E', '#C1D32F'),
('1610612738', 'BOS', 'Boston Celtics', 'East', 'Atlantic', 'Boston',
 'https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg', '#007A33', '#BA9653'),
('1610612751', 'BKN', 'Brooklyn Nets', 'East', 'Atlantic', 'Brooklyn',
 'https://cdn.nba.com/logos/nba/1610612751/primary/L/logo.svg', '#000000', '#FFFFFF'),
('1610612766', 'CHA', 'Charlotte Hornets', 'East', 'Southeast', 'Charlotte',
 'https://cdn.nba.com/logos/nba/1610612766/primary/L/logo.svg', '#1D1160', '#00788C'),
('1610612741', 'CHI', 'Chicago Bulls', 'East', 'Central', 'Chicago',
 'https://cdn.nba.com/logos/nba/1610612741/primary/L/logo.svg', '#CE1141', '#000000'),
('1610612739', 'CLE', 'Cleveland Cavaliers', 'East', 'Central', 'Cleveland',
 'https://cdn.nba.com/logos/nba/1610612739/primary/L/logo.svg', '#860038', '#FDBB30'),
('1610612742', 'DAL', 'Dallas Mavericks', 'West', 'Southwest', 'Dallas',
 'https://cdn.nba.com/logos/nba/1610612742/primary/L/logo.svg', '#00538C', '#002B5E'),
('1610612743', 'DEN', 'Denver Nuggets', 'West', 'Northwest', 'Denver',
 'https://cdn.nba.com/logos/nba/1610612743/primary/L/logo.svg', '#0E2240', '#FEC524'),
('1610612765', 'DET', 'Detroit Pistons', 'East', 'Central', 'Detroit',
 'https://cdn.nba.com/logos/nba/1610612765/primary/L/logo.svg', '#C8102E', '#1D42BA'),
('1610612744', 'GSW', 'Golden State Warriors', 'West', 'Pacific', 'San Francisco',
 'https://cdn.nba.com/logos/nba/1610612744/primary/L/logo.svg', '#1D428A', '#FFC72C'),
('1610612745', 'HOU', 'Houston Rockets', 'West', 'Southwest', 'Houston',
 'https://cdn.nba.com/logos/nba/1610612745/primary/L/logo.svg', '#CE1141', '#000000'),
('1610612754', 'IND', 'Indiana Pacers', 'East', 'Central', 'Indianapolis',
 'https://cdn.nba.com/logos/nba/1610612754/primary/L/logo.svg', '#002D62', '#FDBB30'),
('1610612746', 'LAC', 'LA Clippers', 'West', 'Pacific', 'Los Angeles',
 'https://cdn.nba.com/logos/nba/1610612746/primary/L/logo.svg', '#C8102E', '#1D428A'),
('1610612747', 'LAL', 'Los Angeles Lakers', 'West', 'Pacific', 'Los Angeles',
 'https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg', '#552583', '#FDB927'),
('1610612763', 'MEM', 'Memphis Grizzlies', 'West', 'Southwest', 'Memphis',
 'https://cdn.nba.com/logos/nba/1610612763/primary/L/logo.svg', '#5D76A9', '#12173F'),
('1610612748', 'MIA', 'Miami Heat', 'East', 'Southeast', 'Miami',
 'https://cdn.nba.com/logos/nba/1610612748/primary/L/logo.svg', '#98002E', '#F9A01B'),
('1610612749', 'MIL', 'Milwaukee Bucks', 'East', 'Central', 'Milwaukee',
 'https://cdn.nba.com/logos/nba/1610612749/primary/L/logo.svg', '#00471B', '#EEE1C6'),
('1610612750', 'MIN', 'Minnesota Timberwolves', 'West', 'Northwest', 'Minneapolis',
 'https://cdn.nba.com/logos/nba/1610612750/primary/L/logo.svg', '#0C2340', '#236192'),
('1610612740', 'NOP', 'New Orleans Pelicans', 'West', 'Southwest', 'New Orleans',
 'https://cdn.nba.com/logos/nba/1610612740/primary/L/logo.svg', '#0C2340', '#C8102E'),
('1610612752', 'NYK', 'New York Knicks', 'East', 'Atlantic', 'New York',
 'https://cdn.nba.com/logos/nba/1610612752/primary/L/logo.svg', '#006BB6', '#F58426'),
('1610612760', 'OKC', 'Oklahoma City Thunder', 'West', 'Northwest', 'Oklahoma City',
 'https://cdn.nba.com/logos/nba/1610612760/primary/L/logo.svg', '#007AC1', '#EF3B24'),
('1610612753', 'ORL', 'Orlando Magic', 'East', 'Southeast', 'Orlando',
 'https://cdn.nba.com/logos/nba/1610612753/primary/L/logo.svg', '#0077C0', '#C4CED4'),
('1610612755', 'PHI', 'Philadelphia 76ers', 'East', 'Atlantic', 'Philadelphia',
 'https://cdn.nba.com/logos/nba/1610612755/primary/L/logo.svg', '#006BB6', '#ED174C'),
('1610612756', 'PHX', 'Phoenix Suns', 'West', 'Pacific', 'Phoenix',
 'https://cdn.nba.com/logos/nba/1610612756/primary/L/logo.svg', '#1D1160', '#E56020'),
('1610612757', 'POR', 'Portland Trail Blazers', 'West', 'Northwest', 'Portland',
 'https://cdn.nba.com/logos/nba/1610612757/primary/L/logo.svg', '#E03A3E', '#000000'),
('1610612758', 'SAC', 'Sacramento Kings', 'West', 'Pacific', 'Sacramento',
 'https://cdn.nba.com/logos/nba/1610612758/primary/L/logo.svg', '#5A2D81', '#63727A'),
('1610612759', 'SAS', 'San Antonio Spurs', 'West', 'Southwest', 'San Antonio',
 'https://cdn.nba.com/logos/nba/1610612759/primary/L/logo.svg', '#C4CED4', '#000000'),
('1610612761', 'TOR', 'Toronto Raptors', 'East', 'Atlantic', 'Toronto',
 'https://cdn.nba.com/logos/nba/1610612761/primary/L/logo.svg', '#CE1141', '#000000'),
('1610612762', 'UTA', 'Utah Jazz', 'West', 'Northwest', 'Salt Lake City',
 'https://cdn.nba.com/logos/nba/1610612762/primary/L/logo.svg', '#002B5C', '#00471B'),
('1610612764', 'WAS', 'Washington Wizards', 'East', 'Southeast', 'Washington',
 'https://cdn.nba.com/logos/nba/1610612764/primary/L/logo.svg', '#002B5C', '#E31837')
ON CONFLICT (team_id) DO UPDATE SET
    logo_url = EXCLUDED.logo_url,
    primary_color = EXCLUDED.primary_color,
    secondary_color = EXCLUDED.secondary_color;

CREATE INDEX IF NOT EXISTS idx_teams_abbr ON teams(abbreviation);
CREATE INDEX IF NOT EXISTS idx_teams_conference ON teams(conference);


-- 1.2 Players Master Table
CREATE TABLE IF NOT EXISTS players (
    player_id VARCHAR(10) PRIMARY KEY,
    
    -- Identity
    name VARCHAR(100) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    
    -- Team Assignment
    team_id VARCHAR(10) REFERENCES teams(team_id) ON DELETE SET NULL,
    
    -- Position & Role
    jersey_number VARCHAR(3),
    position VARCHAR(5) CHECK (position IN ('PG', 'SG', 'SF', 'PF', 'C', 'G', 'F')),
    
    -- Physical Attributes
    height_inches INT,
    height_display VARCHAR(10),  -- e.g., "6-8"
    weight_lbs INT,
    
    -- Career Info
    birthdate DATE,
    age INT,
    country VARCHAR(50),
    experience_years INT,
    draft_year INT,
    draft_round INT,
    draft_number INT,
    college VARCHAR(100),
    
    -- Visual Assets (for frontend player cards!)
    headshot_url VARCHAR(200),
    action_photo_url VARCHAR(200),
    
    -- Contract & Status
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_players_position ON players(position);
CREATE INDEX IF NOT EXISTS idx_players_active ON players(is_active);
CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);


-- 1.3 Seasons Reference Table
CREATE TABLE IF NOT EXISTS seasons (
    season_id VARCHAR(10) PRIMARY KEY,
    start_date DATE NOT NULL,
    end_date DATE,
    is_current BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO seasons (season_id, start_date, is_current) 
VALUES ('2025-26', '2025-10-22', TRUE)
ON CONFLICT (season_id) DO UPDATE SET is_current = EXCLUDED.is_current;

CREATE INDEX IF NOT EXISTS idx_seasons_current ON seasons(is_current);


-- ============================================================================
-- SECTION 2: PLAYER PERFORMANCE DATA
-- ============================================================================

-- 2.1 Player Box Scores (for computing team stats)
CREATE TABLE IF NOT EXISTS player_box_scores (
    player_id VARCHAR(10) NOT NULL REFERENCES players(player_id),
    game_id VARCHAR(15) NOT NULL,
    game_date DATE NOT NULL,
    season_id VARCHAR(10) DEFAULT '2025-26',
    team_id VARCHAR(10) REFERENCES teams(team_id),
    opponent_id VARCHAR(10) REFERENCES teams(team_id),
    
    -- Playing Time
    minutes DECIMAL(5,2),
    
    -- Scoring
    pts INT DEFAULT 0,
    fgm INT DEFAULT 0,
    fga INT DEFAULT 0,
    fg_pct DECIMAL(5,3),
    fg3m INT DEFAULT 0,
    fg3a INT DEFAULT 0,
    fg3_pct DECIMAL(5,3),
    ftm INT DEFAULT 0,
    fta INT DEFAULT 0,
    ft_pct DECIMAL(5,3),
    
    -- Rebounding
    oreb INT DEFAULT 0,
    dreb INT DEFAULT 0,
    reb INT DEFAULT 0,
    
    -- Playmaking
    ast INT DEFAULT 0,
    
    -- Defense
    stl INT DEFAULT 0,
    blk INT DEFAULT 0,
    
    -- Errors
    tov INT DEFAULT 0,
    pf INT DEFAULT 0,
    
    -- Advanced
    plus_minus INT,
    team_poss DECIMAL(10,2),
    
    -- Per-100 Stats
    pts_100 DECIMAL(7,2),
    reb_100 DECIMAL(7,2),
    ast_100 DECIMAL(7,2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (player_id, game_id, game_date)
);

CREATE INDEX IF NOT EXISTS idx_box_scores_player_date ON player_box_scores(player_id, game_date DESC);
CREATE INDEX IF NOT EXISTS idx_box_scores_game ON player_box_scores(game_id);
CREATE INDEX IF NOT EXISTS idx_box_scores_team ON player_box_scores(team_id, game_date DESC);


-- 2.2 Player Season Aggregates (powers /api/stats APIs)
CREATE TABLE IF NOT EXISTS player_season_stats (
    player_id VARCHAR(10) REFERENCES players(player_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    
    -- Games Played
    games_played INT DEFAULT 0,
    games_started INT DEFAULT 0,
    minutes_total DECIMAL(10,2) DEFAULT 0,
    
    -- Traditional Stats (Totals)
    pts_total INT DEFAULT 0,
    reb_total INT DEFAULT 0,
    ast_total INT DEFAULT 0,
    stl_total INT DEFAULT 0,
    blk_total INT DEFAULT 0,
    tov_total INT DEFAULT 0,
    
    -- Shooting (Totals)
    fgm_total INT DEFAULT 0,
    fga_total INT DEFAULT 0,
    fg3m_total INT DEFAULT 0,
    fg3a_total INT DEFAULT 0,
    ftm_total INT DEFAULT 0,
    fta_total INT DEFAULT 0,
    
    -- Shooting Percentages
    fg_pct DECIMAL(5,3),
    fg3_pct DECIMAL(5,3),
    ft_pct DECIMAL(5,3),
    
    -- Advanced Shooting (REAL - from Basketball Reference)
    ts_pct DECIMAL(5,3),
    efg_pct DECIMAL(5,3),
    
    -- Per-Game Averages (computed columns)
    ppg DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE WHEN games_played > 0 THEN pts_total / CAST(games_played AS DECIMAL) ELSE 0 END
    ) STORED,
    rpg DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE WHEN games_played > 0 THEN reb_total / CAST(games_played AS DECIMAL) ELSE 0 END
    ) STORED,
    apg DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE WHEN games_played > 0 THEN ast_total / CAST(games_played AS DECIMAL) ELSE 0 END
    ) STORED,
    
    -- Per-100 Possessions (REAL - from Basketball Reference)
    pts_100 DECIMAL(7,2),
    reb_100 DECIMAL(7,2),
    ast_100 DECIMAL(7,2),
    stl_100 DECIMAL(7,2),
    blk_100 DECIMAL(7,2),
    tov_100 DECIMAL(7,2),
    
    -- Per-36 Minutes
    pts_36 DECIMAL(7,2),
    reb_36 DECIMAL(7,2),
    ast_36 DECIMAL(7,2),
    
    -- Impact Metrics (from Basketball Reference)
    bpm DECIMAL(7,3),  -- Box Plus-Minus
    obpm DECIMAL(7,3),  -- Offensive BPM
    dbpm DECIMAL(7,3),  -- Defensive BPM
    vorp DECIMAL(7,3),  -- Value Over Replacement Player
    per DECIMAL(7,2),  -- Player Efficiency Rating
    usage_pct DECIMAL(5,3),  -- Usage Percentage
    win_shares DECIMAL(7,2),
    win_shares_48 DECIMAL(7,3),
    
    -- Custom Impact Metrics
    rapm_total DECIMAL(7,3),  -- Regularized Adjusted Plus-Minus
    lebron_total DECIMAL(7,3),  -- Composite metric
    lebron_offense DECIMAL(7,3),
    lebron_defense DECIMAL(7,3),
    
    -- Metadata
    updated_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (player_id, season_id)
);

CREATE INDEX IF NOT EXISTS idx_player_season_stats_season ON player_season_stats(season_id);
CREATE INDEX IF NOT EXISTS idx_player_season_stats_ppg ON player_season_stats(ppg DESC);
CREATE INDEX IF NOT EXISTS idx_player_season_stats_team ON player_season_stats(team_id, season_id);


-- 2.3 Team Season Aggregates
CREATE TABLE IF NOT EXISTS team_season_stats (
    team_id VARCHAR(10) REFERENCES teams(team_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    
    -- Record
    games_played INT DEFAULT 0,
    wins INT DEFAULT 0,
    losses INT DEFAULT 0,
    win_pct DECIMAL(5,3) GENERATED ALWAYS AS (
        CASE WHEN games_played > 0 THEN wins / CAST(games_played AS DECIMAL) ELSE 0 END
    ) STORED,
    
    -- Scoring
    pts_total INT DEFAULT 0,
    pts_allowed_total INT DEFAULT 0,
    ppg DECIMAL(6,2) GENERATED ALWAYS AS (
        CASE WHEN games_played > 0 THEN pts_total / CAST(games_played AS DECIMAL) ELSE 0 END
    ) STORED,
    opp_ppg DECIMAL(6,2) GENERATED ALWAYS AS (
        CASE WHEN games_played > 0 THEN pts_allowed_total / CAST(games_played AS DECIMAL) ELSE 0 END
    ) STORED,
    
    -- Advanced
    pace DECIMAL(6,2),
    offensive_rating DECIMAL(6,2),
    defensive_rating DECIMAL(6,2),
    net_rating DECIMAL(6,2) GENERATED ALWAYS AS (offensive_rating - defensive_rating) STORED,
    
    -- Metadata
    updated_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (team_id, season_id)
);

CREATE INDEX IF NOT EXISTS idx_team_season_stats_season ON team_season_stats(season_id);
CREATE INDEX IF NOT EXISTS idx_team_season_stats_net_rating ON team_season_stats(net_rating DESC);


-- ============================================================================
-- SECTION 3: SCHEDULE & GAMES
-- ============================================================================

-- 3.1 NBA Schedule (powers /api/schedule endpoints)
CREATE TABLE IF NOT EXISTS nba_schedule (
    game_id VARCHAR(15) PRIMARY KEY,
    season_id VARCHAR(10) DEFAULT '2025-26',
    game_date DATE NOT NULL,
    game_time TIME,
    home_team_id VARCHAR(10) REFERENCES teams(team_id),
    away_team_id VARCHAR(10) REFERENCES teams(team_id),
    home_score INT,
    away_score INT,
    arena VARCHAR(100),
    tv_broadcast VARCHAR(50),
    game_status VARCHAR(20) DEFAULT 'Scheduled' CHECK (game_status IN ('Scheduled', 'Live', 'Final', 'Postponed')),
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_schedule_date ON nba_schedule(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_schedule_season ON nba_schedule(season_id, game_date);
CREATE INDEX IF NOT EXISTS idx_schedule_team_home ON nba_schedule(home_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_schedule_team_away ON nba_schedule(away_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_schedule_status ON nba_schedule(game_status);


-- ============================================================================
-- SECTION 4: INJURIES & DEPTH CHARTS
-- ============================================================================

-- 4.1 Player Injuries (powers /api/injuries endpoint)
CREATE TABLE IF NOT EXISTS player_injuries (
    injury_id SERIAL PRIMARY KEY,
    player_id VARCHAR(10) REFERENCES players(player_id),
    
    -- Injury Details
    status VARCHAR(20) CHECK (status IN ('Out', 'Questionable', 'Doubtful', 'Day-to-Day', 'Probable')),
    injury_type VARCHAR(50),  -- e.g., "Ankle", "Knee", "Rest"
    description TEXT,
    
    -- Dates
    injury_date DATE NOT NULL,
    return_date DATE,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,  -- CRITICAL: Only show active=true on frontend
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_injuries_player ON player_injuries(player_id, injury_date DESC);
CREATE INDEX IF NOT EXISTS idx_injuries_status ON player_injuries(status);
CREATE INDEX IF NOT EXISTS idx_injuries_active ON player_injuries(is_active);


-- 4.2 Team Depth Charts (optional - can compute from MPG)
CREATE TABLE IF NOT EXISTS team_depth_charts (
    team_id VARCHAR(10) REFERENCES teams(team_id),
    player_id VARCHAR(10) REFERENCES players(player_id),
    position VARCHAR(5),
    depth_rank INT CHECK (depth_rank >= 1 AND depth_rank <= 5),
    
    updated_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (team_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_depth_team ON team_depth_charts(team_id, depth_rank);


-- ============================================================================
-- SECTION 5: ML PREDICTIONS & BETTING
-- ============================================================================

-- 5.1 ML Predictions (powers /api/ml/prediction endpoints)
CREATE TABLE IF NOT EXISTS ml_predictions (
    prediction_id SERIAL PRIMARY KEY,
    game_id VARCHAR(15),
    
    -- Prediction Timing
    prediction_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    quarter INT CHECK (quarter >= 1 AND quarter <= 4),
    time_remaining VARCHAR(10),
    
    -- ML Output
    point_forecast DECIMAL(6,2),
    interval_lower DECIMAL(6,2),
    interval_upper DECIMAL(6,2),
    model_confidence DECIMAL(5,3),
    
    -- Features & Importance
    features_extracted JSON,  -- 33 Mamba features
    feature_importance JSON,
    
    -- Market Comparison
    market_spread DECIMAL(6,2),
    edge_detected BOOLEAN DEFAULT FALSE,
    edge_magnitude DECIMAL(6,2),
    
    -- Special Flags
    is_q2_6min BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_game ON ml_predictions(game_id, prediction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_q2_6min ON ml_predictions(is_q2_6min, game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON ml_predictions(prediction_timestamp DESC);


-- ============================================================================
-- SECTION 6: PLAYER LAST 10 GAMES (for /api/stats/player endpoints)
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_last10 (
    player_id VARCHAR(10) REFERENCES players(player_id),
    game_id VARCHAR(15),
    game_date DATE NOT NULL,
    game_rank INT,  -- 1 = most recent, 10 = 10th most recent
    opponent_id VARCHAR(10) REFERENCES teams(team_id),
    
    -- Stats
    pts INT,
    reb INT,
    ast INT,
    stl INT,
    blk INT,
    tov INT,
    fgm INT,
    fga INT,
    fg3m INT,
    minutes DECIMAL(5,2),
    plus_minus INT,
    
    -- Advanced
    pts_100 DECIMAL(7,2),
    ts_pct DECIMAL(5,3),
    
    PRIMARY KEY (player_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_last10_player ON player_last10(player_id, game_rank);
CREATE INDEX IF NOT EXISTS idx_last10_date ON player_last10(game_date DESC);


-- ============================================================================
-- SECTION 7: STANDINGS (optional but nice to have)
-- ============================================================================

CREATE TABLE IF NOT EXISTS standings (
    id SERIAL PRIMARY KEY,
    team_id VARCHAR(10) REFERENCES teams(team_id),
    season_id VARCHAR(10) DEFAULT '2025-26',
    conference VARCHAR(10),
    rank INT,
    wins INT,
    losses INT,
    gb DECIMAL(4,1),  -- Games Behind
    home_record VARCHAR(10),
    away_record VARCHAR(10),
    last_10 VARCHAR(10),
    streak VARCHAR(5),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_standings_date ON standings(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_standings_conference ON standings(conference, rank);
CREATE UNIQUE INDEX IF NOT EXISTS idx_standings_team_date ON standings(team_id, (DATE(updated_at)));


-- ============================================================================
-- SECTION 8: PLAYER TRANSACTIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_transactions (
    transaction_id SERIAL PRIMARY KEY,
    player_id VARCHAR(10) REFERENCES players(player_id) ON DELETE SET NULL,
    player_name VARCHAR(100) NOT NULL,
    
    -- Transaction Details
    transaction_type VARCHAR(20) NOT NULL CHECK (transaction_type IN ('Trade', 'Waiver', 'Signing', 'Release', 'Two-Way', 'G League')),
    transaction_date DATE NOT NULL,
    
    -- Team Movement
    from_team_id VARCHAR(10) REFERENCES teams(team_id) ON DELETE SET NULL,
    to_team_id VARCHAR(10) REFERENCES teams(team_id) ON DELETE SET NULL,
    
    -- Description
    trade_description TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transactions_player ON player_transactions(player_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON player_transactions(transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_from_team ON player_transactions(from_team_id, transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_to_team ON player_transactions(to_team_id, transaction_date DESC);


-- ============================================================================
-- COMMENTS (for documentation)
-- ============================================================================

COMMENT ON TABLE teams IS 'All 30 NBA teams with logos and brand colors for frontend';
COMMENT ON TABLE players IS 'Active NBA players with headshots and career info';
COMMENT ON TABLE player_season_stats IS '2025-26 season stats per player (auto-computed from box scores)';
COMMENT ON TABLE team_season_stats IS '2025-26 season stats per team';
COMMENT ON TABLE player_injuries IS 'Current player injuries (filter by is_active=true)';
COMMENT ON TABLE nba_schedule IS 'NBA schedule for upcoming games';
COMMENT ON TABLE team_depth_charts IS 'Team depth charts (or compute from MPG)';
COMMENT ON TABLE ml_predictions IS 'ML predictions from Mamba model';
COMMENT ON TABLE player_last10 IS 'Last 10 games per player for trends';
COMMENT ON TABLE player_transactions IS 'League-wide player transactions';

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

