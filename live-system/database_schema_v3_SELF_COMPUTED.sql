-- ============================================================================
-- NBA ANALYTICS - SELF-COMPUTED METRICS (100% nba_api → 100% our math)
-- NO EXTERNAL DATA. WE COMPUTE EVERYTHING.
-- ============================================================================

-- ============================================================================
-- 1. FOUNDATION (30 teams, 450 players, seasons)
-- ============================================================================

CREATE TABLE IF NOT EXISTS teams (
    team_id VARCHAR(10) PRIMARY KEY,
    abbreviation VARCHAR(3) UNIQUE NOT NULL,
    full_name VARCHAR(50) NOT NULL,
    conference VARCHAR(10) NOT NULL,
    division VARCHAR(20),
    
    -- Visual Assets (for frontend)
    logo_url VARCHAR(200),  -- Team logo
    primary_color VARCHAR(7),  -- Hex color
    secondary_color VARCHAR(7),
    
    -- Location
    city VARCHAR(50),
    state VARCHAR(30),
    arena VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS seasons (
    season_id VARCHAR(10) PRIMARY KEY,
    start_date DATE,
    is_current BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS players (
    player_id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    
    -- Visual Assets (for frontend)
    headshot_url VARCHAR(200),  -- Player image
    action_photo_url VARCHAR(200),  -- Action shot
    
    -- Bio
    jersey_number VARCHAR(3),
    position VARCHAR(5),
    height_feet INTEGER,
    height_inches INTEGER,
    height_display VARCHAR(10),  -- "6-9"
    weight_lbs INTEGER,
    birthdate DATE,
    age INTEGER,
    country VARCHAR(50),
    
    -- Career
    draft_year INTEGER,
    draft_round INTEGER,
    draft_number INTEGER,
    draft_team VARCHAR(50),
    college VARCHAR(100),
    experience_years INTEGER,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_rookie BOOLEAN DEFAULT FALSE,
    
    -- Social/Marketing
    twitter_handle VARCHAR(50),
    instagram_handle VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- 2. RAW BOX SCORES (from nba_api only)
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_box_scores (
    player_id VARCHAR(10) REFERENCES players(player_id),
    game_id VARCHAR(15) NOT NULL,
    game_date DATE NOT NULL,
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    opponent_id VARCHAR(10) REFERENCES teams(team_id),
    
    -- From boxscoretraditionalv2
    minutes FLOAT,
    pts INTEGER,
    fgm INTEGER,
    fga INTEGER,
    fg3m INTEGER,
    fg3a INTEGER,
    ftm INTEGER,
    fta INTEGER,
    oreb INTEGER,
    dreb INTEGER,
    reb INTEGER,
    ast INTEGER,
    stl INTEGER,
    blk INTEGER,
    tov INTEGER,
    pf INTEGER,
    plus_minus INTEGER,
    
    -- From boxscoreadvancedv2 (team-level)
    team_poss FLOAT,  -- CRITICAL for per-100!
    
    -- On/Off (from advanced box)
    on_court_netrtg FLOAT,
    off_court_netrtg FLOAT,
    
    PRIMARY KEY (player_id, game_id)
) PARTITION BY RANGE (game_date);

-- Partitions (one per season)
CREATE TABLE IF NOT EXISTS player_box_scores_2024_25 PARTITION OF player_box_scores
    FOR VALUES FROM ('2024-10-01') TO ('2025-06-30');

CREATE TABLE IF NOT EXISTS player_box_scores_2025_26 PARTITION OF player_box_scores
    FOR VALUES FROM ('2025-10-01') TO ('2026-06-30');

-- ============================================================================
-- 3. MATERIALIZED VIEW: LAST 10 GAMES (FAST!)
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS player_last10 AS
SELECT 
    player_id,
    game_id,
    game_date,
    team_id,
    opponent_id,
    minutes,
    pts,
    reb,
    ast,
    stl,
    blk,
    tov,
    fgm,
    fga,
    fg3m,
    fg3a,
    ftm,
    fta,
    plus_minus,
    team_poss,
    -- Computed (our math!)
    CASE WHEN fga > 0 THEN (fgm + 0.5 * fg3m)::FLOAT / fga ELSE 0 END AS efg_pct,
    CASE WHEN (fga + 0.44 * fta) > 0 THEN pts::FLOAT / (2 * (fga + 0.44 * fta)) ELSE 0 END AS ts_pct,
    CASE WHEN team_poss > 0 THEN (pts * 100.0) / team_poss ELSE 0 END AS pts_100,
    ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY game_date DESC) AS game_rank
FROM player_box_scores
WHERE game_date >= CURRENT_DATE - INTERVAL '30 days'  -- Rolling 30-day window
;

CREATE UNIQUE INDEX IF NOT EXISTS idx_player_last10_unique ON player_last10(player_id, game_id);
CREATE INDEX IF NOT EXISTS idx_player_last10_player ON player_last10(player_id, game_rank);

-- ============================================================================
-- 4. PLAYER SEASON STATS (ONE ROW - SELF-COMPUTED!)
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_season_stats (
    player_id VARCHAR(10) REFERENCES players(player_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    
    -- Games
    gp INTEGER DEFAULT 0,
    gs INTEGER DEFAULT 0,
    min_total FLOAT DEFAULT 0,
    
    -- Totals (raw from box scores)
    pts_total INTEGER DEFAULT 0,
    reb_total INTEGER DEFAULT 0,
    ast_total INTEGER DEFAULT 0,
    stl_total INTEGER DEFAULT 0,
    blk_total INTEGER DEFAULT 0,
    tov_total INTEGER DEFAULT 0,
    fgm_total INTEGER DEFAULT 0,
    fga_total INTEGER DEFAULT 0,
    fg3m_total INTEGER DEFAULT 0,
    fg3a_total INTEGER DEFAULT 0,
    ftm_total INTEGER DEFAULT 0,
    fta_total INTEGER DEFAULT 0,
    
    -- Per-Game (computed by us!)
    ppg FLOAT GENERATED ALWAYS AS (CASE WHEN gp > 0 THEN pts_total::FLOAT / gp ELSE 0 END) STORED,
    rpg FLOAT GENERATED ALWAYS AS (CASE WHEN gp > 0 THEN reb_total::FLOAT / gp ELSE 0 END) STORED,
    apg FLOAT GENERATED ALWAYS AS (CASE WHEN gp > 0 THEN ast_total::FLOAT / gp ELSE 0 END) STORED,
    
    -- Shooting % (computed by us!)
    fg_pct FLOAT GENERATED ALWAYS AS (CASE WHEN fga_total > 0 THEN fgm_total::FLOAT / fga_total ELSE 0 END) STORED,
    fg3_pct FLOAT GENERATED ALWAYS AS (CASE WHEN fg3a_total > 0 THEN fg3m_total::FLOAT / fg3a_total ELSE 0 END) STORED,
    ft_pct FLOAT GENERATED ALWAYS AS (CASE WHEN fta_total > 0 THEN ftm_total::FLOAT / fta_total ELSE 0 END) STORED,
    
    -- Advanced (computed by us!)
    ts_pct FLOAT GENERATED ALWAYS AS (
        CASE WHEN (fga_total + 0.44 * fta_total) > 0 
        THEN pts_total::FLOAT / (2 * (fga_total + 0.44 * fta_total)) 
        ELSE 0 END
    ) STORED,
    
    efg_pct FLOAT GENERATED ALWAYS AS (
        CASE WHEN fga_total > 0 
        THEN (fgm_total + 0.5 * fg3m_total)::FLOAT / fga_total 
        ELSE 0 END
    ) STORED,
    
    -- Team possessions (from advanced box)
    team_poss_total FLOAT DEFAULT 0,
    
    -- Per-100 (computed by us!)
    pts_100 FLOAT GENERATED ALWAYS AS (
        CASE WHEN team_poss_total > 0 THEN (pts_total * 100.0) / team_poss_total ELSE 0 END
    ) STORED,
    reb_100 FLOAT GENERATED ALWAYS AS (
        CASE WHEN team_poss_total > 0 THEN (reb_total * 100.0) / team_poss_total ELSE 0 END
    ) STORED,
    ast_100 FLOAT GENERATED ALWAYS AS (
        CASE WHEN team_poss_total > 0 THEN (ast_total * 100.0) / team_poss_total ELSE 0 END
    ) STORED,
    tov_100 FLOAT GENERATED ALWAYS AS (
        CASE WHEN team_poss_total > 0 THEN (tov_total * 100.0) / team_poss_total ELSE 0 END
    ) STORED,
    
    -- Per-36 (computed by us!)
    pts_36 FLOAT GENERATED ALWAYS AS (
        CASE WHEN min_total > 0 THEN (pts_total * 36.0) / min_total ELSE 0 END
    ) STORED,
    reb_36 FLOAT GENERATED ALWAYS AS (
        CASE WHEN min_total > 0 THEN (reb_total * 36.0) / min_total ELSE 0 END
    ) STORED,
    ast_36 FLOAT GENERATED ALWAYS AS (
        CASE WHEN min_total > 0 THEN (ast_total * 36.0) / min_total ELSE 0 END
    ) STORED,
    
    -- RAPM (computed by us weekly!)
    rapm_offense FLOAT,
    rapm_defense FLOAT,
    rapm_total FLOAT GENERATED ALWAYS AS (COALESCE(rapm_offense, 0) + COALESCE(rapm_defense, 0)) STORED,
    
    -- LEBRON (computed by us!)
    lebron_offense FLOAT,
    lebron_defense FLOAT,
    lebron_total FLOAT GENERATED ALWAYS AS (COALESCE(lebron_offense, 0) + COALESCE(lebron_defense, 0)) STORED,
    
    updated_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (player_id, season_id)
);

-- ============================================================================
-- 5. TEAM SEASON STATS (ONE ROW - SELF-COMPUTED!)
-- ============================================================================

CREATE TABLE IF NOT EXISTS team_season_stats (
    team_id VARCHAR(10) REFERENCES teams(team_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    
    -- Record
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    win_pct FLOAT GENERATED ALWAYS AS (
        CASE WHEN (wins + losses) > 0 THEN wins::FLOAT / (wins + losses) ELSE 0 END
    ) STORED,
    
    -- Totals (from our box scores)
    pts_total INTEGER DEFAULT 0,
    opp_pts_total INTEGER DEFAULT 0,
    poss_total FLOAT DEFAULT 0,
    opp_poss_total FLOAT DEFAULT 0,
    
    -- Per-Game
    ppg FLOAT GENERATED ALWAYS AS (
        CASE WHEN (wins + losses) > 0 THEN pts_total::FLOAT / (wins + losses) ELSE 0 END
    ) STORED,
    
    -- Ratings (COMPUTED BY US!)
    ortg FLOAT GENERATED ALWAYS AS (
        CASE WHEN poss_total > 0 THEN (pts_total * 100.0) / poss_total ELSE 0 END
    ) STORED,
    drtg FLOAT GENERATED ALWAYS AS (
        CASE WHEN opp_poss_total > 0 THEN (opp_pts_total * 100.0) / opp_poss_total ELSE 0 END
    ) STORED,
    netrtg FLOAT GENERATED ALWAYS AS (
        CASE WHEN poss_total > 0 AND opp_poss_total > 0 
        THEN ((pts_total * 100.0) / poss_total) - ((opp_pts_total * 100.0) / opp_poss_total)
        ELSE 0 END
    ) STORED,
    
    -- Pace (COMPUTED!)
    pace FLOAT GENERATED ALWAYS AS (
        CASE WHEN (wins + losses) > 0 THEN poss_total / (wins + losses) ELSE 0 END
    ) STORED,
    
    -- Luck (Pythagorean expectation - COMPUTED!)
    expected_wins FLOAT,  -- Calculated separately (power formula)
    luck FLOAT GENERATED ALWAYS AS (
        CASE WHEN (wins + losses) > 0 
        THEN (wins::FLOAT / (wins + losses)) - COALESCE(expected_wins, 0) 
        ELSE 0 END
    ) STORED,
    
    updated_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (team_id, season_id)
);

-- ============================================================================
-- 6. STANDINGS (Daily snapshot - PROPER PK!)
-- ============================================================================

CREATE TABLE IF NOT EXISTS standings_daily (
    snapshot_date DATE NOT NULL,
    team_id VARCHAR(10) REFERENCES teams(team_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    
    conference VARCHAR(10),
    rank INTEGER,
    wins INTEGER,
    losses INTEGER,
    gb FLOAT,
    streak VARCHAR(5),
    
    PRIMARY KEY (snapshot_date, team_id)  -- PROPER PK!
);

-- ============================================================================
-- 7. RAPM STINT DATA (for weekly RAPM calculation)
-- ============================================================================

CREATE TABLE IF NOT EXISTS rapm_stints (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(15) NOT NULL,
    stint_num INTEGER NOT NULL,
    
    -- 5 players ON for each team (10 total)
    home_p1 VARCHAR(10) REFERENCES players(player_id),
    home_p2 VARCHAR(10) REFERENCES players(player_id),
    home_p3 VARCHAR(10) REFERENCES players(player_id),
    home_p4 VARCHAR(10) REFERENCES players(player_id),
    home_p5 VARCHAR(10) REFERENCES players(player_id),
    away_p1 VARCHAR(10) REFERENCES players(player_id),
    away_p2 VARCHAR(10) REFERENCES players(player_id),
    away_p3 VARCHAR(10) REFERENCES players(player_id),
    away_p4 VARCHAR(10) REFERENCES players(player_id),
    away_p5 VARCHAR(10) REFERENCES players(player_id),
    
    -- Stint results
    possessions FLOAT,
    home_pts INTEGER,
    away_pts INTEGER,
    netrtg_per_100 FLOAT,  -- (home_pts - away_pts) * 100 / possessions
    
    UNIQUE(game_id, stint_num)
);

-- ============================================================================
-- 8. DAILY HEALTH CHECK
-- ============================================================================

CREATE TABLE IF NOT EXISTS daily_snapshots (
    snapshot_date DATE PRIMARY KEY,
    games_processed INTEGER,
    players_updated INTEGER,
    status VARCHAR(20),
    finished_at TIMESTAMP,
    duration_seconds INTEGER
);

-- ============================================================================
-- 9. INDEXES (Fast queries!)
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_box_scores_player ON player_box_scores(player_id, game_date DESC);
CREATE INDEX IF NOT EXISTS idx_box_scores_date ON player_box_scores(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_season_stats_ppg ON player_season_stats(ppg DESC);
CREATE INDEX IF NOT EXISTS idx_season_stats_lebron ON player_season_stats(lebron_total DESC);
CREATE INDEX IF NOT EXISTS idx_team_stats_netrtg ON team_season_stats(netrtg DESC);
CREATE INDEX IF NOT EXISTS idx_standings_date ON standings_daily(snapshot_date DESC);

-- ============================================================================
-- 10. HELPER FUNCTIONS
-- ============================================================================

-- DON'T PRUNE! Keep ALL games for ML training
-- Only prune games older than 3 years (for storage management)
CREATE OR REPLACE FUNCTION prune_very_old_games() RETURNS void AS $$
BEGIN
    DELETE FROM player_box_scores
    WHERE game_date < CURRENT_DATE - INTERVAL '3 years';
    -- Keeps 3 seasons of data (246 games/season × 3 = 738 games per player)
END;
$$ LANGUAGE plpgsql;

-- Refresh last 10 games view
CREATE OR REPLACE FUNCTION refresh_last10() RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY player_last10;
END;
$$ LANGUAGE plpgsql;

-- Calculate Pythagorean expected wins
CREATE OR REPLACE FUNCTION calculate_expected_wins() RETURNS void AS $$
DECLARE
    team_record RECORD;
    exp_wins FLOAT;
BEGIN
    FOR team_record IN 
        SELECT team_id, season_id, pts_total, opp_pts_total, wins, losses
        FROM team_season_stats
        WHERE (wins + losses) > 0
    LOOP
        -- Pythagorean formula: pts^14 / (pts^14 + opp_pts^14)
        exp_wins := (
            POWER(team_record.pts_total, 14) / 
            (POWER(team_record.pts_total, 14) + POWER(team_record.opp_pts_total, 14))
        ) * (team_record.wins + team_record.losses);
        
        UPDATE team_season_stats
        SET expected_wins = exp_wins
        WHERE team_id = team_record.team_id AND season_id = team_record.season_id;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 11. INITIAL DATA
-- ============================================================================

INSERT INTO seasons (season_id, start_date, is_current) VALUES
('2024-25', '2024-10-22', TRUE)
ON CONFLICT DO NOTHING;

-- Insert all 30 teams
INSERT INTO teams (team_id, abbreviation, full_name, conference) VALUES
('1610612737', 'ATL', 'Atlanta Hawks', 'East'),
('1610612738', 'BOS', 'Boston Celtics', 'East'),
('1610612751', 'BKN', 'Brooklyn Nets', 'East'),
('1610612766', 'CHA', 'Charlotte Hornets', 'East'),
('1610612741', 'CHI', 'Chicago Bulls', 'East'),
('1610612739', 'CLE', 'Cleveland Cavaliers', 'East'),
('1610612742', 'DAL', 'Dallas Mavericks', 'West'),
('1610612743', 'DEN', 'Denver Nuggets', 'West'),
('1610612765', 'DET', 'Detroit Pistons', 'East'),
('1610612744', 'GSW', 'Golden State Warriors', 'West'),
('1610612745', 'HOU', 'Houston Rockets', 'West'),
('1610612754', 'IND', 'Indiana Pacers', 'East'),
('1610612746', 'LAC', 'LA Clippers', 'West'),
('1610612747', 'LAL', 'Los Angeles Lakers', 'West'),
('1610612763', 'MEM', 'Memphis Grizzlies', 'West'),
('1610612748', 'MIA', 'Miami Heat', 'East'),
('1610612749', 'MIL', 'Milwaukee Bucks', 'East'),
('1610612750', 'MIN', 'Minnesota Timberwolves', 'West'),
('1610612740', 'NOP', 'New Orleans Pelicans', 'West'),
('1610612752', 'NYK', 'New York Knicks', 'East'),
('1610612760', 'OKC', 'Oklahoma City Thunder', 'West'),
('1610612753', 'ORL', 'Orlando Magic', 'East'),
('1610612755', 'PHI', 'Philadelphia 76ers', 'East'),
('1610612756', 'PHX', 'Phoenix Suns', 'West'),
('1610612757', 'POR', 'Portland Trail Blazers', 'West'),
('1610612758', 'SAC', 'Sacramento Kings', 'West'),
('1610612759', 'SAS', 'San Antonio Spurs', 'West'),
('1610612761', 'TOR', 'Toronto Raptors', 'East'),
('1610612762', 'UTA', 'Utah Jazz', 'West'),
('1610612764', 'WAS', 'Washington Wizards', 'East')
ON CONFLICT (team_id) DO UPDATE SET
    full_name = EXCLUDED.full_name;

SELECT 'Schema V3 deployed - Self-computed metrics ready!' AS status;

