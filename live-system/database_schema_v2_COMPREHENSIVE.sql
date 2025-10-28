-- ============================================================================
-- NBA ANALYTICS PLATFORM - COMPREHENSIVE SCHEMA V2
-- Captures EVERY statistic from nba_api with full audit trail
-- ============================================================================

-- ============================================================================
-- 1. CORE ENTITIES (Static/Slowly Changing)
-- ============================================================================

CREATE TABLE IF NOT EXISTS teams (
    team_id VARCHAR(10) PRIMARY KEY,
    abbreviation VARCHAR(3) UNIQUE NOT NULL,
    full_name VARCHAR(50) NOT NULL,
    conference VARCHAR(10) NOT NULL,
    division VARCHAR(20) NOT NULL,
    city VARCHAR(50),
    state VARCHAR(30),
    arena VARCHAR(100),
    year_founded INTEGER,
    owner VARCHAR(100),
    gm VARCHAR(100),
    head_coach VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS seasons (
    season_id VARCHAR(10) PRIMARY KEY,  -- e.g., '2024-25'
    start_date DATE NOT NULL,
    end_date DATE,
    is_current BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS players (
    player_id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    birthdate DATE,
    country VARCHAR(50),
    height_inches INTEGER,
    weight_lbs INTEGER,
    draft_year INTEGER,
    draft_round INTEGER,
    draft_number INTEGER,
    college VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Player-Team associations (handles trades!)
CREATE TABLE IF NOT EXISTS player_team_history (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(10) REFERENCES players(player_id),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    start_date DATE NOT NULL,
    end_date DATE,  -- NULL if current
    jersey_number VARCHAR(3),
    position VARCHAR(5),
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(player_id, team_id, season_id, start_date)
);

-- ============================================================================
-- 2. GAMES (Core fact table)
-- ============================================================================

CREATE TABLE IF NOT EXISTS games (
    game_id VARCHAR(15) PRIMARY KEY,
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    game_date DATE NOT NULL,
    game_time TIME,
    
    -- Teams
    home_team_id VARCHAR(10) REFERENCES teams(team_id),
    away_team_id VARCHAR(10) REFERENCES teams(team_id),
    
    -- Venue
    arena VARCHAR(100),
    city VARCHAR(50),
    attendance INTEGER,
    
    -- Game Context
    game_type VARCHAR(20),  -- 'Regular Season', 'Playoffs', 'Play-In'
    playoff_series_id VARCHAR(20),
    playoff_game_number INTEGER,
    
    -- Final Score
    home_score INTEGER,
    away_score INTEGER,
    
    -- Status
    status VARCHAR(20),  -- 'scheduled', 'live', 'final'
    status_detail VARCHAR(50),
    
    -- Timing
    periods_played INTEGER DEFAULT 4,
    is_overtime BOOLEAN DEFAULT FALSE,
    
    -- Officials
    officials JSONB,  -- [{name, jersey_num}]
    
    -- Additional metadata
    broadcast_networks JSONB,  -- ['ESPN', 'TNT']
    game_notes TEXT,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(game_id)
);

-- ============================================================================
-- 3. GAME BOX SCORES (Team-level)
-- ============================================================================

CREATE TABLE IF NOT EXISTS game_team_stats (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(15) REFERENCES games(game_id),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    is_home BOOLEAN NOT NULL,
    
    -- Basic Box Score
    points INTEGER,
    fgm INTEGER,
    fga INTEGER,
    fg_pct FLOAT,
    fg3m INTEGER,
    fg3a INTEGER,
    fg3_pct FLOAT,
    ftm INTEGER,
    fta INTEGER,
    ft_pct FLOAT,
    
    -- Rebounds
    oreb INTEGER,
    dreb INTEGER,
    reb INTEGER,
    
    -- Other
    ast INTEGER,
    stl INTEGER,
    blk INTEGER,
    tov INTEGER,
    pf INTEGER,
    
    -- Advanced
    possessions FLOAT,
    pace FLOAT,
    offensive_rating FLOAT,
    defensive_rating FLOAT,
    net_rating FLOAT,
    efg_pct FLOAT,
    ts_pct FLOAT,
    
    -- Four Factors
    efg FLOAT,
    tov_pct FLOAT,
    oreb_pct FLOAT,
    ft_rate FLOAT,
    
    -- Quarter Scores
    q1_score INTEGER,
    q2_score INTEGER,
    q3_score INTEGER,
    q4_score INTEGER,
    ot_scores INTEGER[],
    
    -- All other stats in JSONB
    extended_stats JSONB,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(game_id, team_id)
);

-- ============================================================================
-- 4. PLAYER GAME STATS (Box Scores)
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_game_stats (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(15) REFERENCES games(game_id),
    player_id VARCHAR(10) REFERENCES players(player_id),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    game_date DATE NOT NULL,
    
    -- Playing Time
    started BOOLEAN DEFAULT FALSE,
    minutes FLOAT,
    seconds INTEGER,
    
    -- Basic Stats
    points INTEGER,
    fgm INTEGER,
    fga INTEGER,
    fg_pct FLOAT,
    fg3m INTEGER,
    fg3a INTEGER,
    fg3_pct FLOAT,
    ftm INTEGER,
    fta INTEGER,
    ft_pct FLOAT,
    
    -- Rebounds
    oreb INTEGER,
    dreb INTEGER,
    reb INTEGER,
    
    -- Playmaking/Defense
    ast INTEGER,
    stl INTEGER,
    blk INTEGER,
    tov INTEGER,
    pf INTEGER,
    
    -- Impact
    plus_minus INTEGER,
    
    -- Advanced (calculated)
    ts_pct FLOAT,
    efg_pct FLOAT,
    usage_pct FLOAT,
    assist_pct FLOAT,
    
    -- Extended stats (tracking, hustle, etc.)
    extended_stats JSONB,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(player_id, game_id)
) PARTITION BY RANGE (game_date);

-- Partitions for player_game_stats (one per season)
CREATE TABLE IF NOT EXISTS player_game_stats_2024_25 PARTITION OF player_game_stats
    FOR VALUES FROM ('2024-10-01') TO ('2025-06-30');

CREATE TABLE IF NOT EXISTS player_game_stats_2025_26 PARTITION OF player_game_stats
    FOR VALUES FROM ('2025-10-01') TO ('2026-06-30');

-- ============================================================================
-- 5. SEASON AGGREGATES (One row per player/team per season)
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_season_stats (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(10) REFERENCES players(player_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    
    -- Games
    games_played INTEGER DEFAULT 0,
    games_started INTEGER DEFAULT 0,
    minutes_total FLOAT DEFAULT 0,
    
    -- Per-Game Stats
    ppg FLOAT DEFAULT 0,
    rpg FLOAT DEFAULT 0,
    apg FLOAT DEFAULT 0,
    spg FLOAT DEFAULT 0,
    bpg FLOAT DEFAULT 0,
    tov_pg FLOAT DEFAULT 0,
    
    -- Shooting
    fgm_pg FLOAT DEFAULT 0,
    fga_pg FLOAT DEFAULT 0,
    fg_pct FLOAT DEFAULT 0,
    fg3m_pg FLOAT DEFAULT 0,
    fg3a_pg FLOAT DEFAULT 0,
    fg3_pct FLOAT DEFAULT 0,
    ftm_pg FLOAT DEFAULT 0,
    fta_pg FLOAT DEFAULT 0,
    ft_pct FLOAT DEFAULT 0,
    
    -- Totals (for verification)
    points_total INTEGER DEFAULT 0,
    rebounds_total INTEGER DEFAULT 0,
    assists_total INTEGER DEFAULT 0,
    
    -- Advanced (recalculated nightly)
    ts_pct FLOAT,
    efg_pct FLOAT,
    usage_pct FLOAT,
    per FLOAT,
    
    -- All other stats in JSONB
    advanced_stats JSONB,
    
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(player_id, season_id)
);

CREATE TABLE IF NOT EXISTS team_season_stats (
    id SERIAL PRIMARY KEY,
    team_id VARCHAR(10) REFERENCES teams(team_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    
    -- Record
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    win_pct FLOAT DEFAULT 0,
    
    -- Per Game
    ppg FLOAT DEFAULT 0,
    opp_ppg FLOAT DEFAULT 0,
    fg_pct FLOAT DEFAULT 0,
    fg3_pct FLOAT DEFAULT 0,
    ft_pct FLOAT DEFAULT 0,
    
    -- Advanced
    pace FLOAT,
    offensive_rating FLOAT,
    defensive_rating FLOAT,
    net_rating FLOAT,
    
    -- Four Factors
    efg FLOAT,
    tov_pct FLOAT,
    oreb_pct FLOAT,
    ft_rate FLOAT,
    
    -- KenPom-style
    adj_offensive_efficiency FLOAT,
    adj_defensive_efficiency FLOAT,
    sos FLOAT,  -- Strength of Schedule
    
    -- Extended stats
    advanced_stats JSONB,
    
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(team_id, season_id)
);

-- ============================================================================
-- 6. ADVANCED METRICS (Calculated nightly)
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_advanced_metrics (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(10) REFERENCES players(player_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    calculated_date DATE NOT NULL,
    
    -- Impact Metrics
    bpm FLOAT,  -- Box Plus/Minus
    obpm FLOAT,  -- Offensive BPM
    dbpm FLOAT,  -- Defensive BPM
    vorp FLOAT,  -- Value Over Replacement Player
    win_shares FLOAT,
    win_shares_per_48 FLOAT,
    
    -- Shooting Efficiency
    ts_pct FLOAT,
    efg_pct FLOAT,
    true_3p_pct FLOAT,
    
    -- Usage & Creation
    usage_pct FLOAT,
    assist_pct FLOAT,
    ast_to_tov FLOAT,
    
    -- Rebounding
    oreb_pct FLOAT,
    dreb_pct FLOAT,
    reb_pct FLOAT,
    
    -- Defense
    stl_pct FLOAT,
    blk_pct FLOAT,
    def_rating FLOAT,
    
    -- KenPom-inspired
    offensive_load FLOAT,
    defensive_load FLOAT,
    
    -- RAPTOR-style (placeholder for future ML)
    raptor_offense FLOAT,
    raptor_defense FLOAT,
    raptor_total FLOAT,
    
    -- RAPM (requires stint data, placeholder)
    rapm_offense FLOAT,
    rapm_defense FLOAT,
    rapm_total FLOAT,
    
    -- Extended metrics in JSONB
    extended_metrics JSONB,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(player_id, season_id, calculated_date)
);

CREATE TABLE IF NOT EXISTS team_advanced_metrics (
    id SERIAL PRIMARY KEY,
    team_id VARCHAR(10) REFERENCES teams(team_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    calculated_date DATE NOT NULL,
    
    -- Four Factors
    offensive_four_factors JSONB,
    defensive_four_factors JSONB,
    
    -- KenPom-style
    adj_tempo FLOAT,
    adj_offensive_efficiency FLOAT,
    adj_defensive_efficiency FLOAT,
    pythag_wins FLOAT,
    luck_factor FLOAT,
    
    -- Strength metrics
    sos FLOAT,
    sos_offensive FLOAT,
    sos_defensive FLOAT,
    
    -- Extended
    extended_metrics JSONB,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(team_id, season_id, calculated_date)
);

-- ============================================================================
-- 7. STANDINGS (Daily snapshots with proper PK)
-- ============================================================================

CREATE TABLE IF NOT EXISTS standings (
    team_id VARCHAR(10) REFERENCES teams(team_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    snapshot_date DATE NOT NULL,
    
    -- Standing
    conference VARCHAR(10),
    division VARCHAR(20),
    conference_rank INTEGER,
    division_rank INTEGER,
    
    -- Record
    wins INTEGER,
    losses INTEGER,
    win_pct FLOAT,
    gb FLOAT,
    
    -- Splits
    home_wins INTEGER,
    home_losses INTEGER,
    away_wins INTEGER,
    away_losses INTEGER,
    
    -- Recent
    last_10 VARCHAR(10),
    streak VARCHAR(10),
    
    -- Playoff Picture
    playoff_seed INTEGER,
    clinched_playoffs BOOLEAN DEFAULT FALSE,
    eliminated BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (team_id, season_id, snapshot_date)  -- PROPER PK!
);

-- ============================================================================
-- 8. LINEUPS (5-man units, for RAPM)
-- ============================================================================

CREATE TABLE IF NOT EXISTS lineups (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(15) REFERENCES games(game_id),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    
    -- 5 players (sorted alphabetically)
    player_1_id VARCHAR(10) REFERENCES players(player_id),
    player_2_id VARCHAR(10) REFERENCES players(player_id),
    player_3_id VARCHAR(10) REFERENCES players(player_id),
    player_4_id VARCHAR(10) REFERENCES players(player_id),
    player_5_id VARCHAR(10) REFERENCES players(player_id),
    
    -- Performance
    minutes_played FLOAT,
    plus_minus INTEGER,
    possessions FLOAT,
    points_for INTEGER,
    points_against INTEGER,
    
    -- Efficiency
    offensive_rating FLOAT,
    defensive_rating FLOAT,
    net_rating FLOAT,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(game_id, team_id, player_1_id, player_2_id, player_3_id, player_4_id, player_5_id)
);

-- ============================================================================
-- 9. PLAY-BY-PLAY (For pattern analysis)
-- ============================================================================

CREATE TABLE IF NOT EXISTS play_by_play (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(15) REFERENCES games(game_id),
    event_num INTEGER,
    period INTEGER,
    clock VARCHAR(20),
    time_elapsed_seconds INTEGER,
    
    -- Event details
    event_type VARCHAR(50),
    event_subtype VARCHAR(50),
    description TEXT,
    
    -- Players involved
    player_1_id VARCHAR(10) REFERENCES players(player_id),
    player_2_id VARCHAR(10) REFERENCES players(player_id),
    player_3_id VARCHAR(10) REFERENCES players(player_id),
    
    -- Teams
    team_id VARCHAR(10) REFERENCES teams(team_id),
    
    -- Score after event
    home_score INTEGER,
    away_score INTEGER,
    score_margin INTEGER,
    
    -- Full event data
    event_data JSONB,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(game_id, event_num)
) PARTITION BY RANGE (time_elapsed_seconds);

-- PBP Partitions (for performance)
CREATE TABLE IF NOT EXISTS play_by_play_q1 PARTITION OF play_by_play
    FOR VALUES FROM (0) TO (720);  -- 0-12 minutes

CREATE TABLE IF NOT EXISTS play_by_play_q2 PARTITION OF play_by_play
    FOR VALUES FROM (720) TO (1440);  -- 12-24 minutes

CREATE TABLE IF NOT EXISTS play_by_play_q3 PARTITION OF play_by_play
    FOR VALUES FROM (1440) TO (2160);  -- 24-36 minutes

CREATE TABLE IF NOT EXISTS play_by_play_q4_ot PARTITION OF play_by_play
    FOR VALUES FROM (2160) TO (4000);  -- 36+ minutes

-- ============================================================================
-- 10. TRACKING STATS (Advanced nba_api data)
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_tracking_stats (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(10) REFERENCES players(player_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    stat_type VARCHAR(50),  -- 'speed', 'distance', 'touches', etc.
    
    -- Speed & Distance
    avg_speed FLOAT,
    distance_miles FLOAT,
    avg_distance_miles FLOAT,
    
    -- Touches
    touches INTEGER,
    front_court_touches INTEGER,
    avg_seconds_per_touch FLOAT,
    avg_dribbles_per_touch FLOAT,
    
    -- Catch & Shoot
    catch_shoot_pts INTEGER,
    catch_shoot_fgm INTEGER,
    catch_shoot_fga INTEGER,
    catch_shoot_pct FLOAT,
    
    -- Pull-up
    pull_up_pts INTEGER,
    pull_up_fgm INTEGER,
    pull_up_fga INTEGER,
    pull_up_pct FLOAT,
    
    -- Defense
    def_deflections INTEGER,
    def_loose_balls_recovered INTEGER,
    def_contested_shots INTEGER,
    
    -- Full tracking data
    tracking_data JSONB,
    
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(player_id, season_id, stat_type)
);

-- ============================================================================
-- 11. HUSTLE STATS
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_hustle_stats (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(10) REFERENCES players(player_id),
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    
    -- Hustle plays
    screen_assists INTEGER,
    screen_assists_points INTEGER,
    deflections INTEGER,
    loose_balls_recovered INTEGER,
    charges_drawn INTEGER,
    contested_shots INTEGER,
    contested_shots_2pt INTEGER,
    contested_shots_3pt INTEGER,
    
    -- Box outs
    box_outs INTEGER,
    box_outs_offensive INTEGER,
    box_outs_defensive INTEGER,
    
    -- Full hustle data
    hustle_data JSONB,
    
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(player_id, season_id)
);

-- ============================================================================
-- 12. SIMILARITY SCORES (Pre-calculated)
-- ============================================================================

CREATE TABLE IF NOT EXISTS similarity_scores (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(10) NOT NULL,  -- 'player' or 'team'
    entity_id_1 VARCHAR(10) NOT NULL,
    entity_id_2 VARCHAR(10) NOT NULL,
    
    -- Similarity metrics
    overall_similarity FLOAT,
    statistical_similarity FLOAT,
    style_similarity FLOAT,
    
    -- Time window
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    time_window VARCHAR(20),  -- 'last_10', 'season', 'career'
    
    -- Breakdown
    similarity_breakdown JSONB,
    
    calculated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(entity_type, entity_id_1, entity_id_2, season_id, time_window)
);

-- ============================================================================
-- 13. AUDIT / SNAPSHOTS (Historical tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS daily_snapshots (
    snapshot_date DATE PRIMARY KEY,
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    
    -- Counts
    teams_count INTEGER,
    players_count INTEGER,
    games_today INTEGER,
    games_total_season INTEGER,
    
    -- Collection status
    collection_status VARCHAR(20) DEFAULT 'success',
    error_message TEXT,
    
    -- Stats collected
    stats_types_collected VARCHAR[],
    
    -- Timing
    collection_start_time TIMESTAMP,
    collection_end_time TIMESTAMP,
    duration_seconds INTEGER,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- 14. COMPREHENSIVE INDEXES
-- ============================================================================

-- Teams
CREATE INDEX IF NOT EXISTS idx_teams_conference ON teams(conference);
CREATE INDEX IF NOT EXISTS idx_teams_division ON teams(division);

-- Players
CREATE INDEX IF NOT EXISTS idx_players_active ON players(is_active);
CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);
CREATE INDEX IF NOT EXISTS idx_players_country ON players(country);

-- Player-Team History
CREATE INDEX IF NOT EXISTS idx_player_team_current ON player_team_history(player_id, is_current);
CREATE INDEX IF NOT EXISTS idx_player_team_season ON player_team_history(season_id);

-- Games
CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season_id);
CREATE INDEX IF NOT EXISTS idx_games_home_team ON games(home_team_id);
CREATE INDEX IF NOT EXISTS idx_games_away_team ON games(away_team_id);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);
CREATE INDEX IF NOT EXISTS idx_games_date_teams ON games(game_date, home_team_id, away_team_id);

-- Game Team Stats
CREATE INDEX IF NOT EXISTS idx_game_team_stats_game ON game_team_stats(game_id);
CREATE INDEX IF NOT EXISTS idx_game_team_stats_team ON game_team_stats(team_id);

-- Player Game Stats
CREATE INDEX IF NOT EXISTS idx_player_game_stats_player ON player_game_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_player_game_stats_date ON player_game_stats(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_player_game_stats_player_date ON player_game_stats(player_id, game_date DESC);

-- Player Season Stats
CREATE INDEX IF NOT EXISTS idx_player_season_stats_player ON player_season_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_player_season_stats_season ON player_season_stats(season_id);
CREATE INDEX IF NOT EXISTS idx_player_season_stats_team ON player_season_stats(team_id);
CREATE INDEX IF NOT EXISTS idx_player_season_stats_ppg ON player_season_stats(ppg DESC);

-- Team Season Stats
CREATE INDEX IF NOT EXISTS idx_team_season_stats_team ON team_season_stats(team_id);
CREATE INDEX IF NOT EXISTS idx_team_season_stats_season ON team_season_stats(season_id);

-- Standings
CREATE INDEX IF NOT EXISTS idx_standings_date ON standings(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_standings_season ON standings(season_id);
CREATE INDEX IF NOT EXISTS idx_standings_conference_rank ON standings(conference, conference_rank);

-- Lineups
CREATE INDEX IF NOT EXISTS idx_lineups_game ON lineups(game_id);
CREATE INDEX IF NOT EXISTS idx_lineups_team ON lineups(team_id);
CREATE INDEX IF NOT EXISTS idx_lineups_season ON lineups(season_id);

-- Play-by-Play
CREATE INDEX IF NOT EXISTS idx_pbp_game ON play_by_play(game_id);
CREATE INDEX IF NOT EXISTS idx_pbp_player1 ON play_by_play(player_1_id);
CREATE INDEX IF NOT EXISTS idx_pbp_event_type ON play_by_play(event_type);

-- Tracking Stats
CREATE INDEX IF NOT EXISTS idx_tracking_player ON player_tracking_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_tracking_season ON player_tracking_stats(season_id);
CREATE INDEX IF NOT EXISTS idx_tracking_type ON player_tracking_stats(stat_type);

-- Hustle Stats
CREATE INDEX IF NOT EXISTS idx_hustle_player ON player_hustle_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_hustle_season ON player_hustle_stats(season_id);

-- Similarity
CREATE INDEX IF NOT EXISTS idx_similarity_entity1 ON similarity_scores(entity_id_1);
CREATE INDEX IF NOT EXISTS idx_similarity_entity2 ON similarity_scores(entity_id_2);
CREATE INDEX IF NOT EXISTS idx_similarity_type ON similarity_scores(entity_type);

-- JSONB Indexes (for fast queries on extended stats)
CREATE INDEX IF NOT EXISTS idx_game_team_extended_stats ON game_team_stats USING GIN (extended_stats);
CREATE INDEX IF NOT EXISTS idx_player_game_extended_stats ON player_game_stats USING GIN (extended_stats);
CREATE INDEX IF NOT EXISTS idx_player_advanced_stats ON player_season_stats USING GIN (advanced_stats);
CREATE INDEX IF NOT EXISTS idx_team_advanced_stats ON team_season_stats USING GIN (advanced_stats);

-- ============================================================================
-- 15. INITIAL DATA
-- ============================================================================

-- Insert current season
INSERT INTO seasons (season_id, start_date, is_current) VALUES
('2024-25', '2024-10-22', TRUE)
ON CONFLICT (season_id) DO NOTHING;

-- Insert all 30 NBA teams
INSERT INTO teams (team_id, abbreviation, full_name, conference, division, city, state) VALUES
('1610612737', 'ATL', 'Atlanta Hawks', 'East', 'Southeast', 'Atlanta', 'GA'),
('1610612738', 'BOS', 'Boston Celtics', 'East', 'Atlantic', 'Boston', 'MA'),
('1610612751', 'BKN', 'Brooklyn Nets', 'East', 'Atlantic', 'Brooklyn', 'NY'),
('1610612766', 'CHA', 'Charlotte Hornets', 'East', 'Southeast', 'Charlotte', 'NC'),
('1610612741', 'CHI', 'Chicago Bulls', 'East', 'Central', 'Chicago', 'IL'),
('1610612739', 'CLE', 'Cleveland Cavaliers', 'East', 'Central', 'Cleveland', 'OH'),
('1610612742', 'DAL', 'Dallas Mavericks', 'West', 'Southwest', 'Dallas', 'TX'),
('1610612743', 'DEN', 'Denver Nuggets', 'West', 'Northwest', 'Denver', 'CO'),
('1610612765', 'DET', 'Detroit Pistons', 'East', 'Central', 'Detroit', 'MI'),
('1610612744', 'GSW', 'Golden State Warriors', 'West', 'Pacific', 'San Francisco', 'CA'),
('1610612745', 'HOU', 'Houston Rockets', 'West', 'Southwest', 'Houston', 'TX'),
('1610612754', 'IND', 'Indiana Pacers', 'East', 'Central', 'Indianapolis', 'IN'),
('1610612746', 'LAC', 'LA Clippers', 'West', 'Pacific', 'Los Angeles', 'CA'),
('1610612747', 'LAL', 'Los Angeles Lakers', 'West', 'Pacific', 'Los Angeles', 'CA'),
('1610612763', 'MEM', 'Memphis Grizzlies', 'West', 'Southwest', 'Memphis', 'TN'),
('1610612748', 'MIA', 'Miami Heat', 'East', 'Southeast', 'Miami', 'FL'),
('1610612749', 'MIL', 'Milwaukee Bucks', 'East', 'Central', 'Milwaukee', 'WI'),
('1610612750', 'MIN', 'Minnesota Timberwolves', 'West', 'Northwest', 'Minneapolis', 'MN'),
('1610612740', 'NOP', 'New Orleans Pelicans', 'West', 'Southwest', 'New Orleans', 'LA'),
('1610612752', 'NYK', 'New York Knicks', 'East', 'Atlantic', 'New York', 'NY'),
('1610612760', 'OKC', 'Oklahoma City Thunder', 'West', 'Northwest', 'Oklahoma City', 'OK'),
('1610612753', 'ORL', 'Orlando Magic', 'East', 'Southeast', 'Orlando', 'FL'),
('1610612755', 'PHI', 'Philadelphia 76ers', 'East', 'Atlantic', 'Philadelphia', 'PA'),
('1610612756', 'PHX', 'Phoenix Suns', 'West', 'Pacific', 'Phoenix', 'AZ'),
('1610612757', 'POR', 'Portland Trail Blazers', 'West', 'Northwest', 'Portland', 'OR'),
('1610612758', 'SAC', 'Sacramento Kings', 'West', 'Pacific', 'Sacramento', 'CA'),
('1610612759', 'SAS', 'San Antonio Spurs', 'West', 'Southwest', 'San Antonio', 'TX'),
('1610612761', 'TOR', 'Toronto Raptors', 'East', 'Atlantic', 'Toronto', 'ON'),
('1610612762', 'UTA', 'Utah Jazz', 'West', 'Northwest', 'Salt Lake City', 'UT'),
('1610612764', 'WAS', 'Washington Wizards', 'East', 'Southeast', 'Washington', 'DC')
ON CONFLICT (team_id) DO UPDATE SET
    full_name = EXCLUDED.full_name,
    city = EXCLUDED.city,
    state = EXCLUDED.state;

-- ============================================================================
-- 16. VIEWS (For common queries)
-- ============================================================================

-- Current season player stats with team info
CREATE OR REPLACE VIEW v_current_player_stats AS
SELECT 
    p.player_id,
    p.name,
    t.abbreviation AS team,
    pss.games_played,
    pss.ppg,
    pss.rpg,
    pss.apg,
    pss.fg_pct,
    pss.fg3_pct,
    pss.ft_pct,
    pam.bpm,
    pam.vorp,
    pss.updated_at
FROM player_season_stats pss
JOIN players p ON pss.player_id = p.player_id
LEFT JOIN teams t ON pss.team_id = t.team_id
LEFT JOIN player_advanced_metrics pam ON pss.player_id = pam.player_id 
    AND pss.season_id = pam.season_id
    AND pam.calculated_date = (SELECT MAX(calculated_date) FROM player_advanced_metrics WHERE player_id = pss.player_id)
WHERE pss.season_id = (SELECT season_id FROM seasons WHERE is_current = TRUE);

-- Current standings
CREATE OR REPLACE VIEW v_current_standings AS
SELECT 
    t.abbreviation,
    t.full_name,
    s.conference,
    s.conference_rank,
    s.wins,
    s.losses,
    s.win_pct,
    s.gb,
    s.streak,
    s.last_10
FROM standings s
JOIN teams t ON s.team_id = t.team_id
WHERE s.snapshot_date = (SELECT MAX(snapshot_date) FROM standings)
ORDER BY s.conference, s.conference_rank;

-- Player game log (last 10 games)
CREATE OR REPLACE VIEW v_player_recent_games AS
SELECT 
    pgs.player_id,
    p.name,
    g.game_date,
    CASE WHEN g.home_team_id = pgs.team_id 
         THEN 'vs ' || away.abbreviation 
         ELSE '@ ' || home.abbreviation END AS opponent,
    pgs.points,
    pgs.rebounds,
    pgs.assists,
    pgs.minutes,
    pgs.plus_minus,
    g.game_id
FROM player_game_stats pgs
JOIN players p ON pgs.player_id = p.player_id
JOIN games g ON pgs.game_id = g.game_id
JOIN teams home ON g.home_team_id = home.team_id
JOIN teams away ON g.away_team_id = away.team_id
ORDER BY pgs.player_id, g.game_date DESC;

-- ============================================================================
-- 17. MATERIALIZED VIEWS (For performance)
-- ============================================================================

-- Top performers (refreshed nightly)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_top_scorers AS
SELECT 
    p.player_id,
    p.name,
    t.abbreviation AS team,
    pss.ppg,
    pss.games_played,
    pss.updated_at
FROM player_season_stats pss
JOIN players p ON pss.player_id = p.player_id
JOIN teams t ON pss.team_id = t.team_id
WHERE pss.season_id = (SELECT season_id FROM seasons WHERE is_current = TRUE)
ORDER BY pss.ppg DESC
LIMIT 100;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_top_scorers_player ON mv_top_scorers(player_id);

-- ============================================================================
-- 18. CLEANUP FUNCTIONS
-- ============================================================================

-- Prune old play-by-play data (keep last 30 days only)
CREATE OR REPLACE FUNCTION cleanup_old_pbp() RETURNS void AS $$
BEGIN
    DELETE FROM play_by_play
    WHERE game_id IN (
        SELECT game_id FROM games
        WHERE game_date < CURRENT_DATE - INTERVAL '30 days'
    );
END;
$$ LANGUAGE plpgsql;

-- Refresh materialized views
CREATE OR REPLACE FUNCTION refresh_all_mv() RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_top_scorers;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE teams IS 'All 30 NBA teams with metadata';
COMMENT ON TABLE players IS 'All NBA players (active and historical)';
COMMENT ON TABLE games IS 'Every NBA game with final scores and metadata';
COMMENT ON TABLE player_game_stats IS 'Individual player box scores (partitioned by date)';
COMMENT ON TABLE player_season_stats IS 'Aggregated season stats per player (rolling updates)';
COMMENT ON TABLE player_advanced_metrics IS 'Advanced metrics (BPM, VORP, RAPTOR, RAPM)';
COMMENT ON TABLE player_tracking_stats IS 'SportVU tracking data (speed, distance, touches)';
COMMENT ON TABLE player_hustle_stats IS 'Hustle metrics (deflections, charges, screens)';
COMMENT ON TABLE lineups IS '5-man lineups for RAPM calculation';
COMMENT ON TABLE play_by_play IS 'Play-by-play events (partitioned by time)';
COMMENT ON TABLE similarity_scores IS 'Pre-calculated similarity between players/teams';
COMMENT ON TABLE standings IS 'Daily standings snapshots (proper PK prevents duplicates)';
COMMENT ON TABLE daily_snapshots IS 'Collection status audit trail';

-- ============================================================================
-- SCHEMA VERSION
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    description TEXT,
    applied_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO schema_version (version, description) VALUES
(2, 'Comprehensive schema v2 - All nba_api stats with partitioning, JSONB, proper indexes')
ON CONFLICT (version) DO NOTHING;

-- ============================================================================
-- DONE!
-- ============================================================================

SELECT 'Schema deployed successfully!' AS status;

