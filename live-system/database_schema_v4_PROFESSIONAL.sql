-- ============================================================================
-- ONTOLOGIC XYZ - PROFESSIONAL NBA ANALYTICS DATABASE
-- Data Science Company Standard Schema
-- Version: 4.0
-- Last Updated: 2025-10-28
-- ============================================================================

-- ============================================================================
-- SECTION 1: CORE ENTITIES (Teams, Players, Games)
-- ============================================================================

-- 1.1 Teams Master Table
CREATE TABLE IF NOT EXISTS teams (
    team_id VARCHAR(10) PRIMARY KEY,
    abbreviation VARCHAR(3) UNIQUE NOT NULL,
    full_name VARCHAR(50) NOT NULL,
    conference VARCHAR(10) NOT NULL CHECK (conference IN ('East', 'West')),
    division VARCHAR(20) NOT NULL,
    
    -- Visual Identity
    logo_url VARCHAR(200),
    primary_color VARCHAR(7),
    secondary_color VARCHAR(7),
    
    -- Location & Venue
    city VARCHAR(50),
    state VARCHAR(30),
    arena VARCHAR(100),
    arena_capacity INT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for teams table
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
    weight_lbs INT,
    
    -- Career Info
    birth_date DATE,
    draft_year INT,
    draft_pick INT,
    college VARCHAR(100),
    country VARCHAR(50),
    years_experience INT,
    
    -- Visual Assets
    headshot_url VARCHAR(200),
    action_photo_url VARCHAR(200),
    
    -- Contract & Status
    is_active BOOLEAN DEFAULT TRUE,
    contract_year INT,
    salary BIGINT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for players table
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
    playoff_start_date DATE,
    finals_start_date DATE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for seasons table
CREATE INDEX IF NOT EXISTS idx_seasons_current ON seasons(is_current);

-- 1.4 Games Master Table
CREATE TABLE IF NOT EXISTS games (
    game_id VARCHAR(15) PRIMARY KEY,
    season_id VARCHAR(10) REFERENCES seasons(season_id),
    game_date DATE NOT NULL,
    game_time TIMESTAMP,
    
    -- Teams
    home_team_id VARCHAR(10) REFERENCES teams(team_id),
    away_team_id VARCHAR(10) REFERENCES teams(team_id),
    
    -- Final Score
    home_score INT,
    away_score INT,
    
    -- Game Status
    status VARCHAR(20) CHECK (status IN ('Scheduled', 'Live', 'Final', 'Postponed', 'Cancelled')),
    is_playoff BOOLEAN DEFAULT FALSE,
    
    -- Venue
    arena VARCHAR(100),
    attendance INT,
    
    -- Broadcast
    tv_network VARCHAR(50),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for games table
CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season_id, game_date);
CREATE INDEX IF NOT EXISTS idx_games_team_home ON games(home_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_games_team_away ON games(away_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);


-- ============================================================================
-- SECTION 2: PLAYER PERFORMANCE DATA
-- ============================================================================

-- 2.1 Player Box Scores (Partitioned by Season for Performance)
CREATE TABLE IF NOT EXISTS player_box_scores (
    player_id VARCHAR(10) NOT NULL REFERENCES players(player_id),
    game_id VARCHAR(15) NOT NULL REFERENCES games(game_id),
    game_date DATE NOT NULL,
    season_id VARCHAR(10) DEFAULT '2025-26',
    team_id VARCHAR(10) REFERENCES teams(team_id),
    
    -- Playing Time
    minutes DECIMAL(5,2),
    seconds_played INT,
    
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
    
    -- Advanced Metrics (auto-computed)
    plus_minus INT,
    team_poss DECIMAL(10,2),
    
    -- Per-100 Stats (REAL - calculated from team_poss)
    pts_100 DECIMAL(7,2),
    reb_100 DECIMAL(7,2),
    ast_100 DECIMAL(7,2),
    stl_100 DECIMAL(7,2),
    blk_100 DECIMAL(7,2),
    tov_100 DECIMAL(7,2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (player_id, game_id, game_date)
) PARTITION BY RANGE (game_date);

-- Create indexes for player_box_scores table
CREATE INDEX IF NOT EXISTS idx_box_scores_player_date ON player_box_scores(player_id, game_date DESC);
CREATE INDEX IF NOT EXISTS idx_box_scores_game ON player_box_scores(game_id);
CREATE INDEX IF NOT EXISTS idx_box_scores_team ON player_box_scores(team_id, game_date DESC);

-- Create partitions for each season
CREATE TABLE IF NOT EXISTS player_box_scores_2024_25 
    PARTITION OF player_box_scores 
    FOR VALUES FROM ('2024-10-01') TO ('2025-06-30');

CREATE TABLE IF NOT EXISTS player_box_scores_2025_26 
    PARTITION OF player_box_scores 
    FOR VALUES FROM ('2025-10-01') TO ('2026-06-30');

CREATE TABLE IF NOT EXISTS player_box_scores_2026_27 
    PARTITION OF player_box_scores 
    FOR VALUES FROM ('2026-10-01') TO ('2027-06-30');


-- ============================================================================
-- SECTION 3: ADVANCED ANALYTICS (Self-Computed)
-- ============================================================================

-- 3.1 Player Season Aggregates
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
    
    -- Advanced Metrics (Self-Computed)
    ts_pct DECIMAL(5,3) GENERATED ALWAYS AS (
        CASE 
            WHEN (fga_total + 0.44 * fta_total) > 0 
            THEN pts_total / (2.0 * (fga_total + 0.44 * fta_total))
            ELSE NULL
        END
    ) STORED,
    
    efg_pct DECIMAL(5,3) GENERATED ALWAYS AS (
        CASE 
            WHEN fga_total > 0 
            THEN (fgm_total + 0.5 * fg3m_total) / CAST(fga_total AS DECIMAL)
            ELSE NULL
        END
    ) STORED,
    
    -- Per-Game Averages
    ppg DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE WHEN games_played > 0 THEN pts_total / CAST(games_played AS DECIMAL) ELSE 0 END
    ) STORED,
    rpg DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE WHEN games_played > 0 THEN reb_total / CAST(games_played AS DECIMAL) ELSE 0 END
    ) STORED,
    apg DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE WHEN games_played > 0 THEN ast_total / CAST(games_played AS DECIMAL) ELSE 0 END
    ) STORED,
    mpg DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE WHEN games_played > 0 THEN minutes_total / CAST(games_played AS DECIMAL) ELSE 0 END
    ) STORED,
    
    -- Per-100 Possessions (REAL - from boxscoreadvancedv2)
    total_team_possessions INT,  -- Sum of team possessions across all games
    pts_100 DECIMAL(7,2),
    reb_100 DECIMAL(7,2),
    ast_100 DECIMAL(7,2),
    stl_100 DECIMAL(7,2),
    blk_100 DECIMAL(7,2),
    tov_100 DECIMAL(7,2),
    
    -- Impact Metrics
    rapm DECIMAL(7,3),  -- Regularized Adjusted Plus-Minus
    lebron DECIMAL(7,3),  -- Our composite metric
    bpm DECIMAL(7,3),  -- Box Plus-Minus
    obpm DECIMAL(7,3),  -- Offensive BPM
    dbpm DECIMAL(7,3),  -- Defensive BPM
    vorp DECIMAL(7,3),  -- Value Over Replacement Player
    
    -- Advanced Metrics (from Basketball Reference)
    per DECIMAL(7,2),  -- Player Efficiency Rating
    usage_pct DECIMAL(5,3),  -- Usage Percentage
    win_shares DECIMAL(7,2),  -- Win Shares
    win_shares_48 DECIMAL(7,3),  -- Win Shares per 48 minutes
    
    -- Metadata
    updated_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (player_id, season_id)
);

-- Create indexes for player_season_stats table
CREATE INDEX IF NOT EXISTS idx_player_season_stats_season ON player_season_stats(season_id);
CREATE INDEX IF NOT EXISTS idx_player_season_stats_ppg ON player_season_stats(ppg DESC);
CREATE INDEX IF NOT EXISTS idx_player_season_stats_rapm ON player_season_stats(rapm DESC);

-- 3.2 Team Season Aggregates
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
    
    -- Four Factors
    efg_pct DECIMAL(5,3),
    tov_pct DECIMAL(5,3),
    oreb_pct DECIMAL(5,3),
    ft_rate DECIMAL(5,3),
    
    -- Pace & Efficiency
    pace DECIMAL(6,2),
    offensive_rating DECIMAL(6,2),
    defensive_rating DECIMAL(6,2),
    net_rating DECIMAL(6,2) GENERATED ALWAYS AS (offensive_rating - defensive_rating) STORED,
    
    -- Luck Index
    pythagorean_wins DECIMAL(6,2),
    luck DECIMAL(6,3) GENERATED ALWAYS AS (
        CASE WHEN games_played > 0 
        THEN (wins / CAST(games_played AS DECIMAL)) - (pythagorean_wins / CAST(games_played AS DECIMAL))
        ELSE 0 END
    ) STORED,
    
    -- Metadata
    updated_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (team_id, season_id)
);

-- Create indexes for team_season_stats table
CREATE INDEX IF NOT EXISTS idx_team_season_stats_season ON team_season_stats(season_id);
CREATE INDEX IF NOT EXISTS idx_team_season_stats_net_rating ON team_season_stats(net_rating DESC);


-- ============================================================================
-- SECTION 4: REAL-TIME GAME DATA
-- ============================================================================

-- 4.1 Player Injuries & Availability
CREATE TABLE IF NOT EXISTS player_injuries (
    player_id VARCHAR(10) REFERENCES players(player_id),
    injury_date DATE NOT NULL,
    status VARCHAR(20) CHECK (status IN ('Out', 'Questionable', 'Doubtful', 'Day-to-Day', 'Probable')),
    description TEXT,
    return_date DATE,
    games_missed INT DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(player_id, injury_date)
);

-- Create indexes for player_injuries table
CREATE INDEX IF NOT EXISTS idx_injuries_player ON player_injuries(player_id, injury_date DESC);
CREATE INDEX IF NOT EXISTS idx_injuries_status ON player_injuries(status);

-- 4.2 Team Depth Charts
CREATE TABLE IF NOT EXISTS team_depth_charts (
    team_id VARCHAR(10) REFERENCES teams(team_id),
    player_id VARCHAR(10) REFERENCES players(player_id),
    position VARCHAR(5),
    depth_rank INT CHECK (depth_rank >= 1 AND depth_rank <= 5),
    minutes_projection DECIMAL(5,2),
    
    updated_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (team_id, player_id, position)
);

-- Create indexes for team_depth_charts table
CREATE INDEX IF NOT EXISTS idx_depth_team_pos ON team_depth_charts(team_id, position, depth_rank);

-- 4.3 Game Schedule
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
    game_status VARCHAR(20) DEFAULT 'Scheduled',
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for nba_schedule table
CREATE INDEX IF NOT EXISTS idx_schedule_date ON nba_schedule(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_schedule_season ON nba_schedule(season_id, game_date);
CREATE INDEX IF NOT EXISTS idx_schedule_team_home ON nba_schedule(home_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_schedule_team_away ON nba_schedule(away_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_schedule_status ON nba_schedule(game_status);


-- ============================================================================
-- SECTION 5: ML PREDICTION SYSTEM
-- ============================================================================

-- 5.1 ML Model Registry
CREATE TABLE IF NOT EXISTS ml_models (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) CHECK (model_type IN ('Ensemble', 'LSTM', 'XGBoost', 'Conformal', 'RAPM')),
    
    -- Model Artifacts (reference, not storage)
    storage_path TEXT,  -- e.g., GCS, S3, or local path
    file_hash VARCHAR(64),  -- SHA-256 for verification
    file_size_mb DECIMAL(10,2),
    
    -- Performance Metrics
    mae DECIMAL(6,3),
    rmse DECIMAL(6,3),
    r2_score DECIMAL(6,4),
    coverage_90 DECIMAL(6,4),
    
    -- Training Info
    training_start_date DATE,
    training_end_date DATE,
    num_games_trained INT,
    features_used INT,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    deployed_at TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),
    notes TEXT,
    
    UNIQUE (model_name, model_version)
);

-- Create indexes for ml_models table
CREATE INDEX IF NOT EXISTS idx_models_active ON ml_models(is_active, model_version DESC);

-- 5.2 ML Predictions Log (Real-time)
CREATE TABLE IF NOT EXISTS ml_predictions (
    prediction_id SERIAL PRIMARY KEY,
    game_id VARCHAR(15) REFERENCES games(game_id),
    model_id INT REFERENCES ml_models(model_id),
    
    -- Prediction Timing
    prediction_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    quarter INT CHECK (quarter >= 1 AND quarter <= 4),
    time_remaining VARCHAR(10),  -- e.g., "06:00"
    game_state JSON,  -- Current scores, possessions, etc.
    
    -- ML Output
    point_forecast DECIMAL(6,2),
    interval_lower DECIMAL(6,2),
    interval_upper DECIMAL(6,2),
    coverage_probability DECIMAL(5,3),
    
    -- Confidence & Features
    model_confidence DECIMAL(5,3),
    features_extracted JSON,  -- 33 Mamba features
    feature_importance JSON,  -- Top contributing features
    
    -- Market Comparison
    market_spread DECIMAL(6,2),
    edge_detected BOOLEAN DEFAULT FALSE,
    edge_magnitude DECIMAL(6,2),
    
    -- Outcome (filled post-game)
    actual_outcome DECIMAL(6,2),
    prediction_error DECIMAL(6,2),
    within_interval BOOLEAN,
    
    -- Special Flags
    is_q2_6min BOOLEAN DEFAULT FALSE,  -- The golden prediction
    is_trade_signal BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for ml_predictions table
CREATE INDEX IF NOT EXISTS idx_predictions_game ON ml_predictions(game_id, prediction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_q2_6min ON ml_predictions(is_q2_6min, game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_edge ON ml_predictions(edge_detected, prediction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON ml_predictions(prediction_timestamp DESC);

-- 5.3 ML Prediction Performance Tracking
CREATE TABLE IF NOT EXISTS prediction_performance (
    performance_id SERIAL PRIMARY KEY,
    model_id INT REFERENCES ml_models(model_id),
    evaluation_date DATE NOT NULL,
    
    -- Daily Metrics
    games_predicted INT,
    mean_error DECIMAL(6,3),
    median_error DECIMAL(6,3),
    mae DECIMAL(6,3),
    rmse DECIMAL(6,3),
    
    -- Interval Coverage
    coverage_90_actual DECIMAL(6,4),
    coverage_80_actual DECIMAL(6,4),
    
    -- Edge Detection Performance
    edges_detected INT,
    edges_profitable INT,
    edge_win_rate DECIMAL(6,4),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE (model_id, evaluation_date)
);

-- Create indexes for prediction_performance table
CREATE INDEX IF NOT EXISTS idx_performance_model_date ON prediction_performance(model_id, evaluation_date DESC);


-- ============================================================================
-- SECTION 6: ROLLING WINDOWS & MATERIALIZED VIEWS
-- ============================================================================

-- 6.1 Last 10 Games Rolling Window (FAST ACCESS)
CREATE MATERIALIZED VIEW IF NOT EXISTS player_last10 AS
WITH ranked_games AS (
    SELECT 
        player_id,
        game_id,
        game_date,
        pts, reb, ast, stl, blk, tov,
        minutes,
        ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY game_date DESC) as rn
    FROM player_box_scores
)
SELECT *
FROM ranked_games
WHERE rn <= 10;

CREATE INDEX idx_last10_player ON player_last10(player_id, rn);

-- Refresh schedule: Run this every night at 3:30 AM
-- REFRESH MATERIALIZED VIEW CONCURRENTLY player_last10;


-- ============================================================================
-- SECTION 7: AUDIT & MONITORING
-- ============================================================================

-- 7.1 Data Pipeline Runs
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(100) NOT NULL,
    run_date DATE NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR(20) CHECK (status IN ('Running', 'Success', 'Failed', 'Partial')),
    
    -- Metrics
    games_processed INT,
    players_updated INT,
    errors_count INT,
    error_log TEXT,
    
    -- Metadata
    triggered_by VARCHAR(50)  -- 'scheduled', 'manual', 'api'
);

-- Create indexes for pipeline_runs table
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_date ON pipeline_runs(run_date DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(status, run_date);

-- 7.2 API Request Log (for rate limiting & monitoring)
CREATE TABLE IF NOT EXISTS api_requests (
    request_id SERIAL PRIMARY KEY,
    endpoint VARCHAR(200) NOT NULL,
    request_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    response_time_ms INT,
    status_code INT,
    error_message TEXT
);

-- Create indexes for api_requests table
CREATE INDEX IF NOT EXISTS idx_api_requests_timestamp ON api_requests(request_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_api_requests_endpoint ON api_requests(endpoint, request_timestamp DESC);


-- ============================================================================
-- SECTION 8: HELPER FUNCTIONS
-- ============================================================================

-- 8.1 Calculate Team Possessions
CREATE OR REPLACE FUNCTION calculate_team_possessions(
    fga INT,
    fta INT,
    oreb INT,
    tov INT
) RETURNS DECIMAL(10,2) AS $$
BEGIN
    RETURN fga + 0.44 * fta - oreb + tov;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- 8.2 Calculate True Shooting %
CREATE OR REPLACE FUNCTION calculate_ts_pct(
    pts INT,
    fga INT,
    fta INT
) RETURNS DECIMAL(5,3) AS $$
BEGIN
    IF (fga + 0.44 * fta) = 0 THEN
        RETURN NULL;
    END IF;
    RETURN pts / (2.0 * (fga + 0.44 * fta));
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ============================================================================
-- SECTION 9: SEED DATA
-- ============================================================================

-- Insert current season
INSERT INTO seasons (season_id, start_date, is_current) 
VALUES ('2025-26', '2025-10-22', TRUE)
ON CONFLICT (season_id) DO NOTHING;

-- Insert active ML model
INSERT INTO ml_models (
    model_name, 
    model_version, 
    model_type,
    storage_path,
    file_size_mb,
    mae,
    is_active,
    deployed_at,
    created_by,
    notes
) VALUES (
    'MAMBA_MENTALITY_SYSTEM',
    '1.0.0',
    'Ensemble',
    'https://drive.google.com/uc?export=download&id=1Zi9OnSc3mMVDZyXVL0tGvvkIa9-O7JFG',
    307.5,
    5.39,
    TRUE,
    NOW(),
    'Ontologic XYZ',
    'Dejavu + LSTM + Conformal Prediction ensemble. Trained on 18min PBP windows. Optimal at Q2 6:00.'
)
ON CONFLICT (model_name, model_version) DO UPDATE
SET is_active = TRUE, deployed_at = NOW();


-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

-- Performance optimization: Analyze tables after initial load
-- Run this after data import:
-- ANALYZE player_box_scores;
-- ANALYZE player_season_stats;
-- ANALYZE team_season_stats;
-- ANALYZE ml_predictions;

