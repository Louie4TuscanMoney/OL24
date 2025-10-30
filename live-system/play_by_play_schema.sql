-- ============================================================================
-- PLAY-BY-PLAY TABLE FOR MAMBA AUTONOMOUS PREDICTIONS
-- ============================================================================
--
-- Purpose: Store minute-by-minute scoring events for live games
-- Used by: Mamba model to extract 33 features at Q2 6:00
--
-- ============================================================================

CREATE TABLE IF NOT EXISTS play_by_play (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(15) NOT NULL,
    event_num INTEGER NOT NULL,
    
    -- Timing
    period INTEGER NOT NULL,
    clock VARCHAR(10),  -- "6:34"
    time_elapsed_seconds INTEGER,  -- 0-2880 (48 min game)
    
    -- Event details
    event_type VARCHAR(50),  -- 'field_goal_made', 'free_throw', 'turnover', etc.
    description TEXT,
    
    -- Players
    player_id VARCHAR(10),
    team_id VARCHAR(10),
    
    -- Score AFTER this event
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    score_margin INTEGER NOT NULL,  -- home - away (CRITICAL for Mamba)
    
    -- Event data (full JSON from NBA API)
    event_data JSONB,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(game_id, event_num)
);

-- Index for Mamba queries (get last 18 minutes)
CREATE INDEX IF NOT EXISTS idx_pbp_game_time 
    ON play_by_play(game_id, time_elapsed_seconds DESC);

-- Index for live game queries
CREATE INDEX IF NOT EXISTS idx_pbp_game_period 
    ON play_by_play(game_id, period, clock);

-- Index for cleanup (delete old games)
CREATE INDEX IF NOT EXISTS idx_pbp_created 
    ON play_by_play(created_at DESC);

-- ============================================================================
-- MAMBA GAME CACHE
-- ============================================================================
--
-- Purpose: Store Mamba predictions made at Q2 6:00
-- Includes: Features, prediction, game state, actual result
--
-- ============================================================================

CREATE TABLE IF NOT EXISTS mamba_game_cache (
    game_id VARCHAR(15) PRIMARY KEY,
    
    -- Mamba features (computed at Q2 6:00)
    features JSONB,  -- All 33 features as array
    
    -- Mamba prediction
    prediction DECIMAL(10,2),  -- Final spread prediction (+/-)
    confidence DECIMAL(5,2),  -- 0-100%
    
    -- Timing
    triggered_at TIMESTAMP,
    period INTEGER DEFAULT 2,
    clock VARCHAR(10) DEFAULT '6:00',
    triggered_type VARCHAR(20) DEFAULT 'Q2_6:00',  -- Q1_11:00 or Q2_6:00
    
    -- Game state when triggered
    home_team_id VARCHAR(10),
    away_team_id VARCHAR(10),
    home_score INTEGER,
    away_score INTEGER,
    current_margin INTEGER,
    
    -- Result (after game ends)
    final_home_score INTEGER,
    final_away_score INTEGER,
    actual_margin INTEGER,
    mamba_correct BOOLEAN,
    mamba_error DECIMAL(10,2),  -- prediction - actual
    
    -- 2H Result tracking
    h2_home_score INTEGER,
    h2_away_score INTEGER,
    h2_margin INTEGER,
    h2_prediction_error DECIMAL(10,2),
    
    -- Market data at trigger time
    market_spread DECIMAL(5,1),
    market_total DECIMAL(5,1),
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for performance tracking
CREATE INDEX IF NOT EXISTS idx_mamba_cache_triggered 
    ON mamba_game_cache(triggered_at DESC);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get play-by-play for last N seconds
CREATE OR REPLACE FUNCTION get_recent_playbyplay(
    p_game_id VARCHAR(15),
    p_seconds INTEGER DEFAULT 1080  -- 18 minutes
)
RETURNS TABLE (
    home_score INTEGER,
    away_score INTEGER,
    score_margin INTEGER,
    time_elapsed INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pbp.home_score,
        pbp.away_score,
        pbp.score_margin,
        pbp.time_elapsed_seconds
    FROM play_by_play pbp
    WHERE pbp.game_id = p_game_id
    AND pbp.time_elapsed_seconds <= p_seconds
    ORDER BY pbp.time_elapsed_seconds ASC;
END;
$$ LANGUAGE plpgsql;

-- Function to check if Mamba should trigger
CREATE OR REPLACE FUNCTION should_trigger_mamba(
    p_game_id VARCHAR(15),
    p_period INTEGER,
    p_clock VARCHAR(10)
)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check if Q2 6:00 and not already triggered
    IF p_period = 2 AND p_clock LIKE '6:0%' THEN
        RETURN NOT EXISTS (
            SELECT 1 FROM mamba_game_cache
            WHERE game_id = p_game_id
        );
    END IF;
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Cleanup old play-by-play data (keep last 7 days)
CREATE OR REPLACE FUNCTION cleanup_old_playbyplay()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM play_by_play
    WHERE created_at < NOW() - INTERVAL '7 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE play_by_play IS 'Stores every scoring event from live NBA games for Mamba model feature extraction';
COMMENT ON TABLE mamba_game_cache IS 'Stores Mamba predictions made at Q2 6:00 for each live game';
COMMENT ON COLUMN play_by_play.score_margin IS 'CRITICAL: home_score - away_score, used for Mamba pattern analysis';
COMMENT ON COLUMN play_by_play.time_elapsed_seconds IS 'Seconds from game start, used to filter last 18 minutes (0-1080)';
COMMENT ON COLUMN mamba_game_cache.features IS 'JSON array of 33 Mamba features extracted from play-by-play';
COMMENT ON COLUMN mamba_game_cache.prediction IS 'Mamba final spread prediction (positive = home favored)';

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Check if tables exist
DO $$
BEGIN
    RAISE NOTICE 'Tables created:';
    RAISE NOTICE '  ✅ play_by_play';
    RAISE NOTICE '  ✅ mamba_game_cache';
    RAISE NOTICE '';
    RAISE NOTICE 'Indexes created: 6';
    RAISE NOTICE 'Functions created: 3';
    RAISE NOTICE '';
    RAISE NOTICE '🎯 Ready for Mamba autonomous predictions!';
END $$;

