-- ============================================================================
-- BETTING BACKEND SCHEMA
-- PostgreSQL schema for betting functionality (replaces BetOnline scraper)
-- ============================================================================

-- Betting Lines Table
-- Stores odds from various sources (manual entry, API, or future scrapers)
CREATE TABLE IF NOT EXISTS betting_lines (
    line_id SERIAL PRIMARY KEY,
    game_id VARCHAR(20) NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    market_type VARCHAR(50) NOT NULL,  -- 'moneyline', 'spread', 'total'
    line_value DECIMAL(5,1),  -- Spread or total value (NULL for moneyline)
    home_odds DECIMAL(6,2),   -- American odds (e.g., -110, +150)
    away_odds DECIMAL(6,2),
    over_odds DECIMAL(6,2),   -- For totals
    under_odds DECIMAL(6,2),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    
    CONSTRAINT unique_line UNIQUE(game_id, bookmaker, market_type, line_value)
);

CREATE INDEX IF NOT EXISTS idx_betting_lines_game ON betting_lines(game_id, is_active);
CREATE INDEX IF NOT EXISTS idx_betting_lines_updated ON betting_lines(updated_at DESC);


-- User Bankroll Table
-- Tracks user balance and betting history
CREATE TABLE IF NOT EXISTS user_bankroll (
    user_id VARCHAR(50) PRIMARY KEY,
    balance DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    total_deposited DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    total_withdrawn DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    total_wagered DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    total_profit DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    total_bets INT NOT NULL DEFAULT 0,
    winning_bets INT NOT NULL DEFAULT 0,
    losing_bets INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT positive_balance CHECK (balance >= 0)
);


-- Bets Table
-- Records all placed bets with OntoRisk validation
CREATE TABLE IF NOT EXISTS bets (
    bet_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES user_bankroll(user_id),
    game_id VARCHAR(20) NOT NULL,
    prediction_id INT,  -- Links to mamba_game_cache if ML-based
    
    -- Bet Details
    bet_type VARCHAR(20) NOT NULL,  -- 'moneyline', 'spread', 'total_over', 'total_under'
    selection VARCHAR(10) NOT NULL,  -- 'home', 'away', 'over', 'under'
    line_value DECIMAL(5,1),  -- Spread or total value
    odds DECIMAL(6,2) NOT NULL,  -- American odds
    
    -- Money
    stake DECIMAL(10,2) NOT NULL,
    potential_return DECIMAL(10,2) NOT NULL,
    actual_return DECIMAL(10,2),
    profit_loss DECIMAL(10,2),
    
    -- OntoRisk Validation
    ontorisk_approved BOOLEAN DEFAULT FALSE,
    risk_score DECIMAL(5,2),  -- 0-100 risk score
    kelly_fraction DECIMAL(5,4),  -- Recommended Kelly bet size
    edge DECIMAL(5,4),  -- Estimated edge over bookmaker
    
    -- Status
    status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'won', 'lost', 'push', 'cancelled'
    result_verified BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    placed_at TIMESTAMP DEFAULT NOW(),
    settled_at TIMESTAMP,
    
    -- Constraints
    CONSTRAINT positive_stake CHECK (stake > 0),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'won', 'lost', 'push', 'cancelled'))
);

CREATE INDEX IF NOT EXISTS idx_bets_user ON bets(user_id, placed_at DESC);
CREATE INDEX IF NOT EXISTS idx_bets_game ON bets(game_id);
CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status, placed_at DESC);
CREATE INDEX IF NOT EXISTS idx_bets_prediction ON bets(prediction_id);


-- Bookmaker Configuration
-- Stores available bookmakers and their settings
CREATE TABLE IF NOT EXISTS bookmakers (
    bookmaker_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    website_url VARCHAR(200),
    api_endpoint VARCHAR(200),
    is_active BOOLEAN DEFAULT TRUE,
    default_juice DECIMAL(5,2) DEFAULT -110,  -- Standard vig
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Seed default bookmakers
INSERT INTO bookmakers (name, display_name, website_url, is_active) VALUES
    ('draftkings', 'DraftKings', 'https://sportsbook.draftkings.com', true),
    ('fanduel', 'FanDuel', 'https://sportsbook.fanduel.com', true),
    ('betmgm', 'BetMGM', 'https://sports.betmgm.com', true),
    ('caesars', 'Caesars', 'https://www.caesars.com/sportsbook', true),
    ('pointsbet', 'PointsBet', 'https://pointsbet.com', true)
ON CONFLICT (name) DO NOTHING;


-- Betting Opportunities View
-- Combines Mamba predictions with available betting lines
CREATE OR REPLACE VIEW betting_opportunities AS
SELECT 
    m.game_id,
    ht.abbreviation as home_team,
    at.abbreviation as away_team,
    m.prediction as mamba_prediction,
    m.confidence,
    m.triggered_at,
    m.current_margin,
    m.market_spread,
    m.market_total,
    bl.bookmaker,
    bl.market_type,
    bl.line_value,
    bl.home_odds,
    bl.away_odds,
    bl.line_id,
    -- Calculate implied probability from odds
    CASE 
        WHEN bl.home_odds < 0 THEN ABS(bl.home_odds) / (ABS(bl.home_odds) + 100.0)
        ELSE 100.0 / (bl.home_odds + 100.0)
    END as home_implied_prob,
    CASE 
        WHEN bl.away_odds < 0 THEN ABS(bl.away_odds) / (ABS(bl.away_odds) + 100.0)
        ELSE 100.0 / (bl.away_odds + 100.0)
    END as away_implied_prob
FROM mamba_game_cache m
LEFT JOIN teams ht ON m.home_team_id = ht.team_id
LEFT JOIN teams at ON m.away_team_id = at.team_id
LEFT JOIN betting_lines bl ON m.game_id = bl.game_id AND bl.is_active = TRUE
WHERE m.triggered_at IS NOT NULL  -- Only show games where Mamba has triggered
ORDER BY m.triggered_at DESC;


-- User Statistics View
-- Provides betting performance analytics
CREATE OR REPLACE VIEW user_betting_stats AS
SELECT 
    u.user_id,
    u.balance,
    u.total_bets,
    u.winning_bets,
    u.losing_bets,
    CASE 
        WHEN u.total_bets > 0 THEN ROUND((u.winning_bets::DECIMAL / u.total_bets) * 100, 1)
        ELSE 0 
    END as win_rate_pct,
    u.total_profit,
    CASE 
        WHEN u.total_wagered > 0 THEN ROUND((u.total_profit / u.total_wagered) * 100, 2)
        ELSE 0
    END as roi_pct,
    -- Recent performance (last 10 bets)
    (SELECT COUNT(*) FROM (
        SELECT * FROM bets WHERE user_id = u.user_id AND status = 'won' 
        ORDER BY placed_at DESC LIMIT 10
    ) recent_won) as recent_wins,
    (SELECT COUNT(*) FROM (
        SELECT * FROM bets WHERE user_id = u.user_id AND status = 'lost' 
        ORDER BY placed_at DESC LIMIT 10
    ) recent_lost) as recent_losses,
    -- Best streak
    (SELECT MAX(streak_count) FROM (
        SELECT COUNT(*) as streak_count
        FROM (
            SELECT user_id, status,
                   SUM(CASE WHEN status != 'won' THEN 1 ELSE 0 END) OVER (ORDER BY placed_at) as grp
            FROM bets WHERE user_id = u.user_id
        ) sub
        WHERE status = 'won'
        GROUP BY grp
    ) streaks) as best_win_streak
FROM user_bankroll u;


-- Helper Functions

-- Calculate American odds to decimal
CREATE OR REPLACE FUNCTION american_to_decimal(odds DECIMAL)
RETURNS DECIMAL AS $$
BEGIN
    IF odds < 0 THEN
        RETURN 1 + (100.0 / ABS(odds));
    ELSE
        RETURN 1 + (odds / 100.0);
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Calculate potential return from American odds
CREATE OR REPLACE FUNCTION calculate_return(stake DECIMAL, odds DECIMAL)
RETURNS DECIMAL AS $$
BEGIN
    RETURN stake * american_to_decimal(odds);
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Comments
COMMENT ON TABLE betting_lines IS 'Stores betting odds from various bookmakers (replaces BetOnline scraper)';
COMMENT ON TABLE user_bankroll IS 'Tracks user balance and betting performance';
COMMENT ON TABLE bets IS 'Records all placed bets with OntoRisk validation';
COMMENT ON TABLE bookmakers IS 'Available bookmakers and their configuration';
COMMENT ON VIEW betting_opportunities IS 'Combines Mamba predictions with available betting lines';
COMMENT ON VIEW user_betting_stats IS 'Provides comprehensive user betting analytics';

-- ============================================================================
-- SAMPLE DATA FOR TESTING
-- ============================================================================

-- Create test user
INSERT INTO user_bankroll (user_id, balance, total_deposited) VALUES
    ('test_user', 5000.00, 5000.00)
ON CONFLICT (user_id) DO UPDATE SET balance = 5000.00, total_deposited = 5000.00;

-- Sample betting lines (you'll populate these via API or manual entry)
-- These are examples - replace with real odds
/*
INSERT INTO betting_lines (game_id, bookmaker, market_type, home_odds, away_odds) VALUES
    ('401809993', 'draftkings', 'moneyline', -150, +130),
    ('401809993', 'fanduel', 'moneyline', -145, +125),
    ('401809994', 'draftkings', 'moneyline', +110, -130);
*/

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

