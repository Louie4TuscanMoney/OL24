-- STRYK PRODUCTION SCHEMA
-- PostgreSQL 15+
-- Run this in Railway Postgres console

-- =====================================================
-- CORE TABLES
-- =====================================================

CREATE TABLE IF NOT EXISTS games (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id TEXT NOT NULL,
    sport TEXT NOT NULL DEFAULT 'nba',
    home_team TEXT,
    away_team TEXT,
    home_score INT,
    away_score INT,
    status INT,  -- 1=scheduled, 2=live, 3=final
    period INT,
    clock TEXT,
    game_date TIMESTAMPTZ,
    spread DECIMAL(4,1),
    total DECIMAL(4,1),
    home_ml INT,
    away_ml INT,
    spread_display TEXT,  -- "Lakers -6.0"
    underdog_display TEXT,  -- "Warriors +6.0"
    home_implied_prob DECIMAL(5,4),
    away_implied_prob DECIMAL(5,4),
    home_no_vig_prob DECIMAL(5,4),
    away_no_vig_prob DECIMAL(5,4),
    vig_percentage DECIMAL(5,2),
    source TEXT DEFAULT 'BetOnline',
    is_q2_6min BOOLEAN DEFAULT FALSE,
    can_predict BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(external_id, sport)
);

CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id UUID REFERENCES games(id) ON DELETE CASCADE,
    model_version TEXT DEFAULT 'mamba_v1',
    spread_pred DECIMAL(4,1),
    mamba_score DECIMAL(10,6),
    mamba_probability DECIMAL(5,4),
    market_probability DECIMAL(5,4),
    kelly_edge DECIMAL(5,4),
    kelly_bet_size DECIMAL(12,2),
    recommended_bet TEXT,
    bet_confidence TEXT,  -- 'high', 'medium', 'low'
    archetype TEXT,
    risk_score DECIMAL(5,3),
    max_exposure DECIMAL(12,2),
    is_q2_6min BOOLEAN DEFAULT TRUE,
    period INT,
    clock TEXT,
    features JSONB,  -- Store 67 Mamba features
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mamba_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id UUID REFERENCES games(id) ON DELETE CASCADE,
    external_id TEXT NOT NULL,
    home_team TEXT,
    away_team TEXT,
    mamba_score DECIMAL(10,6),
    period INT,
    clock TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(external_id, period, clock)
);

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT,
    bankroll DECIMAL(12,2) DEFAULT 1000.00,
    total_bet DECIMAL(12,2) DEFAULT 0.00,
    total_profit DECIMAL(12,2) DEFAULT 0.00,
    roi DECIMAL(5,2) DEFAULT 0.00,
    status TEXT DEFAULT 'pending',  -- 'pending', 'approved', 'banned'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    approved_at TIMESTAMPTZ,
    last_login TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS bets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    game_id UUID REFERENCES games(id) ON DELETE CASCADE,
    prediction_id UUID REFERENCES predictions(id) ON DELETE SET NULL,
    bet_type TEXT,  -- 'spread', 'total', 'moneyline'
    bet_side TEXT,  -- 'home', 'away', 'over', 'under'
    stake DECIMAL(12,2),
    odds INT,
    potential_payout DECIMAL(12,2),
    status TEXT DEFAULT 'pending',  -- 'pending', 'won', 'lost', 'push'
    actual_payout DECIMAL(12,2),
    placed_at TIMESTAMPTZ DEFAULT NOW(),
    settled_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS user_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT NOT NULL,
    email TEXT NOT NULL,
    reason TEXT,
    status TEXT DEFAULT 'pending',  -- 'pending', 'approved', 'rejected'
    admin_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);
CREATE INDEX IF NOT EXISTS idx_games_period ON games(period);
CREATE INDEX IF NOT EXISTS idx_games_sport ON games(sport);
CREATE INDEX IF NOT EXISTS idx_games_external_id ON games(external_id);
CREATE INDEX IF NOT EXISTS idx_games_updated_at ON games(updated_at);

CREATE INDEX IF NOT EXISTS idx_predictions_game_id ON predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_kelly_edge ON predictions(kelly_edge);

CREATE INDEX IF NOT EXISTS idx_mamba_scores_game_id ON mamba_scores(game_id);
CREATE INDEX IF NOT EXISTS idx_mamba_scores_external_id ON mamba_scores(external_id);

CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE INDEX IF NOT EXISTS idx_bets_user_id ON bets(user_id);
CREATE INDEX IF NOT EXISTS idx_bets_game_id ON bets(game_id);
CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

CREATE OR REPLACE VIEW live_opportunities AS
SELECT 
    g.id,
    g.external_id,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score,
    g.period,
    g.clock,
    g.spread,
    g.total,
    g.home_ml,
    g.away_ml,
    g.spread_display,
    p.mamba_score,
    p.kelly_edge,
    p.kelly_bet_size,
    p.recommended_bet,
    p.bet_confidence,
    p.archetype,
    p.created_at as prediction_time
FROM games g
LEFT JOIN predictions p ON g.id = p.game_id
WHERE g.status = 2  -- Live games only
  AND g.period >= 2  -- Q2 or later
  AND p.kelly_edge > 0.02  -- Positive edge only
ORDER BY p.kelly_edge DESC;

CREATE OR REPLACE VIEW user_performance AS
SELECT 
    u.id,
    u.username,
    u.bankroll,
    u.total_bet,
    u.total_profit,
    u.roi,
    COUNT(b.id) as total_bets,
    COUNT(CASE WHEN b.status = 'won' THEN 1 END) as wins,
    COUNT(CASE WHEN b.status = 'lost' THEN 1 END) as losses,
    COUNT(CASE WHEN b.status = 'push' THEN 1 END) as pushes
FROM users u
LEFT JOIN bets b ON u.id = b.user_id
GROUP BY u.id;

-- =====================================================
-- FUNCTIONS FOR AUTO-UPDATES
-- =====================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER games_updated_at
BEFORE UPDATE ON games
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

-- =====================================================
-- VERIFY SCHEMA
-- =====================================================

SELECT 'STRYK SCHEMA CREATED SUCCESSFULLY' AS status;
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

