-- ============================================================================
-- TRACKED BETS TABLE FOR TRADING DASHBOARD
-- ============================================================================

CREATE TABLE IF NOT EXISTS tracked_bets (
    bet_id SERIAL PRIMARY KEY,
    game_id VARCHAR(15) NOT NULL,
    
    -- Bet details
    bet_type VARCHAR(20) NOT NULL,  -- 'spread', 'total', 'moneyline', '2h_spread'
    side VARCHAR(10) NOT NULL,  -- 'home', 'away', 'over', 'under'
    odds DECIMAL(10,2) NOT NULL,  -- American odds
    stake DECIMAL(10,2) NOT NULL,
    book VARCHAR(50) DEFAULT 'DraftKings',
    
    -- Mamba data at time of bet
    mamba_prediction DECIMAL(10,2),
    mamba_confidence DECIMAL(5,2),
    expected_value DECIMAL(10,2),
    
    -- Result
    result VARCHAR(10),  -- 'win', 'loss', 'push'
    profit_loss DECIMAL(10,2),
    
    -- Timestamps
    placed_at TIMESTAMP DEFAULT NOW(),
    settled_at TIMESTAMP,
    
    FOREIGN KEY (game_id) REFERENCES mamba_game_cache(game_id) ON DELETE SET NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tracked_bets_game ON tracked_bets(game_id);
CREATE INDEX IF NOT EXISTS idx_tracked_bets_placed ON tracked_bets(placed_at DESC);
CREATE INDEX IF NOT EXISTS idx_tracked_bets_result ON tracked_bets(result);

-- Comments
COMMENT ON TABLE tracked_bets IS 'All bets placed via trading dashboard with Mamba predictions';
COMMENT ON COLUMN tracked_bets.expected_value IS 'Calculated EV at time of bet placement';
COMMENT ON COLUMN tracked_bets.profit_loss IS 'Actual P&L after bet settles';

