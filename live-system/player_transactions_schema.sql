-- ============================================================================
-- PLAYER TRANSACTIONS (Trades, Waivers, Signings)
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_transactions (
    transaction_id SERIAL PRIMARY KEY,
    player_id VARCHAR(10),
    player_name VARCHAR(100) NOT NULL,
    
    -- Transaction Details
    transaction_type VARCHAR(20) NOT NULL CHECK (transaction_type IN ('Trade', 'Waiver', 'Signing', 'Release', 'Two-Way', 'G League')),
    transaction_date DATE NOT NULL,
    
    -- Team Movement
    from_team_id VARCHAR(10) REFERENCES teams(team_id) ON DELETE SET NULL,
    to_team_id VARCHAR(10) REFERENCES teams(team_id) ON DELETE SET NULL,
    
    -- Trade Details
    trade_description TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_transactions_player (player_id),
    INDEX idx_transactions_date (transaction_date DESC),
    INDEX idx_transactions_from_team (from_team_id, transaction_date DESC),
    INDEX idx_transactions_to_team (to_team_id, transaction_date DESC),
    INDEX idx_transactions_type (transaction_type)
);

-- Create indexes separately (PostgreSQL syntax)
CREATE INDEX IF NOT EXISTS idx_transactions_player ON player_transactions(player_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON player_transactions(transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_from_team ON player_transactions(from_team_id, transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_to_team ON player_transactions(to_team_id, transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON player_transactions(transaction_type);

-- Sample query: Get all transactions for a team
-- SELECT * FROM player_transactions 
-- WHERE from_team_id = '1610612747' OR to_team_id = '1610612747'
-- ORDER BY transaction_date DESC;

-- Sample query: Get league-wide transactions
-- SELECT 
--     pt.*,
--     t_from.abbreviation as from_team,
--     t_to.abbreviation as to_team
-- FROM player_transactions pt
-- LEFT JOIN teams t_from ON pt.from_team_id = t_from.team_id
-- LEFT JOIN teams t_to ON pt.to_team_id = t_to.team_id
-- ORDER BY transaction_date DESC
-- LIMIT 50;

