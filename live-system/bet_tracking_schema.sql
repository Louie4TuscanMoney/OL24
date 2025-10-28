-- ============================================================================
-- BET TRACKING SYSTEM - Store ALL trades with full ML context
-- ============================================================================

-- User Bets (every trade you make)
CREATE TABLE IF NOT EXISTS user_bets (
    bet_id SERIAL PRIMARY KEY,
    
    -- Game Context
    game_id VARCHAR(20) NOT NULL,
    game_date TIMESTAMP NOT NULL,
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,
    
    -- Bet Details
    bet_type VARCHAR(20) NOT NULL,  -- 'spread', 'total', 'moneyline'
    bet_side VARCHAR(10) NOT NULL,  -- 'home', 'away', 'over', 'under'
    
    -- Market Odds (what you bet)
    market_spread DECIMAL(4,1),
    market_total DECIMAL(5,1),
    market_odds_american INTEGER,
    market_odds_decimal DECIMAL(5,2),
    
    -- ML Prediction at time of bet
    ml_predicted_spread DECIMAL(5,2),
    ml_predicted_total DECIMAL(5,2),
    ml_win_probability DECIMAL(5,4),
    ml_confidence_lower DECIMAL(5,2),
    ml_confidence_upper DECIMAL(5,2),
    ml_model_confidence DECIMAL(5,4),
    
    -- Edge Calculations
    edge_points DECIMAL(5,2),
    edge_percentage DECIMAL(6,2),
    expected_value DECIMAL(6,2),
    kelly_criterion DECIMAL(5,2),
    true_edge_probability DECIMAL(5,4),
    
    -- Bet Sizing
    bet_amount DECIMAL(10,2),
    potential_payout DECIMAL(10,2),
    bankroll_at_time DECIMAL(12,2),
    kelly_percentage DECIMAL(5,2),
    
    -- Game State at Bet Time
    game_quarter INTEGER,
    game_time_remaining VARCHAR(20),
    home_score_at_bet INTEGER,
    away_score_at_bet INTEGER,
    score_diff_at_bet INTEGER,
    
    -- ML Feature Snapshot (33 Mamba features!)
    ml_features JSONB,  -- Store all 33 features for analysis
    
    -- Outcome (filled after game)
    actual_result DECIMAL(5,2),  -- Actual spread/total result
    bet_won BOOLEAN,
    profit_loss DECIMAL(10,2),
    settled_at TIMESTAMP,
    
    -- Metadata
    placed_at TIMESTAMP DEFAULT NOW(),
    placed_via VARCHAR(20) DEFAULT 'manual',  -- 'manual', 'auto', 'ontorisk'
    notes TEXT,
    
    -- Indexes for fast queries
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_bets_game ON user_bets(game_id);
CREATE INDEX IF NOT EXISTS idx_user_bets_date ON user_bets(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_user_bets_outcome ON user_bets(bet_won) WHERE bet_won IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_bets_created ON user_bets(created_at DESC);

-- Performance Tracking (daily summary)
CREATE TABLE IF NOT EXISTS bet_performance (
    performance_id SERIAL PRIMARY KEY,
    
    -- Date
    date DATE NOT NULL UNIQUE,
    
    -- Volume
    total_bets INTEGER DEFAULT 0,
    total_wagered DECIMAL(12,2) DEFAULT 0,
    
    -- Results
    bets_won INTEGER DEFAULT 0,
    bets_lost INTEGER DEFAULT 0,
    bets_push INTEGER DEFAULT 0,
    
    -- Profit/Loss
    gross_profit DECIMAL(12,2) DEFAULT 0,
    gross_loss DECIMAL(12,2) DEFAULT 0,
    net_profit DECIMAL(12,2) DEFAULT 0,
    
    -- ROI Metrics
    roi_percentage DECIMAL(6,2),
    win_rate DECIMAL(5,2),
    avg_bet_size DECIMAL(10,2),
    
    -- Edge Metrics
    avg_edge DECIMAL(5,2),
    avg_ev DECIMAL(6,2),
    avg_kelly DECIMAL(5,2),
    
    -- Sharpness (how well ML predicted)
    ml_accuracy DECIMAL(5,2),  -- % of bets where ML was right
    avg_ml_confidence DECIMAL(5,4),
    
    -- Bankroll
    starting_bankroll DECIMAL(12,2),
    ending_bankroll DECIMAL(12,2),
    bankroll_growth DECIMAL(12,2),
    
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ML Model Performance (track prediction accuracy)
CREATE TABLE IF NOT EXISTS ml_bet_performance (
    id SERIAL PRIMARY KEY,
    
    -- Prediction
    game_id VARCHAR(20) NOT NULL,
    predicted_spread DECIMAL(5,2),
    predicted_win_prob DECIMAL(5,4),
    confidence_interval_lower DECIMAL(5,2),
    confidence_interval_upper DECIMAL(5,2),
    model_confidence DECIMAL(5,4),
    
    -- Actual Outcome
    actual_spread DECIMAL(5,2),
    actual_winner VARCHAR(3),
    prediction_correct BOOLEAN,
    within_confidence_interval BOOLEAN,
    
    -- Error Metrics
    mae DECIMAL(5,2),  -- Mean Absolute Error
    prediction_error DECIMAL(5,2),
    
    -- Context
    game_date TIMESTAMP,
    predicted_at TIMESTAMP DEFAULT NOW(),
    settled_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ml_performance_game ON ml_bet_performance(game_id);
CREATE INDEX IF NOT EXISTS idx_ml_performance_date ON ml_bet_performance(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_ml_performance_correct ON ml_bet_performance(prediction_correct);

-- Bankroll Tracking (every transaction)
CREATE TABLE IF NOT EXISTS bankroll_history (
    id SERIAL PRIMARY KEY,
    
    -- Transaction
    transaction_type VARCHAR(20) NOT NULL,  -- 'bet_placed', 'bet_won', 'bet_lost', 'deposit', 'withdrawal'
    amount DECIMAL(10,2) NOT NULL,
    balance_before DECIMAL(12,2) NOT NULL,
    balance_after DECIMAL(12,2) NOT NULL,
    
    -- Reference
    bet_id INTEGER REFERENCES user_bets(bet_id),
    
    -- Metadata
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bankroll_created ON bankroll_history(created_at DESC);

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Set initial bankroll
INSERT INTO bankroll_history (transaction_type, amount, balance_before, balance_after, notes)
VALUES ('deposit', 10000.00, 0.00, 10000.00, 'Initial bankroll')
ON CONFLICT DO NOTHING;

