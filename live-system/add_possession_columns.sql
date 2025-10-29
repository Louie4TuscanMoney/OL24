-- ============================================================================
-- ADD REAL POSSESSION TRACKING COLUMNS
-- Migration: Add per-100 stats to player_box_scores and total_team_possessions
-- ============================================================================

-- 1. Add per-100 columns to player_box_scores (if not exist)
ALTER TABLE player_box_scores 
ADD COLUMN IF NOT EXISTS pts_100 DECIMAL(7,2),
ADD COLUMN IF NOT EXISTS reb_100 DECIMAL(7,2),
ADD COLUMN IF NOT EXISTS ast_100 DECIMAL(7,2),
ADD COLUMN IF NOT EXISTS stl_100 DECIMAL(7,2),
ADD COLUMN IF NOT EXISTS blk_100 DECIMAL(7,2),
ADD COLUMN IF NOT EXISTS tov_100 DECIMAL(7,2);

-- 2. Add total_team_possessions to player_season_stats (if not exist)
ALTER TABLE player_season_stats
ADD COLUMN IF NOT EXISTS total_team_possessions INT;

-- 3. Create index on team_poss for faster queries
CREATE INDEX IF NOT EXISTS idx_box_scores_team_poss ON player_box_scores(team_poss) 
WHERE team_poss IS NOT NULL;

COMMENT ON COLUMN player_box_scores.team_poss IS 'Real team possessions from boxscoreadvancedv2.POSS';
COMMENT ON COLUMN player_box_scores.pts_100 IS 'Points per 100 possessions (REAL, not estimated)';
COMMENT ON COLUMN player_season_stats.total_team_possessions IS 'Sum of team possessions across all games played';

-- Verification
SELECT 'player_box_scores columns' as table_name, COUNT(*) as count 
FROM information_schema.columns 
WHERE table_name = 'player_box_scores' AND column_name LIKE '%_100';

SELECT 'player_season_stats columns' as table_name, COUNT(*) as count
FROM information_schema.columns
WHERE table_name = 'player_season_stats' AND column_name = 'total_team_possessions';

