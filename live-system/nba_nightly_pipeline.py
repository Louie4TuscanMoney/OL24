"""
NBA NIGHTLY PIPELINE - SELF-COMPUTED METRICS
Runs at 3:30 AM UTC on Railway
100% nba_api → 100% our math → PostgreSQL
Finishes in <4 minutes
"""

import os
import time
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values
import numpy as np

try:
    from nba_api.stats.endpoints import (
        leaguegamelog,
        boxscoretraditionalv2,
        leaguestandingsv3
    )
    from nba_api.stats.static import players as nba_players
    from sklearn.linear_model import Ridge
    NBA_API_AVAILABLE = True
except ImportError as e:
    NBA_API_AVAILABLE = False
    print(f"❌ Missing dependency: {e}")

DATABASE_URL = os.environ.get('DATABASE_URL')


class NBANightlyPipeline:
    """
    Nightly pipeline: nba_api → self-computed metrics → PostgreSQL
    Target: <4 minutes total
    """
    
    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_URL)
        self.cursor = self.conn.cursor()
        self.current_season = '2024-25'
        self.start_time = None
        print(f"✅ Connected to PostgreSQL")
    
    @staticmethod
    def parse_minutes(min_str):
        """Convert 'MM:SS' to decimal minutes"""
        if not min_str or min_str == '' or min_str is None:
            return 0.0
        try:
            if ':' in str(min_str):
                parts = str(min_str).split(':')
                return float(parts[0]) + float(parts[1]) / 60.0
            else:
                return float(min_str)
        except:
            return 0.0
    
    @staticmethod
    def safe_int(val):
        """Safely convert to int, handling None/NaN"""
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return 0
        try:
            return int(val)
        except:
            return 0
    
    def run(self):
        """
        MAIN PIPELINE - 5 steps in <4 minutes
        """
        self.start_time = datetime.now()
        
        print("\n" + "="*80)
        print(f"🌙 NIGHTLY PIPELINE: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        try:
            # STEP 1: Pull games + box scores → player_box_scores
            print("📥 [1/5] Pulling games + box scores...")
            games_count = self.pull_games_and_box_scores()
            print(f"   ✅ {games_count} games processed\n")
            
            # STEP 2: Prune >10 games → refresh player_last10
            print("🗑️  [2/5] Pruning old games + refreshing last10...")
            self.prune_and_refresh_last10()
            print(f"   ✅ Last 10 games refreshed\n")
            
            # STEP 3: Aggregate → player_season_stats (per-100, per-36)
            print("🧮 [3/5] Computing season aggregates...")
            self.compute_season_aggregates()
            print(f"   ✅ Season stats updated\n")
            
            # STEP 4: Compute team metrics → team_season_stats
            print("📊 [4/5] Computing team metrics...")
            self.compute_team_metrics()
            print(f"   ✅ Team stats updated\n")
            
            # STEP 5: Run RAPM → update LEBRON (DAILY!)
            print("🧠 [5/5] Computing RAPM + LEBRON (daily)...")
            self.compute_rapm_and_lebron()
            print(f"   ✅ RAPM + LEBRON updated\n")
            
            # Update standings
            print("🏆 Updating standings...")
            self.update_standings()
            print(f"   ✅ Standings updated\n")
            
            # Create snapshot
            duration = (datetime.now() - self.start_time).total_seconds()
            self.create_snapshot('success', games_count, duration)
            
            # Commit
            self.conn.commit()
            
            print("="*80)
            print(f"✅ PIPELINE COMPLETED in {duration:.1f}s")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            duration = (datetime.now() - self.start_time).total_seconds()
            self.create_snapshot('failed', 0, duration)
            self.conn.rollback()
    
    def pull_games_and_box_scores(self):
        """
        STEP 1: Pull raw box scores from nba_api
        Target: <90 seconds
        """
        if not NBA_API_AVAILABLE:
            return 0
        
        # Get date range (allow override for historical data)
        date_from = os.environ.get('FORCE_DATE_FROM')
        date_to = os.environ.get('FORCE_DATE_TO')
        
        if not date_from or not date_to:
            # Default: yesterday and today
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%m/%d/%Y')
            today = datetime.now().strftime('%m/%d/%Y')
            date_from = yesterday
            date_to = today
        
        print(f"      Fetching games from {date_from} to {date_to}...")
        time.sleep(0.6)
        games_df = leaguegamelog.LeagueGameLog(
            season=self.current_season,
            season_type_all_star='Regular Season',
            date_from_nullable=date_from,
            date_to_nullable=date_to
        ).get_data_frames()[0]
        
        # Get unique game IDs + dates
        game_ids = games_df['GAME_ID'].unique()
        game_dates = {}  # Map game_id → game_date
        for _, row in games_df.iterrows():
            game_dates[row['GAME_ID']] = row['GAME_DATE']
        
        games_count = 0
        for game_id in game_ids:
            try:
                time.sleep(0.6)  # Rate limiting
                game_date = game_dates.get(game_id)
                
                # Get traditional box score (returns 3 dataframes: player, team, starter_bench)
                trad_dfs = boxscoretraditionalv2.BoxScoreTraditionalV2(
                    game_id=game_id
                ).get_data_frames()
                
                player_box = trad_dfs[0]  # Player stats
                team_box = trad_dfs[1]    # Team totals
                
                # Calculate team possessions ourselves (BoxScoreAdvancedV2 is broken!)
                # Formula: Poss ≈ FGA + 0.44×FTA - OREB + TO
                team_poss = {}
                all_teams = []
                for _, row in team_box.iterrows():
                    team_id = str(row['TEAM_ID'])
                    all_teams.append(team_id)
                    poss = row['FGA'] + 0.44 * row['FTA'] - row['OREB'] + row['TO']
                    team_poss[team_id] = poss
                
                # Insert player box scores
                for _, row in player_box.iterrows():
                    player_id = str(row['PLAYER_ID'])
                    team_id = str(row['TEAM_ID'])
                    player_name = row.get('PLAYER_NAME', 'Unknown')
                    
                    # Auto-create player if doesn't exist
                    self.cursor.execute("""
                        INSERT INTO players (player_id, name, team_id, is_active)
                        VALUES (%s, %s, %s, TRUE)
                        ON CONFLICT (player_id) DO UPDATE SET
                            team_id = EXCLUDED.team_id,
                            updated_at = NOW()
                    """, (player_id, player_name, team_id))
                    
                    # Get opponent team_id
                    opponent_id = [t for t in all_teams if t != team_id][0] if len(all_teams) == 2 else None
                    
                    self.cursor.execute("""
                        INSERT INTO player_box_scores (
                            player_id, game_id, game_date, season_id, team_id, opponent_id,
                            minutes, pts, fgm, fga, fg3m, fg3a, ftm, fta,
                            oreb, dreb, reb, ast, stl, blk, tov, pf, plus_minus,
                            team_poss
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (player_id, game_id, game_date) DO UPDATE SET
                            pts = EXCLUDED.pts,
                            team_poss = EXCLUDED.team_poss
                    """, (
                        player_id, game_id, game_date, self.current_season, team_id, opponent_id,
                        self.parse_minutes(row.get('MIN', 0)), 
                        self.safe_int(row.get('PTS')), self.safe_int(row.get('FGM')), self.safe_int(row.get('FGA')), 
                        self.safe_int(row.get('FG3M')), self.safe_int(row.get('FG3A')), self.safe_int(row.get('FTM')), self.safe_int(row.get('FTA')),
                        self.safe_int(row.get('OREB')), self.safe_int(row.get('DREB')), self.safe_int(row.get('REB')), self.safe_int(row.get('AST')), 
                        self.safe_int(row.get('STL')), self.safe_int(row.get('BLK')), self.safe_int(row.get('TO')), self.safe_int(row.get('PF')), self.safe_int(row.get('PLUS_MINUS')),
                        team_poss.get(team_id, 0)
                    ))
                
                games_count += 1
                
                if games_count % 10 == 0:
                    self.conn.commit()
                    print(f"      {games_count} games...")
                    
            except Exception as e:
                print(f"   ⚠️ Game {game_id} failed: {e}")
                continue
        
        return games_count
    
    def prune_and_refresh_last10(self):
        """
        STEP 2: Prune old games, refresh materialized view
        Target: <10 seconds
        """
        # Prune (delete games beyond 10th most recent)
        self.cursor.execute("SELECT prune_old_games()")
        
        # Refresh materialized view
        self.cursor.execute("SELECT refresh_last10()")
        
        self.conn.commit()
    
    def compute_season_aggregates(self):
        """
        STEP 3: Aggregate box scores → player_season_stats
        Target: <30 seconds
        """
        # Aggregate all box scores per player
        self.cursor.execute("""
            SELECT 
                player_id,
                COUNT(*) AS gp,
                SUM(minutes) AS min_total,
                SUM(pts) AS pts_total,
                SUM(reb) AS reb_total,
                SUM(ast) AS ast_total,
                SUM(stl) AS stl_total,
                SUM(blk) AS blk_total,
                SUM(tov) AS tov_total,
                SUM(fgm) AS fgm_total,
                SUM(fga) AS fga_total,
                SUM(fg3m) AS fg3m_total,
                SUM(fg3a) AS fg3a_total,
                SUM(ftm) AS ftm_total,
                SUM(fta) AS fta_total,
                SUM(team_poss) AS team_poss_total
            FROM player_box_scores
            WHERE season_id = %s
            GROUP BY player_id
        """, (self.current_season,))
        
        count = 0
        for row in self.cursor.fetchall():
            # Get current team
            self.cursor.execute("SELECT team_id FROM players WHERE player_id = %s", (row[0],))
            team_result = self.cursor.fetchone()
            team_id = team_result[0] if team_result else None
            
            self.cursor.execute("""
                INSERT INTO player_season_stats (
                    player_id, season_id, team_id,
                    gp, min_total,
                    pts_total, reb_total, ast_total, stl_total, blk_total, tov_total,
                    fgm_total, fga_total, fg3m_total, fg3a_total, ftm_total, fta_total,
                    team_poss_total
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_id, season_id) DO UPDATE SET
                    gp = EXCLUDED.gp,
                    min_total = EXCLUDED.min_total,
                    pts_total = EXCLUDED.pts_total,
                    reb_total = EXCLUDED.reb_total,
                    ast_total = EXCLUDED.ast_total,
                    stl_total = EXCLUDED.stl_total,
                    blk_total = EXCLUDED.blk_total,
                    tov_total = EXCLUDED.tov_total,
                    fgm_total = EXCLUDED.fgm_total,
                    fga_total = EXCLUDED.fga_total,
                    fg3m_total = EXCLUDED.fg3m_total,
                    fg3a_total = EXCLUDED.fg3a_total,
                    ftm_total = EXCLUDED.ftm_total,
                    fta_total = EXCLUDED.fta_total,
                    team_poss_total = EXCLUDED.team_poss_total,
                    updated_at = NOW()
            """, (row[0], self.current_season, team_id, *row[1:]))
            
            count += 1
            
            if count % 100 == 0:
                self.conn.commit()
                print(f"      {count} players...")
        
        print(f"      {count} players total")
    
    def compute_team_metrics(self):
        """
        STEP 4: Compute team-level metrics
        Target: <10 seconds
        """
        # Aggregate team stats from player box scores
        self.cursor.execute("""
            SELECT 
                team_id,
                COUNT(DISTINCT game_id) AS games,
                SUM(pts) AS pts_total,
                SUM(team_poss) / NULLIF(COUNT(DISTINCT game_id), 0) AS avg_team_poss
            FROM player_box_scores
            WHERE season_id = %s
            GROUP BY team_id
        """, (self.current_season,))
        
        for row in self.cursor.fetchall():
            team_id, games, pts_total, avg_poss = row
            
            # Get opponent points (need to query games where this team played)
            self.cursor.execute("""
                SELECT SUM(pts) 
                FROM player_box_scores
                WHERE season_id = %s 
                  AND opponent_id = %s
            """, (self.current_season, team_id))
            
            opp_pts_result = self.cursor.fetchone()
            opp_pts_total = opp_pts_result[0] if opp_pts_result and opp_pts_result[0] else 0
            
            # Calculate wins/losses (from standings or game results)
            # For now, use placeholder
            wins = 0
            losses = 0
            
            self.cursor.execute("""
                INSERT INTO team_season_stats (
                    team_id, season_id,
                    wins, losses,
                    pts_total, opp_pts_total,
                    poss_total, opp_poss_total
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (team_id, season_id) DO UPDATE SET
                    pts_total = EXCLUDED.pts_total,
                    opp_pts_total = EXCLUDED.opp_pts_total,
                    poss_total = EXCLUDED.poss_total,
                    opp_poss_total = EXCLUDED.opp_poss_total,
                    updated_at = NOW()
            """, (
                team_id, self.current_season,
                wins, losses,
                pts_total, opp_pts_total,
                avg_poss * games, avg_poss * games
            ))
        
        # Calculate Pythagorean expected wins
        self.cursor.execute("SELECT calculate_expected_wins()")
        
        self.conn.commit()
    
    def compute_rapm_and_lebron(self):
        """
        STEP 5: Compute RAPM using Ridge Regression
        Then compute LEBRON = 0.6*RAPM + 0.4*BoxPIPM
        Target: <90 seconds
        """
        print("      Building stint matrix...")
        
        # Get all player IDs
        self.cursor.execute("SELECT player_id FROM players WHERE is_active = TRUE")
        all_players = [row[0] for row in self.cursor.fetchall()]
        player_to_idx = {p: i for i, p in enumerate(all_players)}
        
        # Build matrix (this is simplified - full version needs play-by-play)
        # For now, use box score plus-minus as proxy
        
        X = []  # Stint matrix
        y = []  # Plus-minus per 100
        
        self.cursor.execute("""
            SELECT player_id, plus_minus, team_poss
            FROM player_box_scores
            WHERE season_id = %s AND team_poss > 0
        """, (self.current_season,))
        
        for row in self.cursor.fetchall():
            player_id, pm, poss = row
            if player_id in player_to_idx:
                # Create one-hot encoding for this player
                stint_vec = np.zeros(len(all_players))
                stint_vec[player_to_idx[player_id]] = 1
                
                X.append(stint_vec)
                y.append((pm * 100.0) / poss if poss > 0 else 0)
        
        if len(X) > 100:  # Need enough data
            X = np.array(X)
            y = np.array(y)
            
            # Ridge regression (α = 300)
            print(f"      Running Ridge regression ({len(X)} stints, {len(all_players)} players)...")
            model = Ridge(alpha=300)
            model.fit(X, y)
            
            # Extract RAPM coefficients
            rapm_values = model.coef_
            
            # Update player_season_stats with RAPM
            count = 0
            for player_id, rapm in zip(all_players, rapm_values):
                # Simplified split (real version needs defensive stints)
                rapm_off = rapm * 0.6
                rapm_def = rapm * 0.4
                
                # LEBRON = 0.6*RAPM + 0.4*BoxPIPM
                # BoxPIPM = simplified BPM from box scores
                self.cursor.execute("""
                    SELECT ppg, apg, rpg, stl_total, blk_total, tov_total
                    FROM player_season_stats
                    WHERE player_id = %s AND season_id = %s
                """, (player_id, self.current_season))
                
                result = self.cursor.fetchone()
                if result:
                    ppg, apg, rpg, stl, blk, tov = result
                    
                    # Simplified Box PIPM
                    box_pipm = (
                        ppg * 0.3 + apg * 0.5 + rpg * 0.2 + 
                        stl * 0.3 + blk * 0.2 - tov * 0.2
                    ) / 10
                    
                    # LEBRON
                    lebron = 0.6 * rapm + 0.4 * box_pipm
                    lebron_off = 0.6 * rapm_off + 0.4 * box_pipm * 0.6
                    lebron_def = 0.6 * rapm_def + 0.4 * box_pipm * 0.4
                    
                    self.cursor.execute("""
                        UPDATE player_season_stats
                        SET rapm_offense = %s,
                            rapm_defense = %s,
                            lebron_offense = %s,
                            lebron_defense = %s
                        WHERE player_id = %s AND season_id = %s
                    """, (float(rapm_off), float(rapm_def), float(lebron_off), float(lebron_def), player_id, self.current_season))
                    
                    count += 1
            
            print(f"      ✅ Updated RAPM for {count} players")
            return count
        else:
            print(f"      ⚠️ Not enough data for RAPM ({len(X)} stints)")
            return 0
    
    def prune_and_refresh_last10(self):
        """
        STEP 2: Refresh last10 view (DON'T delete games - keep for ML!)
        """
        # DON'T PRUNE! Keep all games for future ML models
        # Only refresh the materialized view for fast queries
        self.cursor.execute("SELECT refresh_last10()")
        self.conn.commit()
        
        # Optional: Prune games older than 3 years (once per month)
        if datetime.now().day == 1:  # First of month
            print("      🗑️  Pruning games >3 years old...")
            self.cursor.execute("SELECT prune_very_old_games()")
            self.conn.commit()
    
    def update_standings(self):
        """
        Update standings snapshot
        """
        if not NBA_API_AVAILABLE:
            return
        
        time.sleep(0.6)
        standings_df = leaguestandingsv3.LeagueStandingsV3().get_data_frames()[0]
        
        today = datetime.now().date()
        
        for _, row in standings_df.iterrows():
            # Map team by abbreviation
            self.cursor.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (row.get('TeamCity', ''),))
            result = self.cursor.fetchone()
            
            if result:
                team_id = result[0]
                
                self.cursor.execute("""
                    INSERT INTO standings_daily (
                        snapshot_date, team_id, season_id,
                        conference, rank, wins, losses, gb, streak
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (snapshot_date, team_id) DO UPDATE SET
                        rank = EXCLUDED.rank,
                        wins = EXCLUDED.wins,
                        losses = EXCLUDED.losses
                """, (
                    today, team_id, self.current_season,
                    row['Conference'], row['PlayoffRank'],
                    row['WINS'], row['LOSSES'],
                    row.get('GB', 0), row.get('strCurrentStreak', 'N/A')
                ))
    
    def create_snapshot(self, status, games_count, duration):
        """Create daily snapshot"""
        today = datetime.now().date()
        
        self.cursor.execute("SELECT COUNT(DISTINCT player_id) FROM player_box_scores WHERE game_date = %s", (today,))
        players_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            INSERT INTO daily_snapshots (
                snapshot_date, games_processed, players_updated, status, finished_at, duration_seconds
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (snapshot_date) DO UPDATE SET
                games_processed = EXCLUDED.games_processed,
                status = EXCLUDED.status,
                finished_at = EXCLUDED.finished_at
        """, (today, games_count, players_count, status, datetime.now(), int(duration)))
        
        self.conn.commit()
    
    def close(self):
        """Close connection"""
        self.cursor.close()
        self.conn.close()


def run_at_3_30am():
    """
    Main entry point - runs at 3:30 AM UTC
    """
    print("\n" + "="*80)
    print("🌙 NBA NIGHTLY PIPELINE - SELF-COMPUTED METRICS")
    print("="*80)
    print(f"Schedule: 3:30 AM UTC daily")
    print(f"Target: <4 minutes")
    print(f"Source: 100% nba_api")
    print(f"Compute: 100% ourselves")
    print("="*80 + "\n")
    
    if not NBA_API_AVAILABLE:
        print("❌ nba_api not installed!")
        return
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return
    
    pipeline = NBANightlyPipeline()
    
    while True:
        try:
            # Wait until 3:30 AM UTC
            now = datetime.utcnow()
            target = now.replace(hour=3, minute=30, second=0, microsecond=0)
            
            if now > target:
                # If past 3:30 AM, schedule for tomorrow
                target += timedelta(days=1)
            
            wait_seconds = (target - now).total_seconds()
            
            print(f"⏰ Next run: {target.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"💤 Sleeping for {wait_seconds / 3600:.1f} hours...\n")
            
            time.sleep(wait_seconds)
            
            # Run pipeline
            pipeline.run()
            
        except KeyboardInterrupt:
            print("\n⚠️ Pipeline stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(3600)  # Wait 1 hour on error
    
    pipeline.close()


if __name__ == "__main__":
    run_at_3_30am()

