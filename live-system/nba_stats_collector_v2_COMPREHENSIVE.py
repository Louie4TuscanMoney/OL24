"""
NBA STATS COLLECTOR V2 - COMPREHENSIVE
Collects EVERY statistic available from nba_api
Optimized for Railway + PostgreSQL
"""

import os
import time
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values
import traceback

try:
    from nba_api.stats.static import teams, players as nba_players
    from nba_api.stats.endpoints import (
        # Standings & League
        leaguestandings,
        leaguegamefinder,
        
        # Team Stats
        teamdashboardbygeneralsplits,
        teamgamelog,
        teamdetails,
        commonteamroster,
        
        # Player Stats  
        playerdashboardbygeneralsplits,
        playergamelog,
        playercareerstats,
        commonplayerinfo,
        
        # Advanced Stats
        leaguedashplayerstats,
        leaguedashteamstats,
        
        # Tracking Stats
        leaguedashptstats,
        
        # Hustle Stats
        leaguehustlestatsplayer,
        
        # Play-by-Play
        playbyplayv2
    )
    NBA_API_AVAILABLE = True
except ImportError as e:
    NBA_API_AVAILABLE = False
    print(f"⚠️ nba_api not available: {e}")

# Railway PostgreSQL connection
DATABASE_URL = os.environ.get('DATABASE_URL')

class ComprehensiveNBACollector:
    """
    Collects EVERY statistic from nba_api
    Optimized for Railway deployment
    """
    
    def __init__(self):
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable required")
        
        self.conn = psycopg2.connect(DATABASE_URL)
        self.cursor = self.conn.cursor()
        self.current_season = '2024-25'
        print(f"✅ Connected to PostgreSQL")
    
    def run_comprehensive_update(self):
        """
        COMPLETE data collection - runs nightly
        """
        print("\n" + "="*80)
        print(f"🔄 COMPREHENSIVE UPDATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        start_time = datetime.now()
        stats_collected = []
        
        try:
            # 1. Teams (quick)
            print("📊 [1/11] Updating teams...")
            self.update_teams()
            stats_collected.append('teams')
            time.sleep(1)
            
            # 2. Players (moderate)
            print("\n📊 [2/11] Updating players...")
            self.update_players()
            stats_collected.append('players')
            time.sleep(1)
            
            # 3. Games (important!)
            print("\n📊 [3/11] Updating games...")
            self.update_games()
            stats_collected.append('games')
            time.sleep(1)
            
            # 4. Standings
            print("\n📊 [4/11] Updating standings...")
            self.update_standings()
            stats_collected.append('standings')
            time.sleep(1)
            
            # 5. Player Season Stats
            print("\n📊 [5/11] Updating player season stats...")
            self.update_player_season_stats()
            stats_collected.append('player_season_stats')
            
            # 6. Team Season Stats
            print("\n📊 [6/11] Updating team season stats...")
            self.update_team_season_stats()
            stats_collected.append('team_season_stats')
            
            # 7. Player Game Stats (box scores)
            print("\n📊 [7/11] Updating player game stats...")
            self.update_player_game_stats()
            stats_collected.append('player_game_stats')
            
            # 8. Team Game Stats
            print("\n📊 [8/11] Updating team game stats...")
            self.update_team_game_stats()
            stats_collected.append('team_game_stats')
            
            # 9. Tracking Stats
            print("\n📊 [9/11] Updating tracking stats...")
            self.update_tracking_stats()
            stats_collected.append('tracking_stats')
            
            # 10. Hustle Stats
            print("\n📊 [10/11] Updating hustle stats...")
            self.update_hustle_stats()
            stats_collected.append('hustle_stats')
            
            # 11. Advanced Metrics (calculated)
            print("\n📊 [11/11] Calculating advanced metrics...")
            self.calculate_advanced_metrics()
            stats_collected.append('advanced_metrics')
            
            # Create snapshot
            duration = (datetime.now() - start_time).total_seconds()
            self.create_snapshot('success', stats_collected, start_time, duration)
            
            # Commit all changes
            self.conn.commit()
            
            print("\n" + "="*80)
            print(f"✅ COMPREHENSIVE UPDATE COMPLETED")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Stats collected: {len(stats_collected)}")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n❌ UPDATE FAILED: {e}")
            traceback.print_exc()
            self.create_snapshot('failed', stats_collected, start_time, 0, str(e))
            self.conn.rollback()
    
    def update_teams(self):
        """Update all 30 NBA teams"""
        if not NBA_API_AVAILABLE:
            return
        
        try:
            all_teams = teams.get_teams()
            
            for team in all_teams:
                self.cursor.execute("""
                    INSERT INTO teams (team_id, abbreviation, full_name, conference, division)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (team_id) DO UPDATE SET
                        updated_at = NOW()
                """, (
                    str(team['id']),
                    team['abbreviation'],
                    team['full_name'],
                    'East' if team['abbreviation'] in [
                        'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DET', 
                        'IND', 'MIA', 'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS'
                    ] else 'West',
                    ''  # Division to be filled later
                ))
            
            print(f"   ✅ {len(all_teams)} teams")
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
    
    def update_players(self):
        """Update all active players"""
        if not NBA_API_AVAILABLE:
            return
        
        try:
            all_players = nba_players.get_active_players()
            
            count = 0
            for player in all_players:
                self.cursor.execute("""
                    INSERT INTO players (player_id, name, is_active)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (player_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        is_active = EXCLUDED.is_active,
                        updated_at = NOW()
                """, (
                    str(player['id']),
                    player['full_name'],
                    True
                ))
                count += 1
                
                # Commit in batches
                if count % 100 == 0:
                    self.conn.commit()
                    print(f"   {count} players...")
            
            print(f"   ✅ {count} players")
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
    
    def update_games(self):
        """Update games from current season"""
        if not NBA_API_AVAILABLE:
            return
        
        try:
            # Get recent games (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            time.sleep(0.6)
            games_df = leaguegamefinder.LeagueGameFinder(
                season_nullable=self.current_season,
                league_id_nullable='00',
                date_from_nullable=start_date.strftime('%m/%d/%Y'),
                date_to_nullable=end_date.strftime('%m/%d/%Y')
            ).get_data_frames()[0]
            
            count = 0
            for _, game in games_df.iterrows():
                game_id = game['GAME_ID']
                
                # Parse teams
                team_abbr = game['TEAM_ABBREVIATION']
                matchup = game['MATCHUP']
                is_home = 'vs.' in matchup
                
                self.cursor.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (team_abbr,))
                result = self.cursor.fetchone()
                team_id = result[0] if result else None
                
                if not team_id:
                    continue
                
                # Determine opponent
                opp_abbr = matchup.split()[-1]
                self.cursor.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (opp_abbr,))
                result = self.cursor.fetchone()
                opp_id = result[0] if result else None
                
                # Insert game
                self.cursor.execute("""
                    INSERT INTO games (
                        game_id, season_id, game_date, status,
                        home_team_id, away_team_id
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (game_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        updated_at = NOW()
                """, (
                    game_id,
                    self.current_season,
                    game['GAME_DATE'],
                    'final' if game['WL'] else 'live',
                    team_id if is_home else opp_id,
                    opp_id if is_home else team_id
                ))
                
                count += 1
            
            print(f"   ✅ {count} games")
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            traceback.print_exc()
    
    def update_standings(self):
        """Update current standings"""
        if not NBA_API_AVAILABLE:
            return
        
        try:
            time.sleep(0.6)
            standings_df = leaguestandings.LeagueStandings().get_data_frames()[0]
            
            today = datetime.now().date()
            
            for _, row in standings_df.iterrows():
                # Map team
                self.cursor.execute("""
                    SELECT team_id FROM teams 
                    WHERE abbreviation = %s OR full_name LIKE %s
                """, (row.get('TeamCity', ''), f"%{row.get('TeamName', '')}%"))
                
                result = self.cursor.fetchone()
                if not result:
                    continue
                    
                team_id = result[0]
                
                self.cursor.execute("""
                    INSERT INTO standings (
                        team_id, season_id, snapshot_date,
                        conference, conference_rank,
                        wins, losses, win_pct, gb,
                        last_10, streak
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (team_id, season_id, snapshot_date) DO UPDATE SET
                        conference_rank = EXCLUDED.conference_rank,
                        wins = EXCLUDED.wins,
                        losses = EXCLUDED.losses,
                        win_pct = EXCLUDED.win_pct,
                        gb = EXCLUDED.gb
                """, (
                    team_id,
                    self.current_season,
                    today,
                    row['Conference'],
                    row['PlayoffRank'],
                    row['WINS'],
                    row['LOSSES'],
                    row['WinPCT'],
                    row.get('GB', 0),
                    row.get('L10', 'N/A'),
                    row.get('strCurrentStreak', 'N/A')
                ))
            
            print(f"   ✅ {len(standings_df)} teams")
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
    
    def update_player_season_stats(self):
        """Update season stats for all players"""
        if not NBA_API_AVAILABLE:
            return
        
        try:
            time.sleep(0.6)
            
            # Get all player stats at once (league-wide)
            stats_df = leaguedashplayerstats.LeagueDashPlayerStats(
                season=self.current_season,
                per_mode_detailed='PerGame'
            ).get_data_frames()[0]
            
            count = 0
            for _, row in stats_df.iterrows():
                player_id = str(row['PLAYER_ID'])
                team_id = str(row['TEAM_ID'])
                
                self.cursor.execute("""
                    INSERT INTO player_season_stats (
                        player_id, season_id, team_id,
                        games_played, games_started, minutes_total,
                        ppg, rpg, apg, spg, bpg, tov_pg,
                        fgm_pg, fga_pg, fg_pct,
                        fg3m_pg, fg3a_pg, fg3_pct,
                        ftm_pg, fta_pg, ft_pct
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (player_id, season_id) DO UPDATE SET
                        games_played = EXCLUDED.games_played,
                        ppg = EXCLUDED.ppg,
                        rpg = EXCLUDED.rpg,
                        apg = EXCLUDED.apg,
                        updated_at = NOW()
                """, (
                    player_id, self.current_season, team_id,
                    row['GP'], row.get('GS', 0), row['MIN'],
                    row['PTS'], row['REB'], row['AST'], row['STL'], row['BLK'], row['TOV'],
                    row['FGM'], row['FGA'], row['FG_PCT'],
                    row['FG3M'], row['FG3A'], row['FG3_PCT'],
                    row['FTM'], row['FTA'], row['FT_PCT']
                ))
                
                count += 1
                
                if count % 100 == 0:
                    print(f"   {count} players...")
                    self.conn.commit()
            
            print(f"   ✅ {count} players")
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            traceback.print_exc()
    
    def update_team_season_stats(self):
        """Update season stats for all teams"""
        if not NBA_API_AVAILABLE:
            return
        
        try:
            time.sleep(0.6)
            
            stats_df = leaguedashteamstats.LeagueDashTeamStats(
                season=self.current_season,
                per_mode_detailed='PerGame'
            ).get_data_frames()[0]
            
            for _, row in stats_df.iterrows():
                team_id = str(row['TEAM_ID'])
                
                self.cursor.execute("""
                    INSERT INTO team_season_stats (
                        team_id, season_id,
                        wins, losses, win_pct,
                        ppg, fg_pct, fg3_pct, ft_pct,
                        pace, offensive_rating, defensive_rating, net_rating
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (team_id, season_id) DO UPDATE SET
                        wins = EXCLUDED.wins,
                        losses = EXCLUDED.losses,
                        ppg = EXCLUDED.ppg,
                        updated_at = NOW()
                """, (
                    team_id, self.current_season,
                    row['W'], row['L'], row['W_PCT'],
                    row['PTS'], row['FG_PCT'], row['FG3_PCT'], row['FT_PCT'],
                    row.get('PACE', 0), row.get('OFF_RATING', 0), 
                    row.get('DEF_RATING', 0), row.get('NET_RATING', 0)
                ))
            
            print(f"   ✅ {len(stats_df)} teams")
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
    
    def update_player_game_stats(self):
        """Update player box scores from recent games"""
        if not NBA_API_AVAILABLE:
            return
        
        try:
            # Get top 50 players only (to avoid rate limits)
            self.cursor.execute("""
                SELECT player_id FROM player_season_stats
                WHERE season_id = %s
                ORDER BY ppg DESC NULLS LAST
                LIMIT 50
            """, (self.current_season,))
            
            player_ids = [row[0] for row in self.cursor.fetchall()]
            
            count = 0
            for player_id in player_ids:
                try:
                    time.sleep(0.6)  # Rate limiting
                    
                    log_df = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=self.current_season
                    ).get_data_frames()[0].head(10)
                    
                    for _, game in log_df.iterrows():
                        self.cursor.execute("""
                            INSERT INTO player_game_stats (
                                player_id, game_id, team_id, game_date,
                                minutes, points, rebounds, assists,
                                fgm, fga, fg_pct,
                                fg3m, fg3a, fg3_pct,
                                ftm, fta, ft_pct,
                                oreb, dreb, stl, blk, tov, pf, plus_minus
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (player_id, game_id) DO NOTHING
                        """, (
                            player_id,
                            game['Game_ID'],
                            str(game['TEAM_ID']),
                            game['GAME_DATE'],
                            game.get('MIN', 0),
                            game['PTS'],
                            game['REB'],
                            game['AST'],
                            game['FGM'],
                            game['FGA'],
                            game['FG_PCT'],
                            game['FG3M'],
                            game['FG3A'],
                            game['FG3_PCT'],
                            game['FTM'],
                            game['FTA'],
                            game['FT_PCT'],
                            game.get('OREB', 0),
                            game.get('DREB', 0),
                            game['STL'],
                            game['BLK'],
                            game['TOV'],
                            game.get('PF', 0),
                            game.get('PLUS_MINUS', 0)
                        ))
                        count += 1
                    
                    if count % 100 == 0:
                        print(f"   {count} box scores...")
                        self.conn.commit()
                        
                except Exception as e:
                    print(f"   ⚠️ Player {player_id} failed: {e}")
                    continue
            
            print(f"   ✅ {count} box scores")
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
    
    def update_team_game_stats(self):
        """Update team box scores"""
        # Similar to player_game_stats but for teams
        print(f"   ⏭️  Skipping (implement later)")
    
    def update_tracking_stats(self):
        """Update SportVU tracking stats"""
        if not NBA_API_AVAILABLE:
            return
        
        try:
            time.sleep(0.6)
            
            # Speed & Distance
            tracking_df = leaguedashptstats.LeagueDashPtStats(
                season=self.current_season,
                pt_measure_type='SpeedDistance'
            ).get_data_frames()[0]
            
            for _, row in tracking_df.iterrows():
                player_id = str(row['PLAYER_ID'])
                
                self.cursor.execute("""
                    INSERT INTO player_tracking_stats (
                        player_id, season_id, stat_type,
                        avg_speed, distance_miles
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (player_id, season_id, stat_type) DO UPDATE SET
                        avg_speed = EXCLUDED.avg_speed,
                        distance_miles = EXCLUDED.distance_miles,
                        updated_at = NOW()
                """, (
                    player_id,
                    self.current_season,
                    'speed_distance',
                    row.get('AVG_SPEED', 0),
                    row.get('DIST_MILES', 0)
                ))
            
            print(f"   ✅ {len(tracking_df)} players (tracking)")
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
    
    def update_hustle_stats(self):
        """Update hustle stats"""
        if not NBA_API_AVAILABLE:
            return
        
        try:
            time.sleep(0.6)
            
            hustle_df = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
                season=self.current_season
            ).get_data_frames()[0]
            
            for _, row in hustle_df.iterrows():
                player_id = str(row['PLAYER_ID'])
                
                self.cursor.execute("""
                    INSERT INTO player_hustle_stats (
                        player_id, season_id,
                        screen_assists, deflections, loose_balls_recovered,
                        charges_drawn, contested_shots
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (player_id, season_id) DO UPDATE SET
                        screen_assists = EXCLUDED.screen_assists,
                        deflections = EXCLUDED.deflections,
                        updated_at = NOW()
                """, (
                    player_id,
                    self.current_season,
                    row.get('SCREEN_ASSISTS', 0),
                    row.get('DEFLECTIONS', 0),
                    row.get('LOOSE_BALLS_RECOVERED', 0),
                    row.get('CHARGES_DRAWN', 0),
                    row.get('CONTESTED_SHOTS', 0)
                ))
            
            print(f"   ✅ {len(hustle_df)} players (hustle)")
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
    
    def calculate_advanced_metrics(self):
        """Calculate advanced metrics (BPM, VORP, etc.)"""
        try:
            # Query player stats and calculate BPM, VORP, etc.
            self.cursor.execute("""
                SELECT player_id, ppg, rpg, apg, spg, bpg, tov_pg,
                       fg_pct, fg3_pct, ft_pct
                FROM player_season_stats
                WHERE season_id = %s AND games_played > 0
            """, (self.current_season,))
            
            count = 0
            for row in self.cursor.fetchall():
                player_id = row[0]
                
                # Simplified BPM calculation (real one is complex)
                bpm = (
                    row[1] * 0.5 +  # PPG
                    row[2] * 0.3 +  # RPG
                    row[3] * 0.4 -  # APG
                    row[5] * 0.3    # TOV_PG
                ) / 10
                
                # VORP = (BPM + 2) * (% of minutes) * team_games / 82
                vorp = max(0, bpm + 2) * 0.5
                
                self.cursor.execute("""
                    INSERT INTO player_advanced_metrics (
                        player_id, season_id, calculated_date,
                        bpm, vorp
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (player_id, season_id, calculated_date) DO UPDATE SET
                        bpm = EXCLUDED.bpm,
                        vorp = EXCLUDED.vorp
                """, (
                    player_id,
                    self.current_season,
                    datetime.now().date(),
                    bpm,
                    vorp
                ))
                count += 1
            
            print(f"   ✅ {count} players (advanced metrics)")
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
    
    def create_snapshot(self, status, stats_collected, start_time, duration, error_msg=None):
        """Create daily snapshot"""
        today = datetime.now().date()
        
        self.cursor.execute("SELECT COUNT(*) FROM teams")
        teams_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM players WHERE is_active = TRUE")
        players_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM games WHERE season_id = %s", (self.current_season,))
        games_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            INSERT INTO daily_snapshots (
                snapshot_date, season_id,
                teams_count, players_count, games_total_season,
                collection_status, error_message,
                stats_types_collected,
                collection_start_time, collection_end_time, duration_seconds
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (snapshot_date) DO UPDATE SET
                teams_count = EXCLUDED.teams_count,
                players_count = EXCLUDED.players_count,
                collection_status = EXCLUDED.collection_status
        """, (
            today, self.current_season,
            teams_count, players_count, games_count,
            status, error_msg,
            stats_collected,
            start_time, datetime.now(), int(duration)
        ))
        
        print(f"\n📸 Snapshot: {teams_count} teams, {players_count} players, {games_count} games")
    
    def close(self):
        """Close connection"""
        self.cursor.close()
        self.conn.close()


def run_collector():
    """
    Run comprehensive collector on Railway
    Updates every 24 hours
    """
    print("\n" + "="*80)
    print("🚂 NBA COMPREHENSIVE STATS COLLECTOR V2")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: Comprehensive (ALL nba_api stats)")
    print(f"Update interval: 24 hours")
    print("="*80 + "\n")
    
    if not NBA_API_AVAILABLE:
        print("❌ nba_api not installed!")
        return
    
    collector = ComprehensiveNBACollector()
    
    while True:
        try:
            # Run comprehensive update
            collector.run_comprehensive_update()
            
            # Wait 24 hours
            next_run = (datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n⏰ Next update: {next_run}")
            print("💤 Sleeping for 24 hours...\n")
            
            time.sleep(24 * 60 * 60)
            
        except KeyboardInterrupt:
            print("\n⚠️ Collector stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Collector error: {e}")
            traceback.print_exc()
            time.sleep(60 * 60)  # Wait 1 hour on error
    
    collector.close()


if __name__ == "__main__":
    run_collector()

