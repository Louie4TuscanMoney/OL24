"""
NBA STATS COLLECTOR - ROLLING MODEL
Runs continuously on Railway, updates PostgreSQL daily
Collects: Teams, Players, Season Stats, Standings
"""

import os
import time
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values

try:
    from nba_api.stats.static import teams, players as nba_players
    from nba_api.stats.endpoints import leaguestandings
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("⚠️ nba_api not available")

# Railway PostgreSQL connection
DATABASE_URL = os.environ.get('DATABASE_URL')

class NBAStatsCollector:
    """
    Rolling NBA stats collector for Railway + PostgreSQL
    """
    
    def __init__(self):
        if not DATABASE_URL:
            print("❌ DATABASE_URL not set!")
            raise ValueError("DATABASE_URL environment variable required")
        
        self.conn = psycopg2.connect(DATABASE_URL)
        self.cursor = self.conn.cursor()
        print(f"✅ Connected to PostgreSQL")
        print(f"   Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'local'}")
    
    def run_daily_update(self):
        """
        Main update - runs once per day
        """
        print("\n" + "="*80)
        print(f"🔄 DAILY UPDATE STARTING: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        try:
            # 1. Update teams (quick, rarely changes)
            self.update_teams()
            
            # 2. Update players (roster changes)
            self.update_players()
            
            # 3. Update standings
            self.update_standings()
            
            # 4. Create snapshot
            self.create_snapshot('success')
            
            self.conn.commit()
            
            print("\n" + "="*80)
            print(f"✅ DAILY UPDATE COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n❌ UPDATE FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            self.create_snapshot('failed', str(e))
            self.conn.rollback()
    
    def update_teams(self):
        """
        Update all 30 NBA teams
        """
        print("📊 Updating teams...")
        
        if not NBA_API_AVAILABLE:
            print("⚠️ NBA API not available, using database teams")
            return
        
        try:
            all_teams = teams.get_teams()
            
            # Bulk upsert
            team_data = [
                (
                    str(team['id']),
                    team['abbreviation'],
                    team['full_name'],
                    'East' if team['abbreviation'] in ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DET', 'IND', 'MIA', 'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS'] else 'West',
                    team.get('city', '')
                )
                for team in all_teams
            ]
            
            execute_values(
                self.cursor,
                """
                INSERT INTO teams (team_id, abbreviation, full_name, conference, city)
                VALUES %s
                ON CONFLICT (team_id) DO UPDATE SET
                    abbreviation = EXCLUDED.abbreviation,
                    full_name = EXCLUDED.full_name,
                    updated_at = NOW()
                """,
                team_data
            )
            
            print(f"✅ Updated {len(all_teams)} teams")
            
        except Exception as e:
            print(f"⚠️ Teams update failed: {e}")
    
    def update_players(self):
        """
        Update all active NBA players
        """
        print("📊 Updating players...")
        
        if not NBA_API_AVAILABLE:
            print("⚠️ NBA API not available")
            return
        
        try:
            all_players = nba_players.get_active_players()
            
            # Prepare player data
            player_data = []
            for player in all_players[:100]:  # Limit to 100 for now to avoid rate limits
                # Get team abbreviation (simplified)
                team_abbr = player.get('team_abbr', '')
                
                # Get team_id from abbreviation
                self.cursor.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (team_abbr,))
                result = self.cursor.fetchone()
                team_id = result[0] if result else None
                
                player_data.append((
                    str(player['id']),
                    player['full_name'],
                    team_id,
                    True
                ))
            
            # Bulk upsert
            execute_values(
                self.cursor,
                """
                INSERT INTO players (player_id, name, team_id, is_active)
                VALUES %s
                ON CONFLICT (player_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    team_id = EXCLUDED.team_id,
                    is_active = EXCLUDED.is_active,
                    updated_at = NOW()
                """,
                player_data
            )
            
            print(f"✅ Updated {len(player_data)} players")
            
        except Exception as e:
            print(f"⚠️ Players update failed: {e}")
            import traceback
            traceback.print_exc()
    
    def update_standings(self):
        """
        Update NBA standings
        """
        print("📊 Updating standings...")
        
        if not NBA_API_AVAILABLE:
            print("⚠️ NBA API not available")
            return
        
        try:
            # Rate limiting
            time.sleep(0.6)
            
            standings_data = leaguestandings.LeagueStandings().get_data_frames()[0]
            
            today = datetime.now().date()
            
            for _, row in standings_data.iterrows():
                # Get team_id from team abbreviation
                team_abbr = row['TeamCity'].split()[-1] if 'TeamCity' in row else row['TeamName']
                
                self.cursor.execute("SELECT team_id FROM teams WHERE abbreviation LIKE %s", (f"%{team_abbr}%",))
                result = self.cursor.fetchone()
                
                if result:
                    team_id = result[0]
                    
                    self.cursor.execute("""
                        INSERT INTO standings (
                            team_id, conference, rank, wins, losses, gb,
                            home_record, away_record, last_10, streak
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (team_id, (updated_at::date))
                        DO UPDATE SET
                            rank = EXCLUDED.rank,
                            wins = EXCLUDED.wins,
                            losses = EXCLUDED.losses,
                            gb = EXCLUDED.gb
                    """, (
                        team_id,
                        row['Conference'],
                        row['PlayoffRank'],
                        row['WINS'],
                        row['LOSSES'],
                        row['WinPCT'],
                        row.get('HOME', 'N/A'),
                        row.get('ROAD', 'N/A'),
                        row.get('L10', 'N/A'),
                        row.get('strCurrentStreak', 'N/A')
                    ))
            
            print(f"✅ Updated standings for {len(standings_data)} teams")
            
        except Exception as e:
            print(f"⚠️ Standings update failed: {e}")
            import traceback
            traceback.print_exc()
    
    def create_snapshot(self, status='success', error_msg=None):
        """
        Create daily snapshot record
        """
        today = datetime.now().date()
        
        self.cursor.execute("SELECT COUNT(*) FROM teams")
        teams_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM players WHERE is_active = TRUE")
        players_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            INSERT INTO daily_snapshots (
                snapshot_date, teams_count, players_count, collection_status, error_message
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (snapshot_date) DO UPDATE SET
                teams_count = EXCLUDED.teams_count,
                players_count = EXCLUDED.players_count,
                collection_status = EXCLUDED.collection_status,
                error_message = EXCLUDED.error_message
        """, (today, teams_count, players_count, status, error_msg))
        
        print(f"📸 Snapshot: {teams_count} teams, {players_count} players [{status}]")
    
    def close(self):
        """Close database connection"""
        self.cursor.close()
        self.conn.close()


def run_continuous_collector():
    """
    Run collector continuously on Railway
    Updates every 24 hours at midnight
    """
    print("\n" + "="*80)
    print("🚂 NBA STATS COLLECTOR - RAILWAY")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: Rolling (continuous daily updates)")
    print(f"Update interval: 24 hours")
    print("="*80 + "\n")
    
    collector = NBAStatsCollector()
    
    while True:
        try:
            # Run daily update
            collector.run_daily_update()
            
            # Wait 24 hours
            next_run = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"⏰ Next update in 24 hours (approximately {next_run})")
            print("💤 Sleeping...\n")
            
            time.sleep(24 * 60 * 60)  # 24 hours
            
        except KeyboardInterrupt:
            print("\n⚠️ Collector stopped by user")
            break
            
        except Exception as e:
            print(f"\n❌ Collector error: {e}")
            import traceback
            traceback.print_exc()
            
            # Wait 1 hour before retrying on error
            print("⏰ Retrying in 1 hour...")
            time.sleep(60 * 60)
    
    collector.close()


if __name__ == "__main__":
    if not NBA_API_AVAILABLE:
        print("❌ nba_api not installed! Install with: pip install nba_api")
        exit(1)
    
    run_continuous_collector()

