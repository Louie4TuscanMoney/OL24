"""
COMPREHENSIVE NBA ANALYTICS SYSTEM
- Advanced Stats Calculator
- Injury Tracker
- Depth Chart Builder
- Schedule Manager
- All for ML + Frontend
"""

import os
import time
import psycopg2
from datetime import datetime, timedelta
import cloudscraper
from bs4 import BeautifulSoup

try:
    from nba_api.stats.endpoints import leaguegamefinder
    from nba_api.live.nba.endpoints import scoreboard
    NBA_API_AVAILABLE = True
except:
    NBA_API_AVAILABLE = False

DATABASE_URL = os.environ.get('DATABASE_URL')

class ComprehensiveNBASystem:
    """
    Complete system for NBA analytics
    """
    
    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_URL)
        self.cursor = self.conn.cursor()
        self.scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'darwin', 'desktop': True})
        self.current_season = '2025-26'
        print(f"✅ Comprehensive NBA System initialized")
    
    # ========================================================================
    # ADVANCED STATS CALCULATOR
    # ========================================================================
    
    def compute_advanced_stats(self):
        """
        Advanced stats are AUTO-COMPUTED by PostgreSQL GENERATED columns!
        This function just verifies they exist.
        """
        print("\n🧮 Verifying advanced stats...")
        
        # Check that player_season_stats has data
        self.cursor.execute("""
            SELECT COUNT(*) FROM player_season_stats 
            WHERE season_id = %s AND pts_100 > 0
        """, (self.current_season,))
        
        count = self.cursor.fetchone()[0]
        
        # Show sample stats
        self.cursor.execute("""
            SELECT name, ppg, pts_100, ts_pct, efg_pct
            FROM player_season_stats ps
            JOIN players p ON p.player_id = ps.player_id
            WHERE ps.season_id = %s
            ORDER BY ppg DESC
            LIMIT 3
        """, (self.current_season,))
        
        print(f"   ✅ {count} players with advanced stats")
        print(f"\n   📊 Sample (top 3 scorers):")
        for row in self.cursor.fetchall():
            name, ppg, pts_100, ts, efg = row
            print(f"      {name}: {ppg:.1f} PPG | {pts_100:.1f} per-100 | TS%: {ts:.1%} | eFG%: {efg:.1%}")
        
        return count
    
    # ========================================================================
    # INJURY TRACKER
    # ========================================================================
    
    def scrape_injuries(self):
        """
        Scrape current injuries from Basketball Reference
        """
        print("\n🏥 Scraping injury reports...")
        
        url = "https://www.basketball-reference.com/friv/injuries.fcgi"
        
        try:
            time.sleep(2)
            response = self.scraper.get(url, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find injury table
            table = soup.find('table', {'id': 'injuries'})
            if not table:
                print("   ⚠️  No injury table found")
                return 0
            
            tbody = table.find('tbody')
            if not tbody:
                return 0
            
            rows = tbody.find_all('tr')
            injuries_added = 0
            
            for row in rows:
                try:
                    # Get player link
                    player_th = row.find('th', {'data-stat': 'player'})
                    if not player_th:
                        continue
                    
                    player_link = player_th.find('a')
                    if not player_link:
                        continue
                    
                    player_id = player_link['href'].split('/')[-1].replace('.html', '')
                    player_name = player_link.text.strip()
                    
                    # Get team
                    team_td = row.find('td', {'data-stat': 'team_name'})
                    team_abbr = team_td.text.strip() if team_td else None
                    
                    # Get injury description
                    desc_td = row.find('td', {'data-stat': 'note'})
                    description = desc_td.text.strip() if desc_td else ''
                    
                    # Parse status and injury type from description
                    status = 'Out'
                    injury_type = description
                    
                    if 'day-to-day' in description.lower():
                        status = 'Day-To-Day'
                    elif 'questionable' in description.lower():
                        status = 'Questionable'
                    elif 'probable' in description.lower():
                        status = 'Probable'
                    elif 'gtd' in description.lower():
                        status = 'GTD'
                    
                    # Check if player exists, skip if not
                    self.cursor.execute("SELECT player_id FROM players WHERE player_id = %s", (player_id,))
                    if not self.cursor.fetchone():
                        continue  # Skip players not in our database yet
                    
                    # Insert/update injury
                    self.cursor.execute("""
                        INSERT INTO player_injuries (
                            player_id, injury_date, status, injury_type, description, is_active
                        ) VALUES (%s, %s, %s, %s, %s, TRUE)
                        ON CONFLICT (player_id, injury_date) DO UPDATE SET
                            status = EXCLUDED.status,
                            description = EXCLUDED.description,
                            updated_at = NOW()
                    """, (player_id, datetime.now().date(), status, injury_type[:100], description))
                    
                    injuries_added += 1
                    
                except Exception as e:
                    print(f"      ⚠️  Row failed: {e}")
                    continue
            
            self.conn.commit()
            print(f"   ✅ Added {injuries_added} injury reports")
            
            return injuries_added
            
        except Exception as e:
            print(f"   ❌ Injury scraping failed: {e}")
            return 0
    
    # ========================================================================
    # SCHEDULE MANAGER
    # ========================================================================
    
    def fetch_nba_schedule(self, days_ahead=30):
        """
        Fetch NBA schedule using NBA_API
        """
        if not NBA_API_AVAILABLE:
            print("   ⚠️  NBA_API not available")
            return 0
        
        print(f"\n📅 Fetching NBA schedule ({days_ahead} days ahead)...")
        
        try:
            # Get scoreboard for upcoming days
            games_added = 0
            
            for i in range(days_ahead):
                date = datetime.now() + timedelta(days=i)
                
                try:
                    time.sleep(0.6)  # Rate limiting
                    board = scoreboard.ScoreBoard()
                    games_data = board.get_dict()
                    
                    if not games_data or 'scoreboard' not in games_data:
                        continue
                    
                    games = games_data['scoreboard'].get('games', [])
                    
                    for game in games:
                        game_id = game.get('gameId', '')
                        game_date_str = game.get('gameTimeUTC', '')
                        home_team = game.get('homeTeam', {})
                        away_team = game.get('awayTeam', {})
                        
                        # Parse game date
                        try:
                            game_dt = datetime.strptime(game_date_str, '%Y-%m-%dT%H:%M:%SZ')
                            game_date = game_dt.date()
                            game_time = game_dt.time()
                        except:
                            game_date = date.date()
                            game_time = None
                        
                        # Get team IDs
                        home_team_abbr = home_team.get('teamTricode', '')
                        away_team_abbr = away_team.get('teamTricode', '')
                        
                        # Map to our team IDs
                        self.cursor.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (home_team_abbr,))
                        home_result = self.cursor.fetchone()
                        home_team_id = home_result[0] if home_result else None
                        
                        self.cursor.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (away_team_abbr,))
                        away_result = self.cursor.fetchone()
                        away_team_id = away_result[0] if away_result else None
                        
                        if not home_team_id or not away_team_id:
                            continue
                        
                        # Insert schedule
                        self.cursor.execute("""
                            INSERT INTO nba_schedule (
                                game_id, game_date, game_time, home_team_id, away_team_id, 
                                season_id, game_status
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (game_id) DO UPDATE SET
                                game_status = EXCLUDED.game_status,
                                updated_at = NOW()
                        """, (game_id, game_date, game_time, home_team_id, away_team_id, self.current_season, 'Scheduled'))
                        
                        games_added += 1
                    
                except Exception as e:
                    print(f"      ⚠️  Date {date.date()} failed: {e}")
                    continue
            
            self.conn.commit()
            print(f"   ✅ Added {games_added} scheduled games")
            
            return games_added
            
        except Exception as e:
            print(f"   ❌ Schedule fetch failed: {e}")
            return 0
    
    # ========================================================================
    # DEPTH CHART BUILDER
    # ========================================================================
    
    def build_depth_charts(self):
        """
        Build depth charts based on MPG from recent games
        Detect starters and assign depth rankings
        """
        print("\n📊 Building depth charts...")
        
        # For each team, analyze players by position and MPG
        self.cursor.execute("SELECT team_id, abbreviation FROM teams")
        teams = self.cursor.fetchall()
        
        total_depth_entries = 0
        
        for team_id, team_abbr in teams:
            # Get players on this team with their recent MPG
            self.cursor.execute("""
                SELECT 
                    p.player_id,
                    p.name,
                    p.position,
                    AVG(pb.minutes) as avg_mpg,
                    COUNT(*) as games
                FROM players p
                JOIN player_box_scores pb ON pb.player_id = p.player_id
                WHERE p.team_id = %s 
                  AND pb.season_id = %s
                  AND pb.minutes > 0
                GROUP BY p.player_id, p.name, p.position
                HAVING COUNT(*) >= 2
                ORDER BY AVG(pb.minutes) DESC
            """, (team_id, self.current_season))
            
            players = self.cursor.fetchall()
            
            # Determine starters (top 5 by MPG)
            starters = players[:5] if len(players) >= 5 else players
            
            # Assign positions and depth
            position_depth = {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1}
            
            for player_id, name, position, avg_mpg, games in players:
                # Use player's position, default to 'SF' if unknown
                pos = position if position in ['PG', 'SG', 'SF', 'PF', 'C'] else 'SF'
                
                is_starter = player_id in [p[0] for p in starters]
                depth_rank = position_depth.get(pos, 1)
                
                # Insert depth chart entry
                self.cursor.execute("""
                    INSERT INTO team_depth_charts (
                        team_id, player_id, position, depth_rank, avg_mpg, is_starter, season_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (team_id, player_id, position, season_id) DO UPDATE SET
                        depth_rank = EXCLUDED.depth_rank,
                        avg_mpg = EXCLUDED.avg_mpg,
                        is_starter = EXCLUDED.is_starter,
                        last_updated = NOW()
                """, (team_id, player_id, pos, depth_rank, float(avg_mpg), is_starter, self.current_season))
                
                position_depth[pos] += 1
                total_depth_entries += 1
        
        self.conn.commit()
        print(f"   ✅ Built depth charts for 30 teams ({total_depth_entries} entries)")
        
        return total_depth_entries
    
    def close(self):
        self.cursor.close()
        self.conn.close()


def main():
    """
    Run all systems
    """
    print("\n" + "="*80)
    print("🏀 COMPREHENSIVE NBA ANALYTICS SYSTEM")
    print("="*80 + "\n")
    
    system = ComprehensiveNBASystem()
    
    # Run all components
    system.compute_advanced_stats()
    system.scrape_injuries()
    system.fetch_nba_schedule(days_ahead=30)
    system.build_depth_charts()
    
    system.close()
    
    print("\n" + "="*80)
    print("✅ COMPLETE SYSTEM UPDATED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

