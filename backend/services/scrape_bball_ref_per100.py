"""
BASKETBALL REFERENCE PER-100 STATS SCRAPER
Scrapes REAL per-100 possession stats for 2024-25 season

Basketball Reference calculates possessions accurately and provides:
- Per Game stats
- Per 100 Poss stats (REAL possessions!)
- Advanced stats (PER, WS, BPM, VORP)
- Usage%, TS%, eFG%

URL Structure:
- League Per-100: https://www.basketball-reference.com/leagues/NBA_2025_per_poss.html
- Team Roster: https://www.basketball-reference.com/teams/{TEAM}/2025.html  
- Player Page: https://www.basketball-reference.com/players/{letter}/{playerid}.html
"""

import os
import time
import cloudscraper
from bs4 import BeautifulSoup
import psycopg2
from datetime import datetime
import re


class BballRefPer100Scraper:
    """Scrape Basketball Reference for per-100 possession stats"""
    
    def __init__(self):
        self.base_url = "https://www.basketball-reference.com"
        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'darwin', 'desktop': True}
        )
        
        # BR uses end year, so 2025-26 = 2026
        self.season_year = 2026  
        self.season_id = '2025-26'
        
        # Team abbreviation mapping (BR → Our DB)
        self.team_map = {
            'BRK': 'BKN',  # Brooklyn
            'PHO': 'PHX',  # Phoenix
            'CHO': 'CHA',  # Charlotte
        }
        
        # Database
        DATABASE_URL = os.getenv('DATABASE_URL')
        if not DATABASE_URL:
            raise Exception("DATABASE_URL not set")
        
        self.conn = psycopg2.connect(DATABASE_URL)
        self.cur = self.conn.cursor()
        
        # Load existing players for name matching
        self.cur.execute("SELECT player_id, name, team_id FROM players")
        self.player_map = {row[1].lower(): (row[0], row[2]) for row in self.cur.fetchall()}
        
        print("="*80)
        print("🏀 BASKETBALL REFERENCE PER-100 SCRAPER")
        print("="*80)
        print(f"   Season: 2025-26 (BR year: {self.season_year})")
        print(f"   URL: {self.base_url}/leagues/NBA_{self.season_year}_per_poss.html")
        print(f"   ✅ Connected to PostgreSQL")
        print(f"   🔓 Cloudflare bypass: ENABLED")
        print()
    
    def scrape_league_per100_stats(self):
        """
        Scrape league-wide per-100 possession stats
        URL: https://www.basketball-reference.com/leagues/NBA_2025_per_poss.html
        
        This has REAL per-100 stats calculated by Basketball Reference!
        """
        
        url = f"{self.base_url}/leagues/NBA_{self.season_year}_per_poss.html"
        
        print(f"📊 Scraping per-100 stats from Basketball Reference...")
        print(f"   URL: {url}")
        print()
        
        time.sleep(3)  # Be polite
        response = self.scraper.get(url, timeout=20)
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch page: {response.status_code}")
            return
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the per-100 stats table
        stats_table = soup.find('table', {'id': 'per_poss_stats'})
        
        if not stats_table:
            print("❌ Could not find per_poss_stats table")
            return
        
        # Parse table rows
        tbody = stats_table.find('tbody')
        rows = tbody.find_all('tr', class_=lambda x: x != 'thead')
        
        print(f"✅ Found {len(rows)} player stat rows")
        print()
        
        players_inserted = 0
        
        for row in rows:
            try:
                # Skip header rows
                if row.get('class') and 'thead' in row.get('class'):
                    continue
                
                # Extract player info
                player_cell = row.find('td', {'data-stat': 'player'})
                if not player_cell:
                    continue
                
                player_name = player_cell.text.strip()
                player_link = player_cell.find('a')
                player_url = player_link['href'] if player_link else None
                
                # Extract player ID from URL (e.g., /players/j/jamesle01.html → jamesle01)
                player_br_id = None
                if player_url:
                    match = re.search(r'/players/[a-z]/([^.]+)\.html', player_url)
                    if match:
                        player_br_id = match.group(1)
                
                # Get team
                team_cell = row.find('td', {'data-stat': 'team_name_abbr'})
                team_abbr_raw = team_cell.text.strip() if team_cell else None
                
                # Map team abbreviation
                team_abbr = self.team_map.get(team_abbr_raw, team_abbr_raw)
                
                # Get our team_id from abbreviation
                self.cur.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (team_abbr,))
                team_row = self.cur.fetchone()
                team_id = team_row[0] if team_row else None
                
                if not team_id:
                    # print(f"      ⚠️  Team {team_abbr} not found in database")
                    continue
                
                # Get stats
                stats = {}
                for stat in ['g', 'mp_per_g', 'fg_per_poss', 'fga_per_poss', 'fg_pct', 
                            'fg3_per_poss', 'fg3a_per_poss', 'fg3_pct',
                            'fg2_per_poss', 'fg2a_per_poss', 'fg2_pct',
                            'ft_per_poss', 'fta_per_poss', 'ft_pct',
                            'orb_per_poss', 'drb_per_poss', 'trb_per_poss',
                            'ast_per_poss', 'stl_per_poss', 'blk_per_poss',
                            'tov_per_poss', 'pf_per_poss', 'pts_per_poss',
                            'ts_pct', 'efg_pct']:
                    
                    cell = row.find('td', {'data-stat': stat})
                    if cell and cell.text.strip():
                        try:
                            stats[stat] = float(cell.text.strip())
                        except:
                            stats[stat] = 0
                    else:
                        stats[stat] = 0
                
                # Convert per-possession to per-100
                pts_100 = stats.get('pts_per_poss', 0) * 100
                reb_100 = stats.get('trb_per_poss', 0) * 100
                ast_100 = stats.get('ast_per_poss', 0) * 100
                stl_100 = stats.get('stl_per_poss', 0) * 100
                blk_100 = stats.get('blk_per_poss', 0) * 100
                tov_100 = stats.get('tov_per_poss', 0) * 100
                
                # Get games played and minutes
                games_played = int(stats.get('g', 0))
                mpg = stats.get('mp_per_g', 0)
                
                # Match player by name to get our player_id
                player_name_lower = player_name.lower()
                player_id_matched = None
                
                if player_name_lower in self.player_map:
                    player_id_matched, db_team_id = self.player_map[player_name_lower]
                else:
                    # Try fuzzy match (e.g., "LeBron James" vs "Lebron James")
                    for db_name, (db_id, db_team) in self.player_map.items():
                        if player_name_lower.replace('.', '') == db_name.replace('.', ''):
                            player_id_matched = db_id
                            break
                
                if not player_id_matched:
                    # Player not in our database - create placeholder
                    # print(f"      ⚠️  {player_name} not in database, creating...")
                    
                    # Generate a temporary ID from BR ID
                    temp_id = player_br_id if player_br_id else player_name.replace(' ', '_').lower()[:10]
                    
                    try:
                        self.cur.execute("""
                            INSERT INTO players (player_id, name, team_id, position)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (player_id) DO NOTHING
                        """, (temp_id, player_name, team_id, 'G'))
                        
                        player_id_matched = temp_id
                    except:
                        continue
                
                print(f"   {player_name:25s} {team_abbr:3s}  {pts_100:5.1f} Pts/100  {games_played:2d} GP")
                
                # Store in database with REAL per-100 stats
                try:
                    self.cur.execute("""
                        INSERT INTO player_season_stats (
                            player_id, season_id, team_id,
                            games_played, minutes_total,
                            pts_100, reb_100, ast_100, stl_100, blk_100, tov_100,
                            ts_pct, efg_pct
                        ) VALUES (
                            %s, %s, %s,
                            %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s
                        )
                        ON CONFLICT (player_id, season_id) DO UPDATE SET
                            games_played = EXCLUDED.games_played,
                            minutes_total = EXCLUDED.minutes_total,
                            pts_100 = EXCLUDED.pts_100,
                            reb_100 = EXCLUDED.reb_100,
                            ast_100 = EXCLUDED.ast_100,
                            stl_100 = EXCLUDED.stl_100,
                            blk_100 = EXCLUDED.blk_100,
                            tov_100 = EXCLUDED.tov_100,
                            ts_pct = EXCLUDED.ts_pct,
                            efg_pct = EXCLUDED.efg_pct
                    """, (
                        player_id_matched,
                        self.season_id,
                        team_id,
                        games_played,
                        mpg * games_played if mpg else 0,
                        round(pts_100, 2),
                        round(reb_100, 2),
                        round(ast_100, 2),
                        round(stl_100, 2),
                        round(blk_100, 2),
                        round(tov_100, 2),
                        stats.get('ts_pct', 0),
                        stats.get('efg_pct', 0)
                    ))
                    
                    self.conn.commit()  # Commit after each player to prevent large rollbacks
                    players_inserted += 1
                    
                except Exception as e:
                    print(f"      ⚠️  Error inserting {player_name}: {e}")
                    self.conn.rollback()
                    pass
            
            except Exception as e:
                print(f"   ⚠️  Error parsing row: {e}")
                continue
        
        self.conn.commit()
        
        print()
        print("="*80)
        print(f"✅ SCRAPED {players_inserted} PLAYERS WITH REAL PER-100 STATS")
        print("="*80)
        print()
        print("🎯 These are REAL per-100 possessions from Basketball Reference!")
        print("   (Not estimates - actual NBA possession tracking)")
        
        return players_inserted
    
    def close(self):
        """Close database connection"""
        self.cur.close()
        self.conn.close()


def main():
    """Run the scraper"""
    scraper = BballRefPer100Scraper()
    
    try:
        count = scraper.scrape_league_per100_stats()
        
        # Verify
        scraper.cur = scraper.conn.cursor()
        scraper.cur.execute("SELECT COUNT(*) FROM player_season_stats WHERE season_id = '2025-26' AND pts_100 > 0")
        verified = scraper.cur.fetchone()[0]
        
        print()
        print(f"🔍 Verification: {verified} players with per-100 stats in database")
        
        # Show top 5
        scraper.cur.execute("""
            SELECT player_id, pts_100, reb_100, ast_100, games_played, team_id
            FROM player_season_stats 
            WHERE season_id = '2025-26' AND pts_100 > 0
            ORDER BY pts_100 DESC
            LIMIT 5
        """)
        
        print()
        print("📊 Top 5 Pts/100:")
        for row in scraper.cur.fetchall():
            print(f"   Player {row[0]:20s} {row[5]:3s}  {row[1]:.1f} Pts/100  {row[2]:.1f} Reb  {row[3]:.1f} Ast  ({row[4]} GP)")
        
    finally:
        scraper.close()


if __name__ == "__main__":
    main()

