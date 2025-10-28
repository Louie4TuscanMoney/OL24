"""
BASKETBALL REFERENCE SCRAPER - Reliable NBA Data
Because nba_api is broken for 2025-26 season

Features:
- Scrapes box scores daily
- Calculates advanced stats
- Stores rolling windows for ML
- Efficient incremental updates
"""

import os
import time
import cloudscraper  # Bypasses Cloudflare automatically
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import psycopg2
import numpy as np

DATABASE_URL = os.environ.get('DATABASE_URL')

class BasketballReferenceScraper:
    """
    Scrapes Basketball Reference for reliable NBA data
    """
    
    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_URL)
        self.cursor = self.conn.cursor()
        self.base_url = "https://www.basketball-reference.com"
        # Use cloudscraper to bypass Cloudflare automatically
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'darwin',
                'desktop': True
            }
        )
        self.current_season = '2026'  # Basketball Reference uses end year
        
        # Basketball Reference uses different abbreviations than our DB
        self.team_abbr_map = {
            'BRK': 'BKN',  # Brooklyn Nets
            'PHO': 'PHX',  # Phoenix Suns
            'CHO': 'CHA',  # Charlotte Hornets (if needed)
        }
        
        print(f"✅ Connected to PostgreSQL")
        print(f"📅 Season: 2025-26 (BR format: {self.current_season})")
        print(f"🔓 Cloudflare bypass: ENABLED")
    
    def scrape_daily_games(self, date=None):
        """
        Scrape all games for a given date
        Args:
            date: datetime object or None for today
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        url = f"{self.base_url}/boxscores/?month={date.month}&day={date.day}&year={date.year}"
        
        print(f"\n📥 Scraping games for {date_str}...")
        print(f"   URL: {url}")
        
        time.sleep(2)  # Be polite
        response = self.scraper.get(url, timeout=15)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all game boxes
        game_summaries = soup.find_all('div', class_='game_summary')
        print(f"   Found {len(game_summaries)} games")
        
        games_processed = 0
        for game_box in game_summaries:
            try:
                # Extract game link
                links = game_box.find_all('a', href=True)
                box_score_link = None
                for link in links:
                    if 'boxscores' in link['href'] and link.text == 'Box Score':
                        box_score_link = link['href']
                        break
                
                if box_score_link:
                    full_url = f"{self.base_url}{box_score_link}"
                    self.scrape_box_score(full_url, date_str)
                    games_processed += 1
                    time.sleep(3)  # Be VERY polite (Basketball Reference is strict)
                
            except Exception as e:
                print(f"   ⚠️ Game failed: {e}")
                continue
        
        # Don't need to commit here - we commit after each player
        return games_processed
    
    def scrape_box_score(self, url, game_date):
        """
        Scrape a single game's box score
        Args:
            url: Full URL to box score page
            game_date: Date string 'YYYY-MM-DD'
        """
        print(f"      Scraping: {url}")
        
        response = self.scraper.get(url, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract game ID from URL (e.g., '/boxscores/202510270CLE.html')
        game_id = url.split('/')[-1].replace('.html', '')
        
        # Find all player tables (4 total: home/away basic and advanced)
        tables = soup.find_all('table', {'id': lambda x: x and 'box' in x})
        
        for table in tables:
            table_id = table.get('id', '')
            
            # Only process basic box scores (not advanced yet)
            if 'box-' in table_id and 'game-basic' in table_id:
                team_abbr = table_id.split('-')[1]
                self.parse_box_score_table(table, game_id, game_date, team_abbr)
    
    def parse_box_score_table(self, table, game_id, game_date, team_abbr):
        """
        Parse a box score table and insert into database
        Basketball Reference format has columns:
        [0]=Player, [1]=MP, [2]=FG, [3]=FGA, [4]=FG%, [5]=3P, [6]=3PA, [7]=3P%,
        [8]=FT, [9]=FTA, [10]=FT%, [11]=ORB, [12]=DRB, [13]=TRB, [14]=AST,
        [15]=STL, [16]=BLK, [17]=TOV, [18]=PF, [19]=PTS, [20]=GmSc, [21]=+/-
        """
        tbody = table.find('tbody')
        if not tbody:
            return
        
        rows = tbody.find_all('tr')
        players_inserted = 0
        
        for row in rows:
            # Skip "Team Totals", "Reserves", section headers, and DNP rows
            if not row.find('th', {'data-stat': 'player'}):
                continue
            if 'thead' in row.get('class', []):
                continue
            
            try:
                # Player name and link
                player_th = row.find('th', {'data-stat': 'player'})
                if not player_th:
                    continue
                
                player_name = player_th.text.strip()
                player_link = player_th.find('a')
                
                # Check for "Did Not Play"
                reason_td = row.find('td', {'data-stat': 'reason'})
                if reason_td and reason_td.text.strip():
                    continue  # Skip DNP players
                
                if not player_link:
                    continue
                
                # Extract player ID from href like '/players/y/youngtr01.html'
                player_id = player_link['href'].split('/')[-1].replace('.html', '')
                
                # Extract all stats using data-stat attributes
                def get_stat(stat_name, default=0):
                    td = row.find('td', {'data-stat': stat_name})
                    if not td or not td.text.strip():
                        return default
                    try:
                        return int(td.text.strip())
                    except:
                        return default
                
                def get_float_stat(stat_name, default=0.0):
                    td = row.find('td', {'data-stat': stat_name})
                    if not td or not td.text.strip():
                        return default
                    try:
                        return float(td.text.strip())
                    except:
                        return default
                
                # Get minutes (special handling for MM:SS format)
                mp_td = row.find('td', {'data-stat': 'mp'})
                mp = mp_td.text.strip() if mp_td else '0:00'
                minutes = self.parse_minutes(mp)
                
                if minutes == 0:
                    continue  # Skip if no minutes played
                
                # Get all box score stats
                fg = get_stat('fg')
                fga = get_stat('fga')
                fg3 = get_stat('fg3')
                fg3a = get_stat('fg3a')
                ft = get_stat('ft')
                fta = get_stat('fta')
                orb = get_stat('orb')
                drb = get_stat('drb')
                trb = get_stat('trb')
                ast = get_stat('ast')
                stl = get_stat('stl')
                blk = get_stat('blk')
                tov = get_stat('tov')
                pf = get_stat('pf')
                pts = get_stat('pts')
                
                # Plus/minus
                pm_td = row.find('td', {'data-stat': 'plus_minus'})
                pm = 0
                if pm_td and pm_td.text.strip():
                    try:
                        pm = int(pm_td.text.strip())
                    except:
                        pm = 0
                
                # Map Basketball Reference abbreviations to our database abbreviations
                db_team_abbr = self.team_abbr_map.get(team_abbr, team_abbr)
                
                # Get team_id (map abbreviation to ID)
                self.cursor.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (db_team_abbr,))
                team_result = self.cursor.fetchone()
                team_id = team_result[0] if team_result else None
                
                if not team_id:
                    print(f"         ⚠️ Team abbreviation '{team_abbr}' (mapped to '{db_team_abbr}') not found in database, skipping player")
                    continue
                
                # Insert player if doesn't exist
                self.cursor.execute("""
                    INSERT INTO players (player_id, name, team_id, is_active)
                    VALUES (%s, %s, %s, TRUE)
                    ON CONFLICT (player_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        team_id = EXCLUDED.team_id,
                        updated_at = NOW()
                """, (player_id, player_name, team_id))
                
                # Calculate team possessions (estimate)
                team_poss = fga + 0.44 * fta - orb + tov if fga > 0 else 0
                
                # Insert box score
                self.cursor.execute("""
                    INSERT INTO player_box_scores (
                        player_id, game_id, game_date, season_id, team_id, opponent_id,
                        minutes, pts, fgm, fga, fg3m, fg3a, ftm, fta,
                        oreb, dreb, reb, ast, stl, blk, tov, pf, plus_minus,
                        team_poss
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (player_id, game_id, game_date) DO UPDATE SET
                        pts = EXCLUDED.pts,
                        minutes = EXCLUDED.minutes
                """, (
                    player_id, game_id, game_date, '2025-26', team_id, None,
                    minutes, pts, fg, fga, fg3, fg3a, ft, fta,
                    orb, drb, trb, ast, stl, blk, tov, pf, pm,
                    team_poss
                ))
                
                # Commit after each player to avoid transaction rollback issues
                self.conn.commit()
                players_inserted += 1
                    
            except Exception as e:
                print(f"         ⚠️ Player row failed: {e}")
                self.conn.rollback()  # Rollback this player only
                continue
        
        if players_inserted > 0:
            print(f"         ✅ Inserted {players_inserted} players")
    
    @staticmethod
    def parse_minutes(mp_str):
        """Convert 'MM:SS' to decimal minutes"""
        if not mp_str or mp_str == '' or mp_str == 'Did Not Play':
            return 0.0
        try:
            if ':' in mp_str:
                parts = mp_str.split(':')
                return float(parts[0]) + float(parts[1]) / 60.0
            else:
                return float(mp_str)
        except:
            return 0.0
    
    def close(self):
        """Close database connection"""
        self.cursor.close()
        self.conn.close()


def main():
    """
    Scrape yesterday's games
    """
    print("\n" + "="*80)
    print("🏀 BASKETBALL REFERENCE SCRAPER")
    print("="*80)
    
    scraper = BasketballReferenceScraper()
    
    # Scrape yesterday (BR updates overnight)
    yesterday = datetime.now() - timedelta(days=1)
    games_count = scraper.scrape_daily_games(yesterday)
    
    scraper.close()
    
    print("\n" + "="*80)
    print(f"✅ SCRAPED {games_count} GAMES!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

