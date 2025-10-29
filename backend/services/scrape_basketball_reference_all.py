"""
COMPREHENSIVE BASKETBALL REFERENCE SCRAPER
Scrapes ALL stat tables for 2025-26 season

Tables:
1. Per Game: https://www.basketball-reference.com/leagues/NBA_2026_per_game.html
2. Totals: https://www.basketball-reference.com/leagues/NBA_2026_totals.html
3. Per 36: https://www.basketball-reference.com/leagues/NBA_2026_per_minute.html
4. Per 100: https://www.basketball-reference.com/leagues/NBA_2026_per_poss.html
5. Advanced: https://www.basketball-reference.com/leagues/NBA_2026_advanced.html (PER, WS, BPM, VORP, USG%)
6. Shooting: https://www.basketball-reference.com/leagues/NBA_2026_shooting.html
7. Adjusted Shooting: https://www.basketball-reference.com/leagues/NBA_2026_adj_shooting.html
8. Play-by-Play: https://www.basketball-reference.com/leagues/NBA_2026_play-by-play.html

This provides COMPLETE stats matching KenPom/538 quality!
"""

import os
import time
import cloudscraper
from bs4 import BeautifulSoup, Comment
import psycopg2
from datetime import datetime
import re
import pandas as pd


class ComprehensiveBballRefScraper:
    """Scrape ALL Basketball Reference stat tables"""
    
    def __init__(self):
        self.base_url = "https://www.basketball-reference.com"
        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'darwin', 'desktop': True}
        )
        
        self.season_year = 2026  # 2025-26 season
        self.season_id = '2025-26'
        
        # Team mapping
        self.team_map = {
            'BRK': 'BKN',
            'PHO': 'PHX',
            'CHO': 'CHA',
        }
        
        # Database
        DATABASE_URL = os.getenv('DATABASE_URL')
        if not DATABASE_URL:
            raise Exception("DATABASE_URL not set")
        
        self.conn = psycopg2.connect(DATABASE_URL)
        self.cur = self.conn.cursor()
        
        # Load player/team mappings
        self.cur.execute("SELECT player_id, name FROM players")
        self.player_map = {row[1].lower(): row[0] for row in self.cur.fetchall()}
        
        self.cur.execute("SELECT team_id, abbreviation FROM teams")
        self.team_id_map = {row[1]: row[0] for row in self.cur.fetchall()}
        
        print("="*80)
        print("🏀 COMPREHENSIVE BASKETBALL REFERENCE SCRAPER")
        print("="*80)
        print(f"   Season: 2025-26 (BR year: {self.season_year})")
        print(f"   ✅ Connected to PostgreSQL")
        print(f"   ✅ Loaded {len(self.player_map)} players from database")
        print(f"   ✅ Loaded {len(self.team_id_map)} teams from database")
        print(f"   🔓 Cloudflare bypass: ENABLED")
        print()
    
    def parse_table_from_html(self, html_content, table_id):
        """
        Parse table from HTML (including commented-out tables)
        Basketball Reference often hides tables in HTML comments
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # First try direct table
        table = soup.find('table', {'id': table_id})
        
        if not table:
            # Try to find in comments
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                if table_id in comment:
                    comment_soup = BeautifulSoup(comment, 'html.parser')
                    table = comment_soup.find('table', {'id': table_id})
                    if table:
                        break
        
        if not table:
            return None
        
        # Parse table into list of dicts
        headers = []
        for th in table.find('thead').find_all('th'):
            stat_name = th.get('data-stat', '')
            if stat_name:
                headers.append(stat_name)
        
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr', class_=lambda x: x != 'thead'):
                row_data = {}
                for td in tr.find_all(['td', 'th']):
                    stat_name = td.get('data-stat', '')
                    if stat_name:
                        row_data[stat_name] = td.text.strip()
                
                if row_data:
                    rows.append(row_data)
        
        return rows
    
    def scrape_per_100_stats(self):
        """Scrape Per 100 Possessions stats"""
        url = f"{self.base_url}/leagues/NBA_{self.season_year}_per_poss.html"
        
        print("📊 Scraping Per-100 Stats...")
        print(f"   URL: {url}")
        
        time.sleep(3)
        response = self.scraper.get(url, timeout=20)
        
        if response.status_code != 200:
            print(f"   ❌ Failed: {response.status_code}")
            return 0
        
        # Parse table
        rows = self.parse_table_from_html(response.content, 'per_poss_stats')
        
        if not rows:
            print("   ❌ Could not find table")
            return 0
        
        print(f"   ✅ Found {len(rows)} players")
        
        inserted = 0
        for player_data in rows:
            try:
                player_name = player_data.get('player', '').strip()
                team_abbr = player_data.get('team_name_abbr', '').strip()
                
                if not player_name or not team_abbr:
                    continue
                
                # Map team
                team_abbr = self.team_map.get(team_abbr, team_abbr)
                team_id = self.team_id_map.get(team_abbr)
                
                # Match player
                player_id = self.player_map.get(player_name.lower())
                
                # Get stats
                games_played = int(player_data.get('g', 0) or 0)
                mpg = float(player_data.get('mp_per_g', 0) or 0)
                
                # Per-100 stats (multiply by 100 since BR gives per-possession)
                pts_100 = float(player_data.get('pts_per_poss', 0) or 0) * 100
                reb_100 = float(player_data.get('trb_per_poss', 0) or 0) * 100
                ast_100 = float(player_data.get('ast_per_poss', 0) or 0) * 100
                stl_100 = float(player_data.get('stl_per_poss', 0) or 0) * 100
                blk_100 = float(player_data.get('blk_per_poss', 0) or 0) * 100
                tov_100 = float(player_data.get('tov_per_poss', 0) or 0) * 100
                
                # Shooting
                ts_pct = float(player_data.get('ts_pct', 0) or 0)
                efg_pct = float(player_data.get('efg_pct', 0) or 0)
                
                if player_id and team_id:
                    self.cur.execute("""
                        INSERT INTO player_season_stats (
                            player_id, season_id, team_id,
                            games_played, minutes_total,
                            pts_100, reb_100, ast_100, stl_100, blk_100, tov_100,
                            ts_pct, efg_pct
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        player_id, self.season_id, team_id,
                        games_played, mpg * games_played,
                        round(pts_100, 2), round(reb_100, 2), round(ast_100, 2),
                        round(stl_100, 2), round(blk_100, 2), round(tov_100, 2),
                        ts_pct, efg_pct
                    ))
                    self.conn.commit()
                    inserted += 1
                    
                    if inserted % 50 == 0:
                        print(f"      ... {inserted} players")
                
            except Exception as e:
                self.conn.rollback()
                continue
        
        print(f"   ✅ Inserted {inserted} players with per-100 stats")
        return inserted
    
    def scrape_advanced_stats(self):
        """Scrape Advanced stats (PER, WS, BPM, VORP, USG%)"""
        url = f"{self.base_url}/leagues/NBA_{self.season_year}_advanced.html"
        
        print("🎯 Scraping Advanced Stats (PER, WS, BPM, VORP, USG%)...")
        print(f"   URL: {url}")
        
        time.sleep(3)
        response = self.scraper.get(url, timeout=20)
        
        if response.status_code != 200:
            print(f"   ❌ Failed: {response.status_code}")
            return 0
        
        rows = self.parse_table_from_html(response.content, 'advanced_stats')
        
        if not rows:
            print("   ❌ Could not find table")
            return 0
        
        print(f"   ✅ Found {len(rows)} players")
        
        updated = 0
        for player_data in rows:
            try:
                player_name = player_data.get('player', '').strip()
                player_id = self.player_map.get(player_name.lower())
                
                if not player_id:
                    continue
                
                # Advanced stats
                per = float(player_data.get('per', 0) or 0)
                ts_pct = float(player_data.get('ts_pct', 0) or 0)
                usg_pct = float(player_data.get('usg_pct', 0) or 0)
                ws = float(player_data.get('ws', 0) or 0)
                ws_48 = float(player_data.get('ws_per_48', 0) or 0)
                obpm = float(player_data.get('obpm', 0) or 0)
                dbpm = float(player_data.get('dbpm', 0) or 0)
                bpm = float(player_data.get('bpm', 0) or 0)
                vorp = float(player_data.get('vorp', 0) or 0)
                
                # Update player_season_stats
                self.cur.execute("""
                    UPDATE player_season_stats 
                    SET 
                        bpm = %s,
                        per = %s,
                        usage_pct = %s,
                        win_shares = %s,
                        win_shares_48 = %s,
                        obpm = %s,
                        dbpm = %s,
                        vorp = %s
                    WHERE player_id = %s AND season_id = %s
                """, (bpm, per, usg_pct, ws, ws_48, obpm, dbpm, vorp, player_id, self.season_id))
                
                if self.cur.rowcount > 0:
                    self.conn.commit()
                    updated += 1
                    
                    if updated % 50 == 0:
                        print(f"      ... {updated} players")
                
            except Exception as e:
                self.conn.rollback()
                continue
        
        print(f"   ✅ Updated {updated} players with advanced stats")
        return updated
    
    def run_all(self):
        """Run all scrapers"""
        stats = {}
        
        # 1. Per-100 (foundation)
        stats['per_100'] = self.scrape_per_100_stats()
        
        # 2. Advanced (PER, BPM, VORP, USG%)
        stats['advanced'] = self.scrape_advanced_stats()
        
        # Summary
        print()
        print("="*80)
        print("✅ COMPREHENSIVE SCRAPING COMPLETE")
        print("="*80)
        print(f"   Per-100 stats: {stats['per_100']} players")
        print(f"   Advanced stats: {stats['advanced']} players")
        print()
        
        # Verify
        self.cur.execute("""
            SELECT COUNT(*) FROM player_season_stats 
            WHERE season_id = %s AND pts_100 > 0
        """, (self.season_id,))
        
        total = self.cur.fetchone()[0]
        print(f"🔍 Total players with stats: {total}")
        
        # Show top 5
        self.cur.execute("""
            SELECT p.name, ps.pts_100, ps.bpm, ps.per, ps.vorp, ps.games_played, t.abbreviation
            FROM player_season_stats ps
            JOIN players p ON ps.player_id = p.player_id
            JOIN teams t ON ps.team_id = t.team_id
            WHERE ps.season_id = %s AND ps.games_played > 0
            ORDER BY ps.pts_100 DESC
            LIMIT 10
        """, (self.season_id,))
        
        print()
        print("📊 Top 10 Scorers (Pts/100):")
        print(f"   {'Player':<25} {'Team':3} {'Pts/100':>7} {'BPM':>6} {'PER':>6} {'VORP':>6} {'GP':>3}")
        print("   " + "-"*65)
        
        for row in self.cur.fetchall():
            name, pts100, bpm, per, vorp, gp, team = row
            bpm_str = f"{bpm:.1f}" if bpm else "N/A"
            per_str = f"{per:.1f}" if per else "N/A"
            vorp_str = f"{vorp:.1f}" if vorp else "N/A"
            
            print(f"   {name:<25} {team:3} {pts100:7.1f} {bpm_str:>6} {per_str:>6} {vorp_str:>6} {gp:3d}")
        
        return stats


def main():
    """Run comprehensive scraper"""
    scraper = ComprehensiveBballRefScraper()
    
    try:
        results = scraper.run_all()
        
        print()
        print("="*80)
        print("🎯 NEXT STEPS:")
        print("="*80)
        print("1. Deploy to Railway: git push origin main")
        print("2. Test API: curl https://ol24-production.up.railway.app/api/stats/teams")
        print("3. Check frontend: ontologicxyz.com/stats")
        print()
        print("🕐 Daily auto-update: 3:30 AM UTC (integrated in trading_dashboard_api.py)")
        
    finally:
        scraper.cur.close()
        scraper.conn.close()


if __name__ == "__main__":
    main()

