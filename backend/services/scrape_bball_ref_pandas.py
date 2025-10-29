"""
BASKETBALL REFERENCE SCRAPER - Using Pandas
Pandas can automatically parse Basketball Reference tables!

This is more reliable than BeautifulSoup for BR's complex HTML structure.
"""

import os
import time
import pandas as pd
import psycopg2
from datetime import datetime


class BballRefPandasScraper:
    """Use pandas.read_html() to scrape Basketball Reference"""
    
    def __init__(self):
        self.base_url = "https://www.basketball-reference.com"
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
        
        # Load mappings
        self.cur.execute("SELECT player_id, name FROM players")
        self.player_map = {row[1].lower(): row[0] for row in self.cur.fetchall()}
        
        self.cur.execute("SELECT team_id, abbreviation FROM teams")
        self.team_id_map = {row[1]: row[0] for row in self.cur.fetchall()}
        
        print("="*80)
        print("🏀 BASKETBALL REFERENCE SCRAPER (Pandas)")
        print("="*80)
        print(f"   Season: 2025-26 (BR year: {self.season_year})")
        print(f"   ✅ {len(self.player_map)} players loaded")
        print(f"   ✅ {len(self.team_id_map)} teams loaded")
        print()
    
    def scrape_per_100_stats(self):
        """Scrape per-100 possession stats"""
        url = f"{self.base_url}/leagues/NBA_{self.season_year}_per_poss.html"
        
        print("📊 Scraping Per-100 Stats...")
        print(f"   URL: {url}")
        
        try:
            time.sleep(3)
            
            # pandas.read_html() automatically finds all tables
            dfs = pd.read_html(url)
            
            print(f"   ✅ Found {len(dfs)} tables")
            
            # The stats table is usually the first one
            df = dfs[0]
            
            print(f"   ✅ Parsed table: {len(df)} rows, {len(df.columns)} columns")
            print(f"   Columns: {list(df.columns)[:10]}")
            print()
            
            inserted = 0
            
            for _, row in df.iterrows():
                try:
                    # Handle multi-index columns (BR uses them)
                    player_name = str(row.get('Player', row.iloc[1] if len(row) > 1 else '')).strip()
                    
                    if not player_name or player_name == 'Player':
                        continue
                    
                    # Get team (might be nested column)
                    team_abbr = None
                    for col in df.columns:
                        if 'Tm' in str(col) or 'Team' in str(col):
                            team_abbr = str(row[col]).strip()
                            break
                    
                    if not team_abbr or team_abbr in ['TOT', 'nan', '']:
                        continue
                    
                    # Map team
                    team_abbr = self.team_map.get(team_abbr, team_abbr)
                    team_id = self.team_id_map.get(team_abbr)
                    
                    # Match player
                    player_id = self.player_map.get(player_name.lower())
                    
                    if not player_id or not team_id:
                        continue
                    
                    # Get stats (column names vary)
                    games_played = int(row.get('G', row.get(('Unnamed: 5_level_0', 'G'), 0)) or 0)
                    
                    # Per-100 stats
                    pts_100 = float(row.get('PTS', row.get(('Unnamed: 29_level_0', 'PTS'), 0)) or 0)
                    reb_100 = float(row.get('TRB', row.get(('Unnamed: 23_level_0', 'TRB'), 0)) or 0)
                    ast_100 = float(row.get('AST', row.get(('Unnamed: 24_level_0', 'AST'), 0)) or 0)
                    stl_100 = float(row.get('STL', row.get(('Unnamed: 25_level_0', 'STL'), 0)) or 0)
                    blk_100 = float(row.get('BLK', row.get(('Unnamed: 26_level_0', 'BLK'), 0)) or 0)
                    tov_100 = float(row.get('TOV', row.get(('Unnamed: 27_level_0', 'TOV'), 0)) or 0)
                    
                    # Shooting
                    ts_pct = float(row.get('TS%', row.get(('Shooting', 'TS%'), 0)) or 0)
                    efg_pct = float(row.get('eFG%', row.get(('Shooting', 'eFG%'), 0)) or 0)
                    
                    if pts_100 > 0:  # Valid stat
                        self.cur.execute("""
                            INSERT INTO player_season_stats (
                                player_id, season_id, team_id,
                                games_played,
                                pts_100, reb_100, ast_100, stl_100, blk_100, tov_100,
                                ts_pct, efg_pct
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (player_id, season_id) DO UPDATE SET
                                games_played = EXCLUDED.games_played,
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
                            games_played,
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
            
            print(f"   ✅ Inserted {inserted} players")
            return inserted
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def scrape_advanced_stats(self):
        """Scrape advanced stats"""
        url = f"{self.base_url}/leagues/NBA_{self.season_year}_advanced.html"
        
        print("🎯 Scraping Advanced Stats...")
        print(f"   URL: {url}")
        
        try:
            time.sleep(3)
            dfs = pd.read_html(url)
            
            df = dfs[0]
            print(f"   ✅ Parsed table: {len(df)} rows")
            
            updated = 0
            
            for _, row in df.iterrows():
                try:
                    player_name = str(row.get('Player', row.iloc[1] if len(row) > 1 else '')).strip()
                    
                    if not player_name or player_name == 'Player':
                        continue
                    
                    player_id = self.player_map.get(player_name.lower())
                    if not player_id:
                        continue
                    
                    # Advanced stats
                    per = float(row.get('PER', 0) or 0)
                    ts_pct = float(row.get('TS%', 0) or 0)
                    usg_pct = float(row.get('USG%', 0) or 0)
                    ws = float(row.get('WS', 0) or 0)
                    ws_48 = float(row.get('WS/48', 0) or 0)
                    obpm = float(row.get('OBPM', 0) or 0)
                    dbpm = float(row.get('DBPM', 0) or 0)
                    bpm = float(row.get('BPM', 0) or 0)
                    vorp = float(row.get('VORP', 0) or 0)
                    
                    self.cur.execute("""
                        UPDATE player_season_stats 
                        SET 
                            bpm = %s,
                            obpm = %s,
                            dbpm = %s,
                            per = %s,
                            usage_pct = %s,
                            win_shares = %s,
                            win_shares_48 = %s,
                            vorp = %s
                        WHERE player_id = %s AND season_id = %s
                    """, (bpm, obpm, dbpm, per, usg_pct, ws, ws_48, vorp, player_id, self.season_id))
                    
                    if self.cur.rowcount > 0:
                        self.conn.commit()
                        updated += 1
                        
                        if updated % 50 == 0:
                            print(f"      ... {updated} players")
                
                except Exception as e:
                    self.conn.rollback()
                    continue
            
            print(f"   ✅ Updated {updated} players")
            return updated
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0
    
    def run_all(self):
        """Run all scrapers"""
        stats = {}
        
        stats['per_100'] = self.scrape_per_100_stats()
        stats['advanced'] = self.scrape_advanced_stats()
        
        print()
        print("="*80)
        print("✅ SCRAPING COMPLETE")
        print("="*80)
        print(f"   Per-100: {stats['per_100']} players")
        print(f"   Advanced: {stats['advanced']} players")
        
        # Show results
        self.cur.execute("""
            SELECT p.name, ps.pts_100, ps.bpm, ps.vorp, ps.games_played, t.abbreviation
            FROM player_season_stats ps
            JOIN players p ON ps.player_id = p.player_id
            JOIN teams t ON ps.team_id = t.team_id
            WHERE ps.season_id = %s AND ps.pts_100 > 0
            ORDER BY ps.pts_100 DESC
            LIMIT 10
        """, (self.season_id,))
        
        print()
        print("📊 Top 10 Scorers:")
        for row in self.cur.fetchall():
            print(f"   {row[0]:25s} {row[5]:3s}  {row[1]:.1f} Pts/100  BPM: {row[2]:.1f if row[2] else 'N/A':<4}  VORP: {row[3]:.2f if row[3] else 'N/A':<4}  ({row[4]} GP)")
        
        return stats


if __name__ == "__main__":
    scraper = BballRefPandasScraper()
    try:
        scraper.run_all()
    finally:
        scraper.cur.close()
        scraper.conn.close()

