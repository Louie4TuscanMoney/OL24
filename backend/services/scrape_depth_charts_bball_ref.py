#!/usr/bin/env python3
"""
Scrape depth charts for all NBA teams from Basketball Reference
URL pattern: https://www.basketball-reference.com/teams/{TEAM}/2026_depth.html
"""

import os
import cloudscraper
from bs4 import BeautifulSoup
import psycopg2
import time

class DepthChartScraper:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        self.conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        self.cur = self.conn.cursor()
        
        # Load teams
        self.cur.execute("SELECT team_id, abbreviation FROM teams")
        self.teams = {row[1]: row[0] for row in self.cur.fetchall()}
        
        # Load players
        self.cur.execute("SELECT player_id, name, LOWER(name) as name_lower FROM players")
        self.player_map = {row[2]: row[0] for row in self.cur.fetchall()}
        
        print(f"✅ Loaded {len(self.teams)} teams and {len(self.player_map)} players")
    
    def scrape_team_depth_chart(self, team_abbr):
        """Scrape depth chart for a single team"""
        url = f"https://www.basketball-reference.com/teams/{team_abbr}/2026_depth.html"
        
        try:
            response = self.scraper.get(url)
            
            if response.status_code != 200:
                print(f"   ⚠️  Failed to fetch {team_abbr}: {response.status_code}")
                return 0
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find depth chart table
            table = soup.find('table', {'id': 'depth'})
            if not table:
                print(f"   ⚠️  No depth chart table found for {team_abbr}")
                return 0
            
            tbody = table.find('tbody')
            if not tbody:
                return 0
            
            rows = tbody.find_all('tr')
            
            inserted = 0
            team_id = self.teams.get(team_abbr)
            
            if not team_id:
                print(f"   ⚠️  Team ID not found for {team_abbr}")
                return 0
            
            for row in rows:
                try:
                    # Get position
                    pos_cell = row.find('th', {'data-stat': 'pos'})
                    if not pos_cell:
                        continue
                    position = pos_cell.get_text(strip=True)
                    
                    # Get players at each depth (1st, 2nd, 3rd, etc.)
                    depth_rank = 1
                    for cell in row.find_all('td'):
                        player_links = cell.find_all('a')
                        for link in player_links:
                            player_name = link.get_text(strip=True)
                            
                            # Match player
                            player_id = self.player_map.get(player_name.lower())
                            
                            if not player_id:
                                # Try fuzzy matching
                                name_parts = player_name.lower().split()
                                for db_name, db_id in self.player_map.items():
                                    if all(part in db_name for part in name_parts):
                                        player_id = db_id
                                        break
                            
                            if player_id:
                                # Insert into depth chart
                                self.cur.execute("""
                                    INSERT INTO team_depth_charts (
                                        team_id, player_id, position, depth_rank
                                    ) VALUES (%s, %s, %s, %s)
                                    ON CONFLICT (team_id, player_id) DO UPDATE SET
                                        position = EXCLUDED.position,
                                        depth_rank = EXCLUDED.depth_rank,
                                        updated_at = NOW()
                                """, (team_id, player_id, position, depth_rank))
                                inserted += 1
                        
                        depth_rank += 1
                
                except Exception as e:
                    print(f"   ⚠️  Error parsing row for {team_abbr}: {e}")
                    continue
            
            return inserted
            
        except Exception as e:
            print(f"   ❌ Error scraping {team_abbr}: {e}")
            return 0
    
    def scrape_all_teams(self):
        """Scrape depth charts for all 30 teams"""
        print("📊 Scraping depth charts for all teams...")
        print()
        
        # Clear existing depth charts
        self.cur.execute("DELETE FROM team_depth_charts")
        
        total_inserted = 0
        
        for team_abbr in sorted(self.teams.keys()):
            print(f"   {team_abbr}...", end=" ", flush=True)
            
            count = self.scrape_team_depth_chart(team_abbr)
            total_inserted += count
            
            print(f"✅ {count} players")
            
            # Be nice to Basketball Reference
            time.sleep(0.5)
        
        self.conn.commit()
        
        return total_inserted
    
    def close(self):
        self.cur.close()
        self.conn.close()


if __name__ == '__main__':
    print("="*80)
    print("🏀 SCRAPING NBA DEPTH CHARTS FROM BASKETBALL REFERENCE")
    print("="*80)
    print()
    
    scraper = DepthChartScraper()
    
    total = scraper.scrape_all_teams()
    
    scraper.close()
    
    print()
    print("="*80)
    print(f"✅ DEPTH CHART SCRAPING COMPLETE!")
    print(f"   Total players in depth charts: {total}")
    print("="*80)

