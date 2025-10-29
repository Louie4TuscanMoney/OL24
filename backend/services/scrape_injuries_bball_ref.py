#!/usr/bin/env python3
"""
Scrape current NBA injuries from Basketball Reference
URL: https://www.basketball-reference.com/friv/injuries.fcgi
"""

import os
import cloudscraper
from bs4 import BeautifulSoup
import psycopg2
from datetime import datetime

class InjuryScraper:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        self.conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        self.cur = self.conn.cursor()
        
        # Load player map (name -> player_id)
        self.cur.execute("SELECT player_id, name, LOWER(name) as name_lower FROM players")
        self.player_map = {row[2]: row[0] for row in self.cur.fetchall()}
        
        print(f"✅ Loaded {len(self.player_map)} players from database")
    
    def scrape_injuries(self):
        """Scrape all active injuries from Basketball Reference"""
        url = "https://www.basketball-reference.com/friv/injuries.fcgi"
        
        print(f"📊 Fetching injuries from: {url}")
        response = self.scraper.get(url)
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch injuries: {response.status_code}")
            return 0
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the injuries table
        table = soup.find('table', {'id': 'injuries'})
        if not table:
            print("❌ Could not find injuries table")
            return 0
        
        tbody = table.find('tbody')
        rows = tbody.find_all('tr')
        
        print(f"✅ Found {len(rows)} injuries")
        
        # Clear existing injuries (mark as inactive)
        self.cur.execute("UPDATE player_injuries SET is_active = FALSE")
        
        inserted = 0
        skipped = 0
        
        for row in rows:
            try:
                # Get player name
                player_cell = row.find('td', {'data-stat': 'player'})
                if not player_cell:
                    continue
                
                player_name = player_cell.get_text(strip=True)
                
                # Get team
                team_cell = row.find('td', {'data-stat': 'team_name'})
                team_name = team_cell.get_text(strip=True) if team_cell else ""
                
                # Get update date
                update_cell = row.find('td', {'data-stat': 'date_update'})
                update_str = update_cell.get_text(strip=True) if update_cell else ""
                
                # Get description
                desc_cell = row.find('td', {'data-stat': 'note'})
                description = desc_cell.get_text(strip=True) if desc_cell else ""
                
                # Parse status and injury type from description
                status = "Day-To-Day"
                injury_type = "Unknown"
                
                if "Out" in description:
                    status = "Out"
                elif "Doubtful" in description:
                    status = "Doubtful"
                elif "Questionable" in description:
                    status = "Questionable"
                elif "Probable" in description:
                    status = "Probable"
                
                # Extract injury type (look for body part in parentheses)
                if "(" in description and ")" in description:
                    start = description.index("(") + 1
                    end = description.index(")")
                    injury_type = description[start:end]
                
                # Match player to database
                player_id = self.player_map.get(player_name.lower())
                
                if not player_id:
                    # Try fuzzy matching (first name + last name)
                    name_parts = player_name.lower().split()
                    for db_name, db_id in self.player_map.items():
                        if all(part in db_name for part in name_parts):
                            player_id = db_id
                            break
                
                if player_id:
                    # Insert or update injury
                    self.cur.execute("""
                        INSERT INTO player_injuries (
                            player_id, status, injury_type, description,
                            injury_date, is_active
                        ) VALUES (%s, %s, %s, %s, %s, TRUE)
                        ON CONFLICT (player_id, injury_date) DO UPDATE SET
                            status = EXCLUDED.status,
                            injury_type = EXCLUDED.injury_type,
                            description = EXCLUDED.description,
                            is_active = TRUE,
                            updated_at = NOW()
                    """, (
                        player_id,
                        status,
                        injury_type,
                        description,
                        datetime.now().date()
                    ))
                    inserted += 1
                else:
                    skipped += 1
                    if skipped <= 5:  # Only print first 5
                        print(f"   ⚠️  Player not found: {player_name} ({team_name})")
                
            except Exception as e:
                print(f"   ⚠️  Error parsing injury: {e}")
                continue
        
        self.conn.commit()
        
        if skipped > 5:
            print(f"   ⚠️  ... and {skipped - 5} more players not found")
        
        return inserted
    
    def close(self):
        self.cur.close()
        self.conn.close()


if __name__ == '__main__':
    print("="*80)
    print("🚑 SCRAPING NBA INJURIES FROM BASKETBALL REFERENCE")
    print("="*80)
    print()
    
    scraper = InjuryScraper()
    
    count = scraper.scrape_injuries()
    
    scraper.close()
    
    print()
    print("="*80)
    print(f"✅ INJURY SCRAPING COMPLETE!")
    print(f"   Inserted/updated: {count} injuries")
    print("="*80)

