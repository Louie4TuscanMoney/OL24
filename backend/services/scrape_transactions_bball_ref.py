#!/usr/bin/env python3
"""
Scrape team transactions from Basketball Reference
URL pattern: https://www.basketball-reference.com/teams/{TEAM}/2026_transactions.html
"""

import os
import cloudscraper
from bs4 import BeautifulSoup
import psycopg2
from datetime import datetime
import time
import re

class TransactionsScraper:
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
    
    def parse_transaction_type(self, description):
        """Parse transaction type from description"""
        desc_lower = description.lower()
        
        if "traded" in desc_lower or "acquired" in desc_lower:
            return "Trade"
        elif "waived" in desc_lower:
            return "Waiver"
        elif "signed" in desc_lower:
            return "Signing"
        elif "released" in desc_lower:
            return "Release"
        elif "two-way" in desc_lower:
            return "Two-Way"
        elif "g league" in desc_lower or "g-league" in desc_lower:
            return "G League"
        else:
            return "Other"
    
    def scrape_team_transactions(self, team_abbr):
        """Scrape transactions for a single team"""
        url = f"https://www.basketball-reference.com/teams/{team_abbr}/2026_transactions.html"
        
        try:
            response = self.scraper.get(url)
            
            if response.status_code != 200:
                print(f"   ⚠️  Failed to fetch {team_abbr}: {response.status_code}")
                return 0
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find transactions table
            table = soup.find('table', {'id': 'transactions'})
            if not table:
                # Try finding in div with id 'div_transactions'
                div = soup.find('div', {'id': 'div_transactions'})
                if div:
                    # Transactions might be in a list
                    ul = div.find('ul')
                    if ul:
                        return self._parse_transactions_list(ul, team_abbr)
                
                print(f"   ⚠️  No transactions found for {team_abbr}")
                return 0
            
            tbody = table.find('tbody')
            if not tbody:
                return 0
            
            rows = tbody.find_all('tr')
            
            inserted = 0
            team_id = self.teams.get(team_abbr)
            
            for row in rows:
                try:
                    # Get date
                    date_cell = row.find('td', {'data-stat': 'date_tran'})
                    if not date_cell:
                        continue
                    
                    date_str = date_cell.get_text(strip=True)
                    try:
                        trans_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    except:
                        trans_date = datetime.now().date()
                    
                    # Get transaction description
                    desc_cell = row.find('td', {'data-stat': 'transaction'})
                    if not desc_cell:
                        continue
                    
                    description = desc_cell.get_text(strip=True)
                    
                    # Extract player name (usually first name in description)
                    player_links = desc_cell.find_all('a')
                    if not player_links:
                        continue
                    
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
                            trans_type = self.parse_transaction_type(description)
                            
                            # Insert transaction
                            self.cur.execute("""
                                INSERT INTO player_transactions (
                                    player_id, player_name, transaction_type,
                                    transaction_date, from_team_id, to_team_id,
                                    trade_description
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT DO NOTHING
                            """, (
                                player_id,
                                player_name,
                                trans_type,
                                trans_date,
                                team_id if "from" in description.lower() else None,
                                team_id if "to" in description.lower() or "signed" in description.lower() else None,
                                description
                            ))
                            inserted += 1
                
                except Exception as e:
                    print(f"   ⚠️  Error parsing transaction for {team_abbr}: {e}")
                    continue
            
            return inserted
            
        except Exception as e:
            print(f"   ❌ Error scraping {team_abbr}: {e}")
            return 0
    
    def _parse_transactions_list(self, ul, team_abbr):
        """Parse transactions from a list (alternative format)"""
        inserted = 0
        team_id = self.teams.get(team_abbr)
        
        items = ul.find_all('li')
        for item in items:
            try:
                text = item.get_text(strip=True)
                
                # Extract date (usually at start)
                date_match = re.match(r'(\w+ \d+, \d{4})', text)
                if date_match:
                    date_str = date_match.group(1)
                    try:
                        trans_date = datetime.strptime(date_str, '%B %d, %Y').date()
                    except:
                        trans_date = datetime.now().date()
                else:
                    trans_date = datetime.now().date()
                
                # Extract player names
                links = item.find_all('a')
                for link in links:
                    player_name = link.get_text(strip=True)
                    player_id = self.player_map.get(player_name.lower())
                    
                    if player_id:
                        trans_type = self.parse_transaction_type(text)
                        
                        self.cur.execute("""
                            INSERT INTO player_transactions (
                                player_id, player_name, transaction_type,
                                transaction_date, to_team_id, trade_description
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (
                            player_id,
                            player_name,
                            trans_type,
                            trans_date,
                            team_id,
                            text
                        ))
                        inserted += 1
            
            except Exception as e:
                continue
        
        return inserted
    
    def scrape_all_teams(self):
        """Scrape transactions for all 30 teams"""
        print("📊 Scraping transactions for all teams...")
        print()
        
        # Don't clear existing transactions (they're historical)
        # But we can set a cutoff date for current season
        
        total_inserted = 0
        
        for team_abbr in sorted(self.teams.keys()):
            print(f"   {team_abbr}...", end=" ", flush=True)
            
            count = self.scrape_team_transactions(team_abbr)
            total_inserted += count
            
            print(f"✅ {count} transactions")
            
            # Be nice to Basketball Reference
            time.sleep(0.5)
        
        self.conn.commit()
        
        return total_inserted
    
    def close(self):
        self.cur.close()
        self.conn.close()


if __name__ == '__main__':
    print("="*80)
    print("📋 SCRAPING NBA TRANSACTIONS FROM BASKETBALL REFERENCE")
    print("="*80)
    print()
    
    scraper = TransactionsScraper()
    
    total = scraper.scrape_all_teams()
    
    scraper.close()
    
    print()
    print("="*80)
    print(f"✅ TRANSACTIONS SCRAPING COMPLETE!")
    print(f"   Total transactions: {total}")
    print("="*80)

