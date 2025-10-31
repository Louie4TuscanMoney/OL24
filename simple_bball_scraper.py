#!/usr/bin/env python3

import os
import time
import pandas as pd
import psycopg2
import cloudscraper
from io import StringIO

def main():
    print("="*60)
    print("🏀 SIMPLE BASKETBALL REFERENCE SCRAPER")
    print("="*60)
    
    # Database connection
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set")
        return
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        print("✅ Database connected")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return
    
    # Load player mappings
    cur.execute("SELECT player_id, name FROM players")
    player_map = {row[1].lower(): row[0] for row in cur.fetchall()}
    print(f"✅ Loaded {len(player_map)} players")
    
    cur.execute("SELECT team_id, abbreviation FROM teams")
    team_id_map = {row[1]: row[0] for row in cur.fetchall()}
    print(f"✅ Loaded {len(team_id_map)} teams")
    
    # Web scraping
    scraper = cloudscraper.create_scraper()
    url = "https://www.basketball-reference.com/leagues/NBA_2026_per_poss.html"
    
    print(f"\n📊 Fetching: {url}")
    
    try:
        response = scraper.get(url, timeout=30)
        print(f"✅ Got response: {response.status_code}")
        
        # Parse with pandas
        dfs = pd.read_html(StringIO(response.text))
        print(f"✅ Found {len(dfs)} tables")
        
        if len(dfs) > 0:
            df = dfs[0]
            print(f"✅ Table shape: {df.shape}")
            print(f"✅ Columns: {list(df.columns)[:5]}")
            
            # Process first 10 players
            inserted = 0
            for i in range(min(10, len(df))):
                try:
                    row = df.iloc[i]
                    player_name = str(row.get('Player', row.iloc[1] if len(row) > 1 else '')).strip()
                    
                    if not player_name or player_name == 'Player':
                        continue
                    
                    # Get team
                    team_abbr = None
                    for col in df.columns:
                        if 'Tm' in str(col) or 'Team' in str(col):
                            team_abbr = str(row[col]).strip()
                            break
                    
                    if not team_abbr or team_abbr in ['TOT', 'nan', '']:
                        continue
                    
                    # Map team
                    team_map = {'BRK': 'BKN', 'PHO': 'PHX', 'CHO': 'CHA'}
                    team_abbr = team_map.get(team_abbr, team_abbr)
                    team_id = team_id_map.get(team_abbr)
                    
                    # Match player
                    player_id = player_map.get(player_name.lower())
                    
                    if not player_id or not team_id:
                        print(f"   Skipping {player_name} ({team_abbr}) - not found")
                        continue
                    
                    # Get stats
                    pts_100 = float(row.get('PTS', 0) or 0)
                    reb_100 = float(row.get('TRB', 0) or 0)
                    ast_100 = float(row.get('AST', 0) or 0)
                    
                    if pts_100 > 0:
                        print(f"   ✅ {player_name} ({team_abbr}): {pts_100:.1f} Pts/100")
                        inserted += 1
                
                except Exception as e:
                    print(f"   ❌ Error processing row {i}: {e}")
                    continue
            
            print(f"\n✅ Processed {inserted} players")
        
    except Exception as e:
        print(f"❌ Scraping error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()

