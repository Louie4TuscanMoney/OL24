#!/usr/bin/env python3
"""
INSERT MISSING GAMES - Add today's ESPN games that aren't in database yet
"""

import os
import requests
import psycopg2
from datetime import datetime
from zoneinfo import ZoneInfo

DATABASE_URL = os.getenv('DATABASE_URL')

ESPN_TO_NBA = {
    'GS': 'GSW', 'WSH': 'WAS', 'SA': 'SAS', 'NY': 'NYK', 'NO': 'NOP'
}

def normalize_abbr(espn_abbr):
    return ESPN_TO_NBA.get(espn_abbr, espn_abbr)

def main():
    print("\n" + "="*80)
    print("🕐 INSERTING MISSING GAMES FROM ESPN")
    print("="*80 + "\n")
    
    # Fetch from ESPN
    print("📡 Fetching from ESPN...")
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    response = requests.get(url, timeout=10)
    data = response.json()
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Get team map
    cur.execute("SELECT team_id, abbreviation FROM teams")
    team_map = {row[1]: row[0] for row in cur.fetchall()}
    
    inserted = 0
    updated = 0
    
    for event in data.get('events', []):
        try:
            competition = event.get('competitions', [{}])[0]
            competitors = competition.get('competitors', [])
            
            home_team = next((c for c in competitors if c.get('homeAway') == 'home'), {})
            away_team = next((c for c in competitors if c.get('homeAway') == 'away'), {})
            
            espn_home = home_team.get('team', {}).get('abbreviation', '')
            espn_away = away_team.get('team', {}).get('abbreviation', '')
            
            nba_home = normalize_abbr(espn_home)
            nba_away = normalize_abbr(espn_away)
            
            home_id = team_map.get(nba_home)
            away_id = team_map.get(nba_away)
            
            if not home_id or not away_id:
                print(f"  ⚠️  Teams not found: {nba_away} @ {nba_home}")
                continue
            
            espn_game_id = event.get('id', '')
            date_str = event.get('date', '')
            utc_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            pst_time = utc_time.astimezone(ZoneInfo('America/Los_Angeles'))
            
            # Try to insert or update
            cur.execute("""
                INSERT INTO nba_schedule (
                    game_id, home_team_id, away_team_id,
                    game_date, game_time, game_status,
                    created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (game_id) DO UPDATE SET
                    game_date = EXCLUDED.game_date,
                    game_time = EXCLUDED.game_time,
                    updated_at = NOW()
            """, (
                espn_game_id,
                home_id,
                away_id,
                utc_time.date(),
                utc_time.time(),
                'Scheduled'
            ))
            
            if cur.rowcount > 0:
                # Check if it was insert or update
                cur.execute("SELECT COUNT(*) FROM nba_schedule WHERE game_id = %s", (espn_game_id,))
                if cur.fetchone()[0] == 1:
                    inserted += 1
                    print(f"  ➕ Inserted: {nba_away} @ {nba_home} at {pst_time.strftime('%I:%M %p PST')}")
                else:
                    updated += 1
                    print(f"  ✅ Updated: {nba_away} @ {nba_home} at {pst_time.strftime('%I:%M %p PST')}")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n{'='*80}")
    print(f"✅ DONE!")
    print(f"{'='*80}")
    print(f"   Inserted: {inserted}")
    print(f"   Updated: {updated}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

