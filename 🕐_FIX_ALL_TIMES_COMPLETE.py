#!/usr/bin/env python3
"""
FIX ALL GAME TIMES - Complete solution with ESPN abbreviation mapping
"""

import os
import requests
import psycopg2
from datetime import datetime, date
from zoneinfo import ZoneInfo

DATABASE_URL = os.getenv('DATABASE_URL')

# ESPN to NBA abbreviation mapping
ESPN_TO_NBA = {
    'GS': 'GSW',     # Golden State
    'WSH': 'WAS',    # Washington
    'SA': 'SAS',     # San Antonio  
    'NY': 'NYK',     # New York
    'NO': 'NOP',     # New Orleans
    # All others are the same
}

def normalize_abbr(espn_abbr):
    """Convert ESPN abbreviation to NBA abbreviation"""
    return ESPN_TO_NBA.get(espn_abbr, espn_abbr)

def main():
    print("\n" + "="*80)
    print("🕐 FIXING ALL GAME TIMES WITH ESPN MAPPING")
    print("="*80 + "\n")
    
    # Fetch from ESPN
    print("📡 Fetching from ESPN API...")
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    response = requests.get(url, timeout=10)
    data = response.json()
    
    espn_games = []
    for event in data.get('events', []):
        try:
            competition = event.get('competitions', [{}])[0]
            competitors = competition.get('competitors', [])
            
            home_team = next((c for c in competitors if c.get('homeAway') == 'home'), {})
            away_team = next((c for c in competitors if c.get('homeAway') == 'away'), {})
            
            espn_home = home_team.get('team', {}).get('abbreviation', '')
            espn_away = away_team.get('team', {}).get('abbreviation', '')
            
            # Normalize to NBA abbreviations
            nba_home = normalize_abbr(espn_home)
            nba_away = normalize_abbr(espn_away)
            
            date_str = event.get('date', '')
            utc_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            pst_time = utc_time.astimezone(ZoneInfo('America/Los_Angeles'))
            
            espn_games.append({
                'home': nba_home,
                'away': nba_away,
                'date': utc_time.date(),
                'time': utc_time.time(),
                'pst_display': pst_time.strftime('%I:%M %p PST'),
                'name': f"{nba_away} @ {nba_home}"
            })
            
            print(f"  ✅ {nba_away} @ {nba_home}: {pst_time.strftime('%I:%M %p PST')}")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Found {len(espn_games)} games\n")
    
    # Update database
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return
    
    print("💾 Updating database...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Get all team IDs
    cur.execute("SELECT team_id, abbreviation FROM teams")
    team_map = {row[1]: row[0] for row in cur.fetchall()}
    
    updated = 0
    for game in espn_games:
        try:
            home_id = team_map.get(game['home'])
            away_id = team_map.get(game['away'])
            
            if not home_id or not away_id:
                print(f"  ⚠️  Teams not found: {game['away']} @ {game['home']}")
                continue
            
            # Update ALL games matching this team pair and date
            cur.execute("""
                UPDATE nba_schedule
                SET game_date = %s,
                    game_time = %s,
                    updated_at = NOW()
                WHERE home_team_id = %s
                  AND away_team_id = %s
                  AND game_date = %s
            """, (
                game['date'],
                game['time'],
                home_id,
                away_id,
                game['date']
            ))
            
            if cur.rowcount > 0:
                updated += cur.rowcount
                print(f"  ✅ {game['away']} @ {game['home']}: {game['pst_display']} ({cur.rowcount} entries)")
            else:
                print(f"  ⚠️  No match: {game['away']} @ {game['home']}")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n{'='*80}")
    print(f"✅ ALL GAME TIMES FIXED!")
    print(f"{'='*80}")
    print(f"   Updated {updated} database entries")
    print(f"{'='*80}\n")
    
    # Verify
    print("🔍 Verifying today's schedule...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT away.abbreviation, home.abbreviation,
               s.game_date, s.game_time
        FROM nba_schedule s
        JOIN teams home ON s.home_team_id = home.team_id
        JOIN teams away ON s.away_team_id = away.team_id
        WHERE s.game_date = %s
        ORDER BY s.game_time
    """, (date.today(),))
    
    rows = cur.fetchall()
    print(f"\nToday's games ({date.today()}):")
    for away, home, gdate, gtime in rows:
        if gtime:
            utc_dt = datetime.combine(gdate, gtime).replace(tzinfo=ZoneInfo('UTC'))
            pst_dt = utc_dt.astimezone(ZoneInfo('America/Los_Angeles'))
            print(f"  {away} @ {home}: {pst_dt.strftime('%I:%M %p PST')}")
        else:
            print(f"  {away} @ {home}: NO TIME ❌")
    
    cur.close()
    conn.close()
    
    print(f"\n{'='*80}")
    print("✅ DONE! Check https://ontologicxyz.com/game")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

