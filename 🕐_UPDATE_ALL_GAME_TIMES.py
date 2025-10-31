#!/usr/bin/env python3
"""
UPDATE ALL GAME TIMES - Match by teams and date, update times from ESPN
"""

import os
import requests
import psycopg2
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

DATABASE_URL = os.getenv('DATABASE_URL')

def main():
    print("\n" + "="*80)
    print("🕐 UPDATING ALL GAME TIMES FROM ESPN API")
    print("="*80 + "\n")
    
    # Get games from ESPN for next 7 days
    print("📡 Fetching schedule from ESPN...")
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    
    # ESPN scoreboard only shows today, so we need to fetch calendar
    # For now, just fix today's games
    response = requests.get(url, timeout=10)
    data = response.json()
    
    espn_games = []
    for event in data.get('events', []):
        try:
            competition = event.get('competitions', [{}])[0]
            competitors = competition.get('competitors', [])
            
            home_team = next((c for c in competitors if c.get('homeAway') == 'home'), {})
            away_team = next((c for c in competitors if c.get('homeAway') == 'away'), {})
            
            home_abbr = home_team.get('team', {}).get('abbreviation', '')
            away_abbr = away_team.get('team', {}).get('abbreviation', '')
            
            date_str = event.get('date', '')
            utc_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            pst_time = utc_time.astimezone(ZoneInfo('America/Los_Angeles'))
            
            espn_games.append({
                'home': home_abbr,
                'away': away_abbr,
                'date': utc_time.date(),
                'time': utc_time.time(),
                'pst_display': pst_time.strftime('%I:%M %p PST'),
                'name': event.get('name', '')
            })
            
            print(f"  ✅ {away_abbr} @ {home_abbr}: {pst_time.strftime('%I:%M %p PST')}")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Found {len(espn_games)} games from ESPN\n")
    
    # Update database by matching teams
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return
    
    print("💾 Updating database by team matchup...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    updated = 0
    for game in espn_games:
        try:
            # Get team IDs from abbreviations
            cur.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (game['home'],))
            home_row = cur.fetchone()
            if not home_row:
                print(f"  ⚠️  Home team not found: {game['home']}")
                continue
            
            cur.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (game['away'],))
            away_row = cur.fetchone()
            if not away_row:
                print(f"  ⚠️  Away team not found: {game['away']}")
                continue
            
            home_id = home_row[0]
            away_id = away_row[0]
            
            # Update game by team matchup and date
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
                updated += 1
                print(f"  ✅ {game['away']} @ {game['home']}: {game['pst_display']}")
            else:
                print(f"  ⚠️  No matching game in DB: {game['away']} @ {game['home']}")
        
        except Exception as e:
            print(f"  ❌ Error updating {game['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n{'='*80}")
    print(f"✅ GAME TIMES UPDATED!")
    print(f"{'='*80}")
    print(f"   Updated {updated}/{len(espn_games)} games")
    print(f"{'='*80}\n")
    
    # Verify
    print("🔍 Verifying...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT s.game_id, 
               away.abbreviation, home.abbreviation,
               s.game_date, s.game_time
        FROM nba_schedule s
        JOIN teams home ON s.home_team_id = home.team_id
        JOIN teams away ON s.away_team_id = away.team_id
        WHERE s.game_date = %s
        ORDER BY s.game_time
    """, (date.today(),))
    
    rows = cur.fetchall()
    print(f"\nToday's games in DB:")
    for game_id, away, home, gdate, gtime in rows:
        if gtime:
            utc_dt = datetime.combine(gdate, gtime).replace(tzinfo=ZoneInfo('UTC'))
            pst_dt = utc_dt.astimezone(ZoneInfo('America/Los_Angeles'))
            print(f"  {away} @ {home}: {pst_dt.strftime('%I:%M %p PST')}")
        else:
            print(f"  {away} @ {home}: NO TIME")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()

