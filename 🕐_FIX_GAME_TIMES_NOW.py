#!/usr/bin/env python3
"""
FIX GAME TIMES - Update schedule with correct PST times from ESPN

Problem: Database has wrong times (00:00 instead of 23:00)
Solution: Fetch from ESPN and update database with correct UTC times
"""

import os
import requests
import psycopg2
from datetime import datetime
from zoneinfo import ZoneInfo

DATABASE_URL = os.getenv('DATABASE_URL')

def main():
    print("\n" + "="*80)
    print("🕐 FIXING GAME TIMES FROM ESPN API")
    print("="*80 + "\n")
    
    # 1. Fetch games from ESPN
    print("📡 Fetching today's games from ESPN...")
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    response = requests.get(url, timeout=10)
    data = response.json()
    
    games_to_update = []
    for event in data.get('events', []):
        espn_game_id = event.get('id', '')
        date_str = event.get('date', '')  # ISO format UTC
        name = event.get('name', '')
        
        try:
            # Parse UTC time from ESPN
            utc_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            pst_time = utc_time.astimezone(ZoneInfo('America/Los_Angeles'))
            
            games_to_update.append({
                'espn_id': espn_game_id,
                'utc_datetime': utc_time,
                'pst_display': pst_time.strftime('%I:%M %p PST'),
                'name': name
            })
            
            print(f"  ✅ {name}")
            print(f"     ESPN ID: {espn_game_id}")
            print(f"     PST: {pst_time.strftime('%I:%M %p PST')}")
            
        except Exception as e:
            print(f"  ❌ Error parsing {name}: {e}")
    
    print(f"\n✅ Found {len(games_to_update)} games\n")
    
    # 2. Update database
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return
    
    print("💾 Updating database...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    updated = 0
    for game in games_to_update:
        try:
            # Update by ESPN game_id (stored as string in our DB)
            cur.execute("""
                UPDATE nba_schedule
                SET game_date = %s,
                    game_time = %s,
                    updated_at = NOW()
                WHERE game_id = %s
            """, (
                game['utc_datetime'].date(),
                game['utc_datetime'].time(),
                game['espn_id']
            ))
            
            if cur.rowcount > 0:
                updated += 1
                print(f"  ✅ Updated {game['name']}: {game['pst_display']}")
            else:
                print(f"  ⚠️  Game not found in DB: {game['espn_id']}")
        
        except Exception as e:
            print(f"  ❌ Error updating {game['name']}: {e}")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n{'='*80}")
    print(f"✅ GAME TIMES FIXED!")
    print(f"{'='*80}")
    print(f"   Updated {updated}/{len(games_to_update)} games")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

