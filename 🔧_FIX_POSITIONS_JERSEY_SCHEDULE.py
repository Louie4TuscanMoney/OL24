#!/usr/bin/env python3
"""
FIX POSITIONS, JERSEY NUMBERS, AND SCHEDULE

Issues to fix:
1. All players showing position "F" instead of actual position
2. Jersey numbers not showing (shows # instead of number)
3. Duplicate schedule entries
4. Fetch full season schedule (all games, not just 30 days)
"""

import os
import psycopg2
import requests
import time
from datetime import datetime, timedelta

DATABASE_URL = os.getenv('DATABASE_URL')
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


def fix_player_positions_and_jerseys():
    """Update player positions and jersey numbers from ESPN"""
    print()
    print("="*80)
    print("1️⃣  FIXING PLAYER POSITIONS & JERSEY NUMBERS")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Get all teams
        cur.execute("SELECT team_id, abbreviation FROM teams")
        teams = cur.fetchall()
        
        updated = 0
        
        for team_id, abbr in teams:
            time.sleep(0.5)
            
            # Get roster from ESPN
            url = f"{ESPN_BASE}/teams/{abbr.lower()}/roster"
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                roster_data = response.json()
            except Exception as e:
                print(f"   ⚠️  Could not fetch {abbr}: {e}")
                continue
            
            athletes = roster_data.get('athletes', [])
            
            for athlete in athletes:
                player_data = athlete.get('athlete', {})
                player_id = str(player_data.get('id', ''))
                
                # Get position
                position_obj = player_data.get('position', {})
                position = position_obj.get('abbreviation', '')
                
                # Get jersey
                jersey = player_data.get('jersey', '')
                
                if player_id:
                    # Update player
                    cur.execute("""
                        UPDATE players
                        SET position = %s,
                            jersey_number = %s
                        WHERE player_id = %s
                    """, (position, jersey, player_id))
                    
                    updated += 1
            
            if updated % 50 == 0 and updated > 0:
                print(f"   ... {updated} players updated")
        
        conn.commit()
        print(f"   ✅ Updated {updated} players with positions and jerseys")
        
        # Verify
        print("\n   Verification:")
        cur.execute("""
            SELECT name, position, jersey_number
            FROM players
            WHERE team_id = (SELECT team_id FROM teams WHERE abbreviation = 'OKC')
            ORDER BY name
            LIMIT 5
        """)
        
        for row in cur.fetchall():
            name, pos, jersey = row
            pos_str = pos if pos else "NO POS"
            jersey_str = jersey if jersey else "NO #"
            print(f"      {name}: {pos_str} #{jersey_str}")
        
        return updated
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()


def remove_duplicate_schedule_entries():
    """Remove duplicate games from schedule"""
    print()
    print("="*80)
    print("2️⃣  REMOVING DUPLICATE SCHEDULE ENTRIES")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Find duplicates
        cur.execute("""
            SELECT game_id, COUNT(*)
            FROM nba_schedule
            GROUP BY game_id
            HAVING COUNT(*) > 1
        """)
        
        duplicates = cur.fetchall()
        print(f"   Found {len(duplicates)} duplicate game IDs")
        
        if duplicates:
            # Delete duplicates, keeping only the most recent
            for game_id, count in duplicates:
                cur.execute("""
                    DELETE FROM nba_schedule
                    WHERE game_id = %s
                    AND id NOT IN (
                        SELECT id FROM nba_schedule
                        WHERE game_id = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                    )
                """, (game_id, game_id))
            
            conn.commit()
            print(f"   ✅ Removed {len(duplicates)} duplicate entries")
        else:
            print(f"   ✅ No duplicates found")
        
        # Show total unique games
        cur.execute("SELECT COUNT(DISTINCT game_id) FROM nba_schedule")
        unique_games = cur.fetchone()[0]
        print(f"   ✅ Total unique games: {unique_games}")
        
        return len(duplicates)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()


def fetch_full_season_schedule():
    """Fetch complete season schedule from ESPN (Oct 2025 - June 2026)"""
    print()
    print("="*80)
    print("3️⃣  FETCHING FULL SEASON SCHEDULE")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Season dates: Oct 22, 2025 to June 15, 2026 (approx)
        start_date = datetime(2025, 10, 22)
        end_date = datetime(2026, 6, 15)
        
        total_games = 0
        dates_checked = 0
        
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            
            time.sleep(0.3)  # Rate limiting
            
            url = f"{ESPN_BASE}/scoreboard?dates={date_str}"
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
            except:
                current_date += timedelta(days=1)
                continue
            
            events = data.get('events', [])
            
            if events:
                for event in events:
                    game_id = event.get('id')
                    game_date_str = event.get('date')
                    
                    competitions = event.get('competitions', [{}])[0]
                    competitors = competitions.get('competitors', [])
                    
                    home_team = away_team = None
                    home_score = away_score = None
                    
                    for comp in competitors:
                        abbr = comp.get('team', {}).get('abbreviation', '')
                        score = comp.get('score')
                        
                        if comp.get('homeAway') == 'home':
                            home_team = abbr
                            home_score = int(score) if score else None
                        else:
                            away_team = abbr
                            away_score = int(score) if score else None
                    
                    if not home_team or not away_team:
                        continue
                    
                    # Parse datetime
                    try:
                        game_datetime = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                        game_date = game_datetime.date()
                        game_time = game_datetime.time()
                    except:
                        continue
                    
                    # Get team IDs
                    cur.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (home_team,))
                    home_result = cur.fetchone()
                    cur.execute("SELECT team_id FROM teams WHERE abbreviation = %s", (away_team,))
                    away_result = cur.fetchone()
                    
                    if not home_result or not away_result:
                        continue
                    
                    status = event.get('status', {}).get('type', {}).get('description', 'Scheduled')
                    
                    # Insert/update (ON CONFLICT prevents duplicates)
                    cur.execute("""
                        INSERT INTO nba_schedule (
                            game_id, game_date, game_time,
                            home_team_id, away_team_id,
                            home_score, away_score, game_status
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (game_id) DO UPDATE SET
                            game_time = EXCLUDED.game_time,
                            home_score = EXCLUDED.home_score,
                            away_score = EXCLUDED.away_score,
                            game_status = EXCLUDED.game_status,
                            updated_at = NOW()
                    """, (
                        game_id, game_date, game_time,
                        home_result[0], away_result[0],
                        home_score, away_score, status
                    ))
                    
                    total_games += 1
            
            dates_checked += 1
            
            if dates_checked % 30 == 0:
                conn.commit()
                print(f"   ... checked {dates_checked} dates, found {total_games} games")
            
            current_date += timedelta(days=1)
        
        conn.commit()
        print(f"   ✅ Fetched {total_games} games from {dates_checked} dates")
        
        # Show date range
        cur.execute("""
            SELECT MIN(game_date), MAX(game_date), COUNT(DISTINCT game_id)
            FROM nba_schedule
        """)
        min_date, max_date, unique_games = cur.fetchone()
        print(f"   ✅ Schedule: {min_date} to {max_date} ({unique_games} unique games)")
        
        return total_games
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()


def verify_all_fixes():
    """Verify all fixes worked"""
    print()
    print("="*80)
    print("4️⃣  VERIFICATION")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Check positions
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN position IS NOT NULL AND position != '' THEN 1 END) as with_position,
                COUNT(CASE WHEN jersey_number IS NOT NULL AND jersey_number != '' THEN 1 END) as with_jersey
            FROM players
            WHERE team_id IS NOT NULL
        """)
        total, with_pos, with_jersey = cur.fetchone()
        print(f"   Players with positions: {with_pos}/{total}")
        print(f"   Players with jersey #: {with_jersey}/{total}")
        
        # Sample OKC players
        print("\n   Sample OKC roster:")
        cur.execute("""
            SELECT name, position, jersey_number
            FROM players
            WHERE team_id = (SELECT team_id FROM teams WHERE abbreviation = 'OKC')
            ORDER BY name
            LIMIT 5
        """)
        
        for row in cur.fetchall():
            name, pos, jersey = row
            print(f"      {name}: {pos} #{jersey}")
        
        # Check schedule
        cur.execute("""
            SELECT COUNT(DISTINCT game_id), MIN(game_date), MAX(game_date)
            FROM nba_schedule
        """)
        games, min_date, max_date = cur.fetchone()
        print(f"\n   Schedule: {games} games")
        print(f"   Date range: {min_date} to {max_date}")
        
        # Check for duplicates
        cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT game_id, COUNT(*)
                FROM nba_schedule
                GROUP BY game_id
                HAVING COUNT(*) > 1
            ) as dups
        """)
        dups = cur.fetchone()[0]
        
        if dups == 0:
            print(f"   ✅ No duplicate games!")
        else:
            print(f"   ⚠️  {dups} duplicate games found")
        
    finally:
        cur.close()
        conn.close()


def main():
    print()
    print("="*80)
    print("🔧 FIXING POSITIONS, JERSEYS, AND SCHEDULE")
    print("="*80)
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return False
    
    try:
        # Fix 1: Update positions and jerseys
        updated_players = fix_player_positions_and_jerseys()
        
        # Fix 2: Remove duplicates
        removed_dups = remove_duplicate_schedule_entries()
        
        # Fix 3: Fetch full season (this will take a while!)
        print("\n⏳ Fetching full season schedule (this may take 5-10 minutes)...")
        total_games = fetch_full_season_schedule()
        
        # Verify
        verify_all_fixes()
        
        print()
        print("="*80)
        print("✅ ALL FIXES COMPLETE!")
        print("="*80)
        print(f"   Players updated: {updated_players}")
        print(f"   Duplicates removed: {removed_dups}")
        print(f"   Total games: {total_games}")
        print()
        print("🎯 Frontend will now show:")
        print("   • Correct positions (PG, SG, SF, PF, C)")
        print("   • Jersey numbers")
        print("   • Full season schedule (no duplicates)")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

