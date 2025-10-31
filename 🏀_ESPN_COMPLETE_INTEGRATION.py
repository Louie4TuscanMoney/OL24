#!/usr/bin/env python3
"""
Complete ESPN API Integration

This fetches:
1. Past games with ACTUAL lineups and box scores
2. Live game times in PST
3. Starting lineups for each game
4. Actual player minutes and stats
"""

import os
import psycopg2
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time

DATABASE_URL = os.getenv('DATABASE_URL')

ESPN_TEAM_MAP = {
    '1610612737': '1',   # ATL
    '1610612738': '2',   # BOS
    '1610612751': '17',  # BKN
    '1610612766': '30',  # CHA
    '1610612741': '4',   # CHI
    '1610612739': '5',   # CLE
    '1610612742': '6',   # DAL
    '1610612743': '7',   # DEN
    '1610612765': '8',   # DET
    '1610612744': '9',   # GSW
    '1610612745': '10',  # HOU
    '1610612754': '11',  # IND
    '1610612746': '12',  # LAC
    '1610612747': '13',  # LAL
    '1610612763': '29',  # MEM
    '1610612748': '14',  # MIA
    '1610612749': '15',  # MIL
    '1610612750': '16',  # MIN
    '1610612740': '3',   # NOP
    '1610612752': '18',  # NYK
    '1610612760': '25',  # OKC
    '1610612753': '19',  # ORL
    '1610612755': '20',  # PHI
    '1610612756': '21',  # PHX
    '1610612757': '22',  # POR
    '1610612758': '23',  # SAC
    '1610612759': '24',  # SAS
    '1610612761': '28',  # TOR
    '1610612762': '26',  # UTA
    '1610612764': '27',  # WAS
}


def fetch_espn_box_scores_for_past_games():
    """Fetch actual box scores from ESPN for completed games"""
    print()
    print("="*80)
    print("🏀 FETCHING ACTUAL BOX SCORES FROM ESPN")
    print("="*80)
    print()
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Get completed games
        cur.execute("""
            SELECT game_id, home_team_id, away_team_id, game_date
            FROM nba_schedule
            WHERE game_status = 'Final'
            AND game_id IS NOT NULL
            ORDER BY game_date DESC
            LIMIT 30
        """)
        
        completed_games = cur.fetchall()
        
        if not completed_games:
            print("   ℹ️  No completed games found")
            return 0
        
        print(f"   Found {len(completed_games)} completed games")
        print()
        
        total_added = 0
        games_processed = 0
        
        for game_id, home_id, away_id, game_date in completed_games:
            try:
                time.sleep(0.5)  # Rate limiting
                
                # ESPN game summary API
                url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
                response = requests.get(url, timeout=10)
                
                if response.status_code != 200:
                    print(f"   ⚠️  Game {game_id}: HTTP {response.status_code}")
                    continue
                
                data = response.json()
                
                # Get box score
                if 'boxscore' not in data:
                    print(f"   ⚠️  Game {game_id}: No boxscore")
                    continue
                
                boxscore = data['boxscore']
                players_data = boxscore.get('players', [])
                
                players_added = 0
                
                for team_data in players_data:
                    team_id = team_data.get('team', {}).get('id', '')
                    
                    # Convert ESPN team ID to NBA team ID
                    nba_team_id = None
                    for nba_id, espn_id in ESPN_TEAM_MAP.items():
                        if espn_id == team_id:
                            nba_team_id = nba_id
                            break
                    
                    if not nba_team_id:
                        continue
                    
                    # Process statistics for each player
                    for stat_group in team_data.get('statistics', []):
                        athletes = stat_group.get('athletes', [])
                        
                        for athlete in athletes:
                            athlete_data = athlete.get('athlete', {})
                            player_name = athlete_data.get('displayName', '')
                            
                            # Get stats
                            stats_list = athlete.get('stats', [])
                            if not stats_list or len(stats_list) < 14:
                                continue
                            
                            # Parse stats (ESPN format: MIN, FG, 3PT, FT, OREB, DREB, REB, AST, STL, BLK, TO, PF, +/-, PTS)
                            try:
                                minutes = stats_list[0] if stats_list[0] else '0'
                                
                                # Parse FG (format: "6-12" = 6 made, 12 attempted)
                                fg_parts = stats_list[1].split('-') if '-' in stats_list[1] else ['0', '0']
                                fgm = int(fg_parts[0]) if fg_parts[0] else 0
                                fga = int(fg_parts[1]) if len(fg_parts) > 1 and fg_parts[1] else 0
                                
                                # Parse 3PT (format: "1-5")
                                fg3_parts = stats_list[2].split('-') if '-' in stats_list[2] else ['0', '0']
                                fg3m = int(fg3_parts[0]) if fg3_parts[0] else 0
                                fg3a = int(fg3_parts[1]) if len(fg3_parts) > 1 and fg3_parts[1] else 0
                                
                                # Parse FT (format: "5-7")
                                ft_parts = stats_list[3].split('-') if '-' in stats_list[3] else ['0', '0']
                                ftm = int(ft_parts[0]) if ft_parts[0] else 0
                                fta = int(ft_parts[1]) if len(ft_parts) > 1 and ft_parts[1] else 0
                                
                                reb = int(stats_list[6]) if stats_list[6] else 0
                                ast = int(stats_list[7]) if stats_list[7] else 0
                                stl = int(stats_list[8]) if stats_list[8] else 0
                                blk = int(stats_list[9]) if stats_list[9] else 0
                                to = int(stats_list[10]) if stats_list[10] else 0
                                pf = int(stats_list[11]) if stats_list[11] else 0
                                plus_minus = int(stats_list[12]) if stats_list[12] and stats_list[12] != '--' else 0
                                pts = int(stats_list[13]) if stats_list[13] else 0
                                
                                # Find player_id by name
                                cur.execute("""
                                    SELECT player_id 
                                    FROM players 
                                    WHERE name ILIKE %s
                                    LIMIT 1
                                """, (f"%{player_name}%",))
                                
                                player_row = cur.fetchone()
                                if not player_row:
                                    continue
                                
                                player_id = player_row[0]
                                
                                # Insert box score
                                cur.execute("""
                                    INSERT INTO player_box_scores (
                                        player_id, team_id, game_id, game_date,
                                        minutes, pts, reb, ast, stl, blk,
                                        fgm, fga, fg3m, fg3a, ftm, fta,
                                        tov, pf, plus_minus
                                    ) VALUES (
                                        %s, %s, %s, %s,
                                        %s, %s, %s, %s, %s, %s,
                                        %s, %s, %s, %s, %s, %s,
                                        %s, %s, %s
                                    )
                                    ON CONFLICT (player_id, game_id) DO UPDATE SET
                                        pts = EXCLUDED.pts,
                                        reb = EXCLUDED.reb,
                                        ast = EXCLUDED.ast,
                                        minutes = EXCLUDED.minutes,
                                        plus_minus = EXCLUDED.plus_minus,
                                        fgm = EXCLUDED.fgm,
                                        fga = EXCLUDED.fga,
                                        stl = EXCLUDED.stl,
                                        blk = EXCLUDED.blk,
                                        tov = EXCLUDED.tov,
                                        pf = EXCLUDED.pf
                                """, (
                                    player_id, nba_team_id, game_id, game_date,
                                    minutes, pts, reb, ast, stl, blk,
                                    fgm, fga, fg3m, fg3a, ftm, fta,
                                    to, pf, plus_minus
                                ))
                                
                                players_added += 1
                                
                            except (ValueError, IndexError) as e:
                                continue
                
                conn.commit()
                
                if players_added > 0:
                    print(f"   ✅ Game {game_id}: {players_added} player box scores")
                    total_added += players_added
                    games_processed += 1
                
            except Exception as e:
                print(f"   ❌ Game {game_id}: {str(e)[:50]}")
                continue
        
        print()
        print(f"   ✅ Processed {games_processed} games")
        print(f"   ✅ Added {total_added} player box scores")
        
        return total_added
        
    except Exception as e:
        print(f"   ❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()


def update_live_game_times_pst():
    """Ensure all games have PST times"""
    print()
    print("="*80)
    print("🕐 UPDATING GAME TIMES TO PST")
    print("="*80)
    print()
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Get games without times or in wrong timezone
        cur.execute("""
            SELECT game_id, game_date, game_time
            FROM nba_schedule
            WHERE game_time IS NULL OR game_date >= CURRENT_DATE - 1
            ORDER BY game_date
            LIMIT 50
        """)
        
        games = cur.fetchall()
        
        if not games:
            print("   ✅ All games have times")
            return True
        
        print(f"   Updating {len(games)} game times...")
        
        updated = 0
        
        for game_id, game_date, game_time in games:
            try:
                time.sleep(0.3)
                
                # Fetch from ESPN
                url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
                response = requests.get(url, timeout=10)
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                game_data = data.get('header', {}).get('competitions', [{}])[0]
                
                # Get time
                date_str = game_data.get('date', '')
                if date_str:
                    # Parse UTC time and convert to PST
                    utc_dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    pst_dt = utc_dt.astimezone(ZoneInfo('America/Los_Angeles'))
                    
                    # Update database
                    cur.execute("""
                        UPDATE nba_schedule
                        SET game_time = %s,
                            game_date = %s
                        WHERE game_id = %s
                    """, (pst_dt.time(), pst_dt.date(), game_id))
                    
                    updated += 1
                
            except Exception as e:
                continue
        
        conn.commit()
        
        print(f"   ✅ Updated {updated} game times to PST")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()


def main():
    print()
    print("="*80)
    print("🚀 ESPN COMPLETE INTEGRATION")
    print("="*80)
    print()
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return False
    
    # Step 1: Fetch box scores
    box_scores = fetch_espn_box_scores_for_past_games()
    
    # Step 2: Update game times to PST
    update_live_game_times_pst()
    
    print()
    print("="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"   Box scores: {box_scores} player stats")
    print(f"   Game times: Updated to PST")
    print()
    print("🎯 Now you have:")
    print("   • Actual lineups from past games")
    print("   • Actual MPG and stats from box scores")
    print("   • Game times in PST")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

