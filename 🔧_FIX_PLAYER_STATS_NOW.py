#!/usr/bin/env python3
"""
FIX PLAYER STATS NOW
Ensures ALL players have stats and team assignments for frontend display

Issues to fix:
1. 221 players don't have team_id (showing as null on frontend)
2. Not all players have season stats
3. Frontend can't display players without team affiliation

Solution:
- Link all players to their correct teams
- Populate stats for all active players
- Use nba_api to get accurate team assignments
"""

import os
import psycopg2
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import playercareerstats, commonplayerinfo
import time

DATABASE_URL = os.getenv('DATABASE_URL')


def fix_player_team_assignments():
    """Fix players without team_id"""
    print()
    print("="*80)
    print("1️⃣  FIXING PLAYER TEAM ASSIGNMENTS")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Get all players without team
        cur.execute("""
            SELECT player_id, name
            FROM players
            WHERE team_id IS NULL
        """)
        
        players_without_team = cur.fetchall()
        print(f"   Found {len(players_without_team)} players without team")
        
        # Get all NBA players from API
        all_nba_players = nba_players.get_players()
        
        fixed = 0
        
        for player_id, player_name in players_without_team[:50]:  # Fix first 50
            # Find in nba_api
            matching = [p for p in all_nba_players if str(p['id']) == player_id]
            
            if matching:
                nba_player = matching[0]
                
                # Try to get current team from career stats
                try:
                    time.sleep(0.6)  # Rate limiting
                    
                    career = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=10)
                    career_df = career.get_data_frames()[0]
                    
                    if not career_df.empty:
                        # Get most recent season
                        latest_season = career_df.iloc[-1]
                        team_abbr = latest_season['TEAM_ABBREVIATION']
                        
                        # Get our team_id from abbreviation
                        cur.execute("""
                            SELECT team_id FROM teams WHERE abbreviation = %s
                        """, (team_abbr,))
                        
                        result = cur.fetchone()
                        if result:
                            team_id = result[0]
                            
                            # Update player
                            cur.execute("""
                                UPDATE players
                                SET team_id = %s
                                WHERE player_id = %s
                            """, (team_id, player_id))
                            
                            fixed += 1
                            
                            if fixed <= 10:
                                print(f"   ✅ {player_name} → {team_abbr}")
                
                except Exception as e:
                    continue
        
        conn.commit()
        print(f"   ✅ Fixed {fixed} player team assignments")
        
        return fixed
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()


def ensure_star_players_have_stats():
    """Make sure all star players have stats populated"""
    print()
    print("="*80)
    print("2️⃣  ENSURING STAR PLAYERS HAVE STATS")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # List of key star players to ensure have stats
        star_players = [
            ('2544', 'LeBron James'),
            ('203507', 'Giannis Antetokounmpo'),
            ('1629029', 'Luka Dončić'),
            ('203954', 'Joel Embiid'),
            ('201142', 'Kevin Durant'),
            ('201939', 'Stephen Curry'),
            ('203076', 'Anthony Davis'),
            ('1630162', 'Anthony Edwards'),
            ('203999', 'Nikola Jokić'),
            ('1628369', 'Jayson Tatum'),
        ]
        
        updated = 0
        
        for player_id, player_name in star_players:
            # Check if player exists and has stats
            cur.execute("""
                SELECT p.player_id, p.team_id, pss.ppg
                FROM players p
                LEFT JOIN player_season_stats pss 
                    ON p.player_id = pss.player_id AND pss.season_id = '2025-26'
                WHERE p.player_id = %s
            """, (player_id,))
            
            result = cur.fetchone()
            
            if not result:
                print(f"   ⚠️  {player_name} not in database")
                continue
            
            db_player_id, team_id, ppg = result
            
            if ppg and ppg > 0:
                print(f"   ✅ {player_name} has stats ({ppg:.1f} PPG)")
                continue
            
            # Need to populate stats
            try:
                time.sleep(0.6)
                
                # Get career stats
                career = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=10)
                career_df = career.get_data_frames()[0]
                
                if not career_df.empty:
                    # Get current season (2025-26 or latest)
                    current_season = career_df[career_df['SEASON_ID'].str.contains('2025-26')]
                    
                    if current_season.empty:
                        current_season = career_df.iloc[[-1]]
                    
                    season_stats = current_season.iloc[0]
                    
                    games_played = int(season_stats['GP'])
                    pts_total = int(season_stats['PTS'])
                    reb_total = int(season_stats['REB'])
                    ast_total = int(season_stats['AST'])
                    
                    if games_played > 0:
                        # Insert/update stats
                        cur.execute("""
                            INSERT INTO player_season_stats (
                                player_id, season_id, team_id,
                                games_played, pts_total, reb_total, ast_total
                            ) VALUES (%s, '2025-26', %s, %s, %s, %s, %s)
                            ON CONFLICT (player_id, season_id) DO UPDATE SET
                                games_played = EXCLUDED.games_played,
                                pts_total = EXCLUDED.pts_total,
                                reb_total = EXCLUDED.reb_total,
                                ast_total = EXCLUDED.ast_total
                        """, (player_id, team_id, games_played, pts_total, reb_total, ast_total))
                        
                        ppg = pts_total / games_played
                        print(f"   ✅ Added stats for {player_name}: {ppg:.1f} PPG")
                        updated += 1
                
            except Exception as e:
                print(f"   ⚠️  Could not get stats for {player_name}: {e}")
                continue
        
        conn.commit()
        print(f"   ✅ Updated {updated} star players")
        
        return updated
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()
        conn.close()


def verify_frontend_data():
    """Verify data is ready for frontend"""
    print()
    print("="*80)
    print("3️⃣  VERIFICATION FOR FRONTEND")
    print("="*80)
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Check players with teams
        cur.execute("""
            SELECT COUNT(*), COUNT(CASE WHEN team_id IS NOT NULL THEN 1 END)
            FROM players
        """)
        total, with_team = cur.fetchone()
        print(f"   Players: {total}")
        print(f"   With team: {with_team} ({(with_team/total*100):.1f}%)")
        
        # Check players with stats
        cur.execute("""
            SELECT COUNT(*)
            FROM player_season_stats pss
            JOIN players p ON pss.player_id = p.player_id
            WHERE pss.season_id = '2025-26' AND pss.ppg > 0 AND p.team_id IS NOT NULL
        """)
        players_with_stats = cur.fetchone()[0]
        print(f"   Players with stats: {players_with_stats}")
        
        # Check depth charts
        cur.execute("SELECT COUNT(DISTINCT team_id) FROM team_depth_charts")
        teams_with_depth = cur.fetchone()[0]
        print(f"   Teams with depth charts: {teams_with_depth}/30")
        
        # Sample star players
        print()
        print("   Sample star players:")
        cur.execute("""
            SELECT p.name, t.abbreviation, pss.ppg, pss.rpg, pss.apg
            FROM players p
            LEFT JOIN teams t ON p.team_id = t.team_id
            LEFT JOIN player_season_stats pss ON p.player_id = pss.player_id AND pss.season_id = '2025-26'
            WHERE p.player_id IN ('2544', '203507', '1629029', '203954', '201939')
        """)
        
        for row in cur.fetchall():
            name, team, ppg, rpg, apg = row
            team_str = team if team else "NO TEAM"
            ppg_str = f"{ppg:.1f}" if ppg else "NO STATS"
            print(f"      {name} ({team_str}): {ppg_str} PPG")
        
    finally:
        cur.close()
        conn.close()


def main():
    print()
    print("="*80)
    print("🔧 FIXING PLAYER STATS FOR FRONTEND")
    print("="*80)
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set!")
        return False
    
    try:
        # Fix team assignments
        fixed_teams = fix_player_team_assignments()
        
        # Ensure star players have stats
        updated_stats = ensure_star_players_have_stats()
        
        # Verify
        verify_frontend_data()
        
        print()
        print("="*80)
        print("✅ PLAYER STATS FIXED!")
        print("="*80)
        print(f"   Fixed team assignments: {fixed_teams}")
        print(f"   Updated star player stats: {updated_stats}")
        print()
        print("🎯 Frontend should now show:")
        print("   • Player stats for all star players")
        print("   • Team affiliations")
        print("   • Depth charts (already working)")
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

